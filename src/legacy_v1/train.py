import argparse
import copy

import joblib
import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.networks import GRUModel, LSTMModel

# ---------- Settings (can be overridden from the command line) ----------
parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["gru", "lstm"], default="gru")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--hidden_size", type=int, default=32)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ---------- MLflow setup ----------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("forex-gru-vs-lstm")

# ---------- Load data ----------
data = np.load("data/processed/sequences.npz")
X_train, y_train = torch.from_numpy(data["X_train"]), torch.from_numpy(data["y_train"])
X_val, y_val = torch.from_numpy(data["X_val"]), torch.from_numpy(data["y_val"])
X_test, y_test = torch.from_numpy(data["X_test"]), torch.from_numpy(data["y_test"])
y_scaler = joblib.load("models/y_scaler.joblib")

train_loader = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=args.batch_size, shuffle=True)

# ---------- Model, loss, optimizer ----------
model_class = GRUModel if args.model == "gru" else LSTMModel
model = model_class(input_size=X_train.shape[2], hidden_size=args.hidden_size)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def predict(X):
    model.eval()
    with torch.no_grad():
        return model(X)

def to_returns(scaled):
    return y_scaler.inverse_transform(np.asarray(scaled).reshape(-1, 1)).ravel()

def metrics(X, y):
    pred, true = to_returns(predict(X).numpy()), to_returns(y.numpy())
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae = float(np.mean(np.abs(pred - true)))
    direction = float(np.mean(np.sign(pred) == np.sign(true)))
    base_rmse = float(np.sqrt(np.mean(true ** 2)))
    base_mae = float(np.mean(np.abs(true)))
    return rmse, mae, direction, base_rmse, base_mae, float(np.mean(true > 0))

# ---------- Training (inside an MLflow run) ----------
run_name = f"{args.model}_h{args.hidden_size}_lr{args.lr}_bs{args.batch_size}_seed{args.seed}"

with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_params({
        "model_type": args.model,
        "lookback": X_train.shape[1],
        "n_features": X_train.shape[2],
        "hidden_size": args.hidden_size,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
        "n_parameters": sum(p.numel() for p in model.parameters()),
    })

    best_val, best_state, bad_epochs = float("inf"), None, 0
    print(f"Training {args.model.upper()} | lr={args.lr} batch={args.batch_size} hidden={args.hidden_size} seed={args.seed}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        train_loss = total / len(X_train)
        val_loss = loss_fn(predict(X_val), y_val).item()
        print(f"Epoch {epoch:2d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

        if val_loss < best_val:
            best_val, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch} (best val loss {best_val:.4f})")
                break

    model.load_state_dict(best_state)
    weights_path = f"models/{args.model}.pt"
    torch.save(best_state, weights_path)

    mlflow.log_metric("epochs_run", epoch)
    mlflow.log_metric("best_val_loss", best_val)

    for name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        rmse, mae, direction, b_rmse, b_mae, up_share = metrics(X, y)
        print(f"\n--- {name.upper()} ---")
        print(f"Model    : RMSE {rmse:.5f} | MAE {mae:.5f} | directional accuracy {direction:.3f}")
        print(f"Baseline : RMSE {b_rmse:.5f} | MAE {b_mae:.5f} (always predict 0)")
        print(f"Share of up-days in this period: {up_share:.3f}")
        mlflow.log_metrics({
            f"{name}_rmse": rmse,
            f"{name}_mae": mae,
            f"{name}_direction_acc": direction,
            f"{name}_baseline_rmse": b_rmse,
            f"{name}_rmse_vs_baseline": rmse / b_rmse,
        })

    mlflow.log_artifact(weights_path)
    print(f"\nMLflow run saved: {run_name} (run id {run.info.run_id})")