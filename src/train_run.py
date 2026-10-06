import argparse
import copy
import json
import random
import subprocess
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn

from models import GRUModel, LSTMModel
from windows import get_device, load_split, make_loader


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_scaled(model, ds, device, batch_size=4096):
    model.eval()
    out = [model(xb.to(device)).cpu() for xb, _ in make_loader(ds, batch_size, shuffle=False)]
    return torch.cat(out).numpy()


def metrics_for(split, pred_scaled, data, y_scaler, kind, h, train_mean):
    """All metrics are computed in the original units (returns or prices), not scaled units."""
    ends = data[f"{split}_ends"]
    close = data[f"{split}_close"]
    y_true = data[f"{split}_y_raw"][ends]
    y_pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    last_close = close[ends]

    if kind == "return":
        baselines = {
            "zero": np.zeros_like(y_true),
            "train_mean": np.full_like(y_true, train_mean),
            "persistence": np.log(last_close / close[ends - h]),
        }
        primary = "zero"
        true_change, pred_change = y_true, y_pred
    else:
        baselines = {"last_close": last_close, "train_mean": np.full_like(y_true, train_mean)}
        primary = "last_close"
        true_change, pred_change = y_true - last_close, y_pred - last_close

    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    m = {"rmse": rmse(y_pred, y_true), "mae": float(np.mean(np.abs(y_pred - y_true)))}
    for name, b in baselines.items():
        m[f"baseline_{name}_rmse"] = rmse(b, y_true)
    m["rmse_vs_baseline"] = m["rmse"] / m[f"baseline_{primary}_rmse"]

    constant = np.std(pred_change) == 0 or np.std(true_change) == 0
    m["change_corr"] = 0.0 if constant else float(np.corrcoef(pred_change, true_change)[0, 1])

    nonzero = true_change != 0
    m["direction_acc"] = float(np.mean(np.sign(pred_change[nonzero]) == np.sign(true_change[nonzero])))
    up_share = float(np.mean(true_change[nonzero] > 0))
    m["direction_majority_rate"] = max(up_share, 1 - up_share)
    m["direction_edge"] = m["direction_acc"] - m["direction_majority_rate"]
    m["n_windows"] = int(len(y_true))

    arrays = {"time": data[f"{split}_time"][ends], "y_true": y_true, "y_pred": y_pred,
              "last_close": last_close, "true_change": true_change, "pred_change": pred_change}
    return m, arrays


def print_block(name, m):
    print(f"\n--- {name} ---")
    print(f"Model   : RMSE {m['rmse']:.6f} | MAE {m['mae']:.6f} | change correlation {m['change_corr']:+.3f}")
    base = [f"{k[9:-5]} {v:.6f}" for k, v in m.items() if k.startswith("baseline_") and k.endswith("_rmse")]
    print("Baseline RMSE:", " | ".join(base))
    print(f"RMSE / primary baseline: {m['rmse_vs_baseline']:.4f}  (1.0 = same as the baseline, lower is better)")
    print(f"Direction accuracy {m['direction_acc']:.3f} vs majority-class rate {m['direction_majority_rate']:.3f} "
          f"(edge {m['direction_edge']:+.3f})")


def plot_loss(history, best_epoch, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="train loss")
    ax.plot(epochs, history["val_loss"], label="validation MSE (scaled)")
    ax.axvline(best_epoch, linestyle="--", color="gray", label=f"best epoch {best_epoch}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (scaled units)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_predictions(a, title, path):
    k = min(300, len(a["y_true"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(a["y_true"][-k:], label="actual", linewidth=1)
    axes[0].plot(a["y_pred"][-k:], label="predicted", linewidth=1)
    axes[0].set_title(f"{title}: last {k} windows")
    axes[0].legend()
    axes[1].scatter(a["true_change"], a["pred_change"], s=3, alpha=0.3)
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].axvline(0, color="gray", linewidth=0.5)
    axes[1].set_xlabel("actual change")
    axes[1].set_ylabel("predicted change")
    axes[1].set_title(f"{title}: predicted vs actual change")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train_one(cfg):
    folder = Path(cfg["folder"])
    meta = json.load(open(folder / "meta.json"))
    pre = meta["config"]
    lookback, kind, horizon = pre["lookback"], pre["target"]["kind"], pre["target"]["horizon"]
    assert lookback > horizon, "lookback must be larger than the horizon"

    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    data = np.load(folder / "data.npz")
    y_scaler = joblib.load(folder / "y_scaler.joblib")
    train_ds = load_split(folder, "train", lookback)
    val_ds = load_split(folder, "val", lookback)
    val_y = val_ds.y[val_ds.ends].numpy()
    train_mean = float(data["train_y_raw"][data["train_ends"]].mean())

    model_class = GRUModel if cfg["model"] == "gru" else LSTMModel
    model = model_class(input_size=train_ds.X.shape[1], hidden_size=cfg["hidden_size"],
                        num_layers=cfg["num_layers"], dropout=cfg["dropout"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    train_loss_fn = nn.MSELoss() if cfg["loss"] == "mse" else nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    if mlflow.active_run() is None:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(cfg["experiment"])

    run_name = (f"{cfg['model']}_{pre['name']}_h{cfg['hidden_size']}_l{cfg['num_layers']}_d{cfg['dropout']}"
                f"_lr{cfg['lr']}_bs{cfg['batch_size']}_s{cfg['seed']}")

    with mlflow.start_run(run_name=run_name, nested=mlflow.active_run() is not None) as run:
        out_dir = Path("runs") / run.info.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tags({"mode": cfg["mode"], "model_type": cfg["model"], "dataset": pre["name"],
                                                  "data_folder": folder.name, "git_commit": git_commit()})
        if cfg.get("run_key"):
            mlflow.set_tag("run_key", cfg["run_key"])
        mlflow.log_params({
            "model_type": cfg["model"], "mode": cfg["mode"], "hidden_size": cfg["hidden_size"],
            "num_layers": cfg["num_layers"], "dropout": cfg["dropout"], "learning_rate": cfg["lr"],
            "batch_size": cfg["batch_size"], "max_epochs": cfg["epochs"], "patience": cfg["patience"],
            "grad_clip": cfg["grad_clip"], "loss": cfg["loss"], "seed": cfg["seed"], "device": str(device),
            "n_parameters": n_params, "torch_version": torch.__version__,
            "dataset": pre["name"], "features": str(pre["features"]), "n_features": train_ds.X.shape[1],
            "scaler": pre["scaler"], "lookback": lookback, "target_kind": kind, "horizon": horizon,
            "max_gap": str(pre["data"].get("max_gap")), "raw_file_md5": meta["raw_file_md5"],
            "preprocess_commit": meta["git_commit"], "windows_train": meta["splits"]["train"]["windows"],
            "windows_val": meta["splits"]["val"]["windows"], "windows_test": meta["splits"]["test"]["windows"],
        })
        print(f"Run: {run_name} | device {device} | parameters {n_params} | mode {cfg['mode']}")

        # ---------- Training with early stopping on validation MSE ----------
        history = {"train_loss": [], "val_loss": []}
        best_val, best_state, best_epoch, bad_epochs = float("inf"), None, 0, 0
        start = time.perf_counter()
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            total, count = 0.0, 0
            for xb, yb in make_loader(train_ds, cfg["batch_size"], True, seed=cfg["seed"] + epoch):
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = train_loss_fn(model(xb), yb)
                loss.backward()
                if cfg["grad_clip"] > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()
                total += loss.item() * len(xb)
                count += len(xb)
            train_loss = total / count
            val_loss = float(np.mean((predict_scaled(model, val_ds, device) - val_y) ** 2))
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            print(f"Epoch {epoch:2d} | train loss {train_loss:.4f} | val MSE {val_loss:.4f} "
                  f"| {time.perf_counter() - start:.0f}s elapsed")

            if val_loss < best_val:
                best_val, best_epoch, bad_epochs = val_loss, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad_epochs += 1
                if bad_epochs >= cfg["patience"]:
                    print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, val MSE {best_val:.4f})")
                    break

        train_seconds = time.perf_counter() - start
        epochs_run = len(history["val_loss"])
        model.load_state_dict(best_state)
        mlflow.log_metrics({"best_val_loss": best_val, "best_epoch": best_epoch, "epochs_run": epochs_run,
                            "train_seconds": train_seconds, "seconds_per_epoch": train_seconds / epochs_run})

        # ---------- Evaluation: validation always, test ONLY in final mode ----------
        results = {}
        for split in ["val"] + (["test"] if cfg["mode"] == "final" else []):
            ds = val_ds if split == "val" else load_split(folder, "test", lookback)
            pred = predict_scaled(model, ds, device)
            m, arrays = metrics_for(split, pred, data, y_scaler, kind, horizon, train_mean)
            results[split] = m
            print_block(split.upper(), m)
            mlflow.log_metrics({f"{split}_{k}": v for k, v in m.items()})
            np.savez(out_dir / f"predictions_{split}.npz", **arrays)
            plot_predictions(arrays, split, out_dir / f"pred_vs_actual_{split}.png")

        # ---------- Artifacts ----------
        plot_loss(history, best_epoch, out_dir / "loss_curve.png")
        torch.save(best_state, out_dir / "model_state.pt")
        with open(out_dir / "config.json", "w") as f:
            json.dump({"train": cfg, "preprocess": pre, "git_commit": git_commit()}, f, indent=2, default=str)
        mlflow.log_artifacts(str(out_dir))
        mlflow.log_artifact(str(folder / "meta.json"), "data")

        if cfg["mode"] == "final" and cfg["log_model"]:
            try:
                example = val_ds[[0, 1]][0].numpy()
                cpu_model = model.to("cpu").eval()
                try:
                    mlflow.pytorch.log_model(cpu_model, name="model", input_example=example)
                except TypeError:
                    mlflow.pytorch.log_model(cpu_model, artifact_path="model", input_example=example)
            except Exception as error:
                print("Warning: could not log the MLflow model flavor:", error)

        print(f"\nRun id: {run.info.run_id} | files in {out_dir} | train time {train_seconds:.0f}s")
        return {"run_id": run.info.run_id, "results": results, "best_epoch": best_epoch,
                "epochs_run": epochs_run, "train_seconds": train_seconds}


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", required=True, help="a folder inside data/processed")
    p.add_argument("--model", choices=["gru", "lstm"], default="gru")
    p.add_argument("--mode", choices=["tune", "final"], default="tune",
                   help="tune = validation only; final = also evaluate the test set (once)")
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--loss", choices=["mse", "huber"], default="mse")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--experiment", default="forex-v2")
    p.add_argument("--log_model", action="store_true", help="also log the model to MLflow (final mode)")
    return p


def parse_args():
    return build_parser().parse_args()


if __name__ == "__main__":
    train_one(vars(parse_args()))