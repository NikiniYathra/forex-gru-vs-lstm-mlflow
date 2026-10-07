import argparse
import json
import subprocess
from pathlib import Path

import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch

from src.data.windows import load_split
from src.models.networks import GRUModel, LSTMModel

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default="m1-final")
parser.add_argument("--seed", type=int, default=11, help="which seed's trained model to package")
args = parser.parse_args()

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(args.experiment)
runs = mlflow.search_runs(experiment_names=[args.experiment])
train_runs = runs[(runs["status"] == "FINISHED") & runs["tags.level"].isna()]


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_trained(model_type):
    row = train_runs[(train_runs["params.model_type"] == model_type)
                     & (train_runs["params.seed"] == str(args.seed))].iloc[0]
    run_id = row["run_id"]
    cfg = json.load(open(Path("runs") / run_id / "config.json"))
    t, pre = cfg["train"], cfg["preprocess"]
    ds = load_split(t["folder"], "val", pre["lookback"])
    cls = GRUModel if model_type == "gru" else LSTMModel
    model = cls(input_size=ds.X.shape[1], hidden_size=t["hidden_size"],
                num_layers=t["num_layers"], dropout=t["dropout"])
    model.load_state_dict(torch.load(Path("runs") / run_id / "model_state.pt", map_location="cpu"))
    model.eval()
    example = ds[[0, 1, 2, 3]][0].numpy()      # four real validation windows
    return run_id, model, example


NOTE = (
    "Final comparison of GRU and LSTM on EUR/USD 1-minute data (chronological 70/15/15 split, scalers fitted on "
    "train only, windows never cross a split or a time gap). Configurations were chosen on validation only and "
    "frozen in git before the test set was evaluated; 10 fresh seeds per model. Result: neither model beats the "
    "'predict zero' baseline on the test period (test RMSE ratio about 1.001 for both), and the GRU-vs-LSTM "
    "difference is not distinguishable from zero once test-period uncertainty is included (block bootstrap). "
    "The packaged models are examples (seed " + str(args.seed) + "), not deployment candidates."
)

exp = args.experiment
df = pd.read_csv(f"reports/{exp}_runs.csv")

with mlflow.start_run(run_name="REPORT_GRU_vs_LSTM_final") as report:
    mlflow.set_tags({"level": "report", "freeze_commit": "dc6d791", "git_commit": git_commit(),
                     "mlflow.note.content": NOTE})
    mlflow.log_params({"packaged_seed": args.seed, "split": "70/15/15 chronological", "lookback": 60,
                       "seeds": "11-20", "dataset": "EUR/USD 1-minute, 200,000 rows"})

    # ---------- Package each model with its code, reload it, and check the predictions ----------
    for m in ("gru", "lstm"):
        run_id, model, example = load_trained(m)
        with torch.no_grad():
            expected = model(torch.from_numpy(example)).numpy()
        try:
            info = mlflow.pytorch.log_model(model, name=f"{m}_model", input_example=example, code_paths=["src"])
        except TypeError:
            info = mlflow.pytorch.log_model(model, artifact_path=f"{m}_model", input_example=example, code_paths=["src"])
        loaded = mlflow.pyfunc.load_model(info.model_uri)
        got = np.asarray(loaded.predict(example)).ravel()
        assert np.allclose(expected, got, atol=1e-5), f"{m}: reloaded model gives different predictions"
        mlflow.set_tag(f"{m}_source_run", run_id)
        print(f"{m.upper()} model packaged from run {run_id[:8]}, reloaded, predictions match "
              f"(max difference {np.abs(expected - got).max():.2e})")

    # ---------- Summary metrics across the 10 seeds ----------
    for m in ("gru", "lstm"):
        part = df[df["model"] == m]
        for col in ("val_ratio", "test_ratio", "val_corr", "test_corr", "test_edge"):
            mlflow.log_metric(f"{m}_{col}_mean", float(part[col].mean()))
            mlflow.log_metric(f"{m}_{col}_std", float(part[col].std()))
        mlflow.log_metric(f"{m}_seeds_beating_baseline_on_test", int((part["test_ratio"] < 1).sum()))

    # ---------- Artifacts ----------
    for f in [f"reports/{exp}_comparison.png", f"reports/{exp}_runs.csv", f"reports/{exp}_bootstrap.txt",
              "reports/m1-tune_summary.csv", "reports/m1-tune_configs.png",
              "configs/experiments/m1_tune.yaml", "configs/experiments/m1_final_gru.yaml",
              "configs/experiments/m1_final_lstm.yaml", "configs/m1.yaml"]:
        if Path(f).exists():
            mlflow.log_artifact(f, "report")
        else:
            print("Not found, skipped:", f)

    print("\nReport run id:", report.info.run_id)