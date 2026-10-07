import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default="m1-final")
parser.add_argument("--n_boot", type=int, default=2000)
parser.add_argument("--blocks", default="60,240,1440", help="block lengths in rows (minutes)")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

mlflow.set_tracking_uri("sqlite:///mlflow.db")
runs = mlflow.search_runs(experiment_names=[args.experiment])
runs = runs[(runs["status"] == "FINISHED") & runs["tags.level"].isna()]

# ---------- Load the saved test predictions of every seed ----------
preds, y, times = {"gru": [], "lstm": []}, None, None
for _, r in runs.iterrows():
    d = np.load(Path("runs") / r["run_id"] / "predictions_test.npz")
    if y is None:
        y, times = d["y_true"], d["time"]
    assert np.array_equal(y, d["y_true"]), "runs disagree about the test labels"
    preds[r["params.model_type"]].append(d["y_pred"])
n = len(y)
print(f"Test windows: {n} | period {pd.to_datetime(times[0])} to {pd.to_datetime(times[-1])}")
print(f"Seeds loaded: GRU {len(preds['gru'])}, LSTM {len(preds['lstm'])}")

zero_mse = np.mean(y ** 2)
for m in ("gru", "lstm"):
    singles = [np.sqrt(np.mean((p - y) ** 2) / zero_mse) for p in preds[m]]
    print(f"Check - mean of the individual-seed test ratios, {m.upper()}: {np.mean(singles):.5f}")
ens = {m: np.mean(preds[m], axis=0) for m in preds}


def row_corr(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))


def stats_for(yy, pg, pl):
    """yy, pg, pl have shape (replicates, n)."""
    base = np.mean(yy ** 2, axis=1)
    rg = np.sqrt(np.mean((pg - yy) ** 2, axis=1) / base)
    rl = np.sqrt(np.mean((pl - yy) ** 2, axis=1) / base)
    return {"GRU ratio": rg, "LSTM ratio": rl, "GRU - LSTM ratio": rg - rl,
            "GRU correlation": row_corr(pg, yy), "LSTM correlation": row_corr(pl, yy)}


point = stats_for(y[None], ens["gru"][None], ens["lstm"][None])
print("\n--- Seed-ensemble point estimates on the full test set ---")
for k, v in point.items():
    print(f"{k:18s}: {v[0]:+.5f}")

rng = np.random.default_rng(args.seed)
for block in [int(b) for b in args.blocks.split(",")]:
    k = int(np.ceil(n / block))
    results = {name: [] for name in point}
    for _ in range(args.n_boot // 100):
        starts = rng.integers(0, n - block + 1, size=(100, k))
        idx = (starts[:, :, None] + np.arange(block)).reshape(100, -1)[:, :n]
        out = stats_for(y[idx], ens["gru"][idx], ens["lstm"][idx])
        for name, v in out.items():
            results[name].append(v)
    results = {name: np.concatenate(v) for name, v in results.items()}

    print(f"\n=== Block length {block} minutes ({k} blocks per resample, {len(results['GRU ratio'])} resamples) ===")
    for name, v in results.items():
        low, high = np.percentile(v, [2.5, 97.5])
        print(f"{name:18s}: 95% interval [{low:+.5f}, {high:+.5f}]")
    print(f"Share of resamples where the GRU ratio is below 1.0 (beats the baseline) : {np.mean(results['GRU ratio'] < 1):.3f}")
    print(f"Share of resamples where the LSTM ratio is below 1.0 (beats the baseline): {np.mean(results['LSTM ratio'] < 1):.3f}")
    print(f"Share of resamples where the GRU beats the LSTM                          : {np.mean(results['GRU - LSTM ratio'] < 0):.3f}")