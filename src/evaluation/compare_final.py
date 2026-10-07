import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default="m1-final")
args = parser.parse_args()

mlflow.set_tracking_uri("sqlite:///mlflow.db")
runs = mlflow.search_runs(experiment_names=[args.experiment])
runs = runs[(runs["status"] == "FINISHED") & runs["tags.level"].isna()]

cols = {
    "params.model_type": "model", "params.seed": "seed", "params.n_parameters": "params",
    "metrics.val_rmse_vs_baseline": "val_ratio", "metrics.test_rmse_vs_baseline": "test_ratio",
    "metrics.val_change_corr": "val_corr", "metrics.test_change_corr": "test_corr",
    "metrics.val_direction_edge": "val_edge", "metrics.test_direction_edge": "test_edge",
    "metrics.best_epoch": "best_epoch", "metrics.train_seconds": "seconds",
}
df = runs[list(cols)].rename(columns=cols)
df["params"] = pd.to_numeric(df["params"])
df["minutes"] = df["seconds"] / 60
df.sort_values(["model", "seed"]).to_csv(f"reports/{args.experiment}_runs.csv", index=False)

gru, lstm = df[df["model"] == "gru"], df[df["model"] == "lstm"]
print(f"Experiment: {args.experiment} | GRU runs: {len(gru)} | LSTM runs: {len(lstm)}")
print(f"Parameters: GRU {int(gru['params'].iloc[0])} | LSTM {int(lstm['params'].iloc[0])}")

rows = [("val_ratio", "Validation RMSE / baseline (lower is better)"),
        ("test_ratio", "TEST RMSE / baseline (lower is better)"),
        ("val_corr", "Validation correlation, predicted vs actual change"),
        ("test_corr", "TEST correlation, predicted vs actual change"),
        ("test_edge", "TEST direction accuracy minus majority-class rate"),
        ("best_epoch", "Best epoch"),
        ("minutes", "Training minutes per run")]


def welch(a, b):
    diff = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    low, high = stats.t.interval(0.95, dof, loc=diff, scale=se)
    return diff, low, high, stats.ttest_ind(a, b, equal_var=False).pvalue


print("\n--- GRU vs LSTM, mean / std / std error across seeds ---")
for col, label in rows:
    g, l = gru[col], lstm[col]
    diff, low, high, p = welch(g, l)
    print(f"\n{label}")
    print(f"  GRU : {g.mean():+.5f} | std {g.std():.5f} | se {g.std() / np.sqrt(len(g)):.5f}")
    print(f"  LSTM: {l.mean():+.5f} | std {l.std():.5f} | se {l.std() / np.sqrt(len(l)):.5f}")
    print(f"  GRU - LSTM: {diff:+.5f} | 95% CI [{low:+.5f}, {high:+.5f}] | Welch p = {p:.3f}")

print("\n--- Seeds that beat the 'predict zero' baseline ---")
for name, part in [("GRU", gru), ("LSTM", lstm)]:
    print(f"{name}: validation {int((part['val_ratio'] < 1).sum())} of {len(part)} | "
          f"TEST {int((part['test_ratio'] < 1).sum())} of {len(part)}")

fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
panels = [("val_ratio", "Validation RMSE / baseline\n(lower is better)", 1.0),
          ("test_ratio", "TEST RMSE / baseline\n(lower is better)", 1.0),
          ("test_corr", "TEST correlation\n(predicted vs actual change)", 0.0),
          ("test_edge", "TEST direction accuracy\nminus majority-class rate", 0.0)]
colors = {"gru": "tab:blue", "lstm": "tab:orange"}
rng = np.random.default_rng(0)
for ax, (col, title, ref) in zip(axes, panels):
    for i, m in enumerate(["gru", "lstm"]):
        vals = df[df["model"] == m][col].to_numpy()
        ax.bar(i, vals.mean(), width=0.5, alpha=0.35, color=colors[m])
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, color="black", s=20, zorder=3)
    ax.axhline(ref, color="red", linestyle="--", linewidth=1)
    lo, hi = min(df[col].min(), ref), max(df[col].max(), ref)
    pad = (hi - lo) * 0.2
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["GRU", "LSTM"])
    ax.set_title(title, fontsize=9)
fig.suptitle(f"{args.experiment}: GRU vs LSTM, {len(gru)} seeds each (dots = seeds, bars = means, red = baseline)")
fig.tight_layout()
fig.savefig(f"reports/{args.experiment}_comparison.png", dpi=130)
print(f"\nSaved reports/{args.experiment}_runs.csv and reports/{args.experiment}_comparison.png")