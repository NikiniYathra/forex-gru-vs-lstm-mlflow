import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", required=True, help="e.g. m1-tune")
args = parser.parse_args()

mlflow.set_tracking_uri("sqlite:///mlflow.db")
runs = mlflow.search_runs(experiment_names=[args.experiment])
runs = runs[(runs["status"] == "FINISHED") & runs["tags.level"].isna()]   # training runs only, not the summary parents

cols = {
    "params.model_type": "model", "params.hidden_size": "hidden", "params.num_layers": "layers",
    "params.dropout": "dropout", "params.learning_rate": "lr", "params.batch_size": "batch",
    "params.seed": "seed", "params.n_parameters": "params",
    "metrics.val_rmse_vs_baseline": "val_ratio", "metrics.val_direction_edge": "val_edge",
    "metrics.val_change_corr": "val_corr", "metrics.best_epoch": "best_epoch",
    "metrics.train_seconds": "train_seconds",
}
df = runs[list(cols)].rename(columns=cols)
for c in ["hidden", "layers", "dropout", "lr", "batch", "seed", "params"]:
    df[c] = pd.to_numeric(df[c])

group = ["model", "hidden", "layers", "dropout", "lr", "batch"]
agg = df.groupby(group).agg(
    n_seeds=("seed", "count"), params=("params", "first"),
    ratio_mean=("val_ratio", "mean"), ratio_std=("val_ratio", "std"),
    edge_mean=("val_edge", "mean"), edge_std=("val_edge", "std"),
    corr_mean=("val_corr", "mean"), epochs=("best_epoch", "mean"),
    minutes=("train_seconds", lambda s: s.mean() / 60),
).reset_index().fillna(0.0)

print(f"Experiment: {args.experiment} | training runs: {len(df)} | configurations: {len(agg)}\n")
for _, r in agg.sort_values(["model", "ratio_mean"]).iterrows():
    print(f"{r['model']:4s} h{int(r['hidden']):<3d} L{int(r['layers'])} d{r['dropout']} lr{r['lr']:<7} bs{int(r['batch']):<4d}"
          f" | seeds {int(r['n_seeds'])} | ratio {r['ratio_mean']:.4f} +/- {r['ratio_std']:.4f}"
          f" | edge {r['edge_mean']:+.4f} | corr {r['corr_mean']:+.3f} | best epoch {r['epochs']:.1f} | {r['minutes']:.1f} min")

# ---------- Pre-declared selection rule (validation only) ----------
agg["ratio_r"] = agg["ratio_mean"].round(4)
best = {}
print("\n--- Selected by validation (lowest mean ratio; ties broken by direction edge) ---")
for model, g in agg.groupby("model"):
    pick = g.sort_values(["ratio_r", "edge_mean"], ascending=[True, False]).iloc[0]
    best[model] = {"model": model, "hidden_size": int(pick["hidden"]), "num_layers": int(pick["layers"]),
                   "dropout": float(pick["dropout"]), "lr": float(pick["lr"]), "batch_size": int(pick["batch"])}
    print(f"{model}: {best[model]} | validation ratio {pick['ratio_mean']:.4f}")

agg.drop(columns="ratio_r").to_csv(f"reports/{args.experiment}_summary.csv", index=False)
with open(f"reports/{args.experiment}_best.json", "w") as f:
    json.dump({"experiment": args.experiment, "best": best}, f, indent=2)

# ---------- Chart: mean +/- std across seeds ----------
agg = agg.sort_values(["model", "hidden", "lr"]).reset_index(drop=True)
labels = [f"{r.model}\nh{int(r.hidden)}\nlr{r.lr}" for r in agg.itertuples()]
colors = ["tab:blue" if m == "gru" else "tab:orange" for m in agg["model"]]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
axes[0].bar(range(len(agg)), agg["ratio_mean"], yerr=agg["ratio_std"], color=colors, alpha=0.6, capsize=3)
axes[0].axhline(1.0, color="red", linestyle="--", linewidth=1)
lo = min(agg["ratio_mean"].min() - agg["ratio_std"].max(), 1.0) - 0.0005
hi = max(agg["ratio_mean"].max() + agg["ratio_std"].max(), 1.0) + 0.0005
axes[0].set_ylim(lo, hi)
axes[0].set_title("Validation RMSE / baseline (lower is better, red = predict zero)", fontsize=9)
axes[1].bar(range(len(agg)), agg["edge_mean"], yerr=agg["edge_std"], color=colors, alpha=0.6, capsize=3)
axes[1].axhline(0.0, color="red", linestyle="--", linewidth=1)
axes[1].set_title("Validation direction accuracy minus majority-class rate", fontsize=9)
for ax in axes:
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(labels, fontsize=7)
fig.suptitle(f"{args.experiment}: mean +/- std over seeds (blue = GRU, orange = LSTM)")
fig.tight_layout()
fig.savefig(f"reports/{args.experiment}_configs.png", dpi=130)
print(f"\nSaved reports/{args.experiment}_summary.csv, _best.json and _configs.png")