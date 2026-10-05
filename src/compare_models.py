import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("reports/run_summary.csv")

# Keep only the chosen configuration
chosen = df[(df["hidden"] == 64) & np.isclose(df["lr"], 0.001) & (df["batch"] == 64)]
gru = chosen[chosen["model"] == "gru"]
lstm = chosen[chosen["model"] == "lstm"]
print(f"Runs per model: GRU {len(gru)}, LSTM {len(lstm)}")

metrics = [
    ("val_ratio", "Validation RMSE / baseline (lower is better)"),
    ("test_ratio", "Test RMSE / baseline (lower is better)"),
    ("test_dir", "Test directional accuracy"),
]

print("\n--- GRU vs LSTM (hidden 64, lr 0.001, batch 64) ---")
for col, label in metrics:
    g, l = gru[col], lstm[col]
    t, p = stats.ttest_ind(g, l, equal_var=False)   # Welch t-test
    print(f"\n{label}")
    print(f"  GRU : mean {g.mean():.4f} | std {g.std():.4f} | std error {g.std() / np.sqrt(len(g)):.4f}")
    print(f"  LSTM: mean {l.mean():.4f} | std {l.std():.4f} | std error {l.std() / np.sqrt(len(l)):.4f}")
    print(f"  Difference (GRU - LSTM): {g.mean() - l.mean():+.4f} | Welch t-test p-value: {p:.3f}")

# Chart: every seed as a dot, mean as a bar
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
rng = np.random.default_rng(0)
for ax, (col, label) in zip(axes, metrics):
    for i, (name, part) in enumerate([("GRU", gru), ("LSTM", lstm)]):
        vals = part[col].values
        ax.bar(i, vals.mean(), width=0.5, alpha=0.35)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, color="black", s=22, zorder=3)
    if "ratio" in col:
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1)   # baseline: predict zero
        ax.set_ylim(0.996, 1.007)
    else:
        ax.axhline(0.533, color="red", linestyle="--", linewidth=1)  # always-"down" score on test
        ax.set_ylim(0.46, 0.58)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["GRU", "LSTM"])
    ax.set_title(label, fontsize=9)
fig.suptitle("GRU vs LSTM, 6 seeds each (dots = runs, bars = means, red line = baseline)")
plt.tight_layout()
plt.savefig("reports/gru_vs_lstm.png", dpi=130)
print("\nSaved reports/gru_vs_lstm.png")