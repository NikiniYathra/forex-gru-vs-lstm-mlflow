import mlflow
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
runs = mlflow.search_runs(experiment_names=["forex-gru-vs-lstm"])

# Keep only the columns we need and give them short names
cols = {
    "params.model_type": "model",
    "params.hidden_size": "hidden",
    "params.learning_rate": "lr",
    "params.batch_size": "batch",
    "params.seed": "seed",
    "metrics.val_rmse_vs_baseline": "val_ratio",
    "metrics.test_rmse_vs_baseline": "test_ratio",
    "metrics.val_direction_acc": "val_dir",
    "metrics.test_direction_acc": "test_dir",
    "metrics.epochs_run": "epochs",
}
df = runs[list(cols)].rename(columns=cols)
df["seed"] = df["seed"].astype(int)
df = df.sort_values(["model", "hidden", "lr", "batch", "seed"])

print("Total runs:", len(df))
print("\n--- Every run ---")
print(df.round(4).to_string(index=False))

# Mean and std across seeds for each distinct setting
group_cols = ["model", "hidden", "lr", "batch"]
summary = df.groupby(group_cols)[["val_ratio", "test_ratio", "val_dir", "test_dir", "epochs"]].agg(["mean", "std"])
summary["n_runs"] = df.groupby(group_cols).size()

print("\n--- Mean and std across seeds ---")
print(summary.round(4).to_string())

df.to_csv("reports/run_summary.csv", index=False)
print("\nSaved reports/run_summary.csv")