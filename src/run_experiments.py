import argparse
import hashlib
import itertools
import json

import mlflow
import numpy as np
import yaml

from train_run import build_parser, train_one

parser = argparse.ArgumentParser()
parser.add_argument("--grid", required=True, help="e.g. configs/experiments/daily_tune.yaml")
parser.add_argument("--dry_run", action="store_true", help="show the plan without training")
args = parser.parse_args()

spec = yaml.safe_load(open(args.grid))
defaults = vars(build_parser().parse_args(["--folder", spec["folder"]]))
for key in list(spec.get("fixed", {})) + list(spec["grid"]):
    assert key in defaults, f"Unknown training option in the grid file: {key}"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(spec["experiment"])

base = {**defaults, "folder": spec["folder"], "mode": spec["mode"],
        "experiment": spec["experiment"], **spec.get("fixed", {})}
keys = list(spec["grid"])
combos = [dict(zip(keys, values)) for values in itertools.product(*spec["grid"].values())]


def make_key(cfg):
    """Fingerprint of everything that defines a run (so finished runs can be skipped on restart)."""
    fields = {k: v for k, v in cfg.items() if k not in ("experiment", "device", "run_key")}
    return hashlib.md5(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()[:12]


def already_done(key):
    found = mlflow.search_runs(
        experiment_names=[spec["experiment"]],
        filter_string=f"tags.run_key = '{key}' and attributes.status = 'FINISHED'",
        max_results=1,
    )
    return len(found) > 0


def aggregate(outs):
    rows = {}
    for o in outs:
        for split, m in o["results"].items():
            for k in ("rmse_vs_baseline", "direction_edge", "change_corr"):
                rows.setdefault(f"{split}_{k}", []).append(m[k])
        rows.setdefault("best_epoch", []).append(o["best_epoch"])
        rows.setdefault("train_seconds", []).append(o["train_seconds"])
    flat = {}
    for k, v in rows.items():
        flat[f"mean_{k}"] = float(np.mean(v))
        flat[f"std_{k}"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return flat


# ---------- Build the plan ----------
plan, total, done = [], 0, 0
for model in spec["models"]:
    for combo in combos:
        cfg0 = {**base, **combo, "model": model}
        label = model + "_" + "_".join(f"{k}{v}" for k, v in combo.items())
        pending = []
        for seed in spec["seeds"]:
            cfg = {**cfg0, "seed": seed}
            cfg["run_key"] = make_key(cfg)
            total += 1
            if already_done(cfg["run_key"]):
                done += 1
            else:
                pending.append(cfg)
        plan.append((label, model, combo, pending))

print(f"Planned runs: {total} | already done: {done} | to run: {total - done}")
for label, _, _, pending in plan:
    print(f"PLAN {label}: seeds still to run {[c['seed'] for c in pending]}")
if args.dry_run:
    raise SystemExit(0)

# ---------- Run ----------
for label, model, combo, pending in plan:
    if not pending:
        continue
    with mlflow.start_run(run_name=label):
        mlflow.set_tags({"level": "config", "model_type": model, "mode": spec["mode"]})
        mlflow.log_params({"model_type": model, "mode": spec["mode"], "seeds": str([c["seed"] for c in pending]), **combo})
        outs = [train_one(cfg) for cfg in pending]
        flat = aggregate(outs)
        mlflow.log_metrics(flat)
    key = "val_rmse_vs_baseline"
    print(f"CONFIG {label}: val ratio {flat['mean_' + key]:.4f} +/- {flat['std_' + key]:.4f} "
                    f"| val direction edge {flat['mean_val_direction_edge']:+.4f} | {len(outs)} seeds")
    if "mean_test_rmse_vs_baseline" in flat:
        print(f"CONFIG {label}: TEST ratio {flat['mean_test_rmse_vs_baseline']:.4f} +/- "
              f"{flat['std_test_rmse_vs_baseline']:.4f} | test direction edge {flat['mean_test_direction_edge']:+.4f}")

print("Finished all planned runs for experiment:", spec["experiment"])