import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.data.data_io import load_ohlcv

SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler}

# ---------- Settings: YAML file, optionally overridden from the command line ----------
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="e.g. configs/m1.yaml")
parser.add_argument("--scaler", choices=list(SCALERS))
parser.add_argument("--lookback", type=int)
parser.add_argument("--target_kind", choices=["return", "close"])
parser.add_argument("--horizon", type=int)
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)
if args.scaler:
    cfg["scaler"] = args.scaler
if args.lookback:
    cfg["lookback"] = args.lookback
if args.target_kind:
    cfg["target"]["kind"] = args.target_kind
if args.horizon:
    cfg["target"]["horizon"] = args.horizon

lookback = cfg["lookback"]
horizon = cfg["target"]["horizon"]
kind = cfg["target"]["kind"]
setting_id = hashlib.md5(json.dumps(
    [cfg["features"], cfg["vol_window"], cfg["clean"], cfg["data"], cfg["split"]],
    sort_keys=True, default=str,
).encode()).hexdigest()[:6]
out_dir = Path("data/processed") / f"{cfg['name']}_{cfg['scaler']}_lb{lookback}_{kind}{horizon}_{setting_id}"
out_dir.mkdir(parents=True, exist_ok=True)


# ---------- Helpers ----------
def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def repair_bars(df, tol):
    """Fix bad ticks, then enforce Low <= min(Open, Close) and High >= max(Open, Close)."""
    df = df.copy()
    oc_min = df[["Open", "Close"]].min(axis=1)
    oc_max = df[["Open", "Close"]].max(axis=1)
    bad_low = df["Low"] < oc_min * (1 - tol)
    bad_high = df["High"] > oc_max * (1 + tol)
    df.loc[bad_low, "Low"] = oc_min[bad_low]
    df.loc[bad_high, "High"] = oc_max[bad_high]
    inconsistent = int(((df["Low"] > oc_min) | (df["High"] < oc_max)).sum())
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)
    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    return df, int(bad_low.sum() + bad_high.sum()), inconsistent


def build_features(df, gap_flag, vol_window, names):
    """Every feature uses only the current and earlier rows."""
    close = df["Close"]
    ret = np.log(close / close.shift(1)).where(~gap_flag)   # return across a gap is not a real 1-step return
    clock = np.asarray(df.index.hour + df.index.minute / 60.0)
    table = pd.DataFrame({
        "ret": ret,
        "hl_range": (df["High"] - df["Low"]) / close,
        "oc_change": (close - df["Open"]) / df["Open"],
        "vol_short": ret.rolling(vol_window).std(),
        "log_volume": np.log1p(df["Volume"]),
        "hour_sin": np.sin(2 * np.pi * clock / 24),
        "hour_cos": np.cos(2 * np.pi * clock / 24),
    }, index=df.index)
    return table[names]


def valid_ends(row_ok, y_ok, lookback, horizon):
    """Positions i where rows i-lookback+1..i are all valid, the label exists, and the label bar is inside the split."""
    n = len(row_ok)
    csum = np.concatenate([[0], np.cumsum(row_ok)])
    ends = np.arange(lookback - 1, n - horizon)
    full_window = (csum[ends + 1] - csum[ends + 1 - lookback]) == lookback
    return ends[full_window & y_ok[ends]]


# ---------- Load and clean ----------
df = load_ohlcv(cfg["data"]["path"], cfg["data"].get("format", "auto"))
assert df.index.is_monotonic_increasing and not df.index.has_duplicates
print("Output folder:", out_dir)
print("Loaded:", df.shape, "|", df.index.min(), "to", df.index.max())

df, n_bad, n_inconsistent = repair_bars(df, cfg["clean"]["bad_tick_tol"])
print(f"Bad ticks repaired: {n_bad} | inconsistent bars repaired: {n_inconsistent}")

# ---------- Gaps, features, target ----------
max_gap = cfg["data"].get("max_gap")
if max_gap:
    gap_flag = (df.index.to_series().diff() > pd.Timedelta(max_gap)).fillna(False)
else:
    gap_flag = pd.Series(False, index=df.index)
print("Time gaps longer than the limit:", int(gap_flag.sum()))

feats = build_features(df, gap_flag, cfg["vol_window"], cfg["features"])

close = df["Close"]
future_close = close.shift(-horizon)
y_raw = np.log(future_close / close) if kind == "return" else future_close
future_gap = sum(gap_flag.shift(-k, fill_value=False).astype(int) for k in range(1, horizon + 1)) > 0
y_raw = y_raw.where(~future_gap)   # labels whose horizon crosses a gap are invalid

# ---------- Chronological split (no shuffling) ----------
n = len(df)
a = int(n * cfg["split"]["train"])
b = int(n * (cfg["split"]["train"] + cfg["split"]["val"]))
bounds = {"train": (0, a), "val": (a, b), "test": (b, n)}

parts = {}
for name, (lo, hi) in bounds.items():
    f = feats.iloc[lo:hi]
    row_ok = ~f.isna().any(axis=1).to_numpy()
    y = y_raw.iloc[lo:hi].to_numpy()
    ends = valid_ends(row_ok, ~np.isnan(y), lookback, horizon)
    parts[name] = {
        "feats": f.to_numpy(dtype=np.float64), "row_ok": row_ok, "y": y, "ends": ends,
        "close": close.iloc[lo:hi].to_numpy(),
        "time": df.index[lo:hi].to_numpy().astype("datetime64[ns]").astype("int64"),
    }

# ---------- Leakage checks: labels never cross a split boundary ----------
for left, right in [("train", "val"), ("val", "test")]:
    last_label_pos = bounds[left][0] + parts[left]["ends"].max() + horizon
    assert last_label_pos < bounds[right][0], f"{left} labels reach into {right}"
print("Leakage checks passed: every window and label lies inside a single split.")

# ---------- Scalers: fitted on TRAIN only ----------
train = parts["train"]
x_scaler = SCALERS[cfg["scaler"]]().fit(train["feats"][train["row_ok"]])
y_scaler = SCALERS[cfg["scaler"]]().fit(train["y"][train["ends"]].reshape(-1, 1))

arrays = {}
for name, p in parts.items():
    p["X"] = x_scaler.transform(np.nan_to_num(p["feats"])).astype(np.float32)
    y_scaled = y_scaler.transform(np.nan_to_num(p["y"]).reshape(-1, 1)).ravel().astype(np.float32)
    arrays[f"{name}_X"] = p["X"]
    arrays[f"{name}_y"] = y_scaled
    arrays[f"{name}_y_raw"] = np.nan_to_num(p["y"])
    arrays[f"{name}_ends"] = p["ends"].astype(np.int64)
    arrays[f"{name}_close"] = p["close"]
    arrays[f"{name}_time"] = p["time"]
np.savez(out_dir / "data.npz", **arrays)
joblib.dump(x_scaler, out_dir / "x_scaler.joblib")
joblib.dump(y_scaler, out_dir / "y_scaler.joblib")

# ---------- Report ----------
print(f"\nFeatures: {cfg['features']} | scaler: {cfg['scaler']} | lookback {lookback} | target {kind} (horizon {horizon})")
meta_splits = {}
for name, p in parts.items():
    lo, hi = bounds[name]
    candidates = (hi - lo) - horizon - (lookback - 1)
    kept = len(p["ends"])
    print(f"\n{name.upper():5s}: {hi - lo} rows, {df.index[lo]} to {df.index[hi - 1]}")
    print(f"       windows kept: {kept} of {candidates} possible (dropped for gaps/NaN: {candidates - kept})")
    rows = p["X"][p["row_ok"]]
    print("       scaled mean:", np.round(rows.mean(0), 2))
    print("       scaled std :", np.round(rows.std(0), 2))
    print("       scaled min :", np.round(rows.min(0), 2))
    print("       scaled max :", np.round(rows.max(0), 2))
    meta_splits[name] = {"rows": hi - lo, "start": str(df.index[lo]), "end": str(df.index[hi - 1]), "windows": kept}

meta = {
    "created": datetime.now().isoformat(timespec="seconds"),
    "git_commit": git_commit(),
    "raw_file": cfg["data"]["path"],
    "raw_file_md5": file_md5(cfg["data"]["path"]),
    "config": cfg,
    "bad_ticks_repaired": n_bad,
    "inconsistent_bars_repaired": n_inconsistent,
    "gaps_over_limit": int(gap_flag.sum()),
    "splits": meta_splits,
}
with open(out_dir / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("\nSaved data.npz, scalers and meta.json to", out_dir)