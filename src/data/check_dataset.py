import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.data.windows import get_device, load_split, make_loader

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True, help="a folder inside data/processed")
args = parser.parse_args()

folder = Path(args.folder)
meta = json.load(open(folder / "meta.json"))
cfg = meta["config"]
L, h, kind = cfg["lookback"], cfg["target"]["horizon"], cfg["target"]["kind"]
max_gap = cfg["data"].get("max_gap")
data = np.load(folder / "data.npz")
y_scaler = joblib.load(folder / "y_scaler.joblib")
rng = np.random.default_rng(0)

print("Folder:", folder)
print("Device training would use:", get_device(), "| CUDA available:", torch.cuda.is_available())

for split in ["train", "val", "test"]:
    ds = load_split(folder, split, L)
    ends = data[f"{split}_ends"]
    sample = rng.choice(len(ds), size=min(300, len(ds)), replace=False)

    # 1. Windows contain exactly the intended rows
    for i in sample:
        xb, _ = ds[[int(i)]]
        assert np.array_equal(xb[0].numpy(), data[f"{split}_X"][ends[i] - L + 1: ends[i] + 1])

    # 2. Labels are the true future values (no off-by-one)
    close, y_raw = data[f"{split}_close"], data[f"{split}_y_raw"]
    expected = np.log(close[ends + h] / close[ends]) if kind == "return" else close[ends + h]
    assert np.allclose(expected, y_raw[ends], atol=1e-9)

    # 3. The scaled label is the scaler applied to that value
    scaled = y_scaler.transform(y_raw[ends].reshape(-1, 1)).ravel()
    assert np.allclose(scaled, data[f"{split}_y"][ends], atol=1e-4)

    # 4. No sampled window (or its label bar) spans a time gap above the limit
    if max_gap:
        limit = pd.Timedelta(max_gap).value
        t = data[f"{split}_time"]
        for i in sample:
            span = t[ends[i] - L + 1: ends[i] + h + 1]
            assert np.diff(span).max() <= limit, "window spans a gap"

    xb, yb = next(iter(make_loader(ds, 256, False)))
    print(f"{split:5s}: {len(ds)} windows | first batch X {tuple(xb.shape)} {xb.dtype}, y {tuple(yb.shape)} | checks passed")

# 5. Splits are in time order and labels never cross a boundary
first_row = {s: data[f"{s}_time"][0] for s in ["train", "val", "test"]}
last_label = {s: data[f"{s}_time"][data[f"{s}_ends"].max() + h] for s in ["train", "val", "test"]}
assert last_label["train"] < first_row["val"] and last_label["val"] < first_row["test"]
print("\nLast training label:", pd.to_datetime(last_label["train"]), "| first validation row:", pd.to_datetime(first_row["val"]))
print("Last validation label:", pd.to_datetime(last_label["val"]), "| first test row:", pd.to_datetime(first_row["test"]))

# 6. Same seed gives the same shuffled batches
ds = load_split(folder, "train", L)
a = next(iter(make_loader(ds, 256, True, seed=1)))
b = next(iter(make_loader(ds, 256, True, seed=1)))
assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
print("Seeded shuffling is reproducible.")

# 7. Speed of one full shuffled pass over the training windows
start = time.time()
batches = sum(1 for _ in make_loader(ds, 256, True, seed=0))
print(f"One training epoch of batching: {batches} batches in {time.time() - start:.2f} seconds")