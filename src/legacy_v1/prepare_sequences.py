import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

LOOKBACK = 30
FEATURES = ["ret", "hl_range", "oc_change", "vol5"]

df = pd.read_csv("data/processed/eurusd_features.csv", parse_dates=["Date"], index_col="Date")

# 1. Chronological split (no shuffling)
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)
train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

for name, part in [("train", train), ("val", val), ("test", test)]:
    print(f"{name:5s}: {len(part)} rows, {part.index.min().date()} to {part.index.max().date()}")

# 2. Fit scalers on TRAIN ONLY
x_scaler = StandardScaler().fit(train[FEATURES])
y_scaler = StandardScaler().fit(train[["target"]])

def scale(part):
    x = x_scaler.transform(part[FEATURES])
    y = y_scaler.transform(part[["target"]]).ravel()
    return x, y

# 3. Build sliding windows inside each split
def make_windows(x, y, lookback):
    xs, ys = [], []
    for i in range(len(x) - lookback + 1):
        xs.append(x[i:i + lookback])
        ys.append(y[i + lookback - 1])   # label belongs to the last day of the window
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

X_train, y_train = make_windows(*scale(train), LOOKBACK)
X_val, y_val = make_windows(*scale(val), LOOKBACK)
X_test, y_test = make_windows(*scale(test), LOOKBACK)

print("\nX_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:  ", X_val.shape, "y_val:  ", y_val.shape)
print("X_test: ", X_test.shape, "y_test: ", y_test.shape)

# 4. Sanity checks on scaling
print("\nScaled train features: mean ~", X_train.mean().round(3), "std ~", X_train.std().round(3))
print("Scaled val features:   mean ~", X_val.mean().round(3), "std ~", X_val.std().round(3))

# 5. Save arrays and scalers
np.savez("data/processed/sequences.npz",
         X_train=X_train, y_train=y_train,
         X_val=X_val, y_val=y_val,
         X_test=X_test, y_test=y_test)
joblib.dump(x_scaler, "models/x_scaler.joblib")
joblib.dump(y_scaler, "models/y_scaler.joblib")
print("\nSaved data/processed/sequences.npz and scalers in models/")