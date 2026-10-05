import numpy as np
import pandas as pd

df = pd.read_csv("data/processed/eurusd_clean.csv", parse_dates=["Date"], index_col="Date")

feat = pd.DataFrame(index=df.index)
feat["ret"] = np.log(df["Close"] / df["Close"].shift(1))
feat["hl_range"] = (df["High"] - df["Low"]) / df["Close"]
feat["oc_change"] = (df["Close"] - df["Open"]) / df["Open"]
feat["vol5"] = feat["ret"].rolling(5).std()

# Target: tomorrow's return (shift moves tomorrow's value up into today's row)
feat["target"] = feat["ret"].shift(-1)

# Drop the first rows (no previous day / not enough history) and the last row (no tomorrow)
feat = feat.dropna()

print("Shape:", feat.shape)
print("\n--- First 5 rows ---")
print(feat.head())
print("\n--- Summary statistics ---")
print(feat.describe())
print("\nMissing values:", feat.isna().sum().sum())
print("\n--- Correlation of each feature with the target ---")
print(feat.corr()["target"])

feat.to_csv("data/processed/eurusd_features.csv")
print("\nSaved to data/processed/eurusd_features.csv")