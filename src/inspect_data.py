import argparse

import numpy as np
import pandas as pd

from data_io import load_ohlcv

parser = argparse.ArgumentParser()
parser.add_argument("--path", default="data/raw/eurusd_daily.csv")
parser.add_argument("--format", choices=["auto", "yfinance", "forexsb"], default="auto")
args = parser.parse_args()

pd.set_option("display.width", 200)
df = load_ohlcv(args.path, args.format)

print("File:", args.path)
print("Shape (rows, columns):", df.shape)
print("Date range:", df.index.min(), "to", df.index.max())

print("\n--- Column types ---")
print(df.dtypes)

print("\n--- Missing values per column ---")
print(df.isna().sum())

print("\n--- Duplicate timestamps ---")
print(df.index.duplicated().sum())

print("\n--- Sorted oldest to newest? ---")
print(df.index.is_monotonic_increasing)

# Time between consecutive rows: shows weekends and missing stretches
gaps = df.index.to_series().diff().dropna()
print("\n--- Most common time step between rows ---")
print(gaps.value_counts().head(5))
print("Gaps longer than 1 hour:", int((gaps > pd.Timedelta(hours=1)).sum()))
print("Five largest gaps (timestamp = where the gap ends):")
print(gaps.nlargest(5))

# Bar consistency and suspicious values
oc_min = df[["Open", "Close"]].min(axis=1)
oc_max = df[["Open", "Close"]].max(axis=1)
ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
print("\n--- Bar sanity checks ---")
print("Low above min(Open, Close):", int((df["Low"] > oc_min).sum()))
print("High below max(Open, Close):", int((df["High"] < oc_max).sum()))
print("Zero-range bars (High == Low):", int((df["High"] == df["Low"]).sum()))
print("Largest absolute return between rows:", round(float(ret.abs().max()), 5))
print("Rows with absolute return above 0.5%:", int((ret.abs() > 0.005).sum()))
print("Rows with zero volume:", int((df["Volume"] == 0).sum()))

print("\n--- Summary statistics ---")
print(df.describe())