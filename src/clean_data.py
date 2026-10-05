import pandas as pd

df = pd.read_csv("data/raw/eurusd_daily.csv", parse_dates=["Date"], index_col="Date")

# 1. Volume is always 0 for Forex on Yahoo, so it carries no information
df = df.drop(columns=["Volume"])

oc_min = df[["Open", "Close"]].min(axis=1)
oc_max = df[["Open", "Close"]].max(axis=1)

# 2. Fix bad ticks: Low/High implausibly far (>10%) from the Open/Close range
bad_low = df["Low"] < oc_min * 0.90
bad_high = df["High"] > oc_max * 1.10
print("Bad-tick lows:", bad_low.sum(), "| Bad-tick highs:", bad_high.sum())

print("\nRow 2012-01-27 before:")
print(df.loc["2012-01-27"])

df.loc[bad_low, "Low"] = oc_min[bad_low]
df.loc[bad_high, "High"] = oc_max[bad_high]

# 3. Fix small inconsistencies: Low must be <= Open/Close, High must be >= Open/Close
df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)
df["High"] = df[["Open", "High", "Close"]].max(axis=1)

print("\nRow 2012-01-27 after:")
print(df.loc["2012-01-27"])

# 4. Sanity checks (the script stops with an error if any fail)
assert df.isna().sum().sum() == 0
assert (df["Low"] <= df[["Open", "Close"]].min(axis=1)).all()
assert (df["High"] >= df[["Open", "Close"]].max(axis=1)).all()
assert (df["Low"] >= df["Close"] * 0.90).all(), "A bad Low is still present"
assert (df["High"] <= df["Close"] * 1.10).all(), "A bad High is still present"

print("\nMin Low:", round(df["Low"].min(), 4), "| Max High:", round(df["High"].max(), 4))

# 5. Save the cleaned data. The raw file is left untouched.
df.to_csv("data/processed/eurusd_clean.csv")
print("Saved:", df.shape, "-> data/processed/eurusd_clean.csv")