import pandas as pd

df = pd.read_csv("data/raw/eurusd_daily.csv", parse_dates=["Date"], index_col="Date")

print("Shape (rows, columns):", df.shape)
print("Date range:", df.index.min().date(), "to", df.index.max().date())

print("\n--- Column types ---")
print(df.dtypes)

print("\n--- Missing values per column ---")
print(df.isna().sum())

print("\n--- Duplicate dates ---")
print(df.index.duplicated().sum())

print("\n--- Summary statistics ---")
print(df.describe())

print("\n--- Is the date index sorted oldest to newest? ---")
print(df.index.is_monotonic_increasing)