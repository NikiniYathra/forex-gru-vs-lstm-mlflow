import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/eurusd_daily.csv", parse_dates=["Date"], index_col="Date")

# How far below the Close does the Low go, as a fraction of the Close?
gap = (df["Close"] - df["Low"]) / df["Close"]
print("--- Low-vs-Close gap statistics ---")
print(gap.describe())

# Daily EUR/USD almost never swings more than 3% in one day
print("\n--- Rows where Low is more than 3% below Close ---")
print(df[gap > 0.03])

# Plot the closing price over time and save it
plt.figure(figsize=(12, 5))
df["Close"].plot(title="EUR/USD daily close")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig("reports/eurusd_close.png")
print("\nSaved plot to reports/eurusd_close.png")