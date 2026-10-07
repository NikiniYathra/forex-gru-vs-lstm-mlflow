import yfinance as yf

# Daily EUR/USD from 2010 until today (gives ~4,000 rows)
df = yf.download(
    "EURUSD=X",
    start="2010-01-01",
    auto_adjust=True,
    progress=False,
    multi_level_index=False,   # keeps the CSV header as one clean row
)

df.to_csv("data/raw/eurusd_daily.csv")

print("Shape (rows, columns):", df.shape)
print(df.head())
print(df.tail())