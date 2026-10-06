import numpy as np
import pandas as pd

from data_io import load_ohlcv

df = load_ohlcv("data/raw/eurusd_daily.csv", "yfinance")
year = df.index.year

ret = np.log(df["Close"] / df["Close"].shift(1))
oc = (df["Close"] - df["Open"]) / df["Open"]
open_vs_prev_close = (df["Open"] - df["Close"].shift(1)).abs() / df["Close"].shift(1)

table = pd.DataFrame({
    "oc_std": oc.groupby(year).std(),
    "ret_std": ret.groupby(year).std(),
    "corr_oc_ret": pd.Series({y: oc[g.index].corr(ret[g.index]) for y, g in df.groupby(year)}),
    "open_vs_prev_close": open_vs_prev_close.groupby(year).mean(),
    "open_equals_close": (df["Open"] == df["Close"]).groupby(year).mean(),
})
pd.set_option("display.width", 200)
print(table.round(5))