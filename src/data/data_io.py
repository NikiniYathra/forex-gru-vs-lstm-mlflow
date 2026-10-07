import pandas as pd

COLS = ["Open", "High", "Low", "Close", "Volume"]


def detect_format(path):
    """yfinance CSVs have a header row; forexsb files do not."""
    with open(path, "r", encoding="utf-8-sig") as f:
        first_line = f.readline()
    return "yfinance" if ("Date" in first_line or "Close" in first_line) else "forexsb"


def load_ohlcv(path, fmt="auto"):
    """Load a price file into a DataFrame with a datetime index and columns Open, High, Low, Close, Volume."""
    if fmt == "auto":
        fmt = detect_format(path)

    if fmt == "yfinance":
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    elif fmt == "forexsb":
        raw = pd.read_csv(
            path, sep=r"\s+", header=None,
            names=["date", "time", "Open", "High", "Low", "Close", "Volume"],
        )
        stamps = pd.to_datetime(raw["date"] + " " + raw["time"], format="%Y-%m-%d %H:%M")
        df = raw[COLS].copy()
        df.index = pd.DatetimeIndex(stamps, name="Date")
    else:
        raise ValueError(f"Unknown format: {fmt}")

    return df[COLS].astype("float64")