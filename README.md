# EUR/USD next-day return: GRU vs LSTM (PyTorch + MLflow)

Compares a GRU and an LSTM on daily EUR/USD returns, with every run tracked in MLflow.

## Setup (Windows, PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## Run order (from the project root)
```powershell
python src\download_data.py        # yfinance -> data/raw/eurusd_daily.csv
python src\clean_data.py           # fix bad ticks, drop Volume -> data/processed/eurusd_clean.csv
python src\make_features.py        # features + next-day-return target
python src\prepare_sequences.py    # 70/15/15 chronological split, scaling, 30-day windows
python src\train.py --model gru --hidden_size 64 --lr 0.001 --seed 42
python src\train.py --model lstm --hidden_size 64 --lr 0.001 --seed 42
python src\summarize_runs.py       # all runs -> reports/run_summary.csv
python src\compare_models.py       # GRU vs LSTM stats + reports/gru_vs_lstm.png
mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse runs at http://127.0.0.1:5000
```

## Design choices
- Inputs are returns and volatility features, not raw prices. Target is the next-day log return.
- Chronological split (never shuffled). Scalers are fitted on the training rows only, to avoid leakage.
- Hyperparameters were chosen on validation results only. The test set is a final check.
- Every model is compared with an "always predict 0" baseline (ratio of RMSEs, 1.0 = same).

## Results (hidden 64, lr 0.001, batch 64, 6 seeds each)
| Metric | GRU | LSTM |
|---|---|---|
| Validation RMSE / baseline | 0.9985 | 0.9993 |
| Test RMSE / baseline | 1.0029 | 1.0017 |
| Test directional accuracy | 0.510 | 0.530 |

No difference was statistically convincing (Welch t-test p-values 0.06 to 0.10), and neither model beat the
baseline on the test period. Hidden size and learning rate made little difference.

## Notes
- Data: Yahoo Finance via yfinance. One bad `Low` value (2012-01-27) is repaired in `clean_data.py`.
- CPU-only PyTorch; the models are small and train in about a minute.