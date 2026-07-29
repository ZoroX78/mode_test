import os
import shutil
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import torch

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer

# Override torch.load to handle PyTorch 2.6 default weights_only issue
_original_torch_load = torch.load
def _custom_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _custom_torch_load


def compute_rolling_beta(df_stock: pd.DataFrame, window: int = 60) -> pd.Series:
    cov = df_stock['log_return'].rolling(window).cov(df_stock['nifty_return'])
    var = df_stock['nifty_return'].rolling(window).var()
    beta = cov / (var + 1e-9)
    return beta.fillna(1.0)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def fetch_and_engineer_sensex_data(num_days=180):
    print(f"Fetching recent ~{num_days} days of data for SENSEX (^BSESN) and Nifty 50 (^NSEI)...")
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=num_days)

    sensex_df = yf.download("^BSESN", start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
    nifty_df  = yf.download("^NSEI", start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)

    # Flatten yfinance MultiIndex columns if present
    if isinstance(sensex_df.columns, pd.MultiIndex):
        sensex_df.columns = [c[0] for c in sensex_df.columns]
    if isinstance(nifty_df.columns, pd.MultiIndex):
        nifty_df.columns = [c[0] for c in nifty_df.columns]

    sensex_df = sensex_df.reset_index()
    nifty_df = nifty_df.reset_index()

    sensex_df.columns = [c.lower() for c in sensex_df.columns]
    nifty_df.columns = [c.lower() for c in nifty_df.columns]

    # Align dates
    df = pd.merge(sensex_df[['date', 'close', 'open', 'high', 'low', 'volume']],
                  nifty_df[['date', 'close']].rename(columns={'close': 'nifty_close'}),
                  on='date', how='inner')

    df = df.sort_values('date').reset_index(drop=True)
    df['symbol'] = "SENSEX"
    df['sector'] = "Index"

    # Log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['nifty_return'] = np.log(df['nifty_close'] / df['nifty_close'].shift(1))

    # Features
    df['log_ret_5d'] = df['log_return'].rolling(5).sum()
    df['log_ret_21d'] = df['log_return'].rolling(21).sum()

    df['volatility_10d'] = df['log_return'].rolling(10).std()
    df['volatility_20d'] = df['log_return'].rolling(20).std()
    df['vol_ratio_10_20'] = df['volatility_10d'] / (df['volatility_20d'] + 1e-9)

    df['excess_return_1d'] = df['log_return'] - df['nifty_return']
    df['excess_return_5d'] = df['log_ret_5d'] - df['nifty_return'].rolling(5).sum()
    df['beta_60d'] = compute_rolling_beta(df, window=60)

    df['rsi_14'] = compute_rsi(df['close'], period=14)
    df['rsi_change_3d'] = df['rsi_14'].diff(3)

    # Volume features
    vol_mean_20 = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (vol_mean_20 + 1e-9)
    df['volume_surge'] = df['volume_ratio'] / (df['volume_ratio'].rolling(5).mean() + 1e-9)

    # Calendar features
    df['Date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['Date'].dt.dayofweek.astype(str)
    df['month'] = df['Date'].dt.month.astype(str)
    df['quarter'] = df['Date'].dt.quarter.astype(str)
    df['day_of_month'] = df['Date'].dt.day
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int).astype(str)
    df['is_union_budget_month'] = (df['Date'].dt.month == 2).astype(int).astype(str)

    def is_fno_expiry(d):
        return str(int((d.dayofweek == 3) and ((d + pd.Timedelta(days=7)).month != d.month)))
    df['is_fno_expiry'] = df['Date'].apply(is_fno_expiry)

    # Drop warm-up NaN rows
    df_clean = df.dropna().reset_index(drop=True)
    df_clean['time_idx'] = np.arange(len(df_clean))

    print(f"Engineered clean dataset for SENSEX: {len(df_clean)} trading days available.")
    return df_clean


def main():
    pt_path = os.path.join("tft_checkpoints", "tft-nifty-epoch=04-val_loss=0.0056-v1.pt")
    print(f"Loading PyTorch .pt model from: '{pt_path}'...")
    
    checkpoint_payload = torch.load(pt_path)
    model = checkpoint_payload["full_model"]
    model.eval()

    # Load recent Sensex data
    df = fetch_and_engineer_sensex_data(num_days=200)

    # Select the last 65 trading days for the evaluation window (60 days encoder + 5 days prediction)
    max_encoder_length = 60
    max_prediction_length = 5
    total_eval_window = max_encoder_length + max_prediction_length

    if len(df) < total_eval_window:
        raise ValueError(f"Need at least {total_eval_window} trading days, got {len(df)}.")

    eval_df = df.iloc[-total_eval_window:].copy().reset_index(drop=True)
    eval_df['time_idx'] = np.arange(len(eval_df))

    print(f"\nTarget Evaluation Range:")
    print(f"  Encoder window (60 days): {eval_df['Date'].iloc[0].strftime('%Y-%m-%d')} to {eval_df['Date'].iloc[max_encoder_length-1].strftime('%Y-%m-%d')}")
    print(f"  Forecast window (5 days):  {eval_df['Date'].iloc[max_encoder_length].strftime('%Y-%m-%d')} to {eval_df['Date'].iloc[-1].strftime('%Y-%m-%d')}")

    # Build reference training dataset object matching model's expected dataset parameters
    training_dataset_params = dict(
        time_idx="time_idx",
        target="log_return",
        group_ids=["symbol"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        
        static_categoricals=["symbol", "sector"],
        time_varying_known_categoricals=[
            "day_of_week", "month", "quarter", 
            "is_fno_expiry", "is_month_end", "is_union_budget_month"
        ],
        time_varying_known_reals=["time_idx", "day_of_month"],
        time_varying_unknown_reals=[
            "log_return", "log_ret_5d", "log_ret_21d",
            "volatility_10d", "volatility_20d", "vol_ratio_10_20",
            "excess_return_1d", "excess_return_5d", "beta_60d",
            "rsi_14", "rsi_change_3d", "volume_ratio", "volume_surge",
            "nifty_return"
        ],
        
        target_normalizer=GroupNormalizer(groups=["symbol"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    dataset_template = TimeSeriesDataSet(
        eval_df,
        **training_dataset_params
    )

    # Construct test dataset for out-of-sample prediction
    test_dataset = TimeSeriesDataSet.from_dataset(
        dataset_template,
        eval_df,
        predict=True,
        stop_randomization=True
    )

    test_dataloader = test_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

    # Predict using TFT model loaded from .pt file
    print("\n--- Running 5-Day Forward Prediction on Sensex ---")
    predictions = model.predict(test_dataloader, mode="quantiles", return_x=True)

    # Quantile outputs shape: [1, 5, 3] (quantiles: 0.1, 0.5, 0.9)
    q_pred = predictions.output[0].cpu().numpy()  # shape (5, 3)
    y_true_returns = eval_df['log_return'].iloc[max_encoder_length:].values  # shape (5,)

    pred_ret_p10 = q_pred[:, 0]
    pred_ret_p50 = q_pred[:, 1]  # Median forecast
    pred_ret_p90 = q_pred[:, 2]

    # Reconstruct actual and predicted close prices
    last_encoder_close = eval_df['close'].iloc[max_encoder_length - 1]
    last_encoder_date = eval_df['Date'].iloc[max_encoder_length - 1]

    target_dates = eval_df['Date'].iloc[max_encoder_length:].values
    actual_closes = eval_df['close'].iloc[max_encoder_length:].values

    # Cumulative return projection from T-5 close price
    pred_close_p50 = last_encoder_close * np.exp(np.cumsum(pred_ret_p50))
    pred_close_p10 = last_encoder_close * np.exp(np.cumsum(pred_ret_p10))
    pred_close_p90 = last_encoder_close * np.exp(np.cumsum(pred_ret_p90))

    # Display comparison table
    print("\n" + "="*80)
    print("           SENSEX 5-DAY OUT-OF-SAMPLE MODEL PREDICTION VS ACTUAL")
    print("="*80)
    print(f"Base Date (T-5 Cutoff): {pd.to_datetime(last_encoder_date).strftime('%Y-%m-%d')} | Sensex Base Close: {last_encoder_close:,.2f}\n")
    
    comp_data = []
    for i in range(max_prediction_length):
        d_str = pd.to_datetime(target_dates[i]).strftime('%Y-%m-%d')
        act = actual_closes[i]
        pred = pred_close_p50[i]
        p10 = pred_close_p10[i]
        p90 = pred_close_p90[i]
        diff = pred - act
        err_pct = (diff / act) * 100.0
        
        comp_data.append({
            "Day": f"t+{i+1}",
            "Date": d_str,
            "Actual Close": f"{act:,.2f}",
            "Predicted Close (p50)": f"{pred:,.2f}",
            "90% CI Range": f"[{p10:,.2f} - {p90:,.2f}]",
            "Diff": f"{diff:+,.2f}",
            "Error %": f"{err_pct:+.2f}%"
        })

    comp_df = pd.DataFrame(comp_data)
    print(comp_df.to_string(index=False))

    mae = np.mean(np.abs(pred_close_p50 - actual_closes))
    rmse = np.sqrt(np.mean((pred_close_p50 - actual_closes)**2))
    mape = np.mean(np.abs((pred_close_p50 - actual_closes) / actual_closes)) * 100.0

    # Directional Accuracy (Hit rate on return signs)
    actual_returns = y_true_returns
    hit_rate = np.mean(np.sign(pred_ret_p50) == np.sign(actual_returns)) * 100.0

    print("-" * 80)
    print(f" Summary Metrics over 5-Day Horizon:")
    print(f"  - Mean Absolute Error (MAE):     {mae:,.2f} points")
    print(f"  - Root Mean Squared Error (RMSE): {rmse:,.2f} points")
    print(f"  - Mean Abs % Error (MAPE):      {mape:.2f}%")
    print(f"  - Return Directional Accuracy:    {hit_rate:.1f}%")
    print("="*80 + "\n")

    # ----------------------------------------------------
    # PLOT VISUALIZATION: SENSEX ACTUAL VS PREDICTED
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot historical 65 days actual prices
    x_full = np.arange(len(eval_df))
    date_labels = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in eval_df['Date']]
    closes_full = eval_df['close'].values

    ax.plot(x_full, closes_full, label="Original Sensex Dataset (Actual Close)", color="#1f77b4", linewidth=2.5)

    # Forecast indices & predictions
    x_forecast = np.arange(max_encoder_length - 1, total_eval_window)
    forecast_p50 = np.append([last_encoder_close], pred_close_p50)
    forecast_p10 = np.append([last_encoder_close], pred_close_p10)
    forecast_p90 = np.append([last_encoder_close], pred_close_p90)

    # Plot prediction line & quantile band
    ax.plot(x_forecast, forecast_p50, label="TFT Model Forecast (p50 Median)", color="#ff7f0e", linestyle="--", linewidth=2.5, marker="o")
    ax.fill_between(x_forecast, forecast_p10, forecast_p90, color="#ff7f0e", alpha=0.2, label="90% Quantile Confidence Interval [p10-p90]")

    # Vertical cutoff marker
    cutoff_x = max_encoder_length - 1
    ax.axvline(x=cutoff_x, color="red", linestyle=":", linewidth=2, label="Forecast Input Cutoff (T-5)")

    # Set x-ticks dynamically
    tick_indices = np.linspace(0, len(eval_df) - 1, num=10, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([date_labels[i] for i in tick_indices], rotation=30, ha='right')

    ax.set_title("Sensex (^BSESN) 5-Day Out-of-Sample Forecast vs Original Dataset", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Sensex Index Level (INR)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()

    out_file = "sensex_forecast_vs_actual.png"
    plt.savefig(out_file, dpi=300)
    print(f"Saved forecast comparison chart to '{out_file}'.")

    # Copy chart to artifact directory for embedding in walkthrough
    artifact_dir = r"C:\Users\ramak\.gemini\antigravity-ide\brain\643d8a9c-1976-482c-8c70-85b654c5ba85"
    if os.path.exists(artifact_dir):
        shutil.copy(out_file, os.path.join(artifact_dir, out_file))
        print(f"Copied chart to artifact directory.")

if __name__ == "__main__":
    main()
