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

TOP_10_STOCKS = [
    {"name": "Reliance Industries", "symbol": "RELIANCE.NS", "sector": "Energy"},
    {"name": "Tata Consultancy Services", "symbol": "TCS.NS", "sector": "IT"},
    {"name": "HDFC Bank", "symbol": "HDFCBANK.NS", "sector": "Financials"},
    {"name": "Bharti Airtel", "symbol": "BHARTIARTL.NS", "sector": "Telecom"},
    {"name": "ICICI Bank", "symbol": "ICICIBANK.NS", "sector": "Financials"},
    {"name": "Infosys", "symbol": "INFY.NS", "sector": "IT"},
    {"name": "ITC Ltd.", "symbol": "ITC.NS", "sector": "FMCG"},
    {"name": "Larsen & Toubro", "symbol": "LT.NS", "sector": "Capital Goods"},
    {"name": "Hindustan Unilever", "symbol": "HINDUNILVR.NS", "sector": "FMCG"},
    {"name": "State Bank of India", "symbol": "SBIN.NS", "sector": "Financials"},
]

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

def fetch_and_prep_stock_data(stock_info, num_days=200):
    sym = stock_info["symbol"]
    name = stock_info["name"]
    sec = stock_info["sector"]

    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=num_days)

    stock_df = yf.download(sym, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
    nifty_df = yf.download("^NSEI", start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)

    if isinstance(stock_df.columns, pd.MultiIndex):
        stock_df.columns = [c[0] for c in stock_df.columns]
    if isinstance(nifty_df.columns, pd.MultiIndex):
        nifty_df.columns = [c[0] for c in nifty_df.columns]

    stock_df = stock_df.reset_index()
    nifty_df = nifty_df.reset_index()

    stock_df.columns = [c.lower() for c in stock_df.columns]
    nifty_df.columns = [c.lower() for c in nifty_df.columns]

    df = pd.merge(stock_df[['date', 'close', 'open', 'high', 'low', 'volume']],
                  nifty_df[['date', 'close']].rename(columns={'close': 'nifty_close'}),
                  on='date', how='inner')

    df = df.sort_values('date').reset_index(drop=True)
    df['symbol'] = sym
    df['sector'] = sec

    # Returns
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

    df_clean = df.dropna().reset_index(drop=True)
    df_clean['time_idx'] = np.arange(len(df_clean))

    return df_clean

def evaluate_single_stock(model, stock_info):
    sym = stock_info["symbol"]
    name = stock_info["name"]

    df = fetch_and_prep_stock_data(stock_info, num_days=200)

    max_encoder_length = 60
    max_prediction_length = 5
    total_eval_window = max_encoder_length + max_prediction_length

    if len(df) < total_eval_window:
        raise ValueError(f"Stock {sym} has only {len(df)} days; need {total_eval_window}.")

    eval_df = df.iloc[-total_eval_window:].copy().reset_index(drop=True)
    eval_df['time_idx'] = np.arange(len(eval_df))

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

    dataset_template = TimeSeriesDataSet(eval_df, **training_dataset_params)

    test_dataset = TimeSeriesDataSet.from_dataset(
        dataset_template,
        eval_df,
        predict=True,
        stop_randomization=True
    )

    test_dataloader = test_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
    predictions = model.predict(test_dataloader, mode="quantiles", return_x=True)

    q_pred = predictions.output[0].cpu().numpy()  # shape (5, 3)
    y_true_returns = eval_df['log_return'].iloc[max_encoder_length:].values  # shape (5,)

    pred_ret_p10 = q_pred[:, 0]
    pred_ret_p50 = q_pred[:, 1]
    pred_ret_p90 = q_pred[:, 2]

    last_encoder_close = eval_df['close'].iloc[max_encoder_length - 1]
    actual_closes = eval_df['close'].iloc[max_encoder_length:].values
    target_dates = eval_df['Date'].iloc[max_encoder_length:].values

    pred_close_p50 = last_encoder_close * np.exp(np.cumsum(pred_ret_p50))
    pred_close_p10 = last_encoder_close * np.exp(np.cumsum(pred_ret_p10))
    pred_close_p90 = last_encoder_close * np.exp(np.cumsum(pred_ret_p90))

    mae = np.mean(np.abs(pred_close_p50 - actual_closes))
    rmse = np.sqrt(np.mean((pred_close_p50 - actual_closes)**2))
    mape = np.mean(np.abs((pred_close_p50 - actual_closes) / actual_closes)) * 100.0
    hit_rate = np.mean(np.sign(pred_ret_p50) == np.sign(y_true_returns)) * 100.0

    return {
        "symbol": sym,
        "name": name,
        "sector": stock_info["sector"],
        "eval_df": eval_df,
        "last_encoder_close": last_encoder_close,
        "target_dates": target_dates,
        "actual_closes": actual_closes,
        "pred_close_p50": pred_close_p50,
        "pred_close_p10": pred_close_p10,
        "pred_close_p90": pred_close_p90,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "hit_rate": hit_rate
    }

def main():
    pt_path = os.path.join("tft_checkpoints", "tft-nifty-epoch=04-val_loss=0.0056-v1.pt")
    print(f"Loading PyTorch .pt model from: '{pt_path}'...\n")
    
    checkpoint_payload = torch.load(pt_path)
    model = checkpoint_payload["full_model"]
    model.eval()

    all_results = []

    print("=================================================================================")
    print("      RUNNING 5-DAY OUT-OF-SAMPLE EVALUATION ACROSS TOP 10 INDIAN STOCKS")
    print("=================================================================================\n")

    for stock_info in TOP_10_STOCKS:
        print(f"Evaluating {stock_info['name']} ({stock_info['symbol']})...")
        try:
            res = evaluate_single_stock(model, stock_info)
            all_results.append(res)
            print(f"  -> MAPE: {res['mape']:.2f}% | MAE: {res['mae']:.2f} | Hit Rate: {res['hit_rate']:.1f}%\n")
        except Exception as e:
            print(f"  -> ERROR evaluating {stock_info['symbol']}: {e}\n")

    if not all_results:
        print("No stock results produced.")
        return

    # Print summary table
    summary_rows = []
    for r in all_results:
        act_end = r['actual_closes'][-1]
        pred_end = r['pred_close_p50'][-1]
        diff_end = pred_end - act_end
        summary_rows.append({
            "Stock Name": r['name'],
            "Ticker": r['symbol'],
            "Sector": r['sector'],
            "Base Close (T-5)": f"{r['last_encoder_close']:,.2f}",
            "Actual Close (T)": f"{act_end:,.2f}",
            "Predicted (p50)": f"{pred_end:,.2f}",
            "MAPE (%)": f"{r['mape']:.2f}%",
            "MAE": f"{r['mae']:.2f}",
            "Hit Rate (%)": f"{r['hit_rate']:.1f}%"
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "="*110)
    print("                        TOP 10 INDIAN STOCKS MODEL PERFORMANCE SUMMARY")
    print("="*110)
    print(summary_df.to_string(index=False))

    avg_mape = np.mean([r['mape'] for r in all_results])
    avg_hit_rate = np.mean([r['hit_rate'] for r in all_results])

    print("-" * 110)
    print(f" Overall Portfolio Averages across Top 10 Stocks:")
    print(f"  - Portfolio Mean Absolute Percentage Error (MAPE): {avg_mape:.2f}%")
    print(f"  - Portfolio Average Directional Accuracy (Hit Rate): {avg_hit_rate:.1f}%")
    print("="*110 + "\n")

    # ----------------------------------------------------
    # GENERATE 2x5 MULTI-PANEL VISUALIZATION GRID
    # ----------------------------------------------------
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()

    for idx, r in enumerate(all_results):
        ax = axes[idx]
        eval_df = r['eval_df']
        max_encoder_length = 60
        total_eval_window = len(eval_df)

        x_full = np.arange(total_eval_window)
        date_labels = [pd.to_datetime(d).strftime('%m-%d') for d in eval_df['Date']]
        closes_full = eval_df['close'].values

        ax.plot(x_full, closes_full, label="Actual Close", color="#1f77b4", linewidth=2.0)

        x_forecast = np.arange(max_encoder_length - 1, total_eval_window)
        forecast_p50 = np.append([r['last_encoder_close']], r['pred_close_p50'])
        forecast_p10 = np.append([r['last_encoder_close']], r['pred_close_p10'])
        forecast_p90 = np.append([r['last_encoder_close']], r['pred_close_p90'])

        ax.plot(x_forecast, forecast_p50, label="Predicted (p50)", color="#ff7f0e", linestyle="--", linewidth=2.0, marker="o", markersize=3)
        ax.fill_between(x_forecast, forecast_p10, forecast_p90, color="#ff7f0e", alpha=0.25)

        ax.axvline(x=max_encoder_length - 1, color="red", linestyle=":", linewidth=1.5)

        tick_indices = [0, 20, 40, 59, 64]
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([date_labels[i] for i in tick_indices], fontsize=8, rotation=25)

        ax.set_title(f"{r['name']}\n(MAPE: {r['mape']:.2f}%, Hit: {r['hit_rate']:.0f}%)", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Top 10 Indian Stocks: 5-Day Out-of-Sample Model Predictions vs Actual Prices", fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    grid_file = "top10_stocks_forecast_vs_actual.png"
    plt.savefig(grid_file, dpi=300)
    print(f"Saved 10-stock grid comparison plot to '{grid_file}'.")

    artifact_dir = r"C:\Users\ramak\.gemini\antigravity-ide\brain\643d8a9c-1976-482c-8c70-85b654c5ba85"
    if os.path.exists(artifact_dir):
        shutil.copy(grid_file, os.path.join(artifact_dir, grid_file))
        print("Copied grid chart to artifact directory.")

if __name__ == "__main__":
    main()
