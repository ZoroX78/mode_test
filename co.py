import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer

# ==========================================
# 1. HELPER: FIND BEST MODEL CHECKPOINT
# ==========================================
def get_best_checkpoint(checkpoint_dir="tft_checkpoints"):
    """Finds the checkpoint file with the lowest validation loss."""
    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No .ckpt files found in '{checkpoint_dir}' folder.")
    
    # Sort checkpoints by val_loss in filename
    best_ckpt = min(ckpt_files, key=lambda x: float(x.split("val_loss=")[-1].replace(".ckpt", "")))
    print(f"Selected Checkpoint: '{best_ckpt}'")
    return best_ckpt


# ==========================================
# 2. MAIN EVALUATION & BACKTEST PIPELINE
# ==========================================
def main():
    # Filepaths
    data_path = "nifty100_tft_engineered_dataset.csv"
    ckpt_path = get_best_checkpoint()

    print(f"Loading engineered dataset from '{data_path}'...")
    df = pd.read_csv(data_path)

    # Ensure categorical data types match training setup
    categorical_cols = [
        "symbol", "sector", "day_of_week", "month", 
        "quarter", "is_fno_expiry", "is_month_end", "is_union_budget_month"
    ]
    for col in categorical_cols:
        df[col] = df[col].astype(str)

    df = df.sort_values(by=["symbol", "time_idx"]).reset_index(drop=True)

    # ----------------------------------------------------
    # Reconstruct Dataset Specifications
    # ----------------------------------------------------
    max_encoder_length = 60    # Look back 60 days
    max_prediction_length = 5   # Forecast 5 days ahead

    max_time_idx = df["time_idx"].max()
    val_cutoff = max_time_idx - 20
    training_cutoff = val_cutoff - max_prediction_length

    # Re-create training dataset structure (used as metadata template)
    training_data = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
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

    # Create Out-of-Sample Test Dataset (Last 20 trading days)
    test_data = TimeSeriesDataSet.from_dataset(
        training_data,
        df[lambda x: x.time_idx > training_cutoff],
        predict=True,
        stop_randomization=True
    )

    test_dataloader = test_data.to_dataloader(
        train=False, batch_size=256, num_workers=2
    )

    # ----------------------------------------------------
    # Load Model & Predict
    # ----------------------------------------------------
    print("\n--- Loading TFT Model & Generating Test Predictions ---")
    model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
    model.eval()

    # Predict median forecast (mode="prediction" evaluates quantile q=0.5)
    predictions = model.predict(test_dataloader, mode="prediction", return_x=True)

    # Extract tensors
    y_pred = predictions.output.cpu().numpy()  # Shape: [N_samples, 5]
    y_true = predictions.y[0].cpu().numpy()   # Shape: [N_samples, 5]

    # Evaluate 1-Day Ahead Predictions (t+1 horizon)
    y_pred_1d = y_pred[:, 0]
    y_true_1d = y_true[:, 0]

    # ----------------------------------------------------
    # 1. CALCULATE DIRECTIONAL ACCURACY (HIT RATE)
    # ----------------------------------------------------
    # Match non-zero movement direction signs
    valid_mask = (y_true_1d != 0)
    pred_sign = np.sign(y_pred_1d[valid_mask])
    true_sign = np.sign(y_true_1d[valid_mask])

    correct_direction = (pred_sign == true_sign)
    hit_rate = np.mean(correct_direction) * 100.0

    # High-Confidence Hit Rate (Filtered when predicted return > 0.5%)
    high_conf_mask = (np.abs(y_pred_1d) > 0.005) & valid_mask
    if np.sum(high_conf_mask) > 0:
        high_conf_hit_rate = np.mean(
            np.sign(y_pred_1d[high_conf_mask]) == np.sign(y_true_1d[high_conf_mask])
        ) * 100.0
    else:
        high_conf_hit_rate = 0.0

    # ----------------------------------------------------
    # 2. BACKTEST CUMULATIVE RETURNS
    # ----------------------------------------------------
    # Strategy Rules:
    # - Go LONG (+1) if predicted 1-day log return > 0
    # - Stay in CASH (0) if predicted 1-day log return <= 0
    positions = np.where(y_pred_1d > 0, 1.0, 0.0)

    # Strategy daily returns = Position * Actual Stock Log Return
    strategy_log_returns = positions * y_true_1d
    buy_hold_log_returns = y_true_1d

    # Convert log returns to percentage cumulative returns
    cum_strategy_returns = np.exp(np.cumsum(strategy_log_returns)) - 1.0
    cum_buy_hold_returns = np.exp(np.cumsum(buy_hold_log_returns)) - 1.0

    total_strategy_return = cum_strategy_returns[-1] * 100.0
    total_buy_hold_return = cum_buy_hold_returns[-1] * 100.0

    # Sharpe Ratio Approximation (Annualized, assuming 252 trading days)
    risk_free_rate = 0.06 / 252  # 6% annual risk-free rate
    excess_strat_ret = strategy_log_returns - risk_free_rate
    sharpe_ratio = (np.mean(excess_strat_ret) / (np.std(excess_strat_ret) + 1e-9)) * np.sqrt(252)

    # ----------------------------------------------------
    # PRINT RESULTS
    # ----------------------------------------------------
    print("\n" + "="*50)
    print("        TFT MODEL PERFORMANCE & BACKTEST")
    print("="*50)
    print(f" Total Evaluated Samples:      {len(y_pred_1d):,}")
    print(f" Directional Accuracy (Hit Rate): {hit_rate:.2f}%")
    print(f" High-Confidence Hit Rate (>0.5%): {high_conf_hit_rate:.2f}%")
    print("-" * 50)
    print(f" Cumulative TFT Strategy Return:  {total_strategy_return:+.2f}%")
    print(f" Cumulative Buy & Hold Return:    {total_buy_hold_return:+.2f}%")
    print(f" Outperformance (Alpha):          {total_strategy_return - total_buy_hold_return:+.2f}%")
    print(f" Annualized Sharpe Ratio:         {sharpe_ratio:.2f}")
    print("="*50 + "\n")

    # ----------------------------------------------------
    # 3. PLOT CUMULATIVE RETURN CURVE
    # ----------------------------------------------------
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(cum_strategy_returns * 100, label="TFT Model Strategy (Long / Cash)", color="green", linewidth=2)
        plt.plot(cum_buy_hold_returns * 100, label="Equal-Weighted Buy & Hold", color="gray", linestyle="--", alpha=0.7)
        plt.title("TFT Model Backtest: Out-of-Sample Cumulative Returns (%)")
        plt.xlabel("Test Samples Sequence")
        plt.ylabel("Cumulative Return (%)")
        plt.axhline(0, color="black", linestyle=":", alpha=0.5)
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.3)
        
        plot_filename = "tft_backtest_performance.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        print(f"Performance chart saved to '{plot_filename}'.")
    except Exception as e:
        print(f"Skipping plot saving: {e}")

if __name__ == "__main__":
    main()