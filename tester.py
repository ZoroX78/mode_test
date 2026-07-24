import pandas as pd
import numpy as np

# ==========================================
# 1. FEATURE ENGINEERING FUNCTIONS
# ==========================================

def compute_rolling_beta(df_stock: pd.DataFrame, window: int = 60) -> pd.Series:
    """Calculates 60-day rolling Beta of a stock relative to Nifty 50."""
    cov = df_stock['log_return'].rolling(window).cov(df_stock['nifty_return'])
    var = df_stock['nifty_return'].rolling(window).var()
    beta = cov / (var + 1e-9)
    return beta.fillna(1.0) # Fallback to market beta = 1.0


def engineer_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies technical, momentum, volatility, and market-relative
    feature transformations on a per-symbol basis.
    """
    print("--- Sorting & Grouping Data per Stock Symbol ---")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['symbol', 'Date']).reset_index(drop=True)

    engineered_list = []

    for symbol, group in df.groupby('symbol'):
        group = group.copy()
        
        # ----------------------------------------------------
        # A. MOMENTUM & MULTI-HORIZON RETURNS
        # ----------------------------------------------------
        # Past multi-day cumulative returns (5-day = 1 week, 21-day = 1 month)
        group['log_ret_5d'] = group['log_return'].rolling(5).sum()
        group['log_ret_21d'] = group['log_return'].rolling(21).sum()

        # ----------------------------------------------------
        # B. VOLATILITY & RISK REGIME FEATURES
        # ----------------------------------------------------
        # 10-day & 20-day rolling historical volatility (std dev of log returns)
        group['volatility_10d'] = group['log_return'].rolling(10).std()
        group['volatility_20d'] = group['log_return'].rolling(20).std()
        
        # Volatility ratio: Short-term vs Medium-term volatility
        group['vol_ratio_10_20'] = group['volatility_10d'] / (group['volatility_20d'] + 1e-9)

        # ----------------------------------------------------
        # C. MARKET INTERACTION & ALPHA FEATURES
        # ----------------------------------------------------
        # 1. Excess Return (Alpha): Stock Return minus Nifty Return
        group['excess_return_1d'] = group['log_return'] - group['nifty_return']
        group['excess_return_5d'] = group['log_ret_5d'] - group['nifty_return'].rolling(5).sum()
        
        # 2. Rolling Beta relative to Nifty 50
        group['beta_60d'] = compute_rolling_beta(group, window=60)

        # ----------------------------------------------------
        # D. TECHNICAL INDICATOR DERIVATIVES
        # ----------------------------------------------------
        # RSI Momentum (change in RSI over 3 days)
        group['rsi_change_3d'] = group['rsi_14'].diff(3)

        # Volume Surge Indicator (Ratio of volume_ratio relative to 5-day mean)
        group['volume_surge'] = group['volume_ratio'] / (group['volume_ratio'].rolling(5).mean() + 1e-9)

        engineered_list.append(group)

    # Recombine all stock dataframes
    df_feat = pd.concat(engineered_list, ignore_index=True)

    # --------------------------------------------------------
    # E. CALENDAR & SEASONAL FEATURES (FUTURE KNOWN)
    # --------------------------------------------------------
    print("--- Generating Future-Known Calendar Features ---")
    df_feat['month'] = df_feat['Date'].dt.month.astype(str)
    df_feat['day_of_month'] = df_feat['Date'].dt.day
    df_feat['quarter'] = df_feat['Date'].dt.quarter.astype(str)
    
    # Financial Calendar flags (India specific)
    df_feat['is_month_end'] = df_feat['Date'].dt.is_month_end.astype(int).astype(str)
    df_feat['is_union_budget_month'] = (df_feat['Date'].dt.month == 2).astype(int).astype(str) # February Union Budget

    # --------------------------------------------------------
    # F. CLEANUP & TIME INDEX RE-ALIGNMENT
    # --------------------------------------------------------
    print("--- Cleaning NaNs from Rolling Windows ---")
    # Drop rows where rolling windows generated NaNs (e.g. 60-day rolling beta)
    df_clean = df_feat.dropna().reset_index(drop=True)

    # Re-index time_idx to ensure contiguous sequential integers without gaps
    unique_dates = sorted(df_clean['Date'].unique())
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    df_clean['time_idx'] = df_clean['Date'].map(date_to_idx)

    return df_clean


# ==========================================
# 2. MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    # Load raw dataset generated from the previous script
    input_file = "nifty100_tft_dataset.csv"
    print(f"Loading '{input_file}'...")
    raw_df = pd.read_csv(input_file)

    # Apply feature engineering
    processed_df = engineer_stock_features(raw_df)

    # Export finalized feature matrix
    output_file = "nifty100_tft_engineered_dataset.csv"
    processed_df.to_csv(output_file, index=False)

    print("\n==================================================")
    print(f" FEATURE ENGINEERING COMPLETE!")
    print(f" Saved to: '{output_file}'")
    print(f" Total Records: {len(processed_df):,}")
    print(f" Total Features: {processed_df.shape[1]}")
    print(f" Unique Time Steps (time_idx): 0 to {processed_df['time_idx'].max()}")
    print("==================================================\n")
    print(processed_df[['time_idx', 'symbol', 'log_return', 'volatility_20d', 'beta_60d', 'excess_return_1d']].head())