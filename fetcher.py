import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# 1. DEFINE TOP 100 TICKERS & SECTOR MAP
# ==========================================
STOCKS_BY_SECTOR = {
    # Financial Services & Banking
    "HDFCBANK.NS": "Financials", "ICICIBANK.NS": "Financials", "KOTAKBANK.NS": "Financials",
    "AXISBANK.NS": "Financials", "INDUSINDBK.NS": "Financials", "SBIN.NS": "Financials",
    "BANKBARODA.NS": "Financials", "CANBK.NS": "Financials", "PNB.NS": "Financials",
    "UNIONBANK.NS": "Financials", "BAJFINANCE.NS": "Financials", "BAJAJFINSV.NS": "Financials",
    "SHRIRAMFIN.NS": "Financials", "CHOLAFIN.NS": "Financials", "MUTHOOTFIN.NS": "Financials",
    "JIOFIN.NS": "Financials", "RECLTD.NS": "Financials", "PFC.NS": "Financials",
    "LICI.NS": "Financials", "HDFCLIFE.NS": "Financials", "SBILIFE.NS": "Financials",
    "ICICIGI.NS": "Financials",

    # Information Technology
    "TCS.NS": "IT", "INFY.NS": "IT", "HCLTECH.NS": "IT", "WIPRO.NS": "IT",
    "TECHM.NS": "IT", "LTIM.NS": "IT", "NAUKRI.NS": "IT", "OFSS.NS": "IT",

    # Automobile & Components
    "TATAMOTORS.NS": "Auto", "MARUTI.NS": "Auto", "M&M.NS": "Auto", "BAJAJ-AUTO.NS": "Auto",
    "EICHERMOT.NS": "Auto", "TVSMOTOR.NS": "Auto", "HEROMOTOCO.NS": "Auto",
    "BOSCHLTD.NS": "Auto", "MOTHERSON.NS": "Auto",

    # Energy, Oil & Gas
    "RELIANCE.NS": "Energy", "ONGC.NS": "Energy", "IOC.NS": "Energy", "BPCL.NS": "Energy",
    "GAIL.NS": "Energy", "COALINDIA.NS": "Energy", "ATGL.NS": "Energy", "OIL.NS": "Energy",

    # Power & Green Energy
    "NTPC.NS": "Power", "POWERGRID.NS": "Power", "TATAPOWER.NS": "Power",
    "ADANIPOWER.NS": "Power", "ADANIGREEN.NS": "Power", "ADANIENSOL.NS": "Power", "NHPC.NS": "Power",

    # Consumer Goods / FMCG
    "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG", "NESTLEIND.NS": "FMCG", "BRITANNIA.NS": "FMCG",
    "GODREJCP.NS": "FMCG", "DABUR.NS": "FMCG", "MARICO.NS": "FMCG", "VBL.NS": "FMCG",
    "COLPAL.NS": "FMCG", "UNITDSPR.NS": "FMCG", "TATACONSUM.NS": "FMCG",

    # Pharmaceuticals & Healthcare
    "SUNPHARMA.NS": "Pharma", "CIPLA.NS": "Pharma", "DRREDDY.NS": "Pharma",
    "DIVISLAB.NS": "Pharma", "TORNTPHARM.NS": "Pharma", "ZYDUSLIFE.NS": "Pharma",
    "APOLLOHOSP.NS": "Pharma", "MAXHEALTH.NS": "Pharma",

    # Metals & Mining
    "TATASTEEL.NS": "Metals", "JSWSTEEL.NS": "Metals", "HINDALCO.NS": "Metals",
    "JINDALSTEL.NS": "Metals", "VEDL.NS": "Metals",

    # Capital Goods, Infra, Defence, Cement
    "LT.NS": "Infra", "SIEMENS.NS": "Infra", "ABB.NS": "Infra", "BEL.NS": "Infra",
    "HAL.NS": "Infra", "CGPOWER.NS": "Infra", "SOLARINDS.NS": "Infra", "ADANIPORTS.NS": "Infra",
    "DLF.NS": "Infra", "LODHA.NS": "Infra", "AMBUJACEM.NS": "Infra",
    "ULTRACEMCO.NS": "Infra", "SHREECEM.NS": "Infra", "GRASIM.NS": "Infra",

    # Retail, Services & Telecom
    "DMART.NS": "Services", "TRENT.NS": "Services", "INDIGO.NS": "Services",
    "ADANIENT.NS": "Services", "ZOMATO.NS": "Services", "TITAN.NS": "Services",
    "IRCTC.NS": "Services", "IRFC.NS": "Services", "ASIANPAINT.NS": "Services",
    "BERGEPAINT.NS": "Services", "PIDILITIND.NS": "Services", "BHARTIARTL.NS": "Services"
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates 14-day Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

# ==========================================
# 3. MAIN DATA PIPELINE
# ==========================================
def build_tft_dataset(start_date="2021-01-01", end_date="2026-01-01"):
    print(f"--- Fetching Nifty Benchmark Index (^NSEI) ---")
    nifty_df = yf.download("^NSEI", start=start_date, end=end_date, progress=False)
    
    # Flatten yfinance multi-index columns if present
    if isinstance(nifty_df.columns, pd.MultiIndex):
        nifty_df.columns = nifty_df.columns.get_level_values(0)
        
    nifty_df = nifty_df.reset_index()
    nifty_close = nifty_df['Close'].squeeze()
    nifty_df['nifty_return'] = np.log(nifty_close / nifty_close.shift(1))
    nifty_map = nifty_df.set_index('Date')['nifty_return'].to_dict()

    print(f"--- Downloading 5 Years OHLCV Data for {len(STOCKS_BY_SECTOR)} Stocks ---")
    all_stock_data = []
    
    for ticker, sector in STOCKS_BY_SECTOR.items():
        try:
            stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if stock.empty:
                continue
            
            # Unpack columns
            if isinstance(stock.columns, pd.MultiIndex):
                stock.columns = stock.columns.get_level_values(0)
                
            stock = stock.reset_index()
            stock['symbol'] = ticker
            stock['sector'] = sector
            
            # Feature 1: Log Returns (Target)
            close = stock['Close'].squeeze()
            stock['log_return'] = np.log(close / close.shift(1))
            
            # Feature 2: RSI 14
            stock['rsi_14'] = calculate_rsi(close, period=14)
            
            # Feature 3: Volume Ratio (Current Vol / 20-day SMA Vol)
            vol = stock['Volume'].squeeze()
            vol_sma20 = vol.rolling(window=20).mean()
            stock['volume_ratio'] = vol / (vol_sma20 + 1e-9)
            
            # Feature 4: Nifty Index Return benchmark
            stock['nifty_return'] = stock['Date'].map(nifty_map)
            
            all_stock_data.append(stock)
        except Exception as e:
            print(f"Skipping {ticker} due to download error: {e}")

    # Combine into unified Panel DataFrame
    df = pd.concat(all_stock_data, ignore_index=True)
    
    print("--- Adding Temporal Features & Time Index ---")
    # Clean NaN rows generated by 14-day RSI and log return shifts
    df = df.dropna(subset=['log_return', 'rsi_14', 'volume_ratio', 'nifty_return']).reset_index(drop=True)

    # Global continuous time index (time_idx)
    unique_dates = sorted(df['Date'].unique())
    date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
    df['time_idx'] = df['Date'].map(date_to_idx)
    
    # Calendar features (Future Known)
    df['day_of_week'] = df['Date'].dt.dayofweek.astype(str)
    
    # Last Thursday of the month (F&O Expiry)
    is_thursday = df['Date'].dt.dayofweek == 3
    next_week_different_month = (df['Date'] + pd.Timedelta(days=7)).dt.month != df['Date'].dt.month
    df['is_fno_expiry'] = (is_thursday & next_week_different_month).astype(int).astype(str)

    # Select final TFT-relevant columns
    final_cols = [
        'time_idx', 'symbol', 'sector', 'Date',
        'log_return', 'rsi_14', 'volume_ratio', 'nifty_return',
        'day_of_week', 'is_fno_expiry'
    ]
    df_tft = df[final_cols].copy()
    
    return df_tft

# ==========================================
# 4. RUN PIPELINE & SAVE
# ==========================================
if __name__ == "__main__":
    tft_df = build_tft_dataset(start_date="2021-01-01", end_date="2026-01-01")
    
    # Save to CSV for TFT Model Training
    output_filename = "nifty100_tft_dataset.csv"
    tft_df.to_csv(output_filename, index=False)
    
    print(f"\n==================================================")
    print(f" SUCCESS! Dataset saved to '{output_filename}'")
    print(f" Total Rows: {len(tft_df):,}")
    print(f" Total Stocks: {tft_df['symbol'].nunique()}")
    print(f" Unique Time Steps (Days): {tft_df['time_idx'].nunique()}")
    print(f"==================================================\n")
    print(tft_df.head(10))