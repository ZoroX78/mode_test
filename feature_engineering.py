# feature_engineering.py
import numpy as np, pandas as pd
import pandas_ta as ta   # pip install pandas-ta

df = pd.read_parquet("data/nse_10y_panel.parquet")
meta = pd.read_csv("data/ticker_metadata.csv")
df = df.merge(meta[["symbol", "sector", "industry", "cap_tier"]],
              on="symbol", how="left")

# Sort so per-ticker rolling ops are correct
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

# ---------- 2.1 Returns ----------
g = df.groupby("symbol", group_keys=False)
df["log_ret_1d"]  = np.log(df["close"]).groupby(df["symbol"]).diff()
df["log_ret_5d"]  = np.log(df["close"]).groupby(df["symbol"]).diff(5)
df["log_ret_21d"] = np.log(df["close"]).groupby(df["symbol"]).diff(21)

# ---------- 2.2 Volatility ----------
df["realized_vol_21d"] = (df.groupby("symbol")["log_ret_1d"]
                          .transform(lambda x: x.rolling(21).std()))
# Garman-Klass intraday volatility
df["gk_vol"] = (0.5 * (np.log(df["high"]/df["low"]))**2
                - (2*np.log(2) - 1) * (np.log(df["close"]/df["open"]))**2)
df["gk_vol_21d"] = df.groupby("symbol")["gk_vol"].transform(lambda x: x.rolling(21).mean())

# ---------- 2.3 Technical indicators (per ticker) ----------
def add_indicators(group: pd.DataFrame) -> pd.DataFrame:
    close, high, low, vol = group["close"], group["high"], group["low"], group["volume"]
    out = pd.DataFrame(index=group.index)
    out["rsi_14"]     = ta.rsi(close, length=14)
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    out["macd"]       = macd.iloc[:, 0] if macd is not None else np.nan
    out["macd_signal"]= macd.iloc[:, 1] if macd is not None else np.nan
    out["macd_hist"]  = macd.iloc[:, 2] if macd is not None else np.nan
    bb = ta.bbands(close, length=20, std=2)
    out["bb_upper"], out["bb_mid"], out["bb_lower"] = bb.iloc[:,0], bb.iloc[:,1], bb.iloc[:,2]
    out["bb_width"]   = (out["bb_upper"] - out["bb_lower"]) / out["bb_mid"]
    out["atr_14"]     = ta.atr(high, low, close, length=14)
    out["adx_14"]     = ta.adx(high, low, close, length=14).iloc[:,0]
    out["obv"]        = ta.obv(close, vol)
    out["vwap_20"]    = (close * vol).rolling(20).sum() / vol.rolling(20).sum()
    out["sma_50"]     = ta.sma(close, length=50)
    out["sma_200"]    = ta.sma(close, length=200)
    out["ema_12"]     = ta.ema(close, length=12)
    out["ema_26"]     = ta.ema(close, length=26)
    # Distance from moving averages (normalized)
    out["dist_sma_50"]  = (close - out["sma_50"]) / out["sma_50"]
    out["dist_sma_200"] = (close - out["sma_200"]) / out["sma_200"]
    return out

df = pd.concat([df, df.groupby("symbol", group_keys=False).apply(add_indicators)], axis=1)

# ---------- 2.4 Volume features ----------
df["volume_z_20"] = (df.groupby("symbol")["volume"]
                     .transform(lambda x: (x - x.rolling(20).mean())/x.rolling(20).std()))
df["volume_chg_1d"] = df.groupby("symbol")["volume"].pct_change()

# ---------- 2.5 Calendar features (KNOWN FUTURE) ----------
df["dow"]     = df["date"].dt.dayofweek.astype(int)
df["month"]   = df["date"].dt.month.astype(int)
df["dom"]     = df["date"].dt.day.astype(int)
df["quarter"] = df["date"].dt.quarter.astype(int)
df["is_month_end"]    = df["date"].dt.is_month_end.astype(int)
df["is_quarter_end"]  = df["date"].dt.is_quarter_end.astype(int)
# F&O monthly expiry = last Thursday of month
def is_expiry(d):
    return (d.dayofweek == 3) and (d + pd.Timedelta(days=7)).month != d.month
df["is_expiry"] = df["date"].apply(is_expiry).astype(int)

# ---------- 2.6 Target: 5-day forward log-return ----------
df["target_5d_fwd"] = (df.groupby("symbol")["close"]
                       .transform(lambda x: np.log(x.shift(-5) / x)))

# ---------- 2.7 time_idx per group (TFT requires contiguous integer) ----------
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
df["time_idx"] = df.groupby("symbol").cumcount()

# ---------- 2.8 Drop warmup NaNs ----------
df = df.dropna(subset=["rsi_14", "macd", "atr_14", "sma_200",
                       "realized_vol_21d", "target_5d_fwd"]).reset_index(drop=True)

# Cast categoricals
for c in ["symbol", "sector", "industry", "cap_tier",
          "dow", "month", "dom", "quarter"]:
    df[c] = df[c].astype(str).astype("category")

df.to_parquet("data/tft_features.parquet", index=False)
print(f"Feature panel: {df.shape}, {df['symbol'].nunique()} tickers")