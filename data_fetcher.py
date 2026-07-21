# fetch_data.py
import os, time, re, pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# ---------- 1.1 Raw ticker string (paste full list here) ----------
RAW_TICKERS = """3M India Ltd. (3MINDIA)ABB India Ltd. (ABB)..."""  # paste your full string

# ---------- 1.2 Parse "Name (SYMBOL)" pairs ----------
PATTERN = re.compile(r"\((?P<sym>[A-Z0-9\-&\.]+)\)")
def parse_tickers(raw: str):
    pairs = []
    # Split on closing paren that ends a "(SYMBOL)" token
    parts = re.split(r"\)\s*", raw)
    for p in parts:
        m = PATTERN.search(p)
        if not m:
            continue
        sym = m.group("sym")
        name = p[:m.start()].strip().rstrip(".").strip()
        # drop obvious junk symbols
        if len(sym) < 2 or sym.startswith("`"):
            continue
        pairs.append({"name": name, "symbol": sym})
    return pairs

ticker_df = pd.DataFrame(parse_tickers(RAW_TICKERS))
# Deduplicate (some entries like HITECH appear twice)
ticker_df = ticker_df.drop_duplicates(subset="symbol").reset_index(drop=True)
print(f"Parsed {len(ticker_df)} unique tickers")

# ---------- 1.3 Sector / industry metadata (static categoricals) ----------
def fetch_metadata(sym: str):
    try:
        info = yf.Ticker(f"{sym}.NS").info
        return {
            "symbol": sym,
            "sector":   info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": info.get("marketCap", np.nan),
        }
    except Exception:
        return {"symbol": sym, "sector": "Unknown",
                "industry": "Unknown", "market_cap": np.nan}

meta_rows = []
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = [ex.submit(fetch_metadata, s) for s in ticker_df["symbol"]]
    for f in tqdm(as_completed(futs), total=len(futs), desc="metadata"):
        meta_rows.append(f.result())
meta_df = pd.DataFrame(meta_rows)
# Cap-tier from market cap (large/mid/small)
def cap_tier(mc):
    if pd.isna(mc): return "unknown"
    if mc >= 2e11:  return "large"     # > ₹20,000 cr
    if mc >= 5e10:  return "mid"       # > ₹5,000 cr
    return "small"
meta_df["cap_tier"] = meta_df["market_cap"].apply(cap_tier)
meta_df.to_csv("data/ticker_metadata.csv", index=False)

# ---------- 1.4 Fetch 10 years of OHLCV per ticker ----------
END   = datetime.today()
START = END - timedelta(days=10 * 365 + 5)  # 10y + buffer
CACHE_DIR = Path("data/raw"); CACHE_DIR.mkdir(parents=True, exist_ok=True)

def fetch_one(sym: str, retries: int = 4):
    cache = CACHE_DIR / f"{sym}.parquet"
    if cache.exists():
        return sym, pd.read_parquet(cache)
    yf_sym = f"{sym}.NS"
    for attempt in range(retries):
        try:
            df = yf.download(yf_sym, start=START, end=END,
                             interval="1d", auto_adjust=True,
                             progress=False, threads=False)
            if df is None or df.empty:
                time.sleep(1.5 * (attempt + 1)); continue
            df = df.reset_index().rename(columns=str.lower)
            df.columns = [c.replace(" ", "_") for c in df.columns]
            df["symbol"] = sym
            df.to_parquet(cache)
            return sym, df
        except Exception as e:
            # 429 Too Many Requests -> exponential backoff
            wait = 2 ** attempt + np.random.rand()
            time.sleep(wait)
    return sym, None

results, failed = {}, []
# Batched concurrency: 6 workers keeps Yahoo from rate-limiting
BATCH = 6
with ThreadPoolExecutor(max_workers=BATCH) as ex:
    futs = {ex.submit(fetch_one, s): s for s in ticker_df["symbol"]}
    for f in tqdm(as_completed(futs), total=len(futs), desc="downloading"):
        sym, df = f.result()
        if df is None or df.empty:
            failed.append(sym)
        else:
            results[sym] = df

print(f"OK: {len(results)} | FAILED: {len(failed)} -> {failed}")

# ---------- 1.5 Combine + sanity checks ----------
all_df = pd.concat([df for df in results.values() if df is not None],
                   ignore_index=True)
all_df["date"] = pd.to_datetime(all_df["date"])

# Drop rows with NaN OHLCV (Yahoo sometimes returns NaNs)
all_df = all_df.dropna(subset=["close", "high", "low", "open", "volume"])

# Drop tickers with too few rows (e.g., recent IPOs)
counts = all_df.groupby("symbol").size()
keep = counts[counts >= 1500].index   # >= ~6y of data
all_df = all_df[all_df["symbol"].isin(keep)].reset_index(drop=True)
all_df.to_parquet("data/nse_10y_panel.parquet", index=False)
print(f"Final panel: {all_df['symbol'].nunique()} tickers, "
      f"{len(all_df):,} rows, {all_df['date'].min().date()} → {all_df['date'].max().date()}")