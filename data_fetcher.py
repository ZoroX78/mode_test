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
RAW_TICKERS = """3M India Ltd. (3MINDIA)ABB India Ltd. (ABB)Abbott India Ltd. (ABBOTINDIA)Aditya Birla Capital Ltd. (ABCAPITAL)Aditya Birla Fashion and Retail Ltd. (ABFRL)Aegis Logistics Ltd. (AEGISLOG)Affle (India) Ltd. (AFFLE)Aia Engineering Ltd. (AIAENG)Ajanta Pharma Ltd. (AJANTPHARM)Akzo Nobel India Ltd. (AKZOINDIA)Alembic Pharmaceuticals Ltd. (APLLTD)Alkem Laboratories Ltd. (ALKEM)Allcargo Logistics Ltd. (ALLCARGO)Ambuja Cements Ltd. (AMBUJACEM)Anand Rathi Wealth Ltd. (ANANDRATHI)Apollo Hospitals Enterprise Ltd. (APOLLOHOSP)Apollo Tyres Ltd. (APOLLOTYRE)Archean Chemical Industries Ltd. (ARCHEAN)Asahi India Glass Ltd. (ASAHIINDIA)Ashok Leyland Ltd. (ASHOKLEY)Asian Paints Ltd. (ASIANPAINT)Aster DM Healthcare Ltd. (ASTERDM)Astral Ltd. (ASTRAL)Atul Ltd. (ATUL)AU Small Finance Bank Ltd. (AUBANK)Aurobindo Pharma Ltd. (AUROPHARMA)Automobile Corp of Goa Ltd. (ACGL)Avenue Supermarts Ltd. (DMART)Axis Bank Ltd. (AXISBANK)Bajaj Auto Ltd. (BAJAJ-AUTO)Bajaj Electricals Ltd. (BAJAJELEC)Bajaj Finance Ltd. (BAJFINANCE)Bajaj Finserv Ltd. (BAJAJFINSV)Bajaj Holdings & Investment Ltd. (BAJAJHLDNG)Balkrishna Industries Ltd. (BALKRISIND)Balrampur Chini Mills Ltd. (BALRAMCHIN)Bandhan Bank Ltd. (BANDHANBNK)Bank of Baroda (BANKBARODA)Bank of India (BANKINDIA)BASF India Ltd. (BASF)Bata India Ltd. (BATAINDIA)Bayer Cropscience Ltd. (BAYERCROP)Bharat Electronics Ltd. (BEL)Bharat Forge Ltd. (BHARATFORG)Bharat Heavy Electricals Ltd. (BHEL)Bharat Petroleum Corporation Ltd. (BPCL)Bharti Airtel Ltd. (BHARTIARTL)Biocon Ltd. (BIOCON)Birla Corporation Ltd. (BIRLACORPN)Blue Star Ltd. (BLUESTARCO)Bosch Ltd. (BOSCHLTD)Brigade Enterprises Ltd. (BRIGADE)Britannia Industries Ltd. (BRITANNIA)BSE Ltd. (BSE)Camlin Fine Sciences Ltd. (CAMLINFINE)Can Fin Homes Ltd. (CANFINHOME)Canara Bank (CANBK)Carborundum Universal Ltd. (CARBORUNIV)Castrol India Ltd. (CASTROLIND)CEAT Ltd. (CEATLTD)Central Bank of India (CENTRALBK)Century Plyboards (India) Ltd. (CENTURYPLY)Century Textiles & Industries Ltd. (CENTURYTEX)Cera Sanitaryware Ltd. (CERA)Chalet Hotels Ltd. (CHALET)Chambal Fertilizers & Chemicals Ltd. (CHAMBLFERT)Chemplast Sanmar Ltd. (CHEMPLAST)Cholamandalam Investment and Finance Co Ltd. (CHOLAFIN)Cipla Ltd. (CIPLA)City Union Bank Ltd. (CUB)Clean Science and Technology Ltd. (CLEAN)Coal India Ltd. (COALINDIA)Cochin Shipyard Ltd. (COCHINSHIP)Coforge Ltd. (COFORGE)Colgate-Palmolive (India) Ltd. (COLPAL)Computer Age Management Services Ltd. (CAMS)Container Corporation of India Ltd. (CONCOR)Coromandel International Ltd. (COROMANDEL)Craftsman Automation Ltd. (CRAFTSMAN)CreditAccess Grameen Ltd. (CREDITACC)Crompton Greaves Consumer Electricals Ltd. (CROMPTON)Cummins India Ltd. (CUMMINSIND)Cyient Ltd. (CYIENT)Dabur India Ltd. (DABUR)Dalmia Bharat Ltd. (DALBHARAT)Data Patterns (India) Ltd. (DATAPATTERNS)DCB Bank Ltd. (DCBBANK)Deepak Nitrite Ltd. (DEEPAKNTR)Delhivery Ltd. (DELHIVERY)Delta Corp Ltd. (DELTACORP)Devyani International Ltd. (DEVYANI)Dish TV India Ltd. (DISHTV)Divi's Laboratories Ltd. (DIVISLAB)Dixon Technologies (India) Ltd. (DIXON)DLF Ltd. (DLF)Dollar Industries Ltd. (DOLLAR)Dr. Reddy's Laboratories Ltd. (DRREDDY)eClerx Services Ltd. (ECLERX)Edelweiss Financial Services Ltd. (EDELWEISS)Eicher Motors Ltd. (EICHERMOT)Elgi Equipments Ltd. (ELGIEQUIP)Emami Ltd. (EMAMILTD)Endurance Technologies Ltd. (ENDURANCE)EPL Ltd. (EPL)Equitas Small Finance Bank Ltd. (EQUITASBNK)Escorts Kubota Ltd. (ESCORTS)Exide Industries Ltd. (EXIDEIND)Federal Bank Ltd. (FEDERALBNK)Fiem Industries Ltd. (FIEMIND)Finolex Cables Ltd. (FINCABLES)Finolex Industries Ltd. (FINPIPE)Firstsource Solutions Ltd. (FSL)Five-Star Business Finance Ltd. (FIVESTAR)Fortis Healthcare Ltd. (FORTIS)FSN E-Commerce Ventures Ltd. (NYKAA)Gabriel India Ltd. (GABRIEL)Gail (India) Ltd. (GAIL)Galaxy Surfactants Ltd. (GALAXYSURF)Gangotri Paper Mills Ltd.Garden Reach Shipbuilders & Engineers Ltd. (GRSE)Gateway Distriparks Ltd. (GATEWAY)GE T&D India Ltd. (GET&D)General Insurance Corporation of India (GICRE)GHCL Ltd. (GHCL)Gillette India Ltd. (GILLETTE)GlaxoSmithKline Pharmaceuticals Ltd. (GLAXO)Glenmark Pharmaceuticals Ltd. (GLENMARK)Global Health Ltd. (MEDANTA)GMR Airports Infrastructure Ltd. (GMRINFRA)Godfrey Phillips India Ltd. (GODFRYPHLP)Godrej Agrovet Ltd. (GODREJAGRO)Godrej Consumer Products Ltd. (GODREJCP)Godrej Industries Ltd. (GODREJIND)Godrej Properties Ltd. (GODREJPROP)Goodricke Group Ltd. (GOODRICKE)Granules India Ltd. (GRANULES)Grasim Industries Ltd. (GRASIM)Great Eastern Shipping Co Ltd. (GESHIP)Grindwell Norton Ltd. (GRINDWELL)Gujarat Ambuja Exports Ltd. (GAEL)Gujarat Gas Ltd. (GUJGASLTD)Gujarat Narmada Valley Fertilizers and Chemicals Ltd. (GNFC)Gujarat State Fertilizers & Chemicals Ltd. (GSFC)Gujarat State Petronet Ltd. (GSPL)Haldyn Glass Ltd. (`HALDYNG’)Happiest Minds Technologies Ltd. (HAPPSTMNDS)Hatsun Agro Product Ltd. (HATSUN)Havells India Ltd. (HAVELLS)HCL Technologies Ltd. (HCLTECH)HDFC Asset Management Co Ltd. (HDFCAMC)HDFC Bank Ltd. (HDFCBANK)HDFC Life Insurance Co Ltd. (HDFCLIFE)HeidelbergCement India Ltd. (HEIDELBERG)Hemisphere Properties India Ltd. (HEMIPROP)Heranba Industries Ltd. (HERANBA)Hero MotoCorp Ltd. (HEROMOTOCO)Hester Biosciences Ltd. (HESTERBIO)HFCL Ltd. (HFCL)HG Infra Engineering Ltd. (HGINFRA)Hi-Tech Pipes Ltd. (HITECH)Hikal Ltd. (HIKAL)Himadri Speciality Chemical Ltd. (HSCL)Hindalco Industries Ltd. (HINDALCO)Hind Rectifiers Ltd. (HIRECT)Hindustan Aeronautics Ltd. (HAL)Hindustan Construction Co Ltd. (HCC)Hindustan Copper Ltd. (HINDCOPPER)Hindustan Media Ventures Ltd. (HMVL)Hindustan Oil Exploration Co Ltd. (HOEC)Hindustan Petroleum Corporation Ltd. (HPCL)Hindustan Unilever Ltd. (HINDUNILVR)Hindustan Zinc Ltd. (HINDZINC)Hindware Home Innovation Ltd. (HINDWARE)Hindzen Overseas Ltd.Hitech Corp Ltd. (HITECHCORP)Hi-Tech Gears Ltd. (HITECHGEAR)HLV Ltd. (HLVLTD)HMA Agro Industries Ltd. (HMAAGRO)Home First Finance Company India Ltd. (HOMEFIRST)Honeywell Automation India Ltd. (HONAUT)Honda India Power Products Ltd. (HONDAPOWER)Horizon z"""

# ---------- 1.2 Parse "Name (SYMBOL)" pairs ----------
# Match (TICKER) tokens. Names like "Affle (India) Ltd. (AFFLE)" contain a
# non-ticker paren group; skip those via NON_TICKER_PARENS.
PATTERN = re.compile(r"\((?P<sym>[A-Z0-9\-&\.]+)\)")
NON_TICKER_PARENS = {"INDIA"}  # e.g. "Colgate-Palmolive (India) Ltd."

def parse_tickers(raw: str):
    pairs = []
    last_end = 0
    for m in PATTERN.finditer(raw):
        sym = m.group("sym")
        # drop obvious junk / non-ticker parentheticals
        if len(sym) < 2 or sym.startswith("`") or sym.upper() in NON_TICKER_PARENS:
            continue
        name = raw[last_end:m.start()].strip().rstrip(".").strip()
        last_end = m.end()
        if not name:
            continue
        pairs.append({"name": name, "symbol": sym})
    return pairs

ticker_df = pd.DataFrame(parse_tickers(RAW_TICKERS))
if ticker_df.empty or "symbol" not in ticker_df.columns:
    raise ValueError(
        "No tickers parsed from RAW_TICKERS. "
        "Paste the full 'Name (SYMBOL)' list into RAW_TICKERS."
    )
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
DATA_DIR = Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
meta_df.to_csv(DATA_DIR / "ticker_metadata.csv", index=False)

# ---------- 1.4 Fetch 10 years of OHLCV per ticker ----------
END   = datetime.today()
START = END - timedelta(days=10 * 365 + 5)  # 10y + buffer
CACHE_DIR = DATA_DIR / "raw"; CACHE_DIR.mkdir(parents=True, exist_ok=True)

def normalize_ohlcv(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns and standardize names."""
    # yfinance >=0.2 often returns MultiIndex (Price, Ticker) columns
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            c[0] if isinstance(c, tuple) else c
            for c in df.columns
        ]
    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    # index column may be date / datetime depending on yfinance version
    if "date" not in df.columns:
        for alt in ("datetime", "index"):
            if alt in df.columns:
                df = df.rename(columns={alt: "date"})
                break
    df["symbol"] = sym
    return df

def fetch_one(sym: str, retries: int = 4):
    cache = CACHE_DIR / f"{sym}.parquet"
    if cache.exists():
        return sym, pd.read_parquet(cache)
    yf_sym = f"{sym}.NS"
    last_err = None
    for attempt in range(retries):
        try:
            df = yf.download(yf_sym, start=START, end=END,
                             interval="1d", auto_adjust=True,
                             progress=False, threads=False)
            if df is None or df.empty:
                time.sleep(1.5 * (attempt + 1)); continue
            df = normalize_ohlcv(df, sym)
            df.to_parquet(cache)
            return sym, df
        except Exception as e:
            last_err = e
            # 429 Too Many Requests -> exponential backoff
            wait = 2 ** attempt + np.random.rand()
            time.sleep(wait)
    if last_err is not None:
        print(f"  fail {sym}: {type(last_err).__name__}: {last_err}")
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
if not results:
    raise SystemExit(
        "No OHLCV data downloaded for any ticker. "
        "Check network / Yahoo Finance rate limits and retry."
    )

all_df = pd.concat(list(results.values()), ignore_index=True)
all_df["date"] = pd.to_datetime(all_df["date"])

# Drop rows with NaN OHLCV (Yahoo sometimes returns NaNs)
ohlcv_cols = [c for c in ("close", "high", "low", "open", "volume") if c in all_df.columns]
if len(ohlcv_cols) < 5:
    raise SystemExit(f"Missing OHLCV columns after download; got {list(all_df.columns)}")
all_df = all_df.dropna(subset=ohlcv_cols)

# Drop tickers with too few rows (e.g., recent IPOs)
counts = all_df.groupby("symbol").size()
keep = counts[counts >= 1500].index   # >= ~6y of data
all_df = all_df[all_df["symbol"].isin(keep)].reset_index(drop=True)
all_df.to_parquet(DATA_DIR / "nse_10y_panel.parquet", index=False)
print(f"Final panel: {all_df['symbol'].nunique()} tickers, "
      f"{len(all_df):,} rows, {all_df['date'].min().date()} → {all_df['date'].max().date()}")
