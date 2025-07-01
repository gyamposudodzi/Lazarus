import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import os
import ta

# === Config ===
PAIR = "XAUUSD"
SAVE_PATH = "data/labeled/xauusd_final.csv"
TIMEFRAMES = {
    "m1": mt5.TIMEFRAME_M1,
    "m5": mt5.TIMEFRAME_M5,
    "m15": mt5.TIMEFRAME_M15,
    "m30": mt5.TIMEFRAME_M30
}
N_BARS = 3000  # pull how many candles per timeframe

# === Init MT5 ===
if not mt5.initialize():
    raise RuntimeError("MT5 initialization failed")

# === Pull Data ===
def fetch_data(tf):
    rates = mt5.copy_rates_from_pos(PAIR, TIMEFRAMES[tf], 0, N_BARS)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

# === Feature Generation ===
def add_indicators(df):
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['roc'] = ta.momentum.ROCIndicator(df['close']).roc()
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    return df

# === Load and merge all TFs ===
print("[INFO] Fetching MT5 data...")
df_1m = add_indicators(fetch_data("m1")).add_suffix("_1m")
df_5m = add_indicators(fetch_data("m5")).add_suffix("_5m")
df_15m = add_indicators(fetch_data("m15")).add_suffix("_15m")
df_30m = add_indicators(fetch_data("m30")).add_suffix("_30m")

# === Merge on nearest time index (inner join) ===
print("[INFO] Merging timeframes...")
df = df_5m.copy()
df = df.join(df_1m, how='inner', rsuffix='_1m')
df = df.join(df_15m, how='inner', rsuffix='_15m')
df = df.join(df_30m, how='inner', rsuffix='_30m')

# === Clean and drop NAs ===
df.dropna(inplace=True)

# === Create basic label (e.g. if close increases in next 3 candles) ===
print("[INFO] Labeling data...")
df['future_close'] = df['close_5m'].shift(-3)
df['label'] = (df['future_close'] > df['close_5m']).astype(int)
df.drop(columns=['future_close'], inplace=True)

# === Save dataset ===
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
df.to_csv(SAVE_PATH)
print(f"[OK] Saved labeled dataset to: {SAVE_PATH}")

# === Shutdown MT5 ===
mt5.shutdown()
