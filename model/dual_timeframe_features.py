import MetaTrader5 as mt5
import pandas as pd
import ta
from datetime import datetime

# === Parameters ===
SYMBOL = "XAUUSD"
COUNT = 5000

# === Initialize MT5 ===
if not mt5.initialize():
    raise RuntimeError("MT5 initialization failed")

# === Fetch 15m and 30m data ===
data_15m = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, COUNT)
data_30m = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M30, 0, COUNT)

mt5.shutdown()

# === Convert to DataFrame ===
df_15m = pd.DataFrame(data_15m)
df_30m = pd.DataFrame(data_30m)

# === Convert timestamps ===
df_15m['time'] = pd.to_datetime(df_15m['time'], unit='s')
df_30m['time'] = pd.to_datetime(df_30m['time'], unit='s')

# === Add indicators for 15m ===
df_15m['rsi_15m'] = ta.momentum.RSIIndicator(df_15m['close']).rsi()
df_15m['ema_15m'] = ta.trend.EMAIndicator(df_15m['close'], window=20).ema_indicator()
df_15m['macd_15m'] = ta.trend.MACD(df_15m['close']).macd_diff()

# === Add indicators for 30m ===
df_30m['rsi_30m'] = ta.momentum.RSIIndicator(df_30m['close']).rsi()
df_30m['ema_30m'] = ta.trend.EMAIndicator(df_30m['close'], window=20).ema_indicator()
df_30m['macd_30m'] = ta.trend.MACD(df_30m['close']).macd_diff()

# === Reduce columns ===
features_15m = df_15m[['time', 'close', 'rsi_15m', 'ema_15m', 'macd_15m']]
features_30m = df_30m[['time', 'rsi_30m', 'ema_30m', 'macd_30m']]

# === Merge 15m and 30m on nearest earlier 30m time ===
merged = pd.merge_asof(features_15m.sort_values('time'),
                        features_30m.sort_values('time'),
                        on='time',
                        direction='backward')

# === Drop NaNs and save ===
merged.dropna(inplace=True)
merged.to_csv("data/processed/xauusd_merged.csv", index=False)

print("Feature data saved to data/processed/xauusd_merged.csv")
