import MetaTrader5 as mt5
import pandas as pd
from datetime import timedelta
import ta  # Technical Analysis library
import pandas_ta as pta

# === Configuration ===
symbol = "XAUUSD"
timeframes = {
    "1m": mt5.TIMEFRAME_M1,
    "5m": mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15
}
bar_counts = {
    "1m": 20000,
    "5m": 5000,
    "15m": 2000
}

# === Connect to MT5 ===
if not mt5.initialize():
    raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")

# === Pull data from multiple timeframes ===
dataframes = {}
for label, tf in timeframes.items():
    bars = bar_counts[label]
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"Failed to get data for {label}")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns=lambda col: f"{col}_{label}" if col != "time" else "time")
    dataframes[label] = df

# === Merge all timeframes on time ===
merged = dataframes["1m"]
for label, df in dataframes.items():
    if label == "1m":
        continue
    merged = pd.merge_asof(
        merged.sort_values("time"),
        df.sort_values("time"),
        on="time",
        direction="backward"
    )

# === Drop NaNs caused by alignment ===
merged["label"] = (merged["close_1m"].shift(-30) > merged["close_1m"]).astype(int)
merged.dropna(inplace=True) 



# === Add Technical Indicators ===
merged['rsi_1m'] = pta.rsi(merged['close_1m'], length=14)
macd = pta.macd(merged['close_1m'], fast=12, slow=26, signal=9)
merged['macd_1m'] = macd['MACD_12_26_9']
merged['macd_signal_1m'] = macd['MACDs_12_26_9']
merged['macd_hist_1m'] = macd['MACDh_12_26_9']
merged['ema_1m_20'] = pta.ema(merged['close_1m'], length=20)
merged['sma_1m_20'] = pta.sma(merged['close_1m'], length=20)
merged['atr_1m'] = pta.atr(merged['high_1m'], merged['low_1m'], merged['close_1m'], length=14)




# === Compute indicators using 'ta' library ===
merged = merged.copy()
merged.set_index("time", inplace=True)  # ta prefers indexed DataFrames

# SMA and EMA
merged["sma_20"] = ta.trend.sma_indicator(merged["close_1m"], window=20)
merged["ema_20"] = ta.trend.ema_indicator(merged["close_1m"], window=20)

# RSI
merged["rsi_14"] = ta.momentum.rsi(merged["close_1m"], window=14)

# MACD
macd = ta.trend.macd(merged["close_1m"])
merged["macd"] = macd

# Bollinger Bands
bb = ta.volatility.BollingerBands(merged["close_1m"])
merged["bb_upper"] = bb.bollinger_hband()
merged["bb_lower"] = bb.bollinger_lband()

# ATR
merged["atr"] = ta.volatility.average_true_range(
    merged["high_1m"], merged["low_1m"], merged["close_1m"], window=14
)

# Drop new NaNs after indicators
merged.dropna(inplace=True)

# Save updated processed data
merged.reset_index(inplace=True)
merged.to_csv("data/labeled/xauusd_multi_tf_indicators.csv", index=False)
print("[OK] Saved dataset with indicators to xauusd_multi_tf_indicators.csv")


# === Labeling: TP/SL logic with 30-minute lookahead ===
tp_pips = 10
sl_pips = 10
pip_value = 0.1  # 1 pip = 0.1 for XAUUSD

tp_distance = tp_pips * pip_value
sl_distance = sl_pips * pip_value

labels = []
for i in range(len(merged)):
    entry_price = merged.iloc[i]['close_1m']
    entry_time = merged.iloc[i]['time']
    future_window = merged[
        (merged['time'] > entry_time) &
        (merged['time'] <= entry_time + timedelta(minutes=30))
    ]

    label = 0  # default if nothing is hit
    for _, row in future_window.iterrows():
        price = row['close_1m']
        if price >= entry_price + tp_distance:
            label = 1
            break
        elif price <= entry_price - sl_distance:
            label = 0
            break
    labels.append(label)

merged['label'] = labels

# === Save labeled dataset ===
merged.to_csv("data/labeled/xauusd_final.csv", index=False)
print("[OK] Labeled data saved to data/labeled/xauusd_final.csv")

# === Shutdown MT5 ===
mt5.shutdown()
