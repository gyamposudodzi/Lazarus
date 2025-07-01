import pandas as pd
import MetaTrader5 as mt5
import joblib
import ta
import time
from datetime import datetime

# === Config ===
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
MODEL_PATH = "model/xgboost_model_v1.pkl"
N_BARS = 2000

# === Init MT5 ===
if not mt5.initialize():
    raise RuntimeError("MT5 initialization failed")

# === Load Model ===
model = joblib.load(MODEL_PATH)
expected_features = model.get_booster().feature_names
print("[OK] Loaded model and expecting features:")
print(expected_features)

# === Track last candle times ===
last_1m_candle_time = None
last_5m_candle_time = None
last_15m_candle_time = None

# === Utility Functions ===
def resample(df, tf, label):
    agg = df.resample(tf).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'tick_volume': 'sum', 'spread': 'mean', 'real_volume': 'sum'
    }).dropna()
    agg.columns = [f"{col}_{label}" for col in agg.columns]
    return agg

def add_indicators(df, label):
    df[f'rsi_{label}'] = ta.momentum.rsi(df[f'close_{label}'], window=14)
    df[f'macd_{label}'] = ta.trend.macd(df[f'close_{label}'])
    df[f'macd_signal_{label}'] = ta.trend.macd_signal(df[f'close_{label}'])
    df[f'adx_{label}'] = ta.trend.adx(df[f'high_{label}'], df[f'low_{label}'], df[f'close_{label}'])
    df[f'atr_{label}'] = ta.volatility.average_true_range(df[f'high_{label}'], df[f'low_{label}'], df[f'close_{label}'])
    bb = ta.volatility.BollingerBands(df[f'close_{label}'])
    df[f'bb_high_{label}'] = bb.bollinger_hband()
    df[f'bb_low_{label}'] = bb.bollinger_lband()
    df[f'roc_{label}'] = ta.momentum.roc(df[f'close_{label}'])
    df[f'stoch_k_{label}'] = ta.momentum.stoch(df[f'high_{label}'], df[f'low_{label}'], df[f'close_{label}'])
    df[f'stoch_d_{label}'] = ta.momentum.stoch_signal(df[f'high_{label}'], df[f'low_{label}'], df[f'close_{label}'])

def get_latest_candle_time(timeframe_minutes):
    """Get the latest candle time for a specific timeframe"""
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1)
    if rates is None or len(rates) == 0:
        return None
    
    latest_1m_time = pd.to_datetime(rates[0]['time'], unit='s')
    
    # Calculate the candle time for different timeframes
    if timeframe_minutes == 1:
        return latest_1m_time
    elif timeframe_minutes == 5:
        # Round down to nearest 5-minute interval
        minutes = latest_1m_time.minute
        rounded_minutes = (minutes // 5) * 5
        return latest_1m_time.replace(minute=rounded_minutes, second=0, microsecond=0)
    elif timeframe_minutes == 15:
        # Round down to nearest 15-minute interval
        minutes = latest_1m_time.minute
        rounded_minutes = (minutes // 15) * 15
        return latest_1m_time.replace(minute=rounded_minutes, second=0, microsecond=0)
    
    return None

def make_prediction(signal_type):
    """Make prediction and print signal"""
    try:
        # === Fetch M1 data ===
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, N_BARS)
        if rates is None or len(rates) == 0:
            print("[ERROR] Failed to fetch rates from MT5")
            return
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index("time", inplace=True)

        # === Resample and calculate indicators ===
        df_1m = df.copy()
        df_1m.columns = [f"{col}_1m" for col in df_1m.columns]
        df_5m = resample(df, "5min", "5m")
        df_15m = resample(df, "15min", "15m")
        df_30m = resample(df, "30min", "30m")

        for d, label in [(df_1m, "1m"), (df_5m, "5m"), (df_15m, "15m"), (df_30m, "30m")]:
            add_indicators(d, label)

        # === Merge timeframes ===
        df_merged = df_1m.merge(df_5m, left_index=True, right_index=True, how="left")
        df_merged = df_merged.merge(df_15m, left_index=True, right_index=True, how="left")
        df_merged = df_merged.merge(df_30m, left_index=True, right_index=True, how="left")
        df_merged.dropna(inplace=True)

        if df_merged.empty:
            print("[ERROR] No data available after merging timeframes")
            return

        # === Predict ===
        latest = df_merged.iloc[-1:]
        X_live = latest[expected_features]
        prediction = model.predict(X_live)[0]
        proba = model.predict_proba(X_live)[0][1]

        now = datetime.now()
        print(f"[{now}] {signal_type} Signal: {'BUY' if prediction == 1 else 'SELL'} | Confidence: {proba:.2f}")
        
    except KeyError as e:
        print("[ERROR] Missing features in live data:", e)
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")

# === Start Live Loop ===
print("[INFO] Starting candle-based prediction system...")
print("[INFO] Checking for new 1m candles every 5 seconds")
print("[INFO] Checking for new 5m candles every 1 minute") 
print("[INFO] Checking for new 15m candles every 15 minutes")

check_5m_counter = 0  # Counter to check 5m every 1 minute (12 * 5sec = 60sec)
check_15m_counter = 0 # Counter to check 15m every 15 minutes (180 * 5sec = 900sec)

while True:
    try:
        # === Check for new 1m candle every 5 seconds ===
        current_1m_time = get_latest_candle_time(1)
        if current_1m_time and current_1m_time != last_1m_candle_time:
            print(f"[INFO] New 1m candle detected at {current_1m_time}")
            make_prediction("1m")
            last_1m_candle_time = current_1m_time
        
        # === Check for new 5m candle every 1 minute (every 12 iterations) ===
        check_5m_counter += 1
        if check_5m_counter >= 12:  # 12 * 5 seconds = 60 seconds
            current_5m_time = get_latest_candle_time(5)
            if current_5m_time and current_5m_time != last_5m_candle_time:
                print(f"[INFO] New 5m candle detected at {current_5m_time}")
                make_prediction("5m")
                last_5m_candle_time = current_5m_time
            check_5m_counter = 0
        
        # === Check for new 15m candle every 15 minutes (every 180 iterations) ===
        check_15m_counter += 1
        if check_15m_counter >= 180:  # 180 * 5 seconds = 900 seconds = 15 minutes
            current_15m_time = get_latest_candle_time(15)
            if current_15m_time and current_15m_time != last_15m_candle_time:
                print(f"[INFO] New 15m candle detected at {current_15m_time}")
                make_prediction("15m")
                last_15m_candle_time = current_15m_time
            check_15m_counter = 0
        
        time.sleep(5)  # Wait 5 seconds before next check
        
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
        break
    except Exception as e:
        print(f"[ERROR] Main loop error: {e}")
        time.sleep(5)  # Continue after error

# === Cleanup ===
mt5.shutdown()
print("[INFO] MT5 connection closed")