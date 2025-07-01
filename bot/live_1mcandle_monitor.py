import MetaTrader5 as mt5
from datetime import datetime
import time

SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
CHECK_INTERVAL = 10 # seconds between checks

# Initialize MT5
if not mt5.initialize():
    raise RuntimeError("Failed to initialize MetaTrader 5")

print("[OK] MT5 initialized.")
last_time = None

while True:
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 2)
    if rates is None or len(rates) < 2:
        print("Failed to fetch candles.")
        time.sleep(CHECK_INTERVAL)
        continue

    latest_candle_time = datetime.fromtimestamp(rates[-1]['time'])

    if latest_candle_time != last_time:
        print(f"[{datetime.now()}] New 1m candle detected at {latest_candle_time}")
        last_time = latest_candle_time

        # 🔜 Here is where we’ll later trigger prediction + trading
    else:
        print(f"[{datetime.now()}] No new candle. Waiting...")

    time.sleep(CHECK_INTERVAL)
