# Advanced Supply and Demand MT5 Trading Bot
# Author: ChatGPT (OpenAI)
# Strategy: Multi-timeframe Supply & Demand with Confirmation (patterns, volume, volatility)

import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
import logging
import numpy as np
from typing import List, Dict, Optional
import ta

# === CONFIGURATION ===
SYMBOLS = ["XAUUSD", "GBPUSD", "USOIL", "UKOIL", "USDJPY"]
TIMEFRAMES = [mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_M30]
RISK_PERCENT = 0.05
CANDLE_WINDOW = 100
SLIPPAGE = 5

# === LOGGING ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SupplyDemandBot")

# === MT5 INITIALIZATION ===
if not mt5.initialize():
    logger.error("Failed to initialize MT5")
    quit()
else:
    account = mt5.account_info()
    if account:
        logger.info(f"Connected to MT5 | Account: {account.login} | Balance: {account.balance}")
    else:
        logger.error("Connected to MT5, but no account info found.")

# === STRATEGY UTILITIES ===
def get_rates(symbol, timeframe, n=CANDLE_WINDOW):
    logger.debug(f"Fetching {n} candles for {symbol} at TF={timeframe}")
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    return pd.DataFrame(rates) if rates is not None else pd.DataFrame()

def detect_zones(df: pd.DataFrame) -> List[Dict]:
    logger.debug("Detecting supply and demand zones")
    zones = []
    for i in range(2, len(df)-2):
        if df['low'][i] < df['low'][i-1] and df['low'][i] < df['low'][i+1]:
            zones.append({"type": "demand", "price": df['low'][i], "index": i})
        if df['high'][i] > df['high'][i-1] and df['high'][i] > df['high'][i+1]:
            zones.append({"type": "supply", "price": df['high'][i], "index": i})
    logger.debug(f"Zones found: {zones}")
    return zones

def is_engulfing(df, i):
    if i < 1: return False
    c1 = df.iloc[i-1]
    c2 = df.iloc[i]
    result = c1['close'] < c1['open'] and c2['close'] > c2['open'] and c2['close'] > c1['open'] and c2['open'] < c1['close']
    logger.debug(f"Engulfing pattern at {i}: {result}")
    return result

def volume_spike(df):
    vol = df['tick_volume']
    avg_vol = vol.rolling(20).mean()
    spike = vol.iloc[-1] > avg_vol.iloc[-1] * 1.5
    logger.debug(f"Volume spike: {spike}")
    return spike

def atr_stoploss(df):
    atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    value = atr.iloc[-1] * 2
    logger.debug(f"ATR-based SL: {value}")
    return value

def trend_direction(df):
    direction = "up" if df['close'].iloc[-1] > df['close'].rolling(20).mean().iloc[-1] else "down"
    logger.debug(f"Trend direction: {direction}")
    return direction

def align_trend(lower, higher):
    aligned = trend_direction(lower) == trend_direction(higher)
    logger.debug(f"Trend aligned: {aligned}")
    return aligned

def calculate_lot(symbol, sl_pips, balance):
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info or sl_pips == 0:
        return 0.01
    value_per_pip = symbol_info.trade_tick_value
    risk_amount = balance * RISK_PERCENT
    lot = risk_amount / (sl_pips * value_per_pip)
    return round(min(max(lot, 0.01), 10), 2)



# === TRADE SIGNAL ===
def generate_signal(symbol) -> Optional[Dict]:
    logger.info(f"Generating signal for {symbol}")
    lower_tf = get_rates(symbol, mt5.TIMEFRAME_M5)
    higher_tf = get_rates(symbol, mt5.TIMEFRAME_H1)

    if lower_tf.empty or higher_tf.empty:
        logger.warning("Price data missing")
        return None

    lower_tf['time'] = pd.to_datetime(lower_tf['time'], unit='s')
    higher_tf['time'] = pd.to_datetime(higher_tf['time'], unit='s')

    zones = detect_zones(lower_tf)
    last_candle = lower_tf.iloc[-1]

    for z in reversed(zones):
        price = z['price']
        logger.debug(f"Checking zone {z}")
        if z['type'] == 'demand' and last_candle['low'] <= price:
            if is_engulfing(lower_tf, len(lower_tf) - 1) and volume_spike(lower_tf) and align_trend(lower_tf, higher_tf):
                sl = price - atr_stoploss(lower_tf)
                tp = price + 2 * (price - sl)
                logger.info("BUY signal generated")
                return {"symbol": symbol, "action": "BUY", "entry": last_candle['close'], "sl": sl, "tp": tp}
        elif z['type'] == 'supply' and last_candle['high'] >= price:
            if is_engulfing(lower_tf, len(lower_tf) - 1) and volume_spike(lower_tf) and align_trend(lower_tf, higher_tf):
                sl = price + atr_stoploss(lower_tf)
                tp = price - 2 * (sl - price)
                logger.info("SELL signal generated")
                return {"symbol": symbol, "action": "SELL", "entry": last_candle['close'], "sl": sl, "tp": tp}
    logger.debug("No valid signal found")
    return None

def generate_signal_from_df(symbol, df: pd.DataFrame) -> Optional[Dict]:
    higher_tf = get_rates(symbol, mt5.TIMEFRAME_H1)
    if higher_tf.empty:
        return None

    higher_tf['time'] = pd.to_datetime(higher_tf['time'], unit='s')
    df['time'] = pd.to_datetime(df['time'], unit='s')

    zones = detect_zones(df)
    last_candle = df.iloc[-1]

    for z in reversed(zones):
        price = z['price']
        if z['type'] == 'demand' and last_candle['low'] <= price:
            if is_engulfing(df, len(df) - 1) and volume_spike(df) and align_trend(df, higher_tf):
                sl = price - atr_stoploss(df)
                tp = price + 2 * (price - sl)
                return {"symbol": symbol, "action": "BUY", "entry": last_candle['close'], "sl": sl, "tp": tp}
        elif z['type'] == 'supply' and last_candle['high'] >= price:
            if is_engulfing(df, len(df) - 1) and volume_spike(df) and align_trend(df, higher_tf):
                sl = price + atr_stoploss(df)
                tp = price - 2 * (sl - price)
                return {"symbol": symbol, "action": "SELL", "entry": last_candle['close'], "sl": sl, "tp": tp}
    return None

# === BACKTESTING ===
def backtest_bot(start_date: str, end_date: str):
    logger.info(f"Backtesting from {start_date} to {end_date}")
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_trades = 0
    wins = 0
    losses = 0
    profit = 0.0

    for symbol in SYMBOLS:
        df = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start, end)
        if df is None or len(df) == 0:
            logger.warning(f"No data for {symbol}")
            continue
        df = pd.DataFrame(df)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        for i in range(50, len(df)):
            sub_df = df.iloc[:i].copy()
            sub_df.reset_index(drop=True, inplace=True)
            signal = generate_signal_from_df(symbol, sub_df)
            if signal:
                entry = signal['entry']
                sl = signal['sl']
                tp = signal['tp']
                direction = signal['action']

                future = df.iloc[i:i+10]
                hit_tp = hit_sl = False

                for _, row in future.iterrows():
                    if direction == "BUY":
                        if row['low'] <= sl:
                            hit_sl = True
                            break
                        elif row['high'] >= tp:
                            hit_tp = True
                            break
                    elif direction == "SELL":
                        if row['high'] >= sl:
                            hit_sl = True
                            break
                        elif row['low'] <= tp:
                            hit_tp = True
                            break

                total_trades += 1
                if hit_tp:
                    wins += 1
                    profit += abs(tp - entry)
                elif hit_sl:
                    losses += 1
                    profit -= abs(entry - sl)

    logger.info(f"Backtest Result — Trades: {total_trades}, Wins: {wins}, Losses: {losses}, Net: {profit:.2f}")
    if total_trades > 0:
        win_rate = (wins / total_trades) * 100
        logger.info(f"Win Rate: {win_rate:.2f}%")

# === MAIN LOOP ===
def run_bot():
    logger.info("Starting Supply & Demand Trading Bot")
    while True:
        for symbol in SYMBOLS:
            logger.info(f"Checking {symbol}...")
            signal = generate_signal(symbol)
            if signal:
                logger.info(f"Signal: {signal}")
                execute_trade(signal)
        time.sleep(60)

# === TRADE EXECUTION ===
def execute_trade(signal):
    symbol = signal['symbol']
    price = signal['entry']
    sl = signal['sl']
    tp = signal['tp']
    action = signal['action']

    acc = mt5.account_info()
    if acc is None:
        logger.error("No account info")
        return

    sl_pips = abs(price - sl)
    lot = calculate_lot(symbol, sl_pips, acc.balance)

    order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
    entry_price = mt5.symbol_info_tick(symbol).ask if action == "BUY" else mt5.symbol_info_tick(symbol).bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": entry_price,
        "sl": sl,
        "tp": tp,
        "deviation": SLIPPAGE,
        "magic": 123456890,
        "comment": "SupplyDemandTrade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"Trade executed: {action} {symbol} @ {entry_price}, SL={sl}, TP={tp}")
    else:
        logger.error(f"Order failed: {result}")

# === ENTRY POINT ===
if __name__ == "__main__":
    # Uncomment the one you want to run:
    #run_bot()
    backtest_bot("2024-12-01", "2024-12-31")
