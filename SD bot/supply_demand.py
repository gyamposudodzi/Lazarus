# supply_demand.py

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_zones(symbol: str, timeframe: int, bars: int = 500) -> dict:
    """Identify supply and demand zones from price history."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        return {"supply": [], "demand": []}

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    zones = {"supply": [], "demand": []}

    for i in range(2, len(df) - 2):
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]

        prev_highs = df['high'].iloc[i-2:i]
        next_highs = df['high'].iloc[i+1:i+3]

        prev_lows = df['low'].iloc[i-2:i]
        next_lows = df['low'].iloc[i+1:i+3]

        # Detect supply zone (local high surrounded by lower highs)
        if high > prev_highs.max() and high > next_highs.max():
            zones["supply"].append({
                "price": high,
                "time": df['time'].iloc[i],
                "strength": high - df['open'].iloc[i]  # crude strength
            })

        # Detect demand zone (local low surrounded by higher lows)
        if low < prev_lows.min() and low < next_lows.min():
            zones["demand"].append({
                "price": low,
                "time": df['time'].iloc[i],
                "strength": df['close'].iloc[i] - low
            })

    return zones

def find_nearest_zone(zones: list, price: float, direction: str = "buy") -> dict:
    """Find nearest supply or demand zone based on direction."""
    if not zones:
        return None

    if direction == "buy":
        filtered = [z for z in zones if z['price'] < price]
        if not filtered:
            return None
        return min(filtered, key=lambda z: abs(price - z['price']))

    if direction == "sell":
        filtered = [z for z in zones if z['price'] > price]
        if not filtered:
            return None
        return min(filtered, key=lambda z: abs(price - z['price']))

    return None
