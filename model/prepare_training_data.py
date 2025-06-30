import pandas as pd

# === Load labeled data ===
df = pd.read_csv("data/labeled/xauusd_labeled.csv")

# === Drop redundant features ===
df.drop(columns=[
    'close',        # strongly correlated with ema_15m & ema_30m
    'ema_15m',      # nearly identical to ema_30m
    'rsi_15m',      # highly correlated with rsi_30m
    'macd_15m'      # moderate correlation with macd_30m, but keeping just one
], inplace=True)

# === Optional: reorder columns ===
columns = ['rsi_30m', 'ema_30m', 'macd_30m', 'label']
df = df[columns]

# === Save cleaned dataset ===
df.to_csv("data/labeled/xauusd_final.csv", index=False)
print("[OK] Final training dataset saved to data/labeled/xauusd_final.csv")
