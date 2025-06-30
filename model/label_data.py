import pandas as pd
from datetime import timedelta

# === Load merged features ===
df = pd.read_csv("data/processed/xauusd_merged.csv")
df['time'] = pd.to_datetime(df['time'])

# === Create label column ===
# 1 if price is higher after 4 hours, else 0
future_window = timedelta(hours=4)
labels = []

for idx, current_row in df.iterrows():
    current_time = current_row['time']
    future_time = current_time + future_window

    # Find the closest future row
    future_rows = df[df['time'] >= future_time]
    if not future_rows.empty:
        future_price = future_rows.iloc[0]['close']
        current_price = current_row['close']
        label = 1 if future_price > current_price else 0
    else:
        label = None  # not enough data ahead

    labels.append(label)

# === Attach labels and drop rows without labels ===
df['label'] = labels
df.dropna(inplace=True)
df['label'] = df['label'].astype(int)

# === Save to labeled file ===
df.to_csv("data/labeled/xauusd_labeled.csv", index=False)
print("[OK] Labeled data saved to data/labeled/xauusd_labeled.csv")
