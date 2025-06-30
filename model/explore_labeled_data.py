import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Load labeled data ===
df = pd.read_csv("data/labeled/xauusd_labeled.csv")

# === Label distribution ===
label_counts = df['label'].value_counts()
print("Label Distribution:")
print(label_counts)

# Plot label distribution
sns.countplot(x='label', data=df)
plt.title("Label Distribution (0 = Down/Flat, 1 = Up)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

# === Basic stats ===
print("\nDescriptive Statistics:")
print(df.describe())

# === Correlation matrix ===
plt.figure(figsize=(10, 6))
corr = df.drop(columns=['time']).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()
