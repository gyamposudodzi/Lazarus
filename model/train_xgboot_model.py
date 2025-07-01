import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# === Load final training data ===
df = pd.read_csv("data/labeled/xauusd_multi_tf_indicators.csv")
X = df.drop(columns=["time", "label", "close_1m", "open_1m", "high_1m", "low_1m"], errors='ignore')
y = df["label"]




# === Correlation Heatmap ===
corr = df.corr(numeric_only=True)
plt.figure(figsize=(14, 10))
sns.heatmap(corr, cmap="coolwarm", annot=False, fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === Handle class imbalance ===
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos

# === Determine new version number ===
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)
existing_models = [f for f in os.listdir(model_dir) if f.startswith("xgboost_model_v") and f.endswith(".pkl")]
version = len(existing_models) + 1

# === Define model and train ===
model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss'
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

# === Evaluate ===
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
conf_mat = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# === Save model ===
model_path = f"{model_dir}/xgboost_model_v{version}.pkl"
joblib.dump(model, model_path)

# === Save metadata ===
metadata = {
    "version": version,
    "model_file": os.path.basename(model_path),
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "features": list(X.columns),
    "label_distribution": {
        "class_0": int(neg),
        "class_1": int(pos)
    },
    "accuracy": round(acc, 4),
    "auc": round(auc, 4),
    "confusion_matrix": conf_mat.tolist(),
    "classification_report": report,
    "eval_set": "20% test split",
    "timeframes": ["1m", "5m", "15m"],  # Adjust if necessary
    "bar_count": len(df)
}

with open(f"{model_dir}/xgboost_model_v{version}_meta.json", "w") as f:
    json.dump(metadata, f, indent=4)

# === Log results ===
print(f"[ok] Trained and saved as {model_path}")
print(f"Accuracy: {acc:.2f} | AUC: {auc:.2f}")
print("Confusion Matrix:\n", conf_mat)
print("Classification Report:\n", classification_report(y_test, y_pred))

# === Plot feature importance ===
xgb.plot_importance(model)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()
