import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt

# === Load final training data ===
df = pd.read_csv("data/labeled/xauusd_final.csv")

X = df.drop(columns=["label"])
y = df["label"]

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === Train XGBoost classifier ===
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# === Save model ===
joblib.dump(model, "model/xgboost_model.pkl")
print("[OK] Trained XGBoost model saved as model/xgboost_model.pkl")

# === Evaluate ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# === Plot feature importance ===
xgb.plot_importance(model)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()
