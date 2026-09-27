import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve
)

RANDOM_STATE = 42

DATA_PATH = "malicious_phish_updated.csv"
OUT_DIR = "ml_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================================
# 1. LOAD DATA
# =====================================================================
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("\nLabel distribution:")
print(df["label"].value_counts())

numeric_df = df.select_dtypes(include=[np.number])
feature_cols = [c for c in numeric_df.columns if c != "label"]
print(f"\nUsing {len(feature_cols)} numeric feature columns:")
print(feature_cols)

X = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(0)

y_raw = df["label"]
le = LabelEncoder()
y = le.fit_transform(y_raw)
print("\nEncoded classes:", dict(zip(le.classes_, range(len(le.classes_)))))

# =====================================================================
# 2. TRAIN / TEST SPLIT
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================================
# 3a. BASELINE: Logistic Regression
# =====================================================================
log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION (baseline)")
print("=" * 60)
print(classification_report(y_test, y_pred_lr, target_names=[str(c) for c in le.classes_]))

# =====================================================================
# 3b. Random Forest
# =====================================================================
rf = RandomForestClassifier(
    n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)
print(classification_report(y_test, y_pred_rf, target_names=[str(c) for c in le.classes_]))

cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("Random Forest — Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix_rf.png", dpi=150)
plt.close()

if len(le.classes_) == 2:
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    auc_rf = roc_auc_score(y_test, y_proba_rf)
    print(f"ROC-AUC (Random Forest): {auc_rf:.4f}")
    fpr, tpr, _ = roc_curve(y_test, y_proba_rf)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"RF (AUC={auc_rf:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Random Forest")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/roc_curve_rf.png", dpi=150)
    plt.close()
else:
    auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test), multi_class="ovr", average="macro")
    print(f"ROC-AUC macro (Random Forest): {auc_rf:.4f}")

importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 15 features by importance (Random Forest):")
print(importances.head(15))

plt.figure(figsize=(8, 6))
importances.head(15).iloc[::-1].plot(kind="barh")
plt.title("Top 15 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance_rf.png", dpi=150)
plt.close()

# =====================================================================
# 3c. INNOVATION MODEL: XGBoost
# =====================================================================
try:
    from xgboost import XGBClassifier

    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    print("\n" + "=" * 60)
    print("XGBOOST (innovation model)")
    print("=" * 60)
    print(classification_report(y_test, y_pred_xgb, target_names=[str(c) for c in le.classes_]))

    if len(le.classes_) == 2:
        auc_xgb = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
        print(f"ROC-AUC (XGBoost): {auc_xgb:.4f}  vs  Random Forest: {auc_rf:.4f}")
    else:
        auc_xgb = roc_auc_score(y_test, xgb.predict_proba(X_test), multi_class="ovr", average="macro")
        print(f"ROC-AUC macro (XGBoost): {auc_xgb:.4f}  vs  Random Forest: {auc_rf:.4f}")

except ImportError:
    print("\n[xgboost not installed — run: pip install xgboost --break-system-packages]")

print(f"\nDone. Plots saved to ./{OUT_DIR}/")