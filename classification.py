import os
import string
import numpy as np
import matplotlib
matplotlib.use('Agg')  # file-only backend - we only savefig(), never show(), so skip
                        # the Tkinter GUI backend entirely (avoids the "main thread is
                        # not in main loop" errors from RandomForest's parallel workers)
import matplotlib.pyplot as plot

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, accuracy_score
)

RANDOM_STATE = 42
OUT_DIR = "ml_outputs"

# character-CNN settings
MAX_URL_LEN = 200       # longer URLs get truncated, shorter ones padded
CNN_EPOCHS = 5
CNN_BATCH_SIZE = 256


def classify(df):
    os.makedirs(OUT_DIR, exist_ok=True)

# --------------------Build feature matrix--------------------
    # Use every numeric column except label as a feature.
    # url and tld are text, so they get dropped automatically here.
    numeric_df = df.select_dtypes(include=[np.number])
    feature_cols = [c for c in numeric_df.columns if c != 'label']
    print(f'Using {len(feature_cols)} features:')
    print(feature_cols)

    X = df[feature_cols].fillna(0)
    urls = df['url'].values

    le = LabelEncoder()
    y = le.fit_transform(df['label'])

# --------------------Train/test split--------------------
    # Split on row indices rather than X directly, so the SAME rows end up in
    # train/test for every model below - including the CNN, which uses raw url
    # text instead of X. That keeps the comparison between models fair.
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    url_train, url_test = urls[train_idx], urls[test_idx]

    # Logistic Regression is distance/gradient based so it needs scaled features.
    # Random Forest and XGBoost split on raw values so they don't need scaling.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Track predictions/probabilities for every model that trains successfully,
    # so we can build one combined summary + comparison at the end.
    y_pred_xgb, y_proba_xgb = None, None
    y_pred_cnn, y_proba_cnn = None, None

# =====================================================================
# Model 1 - Logistic Regression (baseline)
# =====================================================================
    log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    log_reg.fit(X_train_scaled, y_train)
    y_pred_lr = log_reg.predict(X_test_scaled)
    y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

    print('\n--- Logistic Regression ---')
    print(classification_report(y_test, y_pred_lr, target_names=[str(c) for c in le.classes_]))
    _save_confusion_matrix(y_test, y_pred_lr, le.classes_, 'Logistic Regression', 'confusion_matrix_lr.png')

# =====================================================================
# Model 2 - Random Forest
# =====================================================================
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]

    print('\n--- Random Forest ---')
    print(classification_report(y_test, y_pred_rf, target_names=[str(c) for c in le.classes_]))
    _save_confusion_matrix(y_test, y_pred_rf, le.classes_, 'Random Forest', 'confusion_matrix_rf.png')
    # NOTE: feature importance analysis is handled separately by a teammate - not duplicated here.

# =====================================================================
# Model 3 (innovation 1) - XGBoost
# =====================================================================
    # Random Forest trains many trees independently and averages the votes.
    # XGBoost trains trees one at a time, where each new tree tries to fix the
    # mistakes of the previous ones (gradient boosting). Still a tree-based
    # model like RF though - see Model 4 below for a genuinely different approach.
    try:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1,
        )
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        y_proba_xgb = xgb.predict_proba(X_test)[:, 1]

        print('\n--- XGBoost ---')
        print(classification_report(y_test, y_pred_xgb, target_names=[str(c) for c in le.classes_]))
        _save_confusion_matrix(y_test, y_pred_xgb, le.classes_, 'XGBoost', 'confusion_matrix_xgb.png')

    except ImportError:
        print('\nxgboost not installed - run: pip install xgboost --break-system-packages')

# =====================================================================
# Model 4 (innovation 2) - Character-level CNN
# =====================================================================
    # All 3 models above rely on OUR hand-crafted features (url_length, has_https,
    # entropy, etc). This model is a different approach entirely: it reads the raw
    # URL character-by-character and learns its own patterns, with no feature
    # engineering at all. The idea is that a small neural network can pick up on
    # character sequences that look suspicious (e.g. "paypal-secure-login",
    # unusual mixes of digits and letters) that our fixed feature list might miss
    # or that only make sense in combination/order, which simple counts can't capture.
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models, callbacks

        # Fixed character vocabulary (printable ASCII) so every run encodes
        # characters the same way, rather than fitting a vocab per run.
        vocab = sorted(set(string.printable))
        char_to_idx = {ch: i + 1 for i, ch in enumerate(vocab)}  # 0 reserved for padding/unknown
        vocab_size = len(vocab) + 1

        def encode_urls(url_array, max_len=MAX_URL_LEN):
            encoded = np.zeros((len(url_array), max_len), dtype=np.int32)
            for i, u in enumerate(url_array):
                u = str(u)[:max_len]
                for j, ch in enumerate(u):
                    encoded[i, j] = char_to_idx.get(ch, 0)
            return encoded

        X_train_cnn = encode_urls(url_train)
        X_test_cnn = encode_urls(url_test)

        cnn = models.Sequential([
            layers.Input(shape=(MAX_URL_LEN,)),
            layers.Embedding(input_dim=vocab_size, output_dim=32),
            layers.Conv1D(64, 5, activation='relu'),
            layers.GlobalMaxPooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid'),
        ])
        cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        cnn.fit(
            X_train_cnn, y_train,
            validation_split=0.1,
            epochs=CNN_EPOCHS, batch_size=CNN_BATCH_SIZE,
            callbacks=[callbacks.EarlyStopping(patience=2, restore_best_weights=True)],
            verbose=2,
        )

        y_proba_cnn = cnn.predict(X_test_cnn, batch_size=512).ravel()
        y_pred_cnn = (y_proba_cnn >= 0.5).astype(int)

        print('\n--- Character-level CNN ---')
        print(classification_report(y_test, y_pred_cnn, target_names=[str(c) for c in le.classes_]))
        _save_confusion_matrix(y_test, y_pred_cnn, le.classes_, 'Character CNN', 'confusion_matrix_cnn.png')

    except ImportError:
        print('\ntensorflow not installed - run: pip install tensorflow --break-system-packages')
        print('NOTE: this can take a while to install and to train (minutes, not seconds) on CPU.')

# =====================================================================
# Evaluation - ROC curves, overall summary, and pairwise comparison
# =====================================================================
    model_entries = [
        ('Logistic Regression', y_pred_lr, y_proba_lr),
        ('Random Forest', y_pred_rf, y_proba_rf),
        ('XGBoost', y_pred_xgb, y_proba_xgb),
        ('Character CNN', y_pred_cnn, y_proba_cnn),
    ]

    # ROC curves, all models on one graph for easy comparison
    plot.figure(figsize=(7, 7))
    for name, _, proba in model_entries:
        if proba is None:
            continue
        auc = roc_auc_score(y_test, proba)
        fpr, tpr, _ = roc_curve(y_test, proba)
        plot.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    plot.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random guess')
    plot.xlabel('False Positive Rate')
    plot.ylabel('True Positive Rate')
    plot.title('ROC Curve Comparison')
    plot.legend()
    plot.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    plot.savefig(f'{OUT_DIR}/roc_curve_comparison.png', dpi=150)
    plot.close()

    # Per-model summary: accuracy + ROC-AUC
    summary = []
    for name, pred, proba in model_entries:
        if pred is None:
            continue
        acc = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, proba) if proba is not None else None
        summary.append((name, acc, auc))

    print('\n' + '=' * 60)
    print('MODEL SUMMARY')
    print('=' * 60)
    print(f"{'Model':<22}{'Accuracy':>12}{'ROC-AUC':>12}")
    for name, acc, auc in summary:
        auc_str = f'{auc:.4f}' if auc is not None else 'N/A'
        print(f'{name:<22}{acc:>12.4f}{auc_str:>12}')

    # Quick "which model wins" line, ranked by ROC-AUC (falls back to accuracy
    # for any model where AUC isn't available).
    ranked = sorted(summary, key=lambda r: r[2] if r[2] is not None else r[1], reverse=True)
    best_name, best_acc, best_auc = ranked[0]
    best_auc_str = f'{best_auc:.4f}' if best_auc is not None else f'{best_acc:.4f} (accuracy)'
    print(f'\nBest model: {best_name}  (ROC-AUC = {best_auc_str})')

    print(f'\nPlots saved to ./{OUT_DIR}/')


def _save_confusion_matrix(y_true, y_pred, class_names, title, filename):
    os.makedirs(OUT_DIR, exist_ok=True)  # guards against the folder being deleted mid-run
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plot.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plot.title(f'{title} - Confusion Matrix')
    plot.tight_layout()
    plot.savefig(f'{OUT_DIR}/{filename}', dpi=150)
    plot.close()