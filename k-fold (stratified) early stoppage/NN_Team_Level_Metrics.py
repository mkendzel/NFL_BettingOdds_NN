import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import numpy as np

# ------- Data import (not included) ----------------
X_full = df_train[base_feats].fillna(df_train[base_feats].median()).values
y_full = df_train['Home_win?'].values

# --------Stratified K-Fold setup ---------
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

train_errors = []
val_errors   = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full), 1):
    print(f"\n=== Fold {fold}/{n_splits} ===")
    X_train, X_val = X_full[train_idx], X_full[val_idx]
    y_train, y_val = y_full[train_idx], y_full[val_idx]

    # ------ scale per fold ------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    # ------ build & compile ------
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )

    # ------ early stopping ------
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # ------ fit ------
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    # ------ get predictions & compute errors ------
    train_pred = (model.predict(X_train) > 0.5).astype(int).flatten()
    val_pred   = (model.predict(X_val)   > 0.5).astype(int).flatten()

    train_err = 1 - accuracy_score(y_train, train_pred)
    val_err   = 1 - accuracy_score(y_val,   val_pred)

    print(f"  train_error: {train_err:.3f}, val_error: {val_err:.3f}")
    train_errors.append(train_err)
    val_errors.append(val_err)

# --- final CV error summary ------
print(f"\nMean train_error: {np.mean(train_errors):.3f} ± {np.std(train_errors):.3f}")
print(f"Mean val_error  : {np.mean(val_errors):.3f} ± {np.std(val_errors):.3f}")
