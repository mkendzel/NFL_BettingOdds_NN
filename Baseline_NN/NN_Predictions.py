import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import numpy as np

# --- 1) Load df for each season and define features of interest ---
# Training and validation data is based on the results of season 2020-2023
# Test data is based on the results of season 2024
df = pd.read_csv(r'\all_seasons_with_features_train.csv')
df_test = pd.read_csv(r'\schedule_2024.csv')

# feature columns
base_feats = [
    'spread_line',     # point spread
    'total_line',      # over/under line
    'under_odds',      
    'over_odds',       
    'div_game',        # intra-division game
    'temp',            # temperature
    'wind'             # wind speed
]

cat_feats = ['roof', 'surface']

# --- 2) Finalize X and Y for validation ---

#X, Train and validation set
X_num = df[base_feats].fillna(df[base_feats].median())
X_cat = pd.get_dummies(df[cat_feats], drop_first=True)

X = pd.concat([X_num, X_cat], axis=1)
y = df['Home_Win?']  # 1 if home team won, 0 otherwise

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    stratify=y,
    test_size=0.2,
    random_state=42
)

# Standardize numeric inputs
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# --- 3) Define and run the NN ---
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC']
)

model.summary()

# --- 4) Train with early stopping ---
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop]
)

# --- 5) Test model to 2024 season

# 1. Load test set
df_test = pd.read_csv(r'C:\NFl_NN\data\schedule_2024.csv')
df_test = df_test.dropna(how='all').reset_index(drop=True)

num_feats = ['spread_line', 'total_line', 'under_odds', 'over_odds', 'div_game', 'temp', 'wind']
cat_feats = ['roof', 'surface']

X_num = df_test[num_feats].fillna(df_test[num_feats].median())
X_cat = pd.get_dummies(df_test[cat_feats], drop_first=True)
train_cols = X.columns
X_test = pd.concat([X_num, X_cat], axis=1).reindex(columns=train_cols, fill_value=0)

# Scale and predict
X_test_scaled = scaler.transform(X_test)
y_proba = model.predict(X_test_scaled).ravel()
y_pred  = (y_proba >= 0.5).astype(int)

# Extract true labels (Something went wrong with import and I got about 7 rows of NA so this drops those)
y_true = df_test['Home_Win?'].astype(int)

# --- 6) Evaluate ----

train_proba = model.predict(X_train).ravel()
train_pred  = (train_proba >= 0.5).astype(int)

val_proba = model.predict(X_val).ravel()
val_pred  = (val_proba >= 0.5).astype(int)

# (you already have these for test)
# y_proba, y_pred on your 2024 test set
# y_true = df_test['Home_Win?'].astype(int)

# 2) Compute error (mis-classification) rate = 1 − accuracy
train_error = 1 - accuracy_score(y_train, train_pred)
val_error   = 1 - accuracy_score(y_val,   val_pred)
test_error  = 1 - accuracy_score(y_true,  y_pred)

print(f"Train error: {train_error:.3f}")
print(f" Val  error: {val_error:.3f}")
print(f" Test error: {test_error:.3f}")
