import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load data
train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"

train = pd.read_csv(train_url)
test = pd.read_csv(test_url)

print("Train shape:", train.shape)
print("Class balance:\n", train['meal'].value_counts())

# prep
def prepare_features(df):
    df = df.copy()

    # Drop identifier column — carries no predictive signal
    df.drop(columns=['id'], inplace=True, errors='ignore')

    # Extract useful time features from DateTime
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['hour'] = df['DateTime'].dt.hour
    df['dow'] = df['DateTime'].dt.dayofweek  # 0=Mon ... 6=Sun
    df['month'] = df['DateTime'].dt.month
    df.drop(columns=['DateTime'], inplace=True)

    return df


train_clean = prepare_features(train.drop(columns=['meal']))
test_clean = prepare_features(test.drop(columns=['meal']) if 'meal' in test.columns else test)

# Align columns (guards against any mismatch between train and test)
train_clean, test_clean = train_clean.align(test_clean, join='left', axis=1, fill_value=0)

y = train['meal'].astype(int)
X = train_clean.astype(float)
X_test = test_clean.astype(float)

#Validation Split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model
# XGBoost handles sparse binary item columns well.
# scale_pos_weight corrects for class imbalance automatically.
neg, pos = (y == 0).sum(), (y == 1).sum()

model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=neg / pos,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
)

# Accuracy Check
model.fit(X_tr, y_tr)
val_preds = model.predict(X_val)
print(f"Validation accuracy: {accuracy_score(y_val, val_preds):.4f}")

#Refit Model
modelFit = model.fit(X, y)

# Test set predictions
pred = modelFit.predict(X_test).astype(int).tolist()

print(f"Predictions generated  : {len(pred)}")
print(f"Predicted meals    (1) : {sum(pred)}")
print(f"Predicted non-meals (0): {len(pred) - sum(pred)}")