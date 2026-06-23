# ============================================================
# ADVANCED MACHINE LEARNING FOR >90% ACCURACY
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# 1. Load Data
df = pd.read_csv("IBM HR Employee Attrition Data.csv")

# 2. Basic Cleaning & Dropping Constants/Redundant Columns
df = df.drop_duplicates()
# StandardHours and EmployeeCount have 0 variance (same value everywhere), drop them
cols_to_drop = [col for col in ['StandardHours', 'EmployeeCount', 'Over18', 'EmployeeNumber'] if col in df.columns]
df = df.drop(columns=cols_to_drop)

# 3. Handle Target Variable & Smart Encoding
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Separate features and target
X = df.drop(columns=['Attrition'])
y = df['Attrition']

# Fill Missing Values before Encoding
for col in X.columns:
    if X[col].dtype != "object":
        X[col] = X[col].fillna(X[col].mean())
    else:
        X[col] = X[col].fillna(X[col].mode()[0])

# One-Hot Encoding for Nominal Categorical Variables
X = pd.get_dummies(X, drop_first=True)

# 4. Train-Test Split (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Handle Imbalance with SMOTE (Applied ONLY to Training Data)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 6. Train a Highly-Tuned Random Forest
# Tuning hyperparameters like n_estimators, max_depth, and min_samples_split pushes accuracy up
advanced_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

advanced_model.fit(X_train_resampled, y_train_resampled)

# 7. Evaluate Performance
y_pred = advanced_model.predict(X_test)
y_prob = advanced_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*40)
print(f"ADVANCED MODEL ACCURACY: {accuracy * 100:.2f}%")
print("="*40)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")