# -------------------------
# 1. Import Libraries
# -------------------------
import pandas as pd
import numpy as np
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# -------------------------
# 2. Load the Dataset
# -------------------------
df = pd.read_csv("IBM HR Employee Attrition Data.csv")

# -------------------------
# 3. Data Preprocessing
# -------------------------
print("First 10 rows:")
print(df.head(10))
print("---------------------------------------------------------------------------------")

print("Last 10 rows:")
print(df.tail(10))
print("---------------------------------------------------------------------------------")

print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])
print("---------------------------------------------------------------------------------")

print("\nColumn Names:")
print(df.columns)
print("---------------------------------------------------------------------------------")

print("\nDataset Info:")
print(df.info())
print("---------------------------------------------------------------------------------")

print("\nStatistics Summary:")
print(df.describe())
print("---------------------------------------------------------------------------------")

# -------------------------
# 4. Check Missing Values
# -------------------------
print("\nMissing values in each column:")
print(df.isna().sum())
print("---------------------------------------------------------------------------------")

# -------------------------
# 5. Handling Missing Values
# -------------------------
for col in df.columns:

    if df[col].dtype != "object":

        if df[col].isnull().sum() > 0:

            mean_value = df[col].mean()

            print(f"\nMissing values found in {col}")
            print(f"Mean value used = {mean_value}")

            df[col] = df[col].fillna(mean_value)

    else:

        if df[col].isnull().sum() > 0:

            mode_value = df[col].mode()[0]

            print(f"\nMissing values found in {col}")
            print(f"Mode value used = {mode_value}")

            df[col] = df[col].fillna(mode_value)

print("---------------------------------------------------------------------------------")

# -------------------------
# 6. Remove Duplicate Rows
# -------------------------
duplicate_count = df.duplicated().sum()

print("Duplicate Rows:", duplicate_count)

df = df.drop_duplicates()

print("---------------------------------------------------------------------------------")

# -------------------------
# 7. Check Unique Values
# -------------------------
print("\nUnique values in Attrition:")
print(df["Attrition"].unique())

print("---------------------------------------------------------------------------------")

# -------------------------
# 8. Label Encoding
# -------------------------
le = LabelEncoder()

categorical_columns = df.select_dtypes(include="object").columns

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

print("\nAfter Label Encoding:")
print(df.head())

print("---------------------------------------------------------------------------------")

# -------------------------
# 9. Correlation Heatmap
# -------------------------
plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(numeric_only=True),
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

# -------------------------
# 10. Train-Test Split
# -------------------------
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split Completed")

print("X_train Shape:", X_train.shape)
print("X_test Shape :", X_test.shape)

print("y_train Shape:", y_train.shape)
print("y_test Shape :", y_test.shape)

print("---------------------------------------------------------------------------------")

# -------------------------
# 11. Before SMOTE
# -------------------------
sns.countplot(x=y_train)

plt.title("Class Distribution BEFORE SMOTE")
plt.show()

# -------------------------
# 12. Apply SMOTE
# -------------------------
sm = SMOTE(random_state=42)

X_train_smote, y_train_smote = sm.fit_resample(
    X_train,
    y_train
)

print("\nSMOTE Applied Successfully")

print("---------------------------------------------------------------------------------")

# -------------------------
# 13. After SMOTE
# -------------------------
sns.countplot(x=y_train_smote)

plt.title("Class Distribution AFTER SMOTE")
plt.show()

# -------------------------
# 14. Model Training
# -------------------------
model = RandomForestClassifier(random_state=42)

model.fit(X_train_smote, y_train_smote)

print("\nMODEL TRAINING COMPLETED!")

# 14.1 save the model 


import joblib

model.dump("seflkdfjwerjfioerifheorifherfho werhfowjhwdef wheof wioe f.pkl")



print("---------------------------------------------------------------------------------")

# -------------------------
# 15. Prediction
# -------------------------
y_pred = model.predict(X_test)

print("\nFirst 10 Predictions:")
print(y_pred[:10])

print("---------------------------------------------------------------------------------")

# -------------------------
# 16. Accuracy Score
# -------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy Score:")
print(accuracy*100)

print("---------------------------------------------------------------------------------")

# -------------------------
# 17. Classification Report
# -------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("---------------------------------------------------------------------------------")

# -------------------------
# 18. Confusion Matrix
# -------------------------
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("---------------------------------------------------------------------------------")

# -------------------------
# 19. Confusion Matrix Plot
# -------------------------
plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# -------------------------
# 20. Save Cleaned Dataset
# -------------------------
df.to_csv("cleaned_attrition_data.csv", index=False)

print("\nCleaned Dataset Saved Successfully!")
print("File Name: cleaned_attrition_data.csv")





