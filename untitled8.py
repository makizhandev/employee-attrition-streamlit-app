# -------------------------
# 1. Import Libraries
# -------------------------
import pandas as pd
import numpy as np
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from catboost import CatBoostClassifier

import seaborn as sns
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")

# -------------------------
# 2. Load Dataset
# -------------------------
df = pd.read_csv("IBM HR Employee Attrition Data.csv")

# -------------------------
# 3. Data Preprocessing
# -------------------------
print("First 10 Rows:")
print(df.head(10))
print("---------------------------------------------------------------------------------")

print("Last 10 Rows:")
print(df.tail(10))
print("---------------------------------------------------------------------------------")

print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])
print("---------------------------------------------------------------------------------")

print("\nColumn Names:")
print(df.columns)
print("---------------------------------------------------------------------------------")

print("\nDataset Info:")
df.info()
print("---------------------------------------------------------------------------------")

print("\nStatistics Summary:")
print(df.describe())
print("---------------------------------------------------------------------------------")

# -------------------------
# 4. Check Missing Values
# -------------------------
print("\nMissing values in each column:")
print(df.isnull().sum())
print("---------------------------------------------------------------------------------")

# -------------------------
# 5. Handle Missing Values
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
# 7. Drop Constant Columns
# -------------------------
df = df.drop(
    ["EmployeeCount", "Over18", "StandardHours"],
    axis=1
)

print("Constant Columns Removed")
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
# 11. Class Distribution
# -------------------------
sns.countplot(x=y_train)

plt.title("Class Distribution")
plt.show()

# -------------------------
# 12. CatBoost Model Training
# -------------------------
model = CatBoostClassifier(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    loss_function='Logloss',
    eval_metric='Accuracy',
    random_state=42,
    verbose=0
)

model.fit(X_train, y_train)

print("\nCATBOOST MODEL TRAINING COMPLETED!")

print("---------------------------------------------------------------------------------")

# -------------------------
# 13. Save Model
# -------------------------
joblib.dump(
    model,
    "catboost_attrition_model.pkl"
)

print("Model Saved Successfully!")

print("---------------------------------------------------------------------------------")

# -------------------------
# 14. Prediction
# -------------------------
y_pred = model.predict(X_test)

print("\nFirst 10 Predictions:")
print(y_pred[:10])

print("---------------------------------------------------------------------------------")

# -------------------------
# 15. Accuracy Score
# -------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy Score:")
print(round(accuracy * 100, 2), "%")

print("---------------------------------------------------------------------------------")

# -------------------------
# 16. Classification Report
# -------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("---------------------------------------------------------------------------------")

# -------------------------
# 17. Confusion Matrix
# -------------------------
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("---------------------------------------------------------------------------------")

# -------------------------
# 18. Confusion Matrix Plot
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
# 19. Save Cleaned Dataset
# -------------------------
df.to_csv(
    "cleaned_attrition_data.csv",
    index=False
)

print("\nCleaned Dataset Saved Successfully!")
print("File Name: cleaned_attrition_data.csv")


# -------------------------
# 20. User Input Prediction
# -------------------------

print("\n========================================")
print("EMPLOYEE ATTRITION PREDICTION")
print("========================================")

Age = int(input("Age: "))
BusinessTravel = int(input("BusinessTravel (0=Non-Travel, 1=Travel_Frequently, 2=Travel_Rarely): "))
DailyRate = int(input("DailyRate: "))
Department = int(input("Department (0=Human Resources, 1=Research & Development, 2=Sales): "))
DistanceFromHome = int(input("DistanceFromHome: "))
Education = int(input("Education (1-5): "))
EducationField = int(input("EducationField (Encoded Value): "))
EmployeeNumber = int(input("EmployeeNumber: "))
EnvironmentSatisfaction = int(input("EnvironmentSatisfaction (1-4): "))
Gender = int(input("Gender (0=Female, 1=Male): "))
HourlyRate = int(input("HourlyRate: "))
JobInvolvement = int(input("JobInvolvement (1-4): "))
JobLevel = int(input("JobLevel: "))
JobRole = int(input("JobRole (Encoded Value): "))
JobSatisfaction = int(input("JobSatisfaction (1-4): "))
MaritalStatus = int(input("MaritalStatus (0=Divorced, 1=Married, 2=Single): "))
MonthlyIncome = int(input("MonthlyIncome: "))
MonthlyRate = int(input("MonthlyRate: "))
NumCompaniesWorked = int(input("NumCompaniesWorked: "))
OverTime = int(input("OverTime (0=No, 1=Yes): "))
PercentSalaryHike = int(input("PercentSalaryHike: "))
PerformanceRating = int(input("PerformanceRating: "))
RelationshipSatisfaction = int(input("RelationshipSatisfaction (1-4): "))
StockOptionLevel = int(input("StockOptionLevel: "))
TotalWorkingYears = int(input("TotalWorkingYears: "))
TrainingTimesLastYear = int(input("TrainingTimesLastYear: "))
WorkLifeBalance = int(input("WorkLifeBalance (1-4): "))
YearsAtCompany = int(input("YearsAtCompany: "))
YearsInCurrentRole = int(input("YearsInCurrentRole: "))
YearsSinceLastPromotion = int(input("YearsSinceLastPromotion: "))
YearsWithCurrManager = int(input("YearsWithCurrManager: "))

# Create DataFrame for Prediction
new_employee = pd.DataFrame({
    "Age": [Age],
    "BusinessTravel": [BusinessTravel],
    "DailyRate": [DailyRate],
    "Department": [Department],
    "DistanceFromHome": [DistanceFromHome],
    "Education": [Education],
    "EducationField": [EducationField],
    "EmployeeNumber": [EmployeeNumber],
    "EnvironmentSatisfaction": [EnvironmentSatisfaction],
    "Gender": [Gender],
    "HourlyRate": [HourlyRate],
    "JobInvolvement": [JobInvolvement],
    "JobLevel": [JobLevel],
    "JobRole": [JobRole],
    "JobSatisfaction": [JobSatisfaction],
    "MaritalStatus": [MaritalStatus],
    "MonthlyIncome": [MonthlyIncome],
    "MonthlyRate": [MonthlyRate],
    "NumCompaniesWorked": [NumCompaniesWorked],
    "OverTime": [OverTime],
    "PercentSalaryHike": [PercentSalaryHike],
    "PerformanceRating": [PerformanceRating],
    "RelationshipSatisfaction": [RelationshipSatisfaction],
    "StockOptionLevel": [StockOptionLevel],
    "TotalWorkingYears": [TotalWorkingYears],
    "TrainingTimesLastYear": [TrainingTimesLastYear],
    "WorkLifeBalance": [WorkLifeBalance],
    "YearsAtCompany": [YearsAtCompany],
    "YearsInCurrentRole": [YearsInCurrentRole],
    "YearsSinceLastPromotion": [YearsSinceLastPromotion],
    "YearsWithCurrManager": [YearsWithCurrManager]
})

# Predict Attrition
prediction = model.predict(new_employee)

print("\n========================================")

if prediction[0] == 1:
    print("Predicted Attrition : YES")
else:
    print("Predicted Attrition : NO")

print("========================================")