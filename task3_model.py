# TASK 3 - MACHINE LEARNING MODELS

import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

# Load Dataset

base_dir = Path(__file__).resolve().parent

train_path = base_dir / "outputs" / "train.csv"
val_path = base_dir / "outputs" / "validation.csv"

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)

print("Training Data Shape:", train_df.shape)
print("Validation Data Shape:", val_df.shape)

print("Columns:", train_df.columns.tolist())

print("Task 3 Dataset Loaded Successfully!")
# Step 2: Prepare Data for Models

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Convert TotalCharges to numeric
train_df["TotalCharges"] = pd.to_numeric(
    train_df["TotalCharges"], errors="coerce"
)

val_df["TotalCharges"] = pd.to_numeric(
    val_df["TotalCharges"], errors="coerce"
)

# Separate Features and Target
# Remove invalid target values


print("Train Churn Values:")
print(train_df["Churn"].value_counts(dropna=False))

print("Validation Churn Values:")
print(val_df["Churn"].value_counts(dropna=False))
# Clean Churn column

for df_temp in [train_df, val_df]:

    df_temp["Churn"] = (
        df_temp["Churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "yes": "1",
            "no": "0"
        })
    )
# Clean and convert Churn column
for df_temp in [train_df, val_df]:

    df_temp["Churn"] = (
        df_temp["Churn"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map({
            "yes": 1,
            "no": 0,
            "1": 1,
            "0": 0,
            "1.0": 1,
            "0.0": 0
        })
    )

    # Remove missing target values
    df_temp.dropna(subset=["Churn"], inplace=True)

    # Convert target to integer
    df_temp["Churn"] = df_temp["Churn"].astype(int)

print("Train missing:", train_df["Churn"].isna().sum())
print("Validation missing:", val_df["Churn"].isna().sum())
    




X_train = train_df.drop(columns=["Churn"])
y_train = train_df["Churn"].astype(int)

X_val = val_df.drop(columns=["Churn"])
y_val = val_df["Churn"].astype(int)


# Handle missing values

# Handle missing values

for col in X_train.columns:

    if X_train[col].isnull().any() or X_val[col].isnull().any():

        if pd.api.types.is_numeric_dtype(X_train[col]):

            fill_value = X_train[col].median()

            if pd.isna(fill_value):
                fill_value = 0

        else:
            mode_value = X_train[col].mode(dropna=True)

            if not mode_value.empty:
                fill_value = mode_value.iloc[0]
            else:
                fill_value = "Unknown"

        X_train[col] = X_train[col].fillna(fill_value)
        X_val[col] = X_val[col].fillna(fill_value)


# Check missing values

print("Train NaN:", X_train.isnull().sum().sum())
print("Validation NaN:", X_val.isnull().sum().sum())

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))



print("Train NaN:", X_train.isnull().sum().sum())
print("Validation NaN:", X_val.isnull().sum().sum())

   
print("Train rows:", len(train_df))
print("Validation rows:", len(val_df))

print("Train missing:", train_df["Churn"].isna().sum())
print("Validation missing:", val_df["Churn"].isna().sum())

print("Train values:")
print(train_df["Churn"].value_counts(dropna=False))

print("Validation values:")
print(val_df["Churn"].value_counts(dropna=False))
X_train = train_df.drop(columns=["Churn"])
y_train = train_df["Churn"].astype(int)

X_val = val_df.drop(columns=["Churn"])
y_val = val_df["Churn"].astype(int)
# Fill missing values

for col in X_train.columns:
    if X_train[col].isnull().any():
        if pd.api.types.is_numeric_dtype(X_train[col]):
            fill_value = X_train[col].median()
        else:
            fill_value = X_train[col].mode()[0]

        X_train[col] = X_train[col].fillna(fill_value)
        X_val[col] = X_val[col].fillna(fill_value)

print("Train NaN:", X_train.isnull().sum().sum())
print("Validation NaN:", X_val.isnull().sum().sum())

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))

print("Train Churn Values:")
print(train_df["Churn"].value_counts(dropna=False))

print("Validation Churn Values:")
print(val_df["Churn"].value_counts(dropna=False))
# Separate Features and Target
# Clean target column

for df_temp in [train_df, val_df]:

    df_temp["Churn"] = (
        df_temp["Churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "yes": "1",
            "no": "0",
            "y": "1",
            "n": "0"
        })
    )

    df_temp["Churn"] = pd.to_numeric(
        df_temp["Churn"], errors="coerce"
    )

    df_temp.dropna(subset=["Churn"], inplace=True)

    df_temp["Churn"] = df_temp["Churn"].astype(int)

print("Train Churn missing:", train_df["Churn"].isna().sum())
print("Validation Churn missing:", val_df["Churn"].isna().sum())
X_train = train_df.drop(columns=["Churn"])
y_train = train_df["Churn"].map({"Yes": 1, "No": 0})

X_val = val_df.drop(columns=["Churn"])
y_val = val_df["Churn"].map({"Yes": 1, "No": 0})

print("Missing y_train:", y_train.isna().sum())
print("Missing y_val:", y_val.isna().sum())

# Identify columns
numeric_cols = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

print("Numeric Columns:", numeric_cols)
print("Categorical Columns:", categorical_cols)

print("Data Preparation Completed!")
print("Training Shape:", X_train.shape)
print("Validation Shape:", X_val.shape)
# Step 3: Create Preprocessing Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

print("Preprocessing Pipeline Created Successfully!")
# Step 4: Logistic Regression Model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

log_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

log_model.fit(X_train, y_train)

log_predictions = log_model.predict(X_val)

print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_val, log_predictions))

print("\nClassification Report:")
print(classification_report(y_val, log_predictions))