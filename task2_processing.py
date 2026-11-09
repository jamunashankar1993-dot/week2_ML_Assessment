import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Load training dataset
base_dir = Path(__file__).resolve().parent
train_path = base_dir / "outputs" / "train.csv"

df = pd.read_csv(train_path)
# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
# Step 11: Feature Engineering

# Feature 1: Tenure in Years
df["tenure_years"] = df["tenure"] / 12

# Feature 2: Monthly Charges Interaction
df["monthly_tenure_interaction"] = (
    df["MonthlyCharges"] * df["tenure"]
)

# Feature 3: Log Transformation
df["log_TotalCharges"] = np.log1p(
    df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    ).clip(lower=0)
)

print("\nFeature Engineering Completed")
print(df[[
    "tenure_years",
    "monthly_tenure_interaction",
    "log_TotalCharges"
]].head())

print("New Dataset Shape:", df.shape)

print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFirst 3 Rows:")
print(df.head(3))
# Task 2 - Missing Values Analysis

print("\n--- Missing Values Analysis ---")

print("Missing values in each column:")
print(df.isnull().sum())

print("\nTotal missing values:")
print(df.isnull().sum().sum())

print("\nNumeric Columns:")
print(df.select_dtypes(include="number").columns.tolist())

print("\nCategorical Columns:")
print(df.select_dtypes(include="object").columns.tolist())
# Separate numeric and categorical columns

numeric_cols = df.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = df.select_dtypes(
    include=["object"]
).columns.tolist()

# Separate target and feature columns

X = df.drop(columns=["Churn"])
y = df["Churn"]

numeric_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("Numeric Columns:", numeric_cols)
print("Categorical Columns:", categorical_cols)
print("Target Shape:", y.shape)
# Step 8: Handle Missing Values

from sklearn.impute import SimpleImputer

# Numeric columns - Median
numeric_imputer = SimpleImputer(strategy="median")

X_numeric = numeric_imputer.fit_transform(
    X[numeric_cols]
)

print("Numeric Imputation Completed")
print("Numeric Shape:", X_numeric.shape)

# Categorical columns - Most Frequent
categorical_imputer = SimpleImputer(
    strategy="most_frequent"
)

X_categorical = categorical_imputer.fit_transform(
    X[categorical_cols]
)

print("Categorical Imputation Completed")
print("Categorical Shape:", X_categorical.shape)
# Step 9: KNN Imputer

from sklearn.impute import KNNImputer

knn_imputer = KNNImputer(n_neighbors=5)

X_numeric_knn = knn_imputer.fit_transform(
    X[numeric_cols]
)

print("KNN Imputation Completed")
print("KNN Shape:", X_numeric_knn.shape)
# Step 10: Encoding

# Convert imputed categorical data to DataFrame
X_cat_df = pd.DataFrame(
    X_categorical,
    columns=categorical_cols
)

# Ordered column
ordered_cols = ["Contract"]

# Remaining categorical columns
nominal_cols = [
    col for col in categorical_cols
    if col not in ordered_cols
]

# Create ColumnTransformer
encoder = ColumnTransformer(
    transformers=[
        ("onehot",
         OneHotEncoder(handle_unknown="ignore"),
         nominal_cols),

        ("ordinal",
         OrdinalEncoder(
             categories=[[
                 "Month-to-month",
                 "One year",
                 "Two year"
             ]],
             handle_unknown="use_encoded_value",
             unknown_value=-1
         ),
         ordered_cols)
    ]
)

X_encoded = encoder.fit_transform(X_cat_df)

print("Encoding Completed")
print("Encoded Shape:", X_encoded.shape)
# Step 12: Save Preprocessed Dataset

import numpy as np
from scipy.sparse import issparse

if issparse(X_encoded):
    X_encoded = X_encoded.toarray()

X_final = np.hstack([X_numeric_knn, X_encoded])

processed_df = pd.DataFrame(X_final)

processed_df["Churn"] = y.reset_index(drop=True)

output_path = base_dir / "outputs" / "preprocessed_train.csv"

processed_df.to_csv(output_path, index=False)

print("Preprocessing Completed Successfully!")
print("Final Dataset Shape:", processed_df.shape)
print("Saved File:", output_path)