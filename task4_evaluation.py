# TASK 4 - EVALUATION AND TUNING

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

print("TASK 4 - Evaluation and Tuning")
print("All imports completed successfully!")
# Step 1: Load Training and Validation Data

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

base_dir = Path(__file__).resolve().parent
output_dir = base_dir / "outputs"

train_df = pd.read_csv(output_dir / "train.csv")
val_df = pd.read_csv(output_dir / "validation.csv")

print("Training Data:", train_df.shape)
print("Validation Data:", val_df.shape)

# Step 2: Clean Target Column

for df in [train_df, val_df]:
    df["Churn"] = (
        df["Churn"].astype(str).str.strip().str.lower()
        .map({"yes": 1, "no": 0, "1": 1, "0": 0,
              "1.0": 1, "0.0": 0})
    )

train_df = train_df.dropna(subset=["Churn"]).copy()
val_df = val_df.dropna(subset=["Churn"]).copy()

train_df["TotalCharges"] = pd.to_numeric(
    train_df["TotalCharges"], errors="coerce"
)
val_df["TotalCharges"] = pd.to_numeric(
    val_df["TotalCharges"], errors="coerce"
)

# Step 3: Separate Features and Target

X_train = train_df.drop(columns=["Churn"])
y_train = train_df["Churn"].astype(int)

X_val = val_df.drop(columns=["Churn"])
y_val = val_df["Churn"].astype(int)

X_train = X_train.drop(columns=["customerID"], errors="ignore")
X_val = X_val.drop(columns=["customerID"], errors="ignore")

# Step 4: Identify Columns

numeric_cols = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

# Step 5: Create Preprocessing Pipeline

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

# Step 6: Model Evaluation

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_val)

print("\nTASK 4 - MODEL EVALUATION")
print("Accuracy:", accuracy_score(y_val, predictions))
print("Precision:", precision_score(y_val, predictions))
print("Recall:", recall_score(y_val, predictions))
print("F1 Score:", f1_score(y_val, predictions))
print("\nClassification Report:")
print(classification_report(y_val, predictions))

# Step 7: Hyperparameter Tuning

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 10]
}

grid_search = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBEST PARAMETERS:")
print(grid_search.best_params_)

# Step 8: Evaluate Best Model

best_predictions = best_model.predict(X_val)

print("\nTUNED MODEL RESULTS")
print("Accuracy:", accuracy_score(y_val, best_predictions))
print("F1 Score:", f1_score(y_val, best_predictions))
print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, best_predictions)
print(cm)

# Step 9: Save Results

results = pd.DataFrame({
    "Actual": y_val,
    "Predicted": best_predictions
})

results.to_csv(
    output_dir / "task4_predictions.csv",
    index=False
)

print("\nTask 4 Evaluation and Tuning Completed Successfully!")
print("Results saved successfully!")