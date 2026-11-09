import pandas as pd
import numpy as np

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

df = pd.read_csv("reports/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.head())
print(df.shape)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df = df.dropna(subset=["Churn"])

# Create new features
df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["New", "Medium", "Long", "Very Long"]
)

df["spend_per_month"] = (
    df["TotalCharges"] / (df["tenure"] + 1)
)

service_cols = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

df["service_count"] = (
    df[service_cols] == "Yes"
).sum(axis=1)

print(df.head())

X = df.drop(
    columns=["Churn", "customerID"],
    errors="ignore"
)

y = df["Churn"].map({"No": 0, "Yes": 1})

num_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns

cat_cols = X.select_dtypes(
    include=["object", "category"]
).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Dummy Classifier": DummyClassifier(
        strategy="most_frequent"
    )
}
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

for name, model in models.items():

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=["roc_auc", "f1"]
    )

    results.append({
        "Model": name,
        "CV ROC-AUC": scores["test_roc_auc"].mean(),
        "CV F1": scores["test_f1"].mean()
    })

results_df = pd.DataFrame(results)

print(results_df)
print("\nModel Performance Results")
print(results_df.to_string(index=False))
results_df.to_csv("model_performance_results.csv", index=False)

print("Results saved successfully!")