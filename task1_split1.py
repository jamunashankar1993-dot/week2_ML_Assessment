import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Find dataset automatically
data_folder = Path(__file__).resolve().parent / "data"

print("Available CSV files:")
for f in data_folder.glob("*.csv"):
    print(f.name)
files = [f for f in data_folder.glob("*.csv") if "Telco" in f.name]
if not files:
    raise FileNotFoundError("Telco CSV file not found")
df = pd.read_csv(files[0])

print("\nDataset Shape:", df.shape)

df = df.drop(columns=["customerID"])

df["Churn"] = (
    df["Churn"].astype(str)
    .str.strip()
    .str.lower()
    .map({"yes": 1, "no": 0, "1": 1, "0": 0})
)

X = df.drop(columns=["Churn"])
y = df["Churn"]

# 60% Train, 20% Validation, 20% Test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("\nTrain Shape:", X_train.shape)
print("Validation Shape:", X_val.shape)
print("Test Shape:", X_test.shape)
print("\nChurn Distribution:")

print("Train:")
print(y_train.value_counts(normalize=True))

print("Validation:")
print(y_val.value_counts(normalize=True))

print("Test:")
print(y_test.value_counts(normalize=True))
# Save train, validation and test datasets

output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(exist_ok=True)

train_df = X_train.copy()
train_df["Churn"] = y_train
train_df.to_csv(output_dir / "train.csv", index=False)

val_df = X_val.copy()
val_df["Churn"] = y_val
val_df.to_csv(output_dir / "validation.csv", index=False)

test_df = X_test.copy()
test_df["Churn"] = y_test
test_df.to_csv(output_dir / "test.csv", index=False)

print("All datasets saved successfully!")