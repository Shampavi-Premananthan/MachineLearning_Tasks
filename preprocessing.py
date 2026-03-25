import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load the dataset
# Using r"" for windows paths
data_path = r"d:\Softwareprojects\ML Task1\Data Set For Task\Churn Prdiction Data\churn-bigml-80.csv"
df = pd.read_csv(data_path)

print("--- Step 1: Handling Missing Values ---")
# Check for missing values
missing_counts = df.isnull().sum()
if missing_counts.sum() == 0:
    print("No missing values detected.")
else:
    print("Missing values found, filling with median/mode...")
    # Fill numeric with median, categorical with mode
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

print("\n--- Step 2: Encoding Categorical Variables ---")
# Binary encoding for Yes/No and True/False
binary_cols = ['International plan', 'Voice mail plan']
le = LabelEncoder()

for col in binary_cols:
    df[col] = list(le.fit_transform(df[col]))  # type: ignore
    print(f"Encoded {col}: {dict(zip(le.classes_, list(le.transform(le.classes_))))}") # type: ignore

# Target encoding (Churn)
df['Churn'] = df['Churn'].astype(int)
print("Encoded Churn: {False: 0, True: 1}")

# For 'State', we'll use Label Encoding to keep the dimensionality low for this task
df['State'] = list(le.fit_transform(df['State']))  # type: ignore
print("Encoded State column using LabelEncoder.")

print("\n--- Step 3: Feature Scaling ---")
# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Identify numeric columns for scaling (all except the ones we just encoded might still be numeric)
# Actually, everything in X is now numeric.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("Numerical features scaled using StandardScaler.")

print("\n--- Step 4: Train-Test Split ---")
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Save the preprocessed data (Optional but good practice)
preprocessed_path = r"d:\Softwareprojects\ML Task1\preprocessed_churn_data.csv"
X_train.to_csv(r"d:\Softwareprojects\ML Task1\X_train.csv", index=False)
X_test.to_csv(r"d:\Softwareprojects\ML Task1\X_test.csv", index=False)
y_train.to_csv(r"d:\Softwareprojects\ML Task1\y_train.csv", index=False)
y_test.to_csv(r"d:\Softwareprojects\ML Task1\y_test.csv", index=False)

print(f"\nPreprocessing Complete! Files saved to d:\\Softwareprojects\\ML Task1\\")
