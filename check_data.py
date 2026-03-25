import pandas as pd

df = pd.read_csv(r"d:\Softwareprojects\ML Task1\Data Set For Task\Churn Prdiction Data\churn-bigml-80.csv")
print("Missing values per column:")
print(df.isnull().sum())
print("\nColumn Types:")
print(df.dtypes)
print("\nUnique values in categorical columns:")
cat_cols = ['State', 'International plan', 'Voice mail plan', 'Churn']
for col in cat_cols:
    print(f"{col}: {df[col].unique()}")
