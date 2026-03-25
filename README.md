# Machine Learning Task 1

This project prepares a telecom churn dataset for machine learning.

## Project Files

- `preprocessing.py`: Main preprocessing pipeline.
- `check_data.py`: Quick data inspection script.
- `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`: Generated train/test outputs.

## What the Preprocessing Does

1. Loads the churn dataset.
2. Checks and handles missing values.
3. Encodes categorical columns:
   - `International plan` and `Voice mail plan` using label encoding.
   - `Churn` converted to numeric (`False -> 0`, `True -> 1`).
   - `State` encoded with label encoding.
4. Scales features using `StandardScaler`.
5. Splits into train/test sets (80/20).
6. Saves processed files in the project root.

## Requirements

- Python 3.10+
- Packages:
  - pandas
  - numpy
  - scikit-learn

Install packages:

```powershell
pip install pandas numpy scikit-learn
```

## Run

From the project root:

```powershell
python check_data.py
python preprocessing.py
```

## Dataset Path Note

Current scripts use an absolute Windows path:

`d:\Softwareprojects\ML Task1\Data Set For Task\Churn Prdiction Data\churn-bigml-80.csv`

If you run this project on another machine, update the path in both scripts to match your local dataset location.

## Public Repository Note

Raw datasets and local environment files are excluded using `.gitignore` to keep this public repository safe and lightweight.
