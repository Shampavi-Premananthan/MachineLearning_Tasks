"""
Level 1, Task 2: Simple Linear Regression Model
Goal: Predict house prices using Linear Regression
Dataset: Company-provided House Prediction Dataset
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


def load_and_preprocess_data():
    """Load and prepare the house prediction dataset"""
    print("=" * 70)
    print("STEP 1: LOADING & PREPROCESSING DATASET")
    print("=" * 70)

    # Load the house prediction dataset from CSV
    # This appears to be the Boston Housing dataset
    dataset_path = r"Data Set For Task\4) house Prediction Data Set.csv"

    # Column names for Boston Housing dataset
    column_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
        'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
    ]

    # The provided file is often a wrapped whitespace text dump of Boston housing values.
    # Parse all numeric tokens and reshape into 14 columns.
    with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    numeric_values = []
    for token in raw_text.replace("\n", " ").split():
        try:
            numeric_values.append(float(token))
        except ValueError:
            continue

    if len(numeric_values) % len(column_names) != 0:
        raise ValueError(
            "Dataset parsing failed: total numeric values are not divisible by 14 columns."
        )

    data = np.array(numeric_values).reshape(-1, len(column_names))
    df = pd.DataFrame(data, columns=column_names)

    print("\nDataset: House Prediction Dataset")
    print(f"Total samples: {len(df)}")
    print(f"Number of features: {len(column_names) - 1}")

    # Separate features and target
    X = df.drop('MEDV', axis=1)  # Features
    y = df['MEDV']  # Target (Median house value in $1000s)

    print(f"\nFeatures: {list(X.columns)}")

    # Check for missing values
    missing_count = X.isnull().sum().sum()
    print(f"\nMissing values: {missing_count}")

    # Handle missing values if any
    if missing_count > 0:
        print("Handling missing values by filling with median...")
        X = X.fillna(X.median())

    # Display basic statistics
    print("\nTarget variable (MEDV - Median House Value) statistics:")
    print(f"  Mean: ${y.mean():.2f}k")
    print(f"  Min: ${y.min():.2f}k")
    print(f"  Max: ${y.max():.2f}k")
    print(f"  Std Dev: ${y.std():.2f}k")

    # Feature descriptions
    print("\n" + "-" * 60)
    print("FEATURE DESCRIPTIONS:")
    print("-" * 60)
    print("  CRIM    - Crime rate per capita")
    print("  ZN      - Proportion of residential land zoned")
    print("  INDUS   - Proportion of non-retail business acres")
    print("  CHAS    - Charles River dummy variable")
    print("  NOX     - Nitric oxides concentration")
    print("  RM      - Average number of rooms per dwelling")
    print("  AGE     - Proportion of owner-occupied units built pre-1940")
    print("  DIS     - Distances to employment centers")
    print("  RAD     - Index of accessibility to radial highways")
    print("  TAX     - Property-tax rate")
    print("  PTRATIO - Pupil-teacher ratio")
    print("  B       - Proportion of African American population")
    print("  LSTAT   - % lower status of the population")
    print("  MEDV    - Median value of homes (TARGET - in $1000s)")

    return X, y, list(X.columns)


def train_model(X_train, y_train):
    """Train the Linear Regression model"""
    print("\n" + "=" * 70)
    print("STEP 2: TRAINING LINEAR REGRESSION MODEL")
    print("=" * 70)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\nModel trained successfully!")
    print(f"Training samples used: {len(X_train)}")

    return model


def interpret_coefficients(model, feature_names):
    """Analyze and interpret the model coefficients"""
    print("\n" + "=" * 70)
    print("STEP 3: INTERPRETING COEFFICIENTS")
    print("=" * 70)

    # Get coefficients and intercept
    coefficients = model.coef_
    intercept = model.intercept_

    print(f"\nModel Intercept: {intercept:.4f}")
    print("\nFeature Coefficients (Impact on House Price):")
    print("-" * 60)

    # Create a DataFrame for better visualization
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    })

    # Sort by absolute value to see most impactful features
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)

    for _, row in coef_df.iterrows():
        feature = row['Feature']
        coef = row['Coefficient']
        impact = "increases" if coef > 0 else "decreases"
        print(f"  {feature:20s}: {coef:8.4f}  ({impact} price)")

    print("\n" + "-" * 60)
    print("INTERPRETATION:")
    top_feature = coef_df.iloc[0]
    print(f"  Most impactful feature: {top_feature['Feature']}")
    print(f"  Coefficient magnitude: {top_feature['Abs_Coefficient']:.4f}")

    # Explain what the model learned
    print("\nWhat the model learned:")
    for _, row in coef_df.head(3).iterrows():
        feature = row['Feature']
        coef = row['Coefficient']
        if coef > 0:
            print(f"  - Higher {feature} -> Higher house prices")
        else:
            print(f"  - Higher {feature} -> Lower house prices")

    return coef_df


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model performance with R^2 and MSE"""
    print("\n" + "=" * 70)
    print("STEP 4: EVALUATING MODEL PERFORMANCE")
    print("=" * 70)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics for training set
    r2_train = r2_score(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)

    # Calculate metrics for test set
    r2_test = r2_score(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)

    print("\n[ TRAINING SET PERFORMANCE ]")
    print("-" * 60)
    print(f"  R-squared: {r2_train:.4f}  (explains {r2_train * 100:.2f}% of variance)")
    print(f"  MSE:       {mse_train:.4f}")
    print(f"  RMSE:      {rmse_train:.4f}  (average error: ${rmse_train:.2f}k)")

    print("\n[ TEST SET PERFORMANCE ]")
    print("-" * 60)
    print(f"  R-squared: {r2_test:.4f}  (explains {r2_test * 100:.2f}% of variance)")
    print(f"  MSE:       {mse_test:.4f}")
    print(f"  RMSE:      {rmse_test:.4f}  (average error: ${rmse_test:.2f}k)")

    # Interpret R-squared score
    print("\n" + "-" * 60)
    print("INTERPRETATION:")
    if r2_test > 0.7:
        performance = "EXCELLENT"
    elif r2_test > 0.5:
        performance = "GOOD"
    elif r2_test > 0.3:
        performance = "MODERATE"
    else:
        performance = "POOR"

    print(f"  Model Performance: {performance}")
    print(f"  The model explains {r2_test * 100:.1f}% of the variance in house prices")
    print(f"  Average prediction error: ${rmse_test:.2f}k (in $1000s)")

    # Show sample predictions
    print("\n" + "-" * 60)
    print("SAMPLE PREDICTIONS (first 5 test samples):")
    print("-" * 60)
    print(f"{'Actual Price':>15} {'Predicted Price':>18} {'Error':>12}")
    print("-" * 60)
    for i in range(min(5, len(y_test))):
        actual = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
        predicted = y_test_pred[i]
        error = abs(actual - predicted)
        print(f"${actual:>6.2f}k         ${predicted:>6.2f}k         +/-${error:>5.2f}k")

    # Save Actual vs Predicted plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5, color='blue')

    # Ensure min/max values are scalar by converting to float
    min_val = float(y_test.min())
    max_val = float(y_test.max())

    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual Prices ($1000s)')
    plt.ylabel('Predicted Prices ($1000s)')
    plt.title('Actual vs Predicted House Prices')
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    print("\nSaved plot to 'actual_vs_predicted.png'")

    return {
        'r2_train': r2_train,
        'r2_test': r2_test,
        'mse_train': mse_train,
        'mse_test': mse_test,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test
    }


def main():
    """Main function to run the complete pipeline"""
    print("\n" + "=" * 70)
    print(" LINEAR REGRESSION MODEL - HOUSE PRICE PREDICTION")
    print("=" * 70)

    # Step 1: Load and preprocess data
    X, y, feature_names = load_and_preprocess_data()

    # Split data into training and test sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nData split: {len(X_train)} training, {len(X_test)} test samples")

    # Step 2: Train the model
    model = train_model(X_train, y_train)

    # Step 3: Interpret coefficients
    coef_df = interpret_coefficients(model, feature_names)

    # Step 4: Evaluate the model
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

    # Final summary
    print("\n" + "=" * 70)
    print(" FINAL SUMMARY")
    print("=" * 70)
    print("\n* Model: Linear Regression")
    print("* Dataset: House Prediction Dataset (Company-provided)")
    print(f"* Features: {len(feature_names)}")
    print(f"* Training samples: {len(X_train)}")
    print(f"* Test samples: {len(X_test)}")
    print("\n* KEY METRICS:")
    print(f"   R-squared (Test): {metrics['r2_test']:.4f}")
    print(f"   MSE (Test):       {metrics['mse_test']:.4f}")
    print(f"   RMSE (Test):      {metrics['rmse_test']:.4f}")
    print("\n* INTERPRETATION:")
    print(f"   The model explains {metrics['r2_test'] * 100:.1f}% of variance in house prices.")

    # Save metrics and summary
    with open("model_summary.txt", "w", encoding="utf-8") as f:
        f.write("LINEAR REGRESSION MODEL - HOUSE PRICE PREDICTION SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"R-squared (Test): {metrics['r2_test']:.4f}\n")
        f.write(f"MSE (Test): {metrics['mse_test']:.4f}\n")
        f.write(f"RMSE (Test): {metrics['rmse_test']:.4f}\n")
        f.write("\nFeatures and Coefficients saved in script output.\n")
        f.write("Plot saved as actual_vs_predicted.png\n")
    print("\nSaved model_summary.txt")
    print(f"   Average prediction error is ${metrics['rmse_test']:.2f}k.")
    print("\n" + "=" * 70)
    print(" MODEL TRAINING COMPLETE!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
