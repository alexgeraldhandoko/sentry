from pathlib import Path

import pandas as pd

import torch

import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# ----------------------------------------------
# Declare constants
# ----------------------------------------------
# The seed to split the dataset, so that the split is repeatable
DATA_SPLIT_SEED = 42 
POLYNOMIAL_TRANSFORMATION_DEGREE = 2
TARGET_COL = "FLAG"

# Columns that should be dropped because they do not contain transaction information
# or is non-numeric
DROP_COLS = [ 
    "Unnamed: 0",
    "Index",
    "Address",
    "ERC20 most sent token type",
    "ERC20_most_rec_token_type"
]

# Declare the paths
ML_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = ML_DIR / "data" / "raw" / "data.csv"
PROCESSED_DIR = ML_DIR / "data" / "processed"

# ----------------------------------------------
# Main processing function
# ----------------------------------------------
def main():

    # Create the processed folder
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Read the csv
    df = pd.read_csv(RAW_PATH)

    # Clean the df column names from leading and trailing spaces
    df.columns = df.columns.str.strip()

    # Check for possibility that the dataset doesn't have target column
    if TARGET_COL not in df.columns:
        raise ValueError(f"Could not find target column: {TARGET_COL}")
    
    # Extract the target column into the label vector y
    y = df[TARGET_COL].astype("float32")

    # Drop the columns we decided to drop and any non-numeric columns
    cols_to_drop = [] 
    for col in DROP_COLS:
        if (col in df.columns):
            cols_to_drop.append(col) # Ensures columns to drop exist in dataset
    X = df.drop(columns=[TARGET_COL] + cols_to_drop)
    X = X.select_dtypes(["number"])

    # Prints to make sure we extracted the correct information
    feature_names = X.columns.tolist()

    print("Raw dataset information:")
    print("--------------------------------------------")
    print(f"Number of examples: {X.shape[0]}")
    print(f"Number of numeric input features: {X.shape[1]}")
    print("Label counts: ")
    print(y.value_counts())

    # Split the dataset from dataframes into dataframes
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=DATA_SPLIT_SEED,
        stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=(0.15/0.85),
        random_state=DATA_SPLIT_SEED,
        stratify=y_temp
    )

    # Define the preprocessing steps
    preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("raw_scaler", StandardScaler()),
            ("poly", PolynomialFeatures(
                degree=POLYNOMIAL_TRANSFORMATION_DEGREE,
                include_bias=False
            )),
            ("transformed_scaler", StandardScaler())
        ]
    )

    # Preprocess the X datasets and turn them into NumPy arrays
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Turn the X datasets from NumPy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_processed, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)

    # Turn the y datasets from pandas df to NumPy arrays to PyTorch tensors
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).view(-1, 1)
    y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32).view(-1, 1)

    # Print the tensor shapes for sanity check
    print("Tensor shapes:")
    print("--------------------------------------------")
    print(f"X_train: {X_train_tensor.shape}")
    print(f"y_train: {y_train_tensor.shape}")
    print(f"X_val: {X_val_tensor.shape}")
    print(f"y_val: {y_val_tensor.shape}")
    print(f"X_test: {X_test_tensor.shape}")
    print(f"y_test: {y_test_tensor.shape}")

    # Store the tensors into their corresponding files
    torch.save(
        {
            "X_train": X_train_tensor,
            "y_train": y_train_tensor
        },
        PROCESSED_DIR / "train.pt"
    )
    torch.save(
        {
            "X_val": X_val_tensor,
            "y_val": y_val_tensor
        },
        PROCESSED_DIR / "val.pt"
    )
    torch.save(
        {
            "X_test": X_test_tensor,
            "y_test": y_test_tensor
        },
        PROCESSED_DIR / "test.pt"
    )

    # Save the preprocessor
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "raw_feature_names": feature_names,
            "target_col": TARGET_COL,
            "dropped_cols": cols_to_drop,
            "polynomial_degree": POLYNOMIAL_TRANSFORMATION_DEGREE,
            "input_dim_after_processing": X_train_tensor.shape[1]
        },
        PROCESSED_DIR / "preprocessor.joblib"
    )

    # Print status message
    print("Preprocessing complete")
    print(f"Saved processed files to: {PROCESSED_DIR}")

if __name__ == "__main__":
    main()