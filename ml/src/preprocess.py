from pathlib import Path

from dataclasses import dataclass

import pandas as pd
import numpy as np

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
# Data containers
# ----------------------------------------------
@dataclass(frozen=True)
class ExtractionResult:
    X: pd.DataFrame
    y: pd.Series
    final_dropped_cols: list[str]
    final_feature_names: list[str]

@dataclass(frozen=True)
class DatasetSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

@dataclass(frozen=True)
class ProcessingResult:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    preprocessor: Pipeline

@dataclass(frozen=True)
class DatasetTensors:
    X_train: torch.Tensor
    X_val: torch.Tensor
    X_test: torch.Tensor
    y_train: torch.Tensor
    y_val: torch.Tensor
    y_test: torch.Tensor

# ----------------------------------------------
# Extract dataframes from raw csv
# ----------------------------------------------
def extract_df_from_csv():
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

    return ExtractionResult(
        X=X, 
        y=y, 
        final_dropped_cols=cols_to_drop, 
        final_feature_names=feature_names
    )

# ----------------------------------------------
# Split the df into train, val, and test dataframes
# ----------------------------------------------
def split_dataset(X, y):
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

    return DatasetSplits(
        X_train=X_train, 
        X_val=X_val, 
        X_test=X_test,
        y_train=y_train, 
        y_val=y_val, 
        y_test=y_test)

# ----------------------------------------------
# Clean and transform the input features
# ----------------------------------------------
def process_X(X_train, X_val, X_test):
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

    return ProcessingResult(
        X_train=X_train_processed, 
        X_val=X_val_processed, 
        X_test=X_test_processed, 
        preprocessor=preprocessor
    )

# ----------------------------------------------
# Convert input features from NumPy arrays to tensors,
# and label Series into tensors
# ----------------------------------------------
def convert_processed_dataset_into_tensors(processing_result: ProcessingResult, 
    dataset_split: DatasetSplits):
    # Turn the X datasets from NumPy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(processing_result.X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(processing_result.X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(processing_result.X_test, dtype=torch.float32)

    # Turn the y datasets from pandas df to NumPy arrays to PyTorch tensors
    y_train_tensor = torch.tensor(dataset_split.y_train.to_numpy(), 
                                  dtype=torch.float32).view(-1, 1)
    y_val_tensor = torch.tensor(dataset_split.y_val.to_numpy(), 
                                dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(dataset_split.y_test.to_numpy(),
                                dtype=torch.float32).view(-1, 1)
    
    return DatasetTensors(
        X_train=X_train_tensor,
        X_val=X_val_tensor,
        X_test=X_test_tensor,
        y_train=y_train_tensor,
        y_val=y_val_tensor,
        y_test=y_test_tensor
    )

# ----------------------------------------------
# Print the train, val, and test tensors
# ----------------------------------------------
def print_dataset_tensors(dataset_tensors: DatasetTensors):
    print("Tensor shapes:")
    print("--------------------------------------------")
    print(f"X_train: {dataset_tensors.X_train.shape}")
    print(f"y_train: {dataset_tensors.y_train.shape}")
    print(f"X_val: {dataset_tensors.X_val.shape}")
    print(f"y_val: {dataset_tensors.y_val.shape}")
    print(f"X_test: {dataset_tensors.X_test.shape}")
    print(f"y_test: {dataset_tensors.y_test.shape}")

# ----------------------------------------------
# Save processed tensors into file
# ----------------------------------------------
def save_dataset_tensors(dataset_tensors: DatasetTensors):
    # Store the tensors into their corresponding files
    torch.save(
        {
            "X_train": dataset_tensors.X_train,
            "y_train": dataset_tensors.y_train
        },
        PROCESSED_DIR / "train.pt"
    )
    torch.save(
        {
            "X_val": dataset_tensors.X_val,
            "y_val": dataset_tensors.y_val
        },
        PROCESSED_DIR / "val.pt"
    )
    torch.save(
        {
            "X_test": dataset_tensors.X_test,
            "y_test": dataset_tensors.y_test
        },
        PROCESSED_DIR / "test.pt"
    )

# ----------------------------------------------
# Save the processing results to file
# ----------------------------------------------
def save_processing_result(processing_result: ProcessingResult,
    extraction_result: ExtractionResult):
    joblib.dump(
        {
            "preprocessor": processing_result.preprocessor,
            "raw_feature_names": extraction_result.final_feature_names,
            "target_col": TARGET_COL,
            "dropped_cols": extraction_result.final_dropped_cols,
            "polynomial_degree": POLYNOMIAL_TRANSFORMATION_DEGREE,
            "input_dim_after_processing": processing_result.X_train.shape[1]
        },
        PROCESSED_DIR / "preprocessor.joblib"
    )

# ----------------------------------------------
# Main processing function
# ----------------------------------------------
def main():
    # Extract csv data into pandas df
    extraction_result = extract_df_from_csv()

    # Split the df into train, val, and test sets
    dataset_split = split_dataset(
        X=extraction_result.X,
        y=extraction_result.y
    )

    # Clean input features and transform them
    processing_result = process_X(
        X_train=dataset_split.X_train, 
        X_val=dataset_split.X_val,
        X_test=dataset_split.X_test
    )

    # Turn the datasets into PyTorch tensors
    dataset_tensors = convert_processed_dataset_into_tensors(
        processing_result,
        dataset_split
    )

    # Print the tensor shapes for sanity check
    print_dataset_tensors(dataset_tensors)

    # Save results
    save_dataset_tensors(dataset_tensors=dataset_tensors)
    save_processing_result(processing_result=processing_result,
                           extraction_result=extraction_result)

    # Print status message
    print("Preprocessing complete")
    print(f"Saved processed files to: {PROCESSED_DIR}")

if __name__ == "__main__":
    main()