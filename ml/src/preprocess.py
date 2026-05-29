from pathlib import Path
import pandas as pd

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

# ----------------------------------------------
# Main processing function
# ----------------------------------------------
def main():

    # Declare the paths
    ml_dir = Path(__file__).resolve().parents[1]
    raw_path = ml_dir / "data" / "raw" / "data.csv"
    processed_dir = ml_dir / "data" / "processed"

    # Create the processed folder
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Read the csv
    df = pd.read_csv(raw_path)

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

main()