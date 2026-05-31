from io import BytesIO
import pandas
from train import build_model
import torch

from pathlib import Path

import joblib

ML_DIR = Path(__file__).resolve().parents[1]
PREPROCESSOR_PATH = ML_DIR / "data" / "processed" / "preprocessor.joblib"
BEST_MODEL_PATH = ML_DIR / "models" / "best.pth"
THRESHOLD = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(csv_bytes):
    # ----------------------------------------------------------
    # Process the csv bytes data into Pandas df
    # ----------------------------------------------------------
    csv_file = BytesIO(csv_bytes)
    transaction_df = pandas.read_csv(csv_file)

    # ----------------------------------------------------------
    # Load the preprocessor bundle
    # ----------------------------------------------------------
    bundle = joblib.load(PREPROCESSOR_PATH)
    preprocessor = bundle["preprocessor"]
    drop_cols = bundle["dropped_cols"]
    target_col = bundle["target_col"]

    # ----------------------------------------------------------
    # Clean the transaction df
    # ----------------------------------------------------------
    # Keep the original df
    original_df = transaction_df

    # Clean the column names
    transaction_df.columns = transaction_df.columns.str.strip()
    
    # Drop the irrelevant columns
    transaction_df = transaction_df.drop(columns=[target_col] + drop_cols)

    # Ensure only numerical columns are kept
    transaction_df = transaction_df.select_dtypes(["number"])

    # ----------------------------------------------------------
    # Preprocess the dataframe based on training data statistics
    # ----------------------------------------------------------
    # Turn df into np array into tensor
    transaction_np = preprocessor.transform(transaction_df)
    transaction_tensor = torch.tensor(
        transaction_np,
        dtype=torch.float32,
        device=device)

    # ----------------------------------------------------------
    # Perform model prediction
    # ----------------------------------------------------------
    # Load the best model
    model = build_model().to(device)
    model_dict = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(model_dict["model_state"])

    # Turn on eval mode
    model.eval()

    # Do forward pass on the PyTorch tensor
    with torch.no_grad():
        logit = model(transaction_tensor)

    # ----------------------------------------------------------
    # Calculate statistics
    # ----------------------------------------------------------
    probability = torch.sigmoid(logit).item()
    classification = "Fraudulent" if probability >= THRESHOLD else "Legitimate"
    true_label = (
        "Fraudulent" if original_df[target_col].iloc[0] == 1 else "Legitimate"
    )
    confidence = (1 - abs((1 if classification == "Fraudulent" else 0) - probability)) * 100

    # ----------------------------------------------------------
    # Return the confidence, classification of the model, and 
    # true label of the test data
    # ----------------------------------------------------------
    return {
        "confidence": confidence,
        "classification": classification,
        "true_label": true_label
    }