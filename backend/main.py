import sys

from fastapi import FastAPI, UploadFile, HTTPException

from pathlib import Path

from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------
# Import predict function from predict
# ---------------------------------
PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1] 
ML_SRC_DIR = PROJECT_ROOT_DIR / "ml" / "src"
sys.path.append(str(ML_SRC_DIR))
from predict import predict as predict_transaction
# ---------------------------------

# ---------------------------------
# Declare constants
# ---------------------------------
THRESHOLD = 0.5
MAX_CSV_FILE_SIZE = 5 * 1024 * 1024

# ---------------------------------
# Declare app
# ---------------------------------
app = FastAPI()

# ---------------------------------
# Set CORS allow list
# ---------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://172.20.10.5"
    ],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"]
)

# ---------------------------------
# Health check
# ---------------------------------
@app.get("/")
def home():
    return {"message": "FastAPI server is running"}

# ---------------------------------
# Prediction route
# ---------------------------------
@app.post("/predict")
async def predict(file: UploadFile):
    # Check if the file is a csv
    if file.filename is None or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400, # Bad request status code
            detail="Bad request. Please upload a CSV file."
        )
    
    # Read the csv file as bytes
    csv_bytes = await file.read()

    # Check if the csv file is empty or too large
    if len(csv_bytes) == 0:
        raise HTTPException(
            status_code=400, # Bad request status code
            detail="Bad request. Uploaded CSV file is empty."
        )
    elif len(csv_bytes) > MAX_CSV_FILE_SIZE:
        raise HTTPException(
            status_code=400, # Bad request status code
            detail="Bad request. Uploaded CSV file exceeds 5MB. Upload a smaller CSV file."
        )

    # Send the csv file bytes to the predict function
    prediction_results = predict_transaction(csv_bytes)

    # Return the output back as JSON
    return {
        "classification": prediction_results["classification"],
        "confidence": prediction_results["confidence"],
        "true_label": prediction_results["true_label"]
    }