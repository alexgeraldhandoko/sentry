import sys
import os

from fastapi import FastAPI, UploadFile, HTTPException

from pathlib import Path

from fastapi.middleware.cors import CORSMiddleware

from web3 import Web3

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
# Declare blockchain constants
# ---------------------------------
BLOCKCHAIN_URL = os.environ["BLOCKCHAIN_URL"]
WALLET_PRIVATE_KEY = os.environ["WALLET_PRIVATE_KEY"]
SMART_CONTRACT_ADDRESS = os.environ["SMART_CONTRACT_ADDRESS"]

BLOCKCHAIN_ABI = [
    {
        "inputs": [
            {"internalType": "string", "name": "classification", "type": "string"},
            {"internalType": "uint256", "name": "confidence", "type": "uint256"},
            {"internalType": "uint256", "name": "trueLabel", "type": "uint256"}
        ],
        "name": "recordTransaction",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }, 
    {
        "inputs": [],
        "name": "getTransactionCount",
        "outputs": [
            {"internalType": "uint256", "name": "transactionCount", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    }, 
    {
        "inputs": [
            {"internalType": "uint256", "name": "index", "type": "uint256"}
        ],
        "name": "getTransactionAtIndex",
        "outputs": [
            {"internalType": "string", "name": "classification", "type": "string"},
            {"internalType": "uint256", "name": "confidence", "type": "uint256"},
            {"internalType": "uint256", "name": "trueLabel", "type": "uint256"},
            {"internalType": "uint256", "name": "timeStamp", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

# ---------------------------------
# Make the blockchain connection
# ---------------------------------
# Connect to web 3
web3 = Web3(Web3.HTTPProvider(BLOCKCHAIN_URL))
if not web3.is_connected():
    raise RuntimeError("FastAPI could not connect to the blockchain node.")

# Make smart contract
smart_contract = web3.eth.contract(
    address=Web3.to_checksum_address(SMART_CONTRACT_ADDRESS),
    abi=BLOCKCHAIN_ABI
)

# Create wallet
wallet = web3.eth.account.from_key(WALLET_PRIVATE_KEY)

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
        "http://172.20.10.5",
        "http://localhost:3000"
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

@app.post("/record-transaction")
def record_transaction(data: dict):
    # Extract the data fields
    classification = data["classification"]
    confidence = data["confidence"]
    true_label = data["true_label"]

    # Convert the confidence percentage to 4 digits
    confidence_integer = int(round(float(confidence) * 100))

    # Build the transaction
    transaction = smart_contract.functions.recordTransaction(
        classification,
        confidence_integer,
        int(true_label)
    ).build_transaction({
        "from": wallet.address,
        "nonce": web3.eth.get_transaction_count(wallet.address),
        "gas": 300000,
        "gasPrice": web3.eth.gas_price,
        "chainId": web3.eth.chain_id
    })

    # Sign the transaction
    signed_transaction = web3.eth.account.sign_transaction(transaction, 
        private_key=WALLET_PRIVATE_KEY)

    # Send the transaction
    transaction_hash = web3.eth.send_raw_transaction(signed_transaction.raw_transaction)

    # Wait for transaction receipt
    transaction_receipt = web3.eth.wait_for_transaction_receipt(
        transaction_hash=transaction_hash)
    if (transaction_receipt.status != 1):
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Blockchain transaction failed."
        )

    # If transaction receipt successfully received, return
    # success message, transaction hash, and block number
    return {
        "message": "Blockchain transaction successful.",
        "transaction_hash": transaction_hash.hex(),
        "block_no": transaction_receipt.blockNumber
    }