import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

import argparse

from preprocess import ML_DIR, RAW_PATH, PROCESSED_DIR

# -------------------------------------------------------------------
# Establish paths
# -------------------------------------------------------------------
MODELS_DIR = ML_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best.pth"
CHECKPOINT_MODEL_PATH = MODELS_DIR / "checkpoint.pth"

# Ensure that the directory to store models exist before Pytorch
# tries to save models inside it
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Figure out device
# -------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------
# Train the model
# -------------------------------------------------------------------
def train():
    print(f"Using {device} to train model")
    # -------------------------------------------------------------------
    # Load the arguments
    # -------------------------------------------------------------------
    args = get_args()
    checkpoint = args.checkpoint
    epochs = args.epochs
    batch_size = args.batch_size
    threshold = args.threshold 
    print("----------------------------------------")
    print(f"Use checkpoint: {checkpoint}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Threshold: {threshold}")
    print("----------------------------------------")

    # -------------------------------------------------------------------
    # Load the saved training dataset
    # -------------------------------------------------------------------
    dataset_dict = torch.load(PROCESSED_DIR / "train.pt")
    X_train = dataset_dict["X_train"].to(device)
    y_train = dataset_dict["y_train"].to(device)

    # Turn the training dataset into a TensorDataset
    train_dataset = TensorDataset(X_train, y_train)

    # Pass the TensorDataset into a DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # -------------------------------------------------------------------
    # Loading information from the last checkpoint, if required
    # -------------------------------------------------------------------
    # Build a consistent model architecture
    model = build_model().to(device)

    start_epoch = 0

    # Load the last checkpoint model if checkpoint argument is used
    if (args.checkpoint):
        checkpoint_data = torch.load(CHECKPOINT_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint_data["model_state"])
        start_epoch = checkpoint_data["epochs"]

    if BEST_MODEL_PATH.exists() and BEST_MODEL_PATH.stat().st_size > 0:
        best_val_f2 = torch.load(BEST_MODEL_PATH)["f2"]
    else:
        best_val_f2 = 0

    # -------------------------------------------------------------------
    # Loss function and optimiser to remember gradient history
    # regardless of epochs
    # -------------------------------------------------------------------
    # Calculate the positive class weight
    num_of_fraud = (y_train == 1).sum()
    num_of_non_fraud = (y_train == 0).sum()
    pos_weight = num_of_non_fraud / num_of_fraud
    pos_weight = pos_weight.reshape(1).to(device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=0.001)

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------
    for i in range (epochs):

        model.train()

        # Repeat (dataset / batch size) number of times:
        for X, y in train_loader:
            # Clear the weight.grads of the current model's weights
            # to prepare training for current batch
            optimizer.zero_grad()

            # Pass a single batch of data into the model
            logits = model(X)

            # Compute loss and accuracy from output
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        # -------------------------------------------------------------------
        # Validate the model
        # -------------------------------------------------------------------
        val_loss, val_accuracy, val_recall, val_precision, val_f2 = validate(model, loss_fn)
        print("----------------------------------------")
        print(f"Validation loss: {val_loss}")
        print(f"Validation accuracy: {val_accuracy}")
        print(f"Validation recall: {val_recall}")
        print(f"Validation precision: {val_precision}")
        print(f"Validation F2: {val_f2}")
        print("----------------------------------------")

        # -------------------------------------------------------------------
        # Save best model
        # -------------------------------------------------------------------
        if (val_f2 > best_val_f2):
            # Save the model into models/best.pth
            best_val_f2 = val_f2
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "accuracy": val_accuracy,
                    "recall": val_recall,
                    "precision": val_precision,
                    "f2": best_val_f2,
                    "epochs": start_epoch + i + 1
                },
                BEST_MODEL_PATH
            )
        
        # -------------------------------------------------------------------
        # Save latest checkpoint
        # -------------------------------------------------------------------
        torch.save(
            {
                "model_state": model.state_dict(),
                "accuracy": val_accuracy,
                "recall": val_recall,
                "precision": val_precision,
                "f2": val_f2,
                "epochs": start_epoch + i + 1
            },
            CHECKPOINT_MODEL_PATH
        )

    # Print complete status as of the most recent epoch
    print("----------------------------------------")
    print("Training complete")
    print(f"Epochs so far: {start_epoch + epochs}")
    print(f"Last checkpoint f2: {torch.load(CHECKPOINT_MODEL_PATH)['f2']}")
    print("----------------------------------------")

    # Print the best model's validation performance
    best_model_dict = torch.load(BEST_MODEL_PATH)
    model.load_state_dict(best_model_dict["model_state"])
    val_loss, val_accuracy, val_recall, val_precision, val_f2 = validate(model, loss_fn)
    print("----------------------------------------")
    print("The best validation performance: ")
    print(f"Validation loss: {val_loss}")
    print(f"Validation accuracy: {val_accuracy}")
    print(f"Validation recall: {val_recall}")
    print(f"Validation precision: {val_precision}")
    print(f"Validation F2: {val_f2}")
    print(f"Using {threshold} threshold")
    print("----------------------------------------")

# -------------------------------------------------------------------
# Obtain the command line arguments that modify the programme execution 
# with flags
# -------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser("Train the fraud detection model")

    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Continue training the model from the last saved checkpoint"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs for this training round"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of training examples per training batch"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Set confidence threshold for model binary classification"
    )

    return parser.parse_args()

# -------------------------------------------------------------------
# Return a PyTorch model with the same architecture
# -------------------------------------------------------------------
def build_model():
    # Load the processed training data
    X_train = torch.load(PROCESSED_DIR / "train.pt")["X_train"]

    # Find number of input features from the X_train tensor
    num_of_input_features = X_train.shape[1] 

    model = nn.Sequential(
        # First layer
        nn.Linear(num_of_input_features, 64),
        nn.ReLU(),

        # Second layer
        nn.Linear(64, 32),
        nn.ReLU(),

        # Third layer
        nn.Linear(32, 16),
        nn.ReLU(),

        # Output layer
        nn.Linear(16, 1)
    )
    return model

# -------------------------------------------------------------------
# Validate the model using validation dataset
# Returns the loss and accuracy
# -------------------------------------------------------------------
def validate(model, loss_fn):
    # Get threshold value from command line arguments
    args = get_args()
    threshold = args.threshold

    # Load validation dataset
    dataset_dict = torch.load(PROCESSED_DIR / "val.pt")
    X_val = dataset_dict["X_val"].to(device)
    y_val = dataset_dict["y_val"].to(device)

    # Put model in evaluation mode
    # This prevents the model from executing training behaviour, like:
    # 1. Randomly dropping some neurons like Dropout
    # 2. Batch normalisation is turned off, use running statistics from
    #    training instead of calculating the statistics from current batch
    model.eval()

    # Turn off gradient calculation during validation
    with torch.no_grad():
        # Pass the validation dataset into the model
        logits = model(X_val)

        # Calculate loss
        loss = loss_fn(logits, y_val)

        probabilities = torch.sigmoid(logits)
        classifications = (probabilities >= threshold).float()

        # Calculate metrics
        tp = ((classifications == 1) & (y_val == 1)).sum()
        tn = ((classifications == 0) & (y_val == 0)).sum()
        fp = ((classifications == 1) & (y_val == 0)).sum()
        fn = ((classifications == 0) & (y_val == 1)).sum()

        # Calculate compound metrics
        eps = 1e-8
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        recall = tp / (tp + fn + eps)
        precision = tp / (tp + fp + eps)
        f2 = 5 * ( (precision * recall) / ((4 * precision) + recall + eps) )

    # Put model back into training mode
    model.train()

    # Return loss, accuracy, recall, precision, f2 from the validation set
    return loss.item(), accuracy.item(), recall.item(), precision.item(), f2.item()

if __name__ == "__main__":
    train()