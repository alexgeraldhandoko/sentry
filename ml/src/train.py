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
    # -------------------------------------------------------------------
    # Load the arguments
    # -------------------------------------------------------------------
    args = get_args()
    checkpoint = args.checkpoint
    epochs = args.epochs
    batch_size = args.batch_size
    print(f"Use checkpoint: {checkpoint}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")

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
        best_val_accuracy = torch.load(BEST_MODEL_PATH)["accuracy"]
    else:
        best_val_accuracy = 0

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
        val_loss, val_accuracy = validate(model, loss_fn)
        print(f"Validation loss: {val_loss}")
        print(f"Validation accuracy: {val_accuracy}")

        # -------------------------------------------------------------------
        # Save best model
        # -------------------------------------------------------------------
        if (val_accuracy > best_val_accuracy):
            # Save the model into models/best.pth
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "accuracy": best_val_accuracy,
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
                "epochs": start_epoch + i + 1
            },
            CHECKPOINT_MODEL_PATH
        )

    # Print complete status as of the most recent epoch
    print("Training complete")
    print(f"Epochs so far: {start_epoch + epochs}")
    print(f"Last checkpoint accuracy: {torch.load(CHECKPOINT_MODEL_PATH)['accuracy']}")

    # Print the best model test performance
    model.load_state_dict(torch.load(BEST_MODEL_PATH)["model_state"])
    _, test_accuracy = test(model, loss_fn)
    print("Test performance:")
    print(f"Accuracy: {test_accuracy}")

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

        # Calculate accuracy
        probabilities = torch.sigmoid(logits)
        classifications = (probabilities >= 0.5).float()
        correct_predictions = (classifications == y_val).sum()
        accuracy = correct_predictions / y_val.shape[0]

    # Put model back into training mode
    model.train()

    # Return the loss and accuracy from the validation set
    return loss.item(), accuracy.item()

# -------------------------------------------------------------------
# Test the model's performance on test data
# -------------------------------------------------------------------
def test(model, loss_fn):
    # Load the test set
    dataset_dict = torch.load(PROCESSED_DIR / "test.pt")
    X_test = dataset_dict["X_test"].to(device)
    y_test = dataset_dict["y_test"].to(device)

    # Put the model into evaluation mode
    model.eval()

    # Turn off gradient calculation during testing
    with torch.no_grad():
        logits = model(X_test)
        probabilities = torch.sigmoid(logits)
        classifications = (probabilities >= 0.5).float()
        accuracy = (classifications == y_test).sum() / y_test.shape[0]

        loss = loss_fn(logits, y_test)

    return loss.item(), accuracy.item()

if __name__ == "__main__":
    train()