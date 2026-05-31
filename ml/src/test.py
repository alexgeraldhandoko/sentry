import torch
from train import device, build_model
from preprocess import PROCESSED_DIR, ML_DIR
import argparse

# -------------------------------------------------------------------
# Test the model's performance on test data
# -------------------------------------------------------------------
def test():
    # Get the argument
    args = get_args()
    threshold = args.threshold

    # Load the test set
    dataset_dict = torch.load(PROCESSED_DIR / "test.pt")
    X_test = dataset_dict["X_test"].to(device)
    y_test = dataset_dict["y_test"].to(device)

    # Load the model
    model = build_model()
    model_dict = torch.load(ML_DIR / "models" / "best.pth")
    model.load_state_dict(model_dict["model_state"])
    model.to(device)

    # Put the model into evaluation mode
    model.eval()

    # Turn off gradient calculation during testing
    with torch.no_grad():
        logits = model(X_test)
        probabilities = torch.sigmoid(logits)
        classifications = (probabilities >= threshold).float()

        # Calculate metrics
        tp = ((classifications == 1) & (y_test == 1)).sum()
        tn = ((classifications == 0) & (y_test == 0)).sum()
        fp = ((classifications == 1) & (y_test == 0)).sum()
        fn = ((classifications == 0) & (y_test == 1)).sum()

        # Calculate compound metrics
        eps = 1e-8
        accuracy = (classifications == y_test).sum() / (y_test.shape[0] + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f2 = 5 * precision * recall / (4 * precision + recall + eps)

    print("----------------------------------------")
    print("Test performance:")
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"F2: {f2}")
    print("----------------------------------------")

    return accuracy.item(), recall.item(), precision.item(), f2.item()

def get_args():
    parser = argparse.ArgumentParser("Test the fraud detection model")

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for model binary classification"
    )

    return parser.parse_args()

if __name__ == "__main__":
    test()