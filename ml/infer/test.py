import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.config.classifier import network_config, class_table
from src.datasets.dataloader import ProcessData, ProcessNewData
from src.models.vgg16 import VGG16
from src.models.t_mds_vit import TemporalMDSViT


def plot_confusion_matrix(model, test_loader, device, save_path=None):
    # Lists to store predictions and true labels
    all_predictions = []
    all_labels = []

    # Set model to evaluation mode
    model.eval()

    # Gather predictions
    with torch.no_grad():
        for data, label in tqdm(test_loader, desc="Computing Confusion Matrix"):
            data = data.to(device)
            label = label.to(device)

            # Get predictions
            outputs = model(data)
            predicted = torch.argmax(outputs.data, 1)

            # Convert one-hot encoded labels to indices
            label_indices = torch.argmax(label, dim=1)

            # Store predictions and labels
            all_predictions.extend(
                [class_table[int(i)] for i in predicted.cpu().numpy()]
            )
            all_labels.extend(
                [class_table[int(i)] for i in label_indices.cpu().numpy()]
            )

    # Compute confusion matrix
    cm = confusion_matrix(
        all_labels, all_predictions, labels=list(class_table.values())
    )

    # Create plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(class_table.values()),
        yticklabels=list(class_table.values()),
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    # Print some metrics
    accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
    print(f"\nTest Set Metrics:")
    print(f"Overall Accuracy: {accuracy:.4f}")

    # Per-class metrics
    for i in range(len(cm)):
        precision = cm[i, i] / np.sum(cm[:, i]) if np.sum(cm[:, i]) != 0 else 0
        recall = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) != 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) != 0
            else 0
        )
        print(f"\nClass {i}:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

    false_alarm_rates = []
    # Loop through each class
    for i in range(len(cm)):
        # False Positives for class i (sum of the column, excluding the diagonal element)
        FP = sum(cm[:, i]) - cm[i, i]

        # True Negatives for class i (sum of all elements excluding row i and column i)
        TN = cm.sum() - (sum(cm[i, :]) + sum(cm[:, i]) - cm[i, i])

        # Compute FPR
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        false_alarm_rates.append(FPR)

    # Print false alarm rates for each class
    for idx, rate in enumerate(false_alarm_rates):
        print(f"False Alarm Rate for Class {idx}: {rate:.4f}")


if __name__ == "__main__":
    batch_size = network_config["batch_size"]
    device = network_config["device"]

    # Load the model
    model = TemporalMDSViT(
        time_dim=55,
        img_size=256,
        in_channels=1,
        num_classes=3,
        embed_dim=768,
        depth=8,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
    ).to(device)
    model = VGG16().to(device)
    model.load_state_dict(
        torch.load(network_config["weights_path"])["model_state_dict"]
    )

    # Load the test data
    test_data = ProcessNewData(type_data="test")
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Plot the confusion matrix
    plot_confusion_matrix(
        model, test_loader, device, save_path="results/confusion_matrix.png"
    )
