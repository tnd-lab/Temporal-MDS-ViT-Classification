import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from src.config.classifier import network_config
from src.datasets.dataloader import ProcessNewData
from src.models.vgg16 import VGG16
from src.models.resnet import ResNet50
from src.models.t_mds_vit import TemporalMDSViT

from src.loss.classifier import compute_loss


def plot_metrics(metrics):
    """Plot training and validation metrics."""
    epochs = range(1, len(metrics["train_loss"]) + 1)

    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    ax1.plot(epochs, metrics["train_loss"], "b-", label="Training Loss")
    ax1.plot(epochs, metrics["val_loss"], "r-", label="Validation Loss")
    ax1.set_title("Loss vs. Epochs")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(epochs, metrics["train_acc"], "b-", label="Training Accuracy")
    ax2.plot(epochs, metrics["val_acc"], "r-", label="Validation Accuracy")
    ax2.set_title("Accuracy vs. Epochs")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("results/training_metrics.png")
    plt.close()


def train_epoch(model, train_loader, criterion, optimizer, device, l2_lambda):
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_l2_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training")
    for data, label in progress_bar:
        data = data.to(device)
        label = label.to(device)

        # Forward pass
        output = model(data)
        label_indices = torch.argmax(label, dim=1)

        # Calculate losses
        loss, ce_loss, l2_loss = compute_loss(
            output, label_indices, model, criterion, l2_lambda, device
        )
        total_loss += loss.item()
        total_ce_loss += ce_loss
        total_l2_loss += l2_loss

        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label_indices).sum().item()

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar
        progress_bar.set_postfix(
            {
                "total_loss": f"{loss.item():.4f}",
                "ce_loss": f"{ce_loss:.4f}",
                "l2_loss": f"{l2_loss:.4f}",
                "acc": f"{100 * correct / total:.2f}%",
            }
        )

    return (
        total_loss / len(train_loader),
        total_ce_loss / len(train_loader),
        total_l2_loss / len(train_loader),
        100 * correct / total,
    )


def test_process(model, test_loader, criterion, device, l2_lambda):
    model.eval()
    total_loss = 0
    total_ce_loss = 0
    total_l2_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, label in tqdm(test_loader, desc="Testing"):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            label_indices = torch.argmax(label, dim=1)

            loss, ce_loss, l2_loss = compute_loss(
                output, label_indices, model, criterion, l2_lambda, device
            )
            total_loss += loss.item()
            total_ce_loss += ce_loss
            total_l2_loss += l2_loss

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label_indices).sum().item()

    return (
        total_loss / len(test_loader),
        total_ce_loss / len(test_loader),
        total_l2_loss / len(test_loader),
        100 * correct / total,
    )


if __name__ == "__main__":
    # Configuration
    epochs = network_config["epoch"]
    batch_size = network_config["batch_size"]
    device = network_config["device"]
    lr = network_config["lr"]
    l2_lambda = 0.1  # L2 regularization strength
    k = 5  # Number of folds for k-fold cross-validation

    # Define models
    models = {
        "TemporalMDSViT": lambda: TemporalMDSViT(
            time_dim=55,
            img_size=256,
            in_channels=1,
            num_classes=3,
            embed_dim=768,
            depth=8,
            num_heads=12,
            mlp_ratio=4.0,
            dropout=0.5,
        ),
        "VGG16": VGG16,
        "ResNet50": ResNet50,
    }

    # Load data for CV (mix train and test data)
    train_data = ProcessNewData(type_data="train")
    test_data = ProcessNewData(type_data="test")
    full_data = torch.utils.data.ConcatDataset([train_data, test_data])
    full_indices = np.arange(len(full_data))
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Dictionary to store mean validation accuracies per model
    mean_accs = {}

    for model_name, model_fn in models.items():
        print(f"\nTraining {model_name} with {k}-fold CV")

        fold_val_accs = []  # List of validation accuracy lists (one per fold)

        for fold, (train_idx, val_idx) in enumerate(kf.split(full_indices)):
            print(f"Fold {fold + 1}/{k}")

            # Create data subsets for this fold
            train_subset = torch.utils.data.Subset(full_data, train_idx)
            val_subset = torch.utils.data.Subset(full_data, val_idx)

            train_loader = DataLoader(
                train_subset, batch_size=batch_size, shuffle=True, num_workers=1
            )
            val_loader = DataLoader(
                val_subset, batch_size=batch_size, shuffle=False, num_workers=1
            )

            # Initialize model, criterion, optimizer
            model = model_fn().to(device)
            criterion = torch.nn.CrossEntropyLoss()
            # criterion = FocalLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #    optimizer, mode="min", patience=3, factor=0.1
            # )

            # Metrics for this fold
            fold_metrics = {
                "train_loss": [],
                "val_loss": [],
                "train_acc": [],
                "val_acc": [],
            }

            for epoch in range(epochs):
                print(f"Epoch {epoch + 1}/{epochs}")

                # Train
                train_loss, train_ce_loss, train_l2_loss, train_acc = train_epoch(
                    model, train_loader, criterion, optimizer, device, l2_lambda
                )

                # Validate
                val_loss, val_ce_loss, val_l2_loss, val_acc = test_process(
                    model, val_loader, criterion, device, l2_lambda
                )

                # Store metrics
                fold_metrics["train_loss"].append(train_loss)
                fold_metrics["val_loss"].append(val_loss)
                fold_metrics["train_acc"].append(train_acc)
                fold_metrics["val_acc"].append(val_acc)

                # Print epoch statistics
                print(
                    f"\nTraining - Total Loss: {train_loss:.4f}, CE Loss: {train_ce_loss:.4f}, "
                    f"L2 Loss: {train_l2_loss:.4f}, Acc: {train_acc:.2f}%"
                )
                print(
                    f"Validation - Total Loss: {val_loss:.4f}, CE Loss: {val_ce_loss:.4f}, "
                    f"L2 Loss: {val_l2_loss:.4f}, Acc: {val_acc:.2f}%"
                )

                # Learning rate scheduling (if enabled)
                # scheduler.step(val_loss)

            # Optional: Plot metrics for this fold
            # plot_metrics(fold_metrics)

            # Collect validation accuracies for this fold
            fold_val_accs.append(fold_metrics["val_acc"])

        # Compute mean validation accuracy per epoch across folds
        mean_val_acc = np.mean(fold_val_accs, axis=0)
        mean_accs[model_name] = mean_val_acc

        # Compute and print average final validation accuracy
        final_mean_acc = np.mean([acc[-1] for acc in fold_val_accs])
        print(
            f"{model_name} average final validation accuracy across folds: {final_mean_acc:.2f}%"
        )

    # Plot mean validation accuracy per epoch for all models
    plt.figure(figsize=(10, 6))
    for model_name, accs in mean_accs.items():
        plt.plot(range(1, epochs + 1), accs, label=model_name)
    plt.title("Mean Validation Accuracy per Epoch for Each Model (from k-fold)")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/model_comparison.png")
    plt.close()

    print("Training and comparison completed!")
