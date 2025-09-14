import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.config.classifier import network_config
from src.datasets.dataloader import ProcessNewData
from src.models.vgg16 import VGG16
from src.models.t_mds_vit import TemporalMDSViT

from src.loss.classifier import compute_loss


def plot_metrics(metrics):
    """Plot training and testing metrics."""
    epochs = range(1, len(metrics["train_loss"]) + 1)

    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    ax1.plot(epochs, metrics["train_loss"], "b-", label="Training Loss")
    ax1.plot(epochs, metrics["test_loss"], "r-", label="Testing Loss")
    ax1.set_title("Loss vs. Epochs")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(epochs, metrics["train_acc"], "b-", label="Training Accuracy")
    ax2.plot(epochs, metrics["test_acc"], "r-", label="Testing Accuracy")
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
    l2_lambda = 0.01  # L2 regularization strength
    # Initialize metrics tracking
    metrics = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    # Data loading
    # train_data = ProcessData(type_data="train")
    # test_data = ProcessData(type_data="test")
    # Data loading
    train_data = ProcessNewData(type_data="train")
    test_data = ProcessNewData(type_data="test")

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=1
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=1
    )

    # Model initialization
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
    # model = VGG16().to(device)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #    optimizer, mode="min", patience=3, factor=0.1
    # )

    # Training loop
    best_test_acc = 0
    patience = 7
    patience_counter = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_loss, train_ce_loss, train_l2_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, l2_lambda
        )

        # Test
        test_loss, test_ce_loss, test_l2_loss, test_acc = test_process(
            model, test_loader, criterion, device, l2_lambda
        )

        # Store metrics
        metrics["train_loss"].append(train_loss)
        metrics["test_loss"].append(test_loss)
        metrics["train_acc"].append(train_acc)
        metrics["test_acc"].append(test_acc)

        plot_metrics(metrics)

        # Print epoch statistics
        print(
            f"\nTraining - Total Loss: {train_loss:.4f}, CE Loss: {train_ce_loss:.4f}, "
            f"L2 Loss: {train_l2_loss:.4f}, Acc: {train_acc:.2f}%"
        )
        print(
            f"Testing - Total Loss: {test_loss:.4f}, CE Loss: {test_ce_loss:.4f}, "
            f"L2 Loss: {test_l2_loss:.4f}, Acc: {test_acc:.2f}%"
        )

        # Learning rate scheduling
        # scheduler.step(test_loss)

        # Early stopping
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
            # Save best model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                },
                network_config["weights_path"],
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                # break

    plot_metrics(metrics)
    print("Training completed!")
