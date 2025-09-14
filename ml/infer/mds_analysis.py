import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from src.datasets.dataloader import ProcessNewData
from src.config.classifier import network_config
from src.models.t_mds_vit import TemporalMDSViT


def plot_surface_matrices(matrices):
    """
    Create a surface plot of multiple 1D arrays

    Parameters:
    matrices: numpy array of shape (55, 65536) where each row is a 1D array
    """
    # Create figure and 3D axes
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Create coordinate matrices
    x = np.arange(matrices.shape[1])
    y = np.arange(matrices.shape[0])
    X, Y = np.meshgrid(x, y)

    # Create surface plot
    surf = ax.plot_surface(X, Y, matrices, cmap="viridis", edgecolor="none", alpha=0.8)

    # Add colorbar
    fig.colorbar(surf, ax=ax, label="Value")

    # Add labels and title
    ax.set_xlabel("Velocity x Time (256 x 256)")
    ax.set_ylabel("Range bins x Angle bins (5 x 11)")
    ax.set_zlabel("Value")
    ax.set_title("Micro-Doppler Signature")

    # Adjust the view angle
    ax.view_init(elev=30, azim=45)

    return fig


# Example usage:
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
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
    ).to(device)
    model.load_state_dict(
        torch.load("./results/vit/best_model.pth")["model_state_dict"]
    )
    model.eval()

    # Generate sample data
    test_data = ProcessNewData(type_data="test")
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    datasets = test_data.datasets
    labels = test_data.labels

    labels = np.argmax(labels, axis=1)

    positions = np.argsort(labels)

    datasets = [datasets[i] for i in positions]
    labels = [int(labels[i]) for i in positions]

    real_label = ["pms", "bms", "cms"]
    # for i, (label, ori_image) in enumerate(zip(labels, datasets)):
    #    fig = plot_surface_matrices(ori_image.transpose(2, 0, 1).reshape(55, -1))
    #    plt.savefig(f'./results/statistics/images/mds_{real_label[label]}_{i}.png')
    #    plt.close()
    #    print(f'Saved ./results/statistics/images/mds_{real_label[label]}_{i}.png')

    # Gather predictions
    counter = [0, 0, 0]
    with torch.no_grad():
        for i, (data, label) in enumerate(
            tqdm(test_loader, desc="Computing Confusion Matrix")
        ):
            data = data.to(device)
            label = label.to(device)

            outputs = model(data)
            predicted = torch.argmax(outputs.data, 1)

            # Convert one-hot encoded labels to indices
            label_indices = torch.argmax(label, dim=1)
            fig = plot_surface_matrices(data.squeeze().cpu().numpy().reshape(55, -1))
            plt.savefig(
                f"./results/statistics/images/mds_{real_label[int(label_indices[0])]}_{counter[int(label_indices[0])]}_{real_label[int(predicted[0])]}_{test_data.paths[i].split('/')[-2]}.png"
            )
            plt.close()
            counter[int(label_indices[0])] += 1
