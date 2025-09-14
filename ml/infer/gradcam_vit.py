import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.config.classifier import network_config, class_table
from src.datasets.dataloader import ProcessNewData
from src.models.t_mds_vit import TemporalMDSViT


class Hook:
    def __init__(self, module):
        self.stored = None
        module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # Store the forward activations
        self.stored = output.detach()


class HookBwd:
    def __init__(self, module):
        self.stored = None
        module.register_full_backward_hook(self.hook_fn)

    def hook_fn(self, module, grad_input, grad_output):
        # Store the gradients during the backward pass
        self.stored = grad_output[0].detach()


class TemporalMDSViTGradCAM:
    """
    Grad-CAM implementation for TemporalMDSViT model using attention layer hooks.
    """

    def __init__(self, model, target_layer=None, device="cpu"):
        self.model = model
        self.device = device

        # If no target layer specified, use the last transformer block
        if target_layer is None:
            target_layer = model.blocks[-1]  # Last transformer block

        # Set up hooks for the attention layer
        self.target_layer = target_layer
        self.hook_activation = Hook(target_layer)
        self.hook_gradient = HookBwd(target_layer)

        self.model.eval()

    def compute_gradcam(self, input_tensor, target_class):
        """
        Compute Grad-CAM using attention layer activations and gradients.

        Args:
            input_tensor: (1, time_dim, H, W) - batch of temporal frames
            target_class: int - target class index for gradients

        Returns:
            gradcam_maps: (time_dim, H, W) - normalized Grad-CAM maps for each frame
        """
        # Clear any existing gradients
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)  # (1, num_classes)

        # Get target class score and compute gradients
        target_score = output[0, target_class]
        target_score.backward()

        # Retrieve stored activations and gradients from the attention layer
        activations = self.hook_activation.stored  # (1, time_dim+1, embed_dim)
        gradients = self.hook_gradient.stored  # (1, time_dim+1, embed_dim)

        if activations is None or gradients is None:
            raise ValueError(
                "Failed to capture activations or gradients. Check target layer."
            )

        # Remove batch dimension and exclude CLS token (first token)
        activations = activations.squeeze(0)[1:]  # (time_dim, embed_dim)
        gradients = gradients.squeeze(0)[1:]  # (time_dim, embed_dim)

        # Compute importance weights by averaging gradients across embedding dimension
        importance_weights = gradients.mean(dim=1, keepdim=True)  # (time_dim, 1)

        # Multiply importance weights with activations
        weighted_activations = importance_weights * activations  # (time_dim, embed_dim)

        # Get activation scores by summing across embedding dimension
        activation_scores = weighted_activations.sum(dim=1)  # (time_dim,)

        # Apply ReLU to remove negative values
        activation_scores = F.relu(activation_scores)

        # Convert to numpy and normalize
        activation_scores = activation_scores.cpu().numpy()

        # Normalize to [0, 1]
        if activation_scores.max() > activation_scores.min():
            activation_scores = (activation_scores - activation_scores.min()) / (
                activation_scores.max() - activation_scores.min()
            )
        else:
            activation_scores = np.zeros_like(activation_scores)

        # Create spatial maps by broadcasting the temporal attention scores
        # Each frame gets the same spatial weight based on its temporal importance
        time_dim, H, W = (
            input_tensor.shape[1],
            input_tensor.shape[2],
            input_tensor.shape[3],
        )
        gradcam_maps = np.zeros((time_dim, H, W))

        for i in range(time_dim):
            # Create a simple spatial mask based on the temporal attention score
            # You could make this more sophisticated by using patch-level attention if available
            gradcam_maps[i] = np.full((H, W), activation_scores[i])

        return gradcam_maps

    def compute_gradcam_with_spatial_detail(self, input_tensor, target_class):
        """
        Alternative method that tries to extract more spatial detail.
        This method computes gradients with respect to patch embeddings and reshapes them.
        """
        # Clear any existing gradients
        self.model.zero_grad()

        # We need to hook into the patch embedding layer to get spatial gradients
        patch_embed_hook = Hook(self.model.patch_embed)
        patch_embed_hook_bwd = HookBwd(self.model.patch_embed)

        # Forward pass
        output = self.model(input_tensor)  # (1, num_classes)

        # Get target class score and compute gradients
        target_score = output[0, target_class]
        target_score.backward()

        # Get patch embeddings and their gradients
        patch_activations = patch_embed_hook.stored  # (1, time_dim, embed_dim)
        patch_gradients = patch_embed_hook_bwd.stored  # (1, time_dim, embed_dim)

        if patch_activations is None or patch_gradients is None:
            # Fallback to the simpler method
            return self.compute_gradcam(input_tensor, target_class)

        # Remove batch dimension
        patch_activations = patch_activations.squeeze(0)  # (time_dim, embed_dim)
        patch_gradients = patch_gradients.squeeze(0)  # (time_dim, embed_dim)

        # Compute importance weights
        importance_weights = patch_gradients.mean(dim=1, keepdim=True)  # (time_dim, 1)

        # Weighted activations
        weighted_activations = (
            importance_weights * patch_activations
        )  # (time_dim, embed_dim)

        # Sum across embedding dimension to get per-frame importance
        frame_importance = weighted_activations.sum(dim=1)  # (time_dim,)
        frame_importance = F.relu(frame_importance)
        frame_importance = frame_importance.cpu().numpy()

        # Normalize
        if frame_importance.max() > frame_importance.min():
            frame_importance = (frame_importance - frame_importance.min()) / (
                frame_importance.max() - frame_importance.min()
            )

        # Create spatial maps with more detail by using input gradients for spatial information
        input_tensor.requires_grad_(True)
        self.model.zero_grad()
        output = self.model(input_tensor)
        target_score = output[0, target_class]
        target_score.backward()

        input_gradients = input_tensor.grad.detach().squeeze(0)  # (time_dim, H, W)

        # Combine frame importance with spatial gradients
        gradcam_maps = np.zeros_like(input_gradients.cpu().numpy())
        for i in range(len(frame_importance)):
            spatial_grad = input_gradients[i].cpu().numpy()
            # Apply ReLU and normalize spatial gradients
            spatial_grad = np.maximum(spatial_grad, 0)
            if spatial_grad.max() > spatial_grad.min():
                spatial_grad = (spatial_grad - spatial_grad.min()) / (
                    spatial_grad.max() - spatial_grad.min()
                )

            # Combine temporal importance with spatial gradients
            gradcam_maps[i] = frame_importance[i] * spatial_grad

        return gradcam_maps


def create_2d_visualization(
    original_data,
    gradcam_maps,
    overlay_data,
    sample_idx,
    target_class,
    save_dir,
    mode="sum",
):
    """
    Create Grad-CAM style 2D visualization:
    1. Original aggregated image
    2. Activation heatmap
    3. Overlay (image + heatmap)

    Parameters:
    - original_data: np.array (C, H, W), raw input
    - gradcam_maps: np.array (C, H, W), attention maps
    - sample_idx: int, index of the sample
    - target_class: int or str, class label
    - save_dir: str, directory to save the figure
    - mode: 'sum' or 'mean' to aggregate over C
    """
    assert mode in ["sum", "mean"], "mode must be 'sum' or 'mean'"

    # Aggregate along C
    if mode == "sum":
        real_image = original_data.sum(axis=0)
        activation_map = gradcam_maps.sum(axis=0)
    else:  # mean
        real_image = original_data.mean(axis=0)
        activation_map = gradcam_maps.mean(axis=0)

    H, W = real_image.shape

    # Normalize real_image for better visualization (grayscale)
    real_image = (real_image - real_image.min()) / (
        real_image.max() - real_image.min() + 1e-8
    )

    # Shared vmin/vmax for consistency
    vmin, vmax = activation_map.min(), activation_map.max()

    # Create figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Original Image (grayscale)
    axs[0].imshow(real_image, alpha=0.9, cmap="jet")
    axs[0].axis("off")
    axs[0].set_title("Input Image")

    # 2. Heatmap (jet colormap, shared vmin/vmax)
    im = axs[1].imshow(
        activation_map,
        alpha=0.4,
        cmap="jet",
        extent=(0, W, H, 0),
        interpolation="bilinear",
    )
    axs[1].axis("off")
    axs[1].set_title("Activation Map")

    # Add colorbar next to heatmap
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    # 3. Overlay (grayscale + jet heatmap, same scale)
    axs[2].imshow(real_image, aspect="auto", alpha=0.9, cmap="jet")
    axs[2].imshow(
        activation_map,
        alpha=0.6,
        cmap="jet",
        extent=(0, W, H, 0),
        interpolation="bilinear",
    )
    axs[2].axis("off")
    axs[2].set_title("Activation Map Overlay")

    plt.tight_layout()

    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir, f"gradcam_style_{mode}_sample_{sample_idx}_class_{target_class}.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved Grad-CAM style visualization to: {save_path}")
    return save_path


def create_3d_visualization(
    original_data,
    gradcam_maps,
    overlay_data,
    sample_idx,
    target_class,
    save_dir,
    mode="sum",
):
    """
    Create 3D Grad-CAM style visualization:
    1. Original aggregated surface
    2. Activation surface
    3. Overlay surface (original + activation)

    Parameters:
    - original_data: np.array (C, H, W), raw input
    - gradcam_maps: np.array (C, H, W), attention maps
    - sample_idx: int, index of the sample
    - target_class: int or str, class label
    - save_dir: str, directory to save the figure
    - mode: 'sum' or 'mean' to aggregate over C
    """
    assert mode in ["sum", "mean"], "mode must be 'sum' or 'mean'"

    # Aggregate along C
    if mode == "sum":
        real_image = original_data.sum(axis=0)  # (H, W)
        activation_map = gradcam_maps.sum(axis=0)  # (H, W)
    else:  # mean
        real_image = original_data.mean(axis=0)
        activation_map = gradcam_maps.mean(axis=0)

    H, W = real_image.shape

    # Normalize real_image for visualization
    real_image = (real_image - real_image.min()) / (
        real_image.max() - real_image.min() + 1e-8
    )

    # Coordinates
    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(22, 6))

    # 1. Original Image Surface
    ax1 = fig.add_subplot(131, projection="3d")
    surf1 = ax1.plot_surface(X, Y, real_image, cmap="jet", edgecolor="none", alpha=0.9)
    ax1.set_title("Original Image (Aggregated)")
    ax1.set_xlabel("Width")
    ax1.set_ylabel("Height")
    ax1.set_zlabel("Power")
    ax1.view_init(elev=30, azim=45)

    # 2. Activation Map Surface
    ax2 = fig.add_subplot(132, projection="3d")
    surf2 = ax2.plot_surface(
        X, Y, activation_map, cmap="jet", edgecolor="none", alpha=0.9
    )
    ax2.set_title("Activation Map (Aggregated)")
    ax2.set_xlabel("Width")
    ax2.set_ylabel("Height")
    ax2.set_zlabel("Power")
    ax2.view_init(elev=30, azim=45)
    fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=12, label="Activation")

    # 3. Overlay Surface
    ax3 = fig.add_subplot(133, projection="3d")
    # Blend original and activation
    overlay_map = real_image + 0.5 * activation_map / (activation_map.max() + 1e-8)
    surf3 = ax3.plot_surface(X, Y, overlay_map, cmap="jet", edgecolor="none", alpha=0.9)
    ax3.set_title("Overlay (Image + Activation)")
    ax3.set_xlabel("Width")
    ax3.set_ylabel("Height")
    ax3.set_zlabel("Power")
    ax3.view_init(elev=30, azim=45)

    plt.tight_layout()

    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir, f"gradcam_3d_{mode}_sample_{sample_idx}_class_{target_class}.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved 3D Grad-CAM visualization to: {save_path}")
    return save_path


def create_3d_visualization_v2(
    original_data, gradcam_maps, overlay_data, sample_idx, target_class, save_dir
):
    """
    Create 3D Grad-CAM style visualization:
    1. Original aggregated surface
    2. Activation surface
    3. Overlay surface (original + activation)

    Parameters:
    - original_data: np.array (C, H, W), raw input
    - gradcam_maps: np.array (C, H, W), attention maps
    - overlay_data: np.array (C, H, W), combined data
    - sample_idx: int, index of the sample
    - target_class: int or str, class label
    - save_dir: str, directory to save the figure
    """
    C, H, W = original_data.shape
    flat_dim = H * W

    # Reshape (C, H, W) -> (C, H*W)
    orig_flat = original_data.reshape(C, flat_dim)
    grad_flat = gradcam_maps.reshape(C, flat_dim)
    over_flat = overlay_data.reshape(C, flat_dim)

    # Coordinates
    x = np.arange(flat_dim)
    y = np.arange(C)
    X, Y = np.meshgrid(x, y)

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(22, 6))

    # 1. Original Data Surface
    ax1 = fig.add_subplot(131, projection="3d")
    surf1 = ax1.plot_surface(X, Y, orig_flat, cmap="jet", edgecolor="none", alpha=0.8)
    ax1.set_title("Original Data")
    ax1.set_xlabel("Velocity x Time (256x256)")
    ax1.set_ylabel("Frames (C=55)")
    ax1.set_zlabel("Value")
    ax1.view_init(elev=30, azim=45)
    fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=12, label="Value")

    # 2. Activation Map Surface
    ax2 = fig.add_subplot(132, projection="3d")
    surf2 = ax2.plot_surface(X, Y, grad_flat, cmap="jet", edgecolor="none", alpha=0.8)
    ax2.set_title("Activation Map")
    ax2.set_xlabel("Velocity x Time (256x256)")
    ax2.set_ylabel("Frames (C=55)")
    ax2.set_zlabel("Value")
    ax2.view_init(elev=30, azim=45)
    fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=12, label="Activation")

    # 3. Overlay Surface
    ax3 = fig.add_subplot(133, projection="3d")
    surf3 = ax3.plot_surface(X, Y, over_flat, cmap="jet", edgecolor="none", alpha=0.8)
    ax3.set_title("Overlay")
    ax3.set_xlabel("Velocity x Time (256x256)")
    ax3.set_ylabel("Frames (C=55)")
    ax3.set_zlabel("Value")
    ax3.view_init(elev=30, azim=45)
    fig.colorbar(surf3, ax=ax3, shrink=0.6, aspect=12, label="Overlay Value")

    plt.tight_layout()

    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir, f"gradcam3d_sample{sample_idx}_class{target_class}.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved 3D Grad-CAM visualization to: {save_path}")
    return save_path


def create_2d_visualization_v2(
    original_data, gradcam_maps, overlay_data, sample_idx, target_class, save_dir
):
    """
    Create 2D Grad-CAM style visualization (scaled like 3D style):
    1. Original data heatmap
    2. Activation heatmap
    3. Overlay heatmap

    Parameters:
    - original_data: np.array (C, H, W), raw input
    - gradcam_maps: np.array (C, H, W), attention maps
    - overlay_data: np.array (C, H, W), combined data
    - sample_idx: int, index of the sample
    - target_class: int or str, class label
    - save_dir: str, directory to save the figure
    """
    C, H, W = original_data.shape
    flat_dim = H * W  # 65536

    # Reshape (C, H, W) -> (C, H*W)
    orig_flat = original_data.reshape(C, flat_dim)
    grad_flat = gradcam_maps.reshape(C, flat_dim)
    over_flat = overlay_data.reshape(C, flat_dim)

    # Scale X-axis (instead of 0..65536, map to 0..256 for readability)
    x_ticks = np.linspace(0, flat_dim, 9)  # 9 tick marks
    x_labels = np.linspace(0, 256, 9, dtype=int)  # map to 256 range

    # Create figure with 3 heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # 1. Original Data Heatmap
    im1 = axes[0].imshow(orig_flat, aspect="auto", cmap="viridis")
    axes[0].set_title("Original Data")
    axes[0].set_xlabel("Velocity x Time (scaled ~256)")
    axes[0].set_ylabel("Frames (C=55)")
    axes[0].set_xticks(x_ticks)
    axes[0].set_xticklabels(x_labels)
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label="Power")

    # 2. Activation Map Heatmap
    im2 = axes[1].imshow(grad_flat, aspect="auto", cmap="plasma")
    axes[1].set_title("Activation Map")
    axes[1].set_xlabel("Velocity x Time (scaled ~256)")
    axes[1].set_ylabel("Frames (C=55)")
    axes[1].set_xticks(x_ticks)
    axes[1].set_xticklabels(x_labels)
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label="Activation")

    # 3. Overlay Heatmap
    im3 = axes[2].imshow(over_flat, aspect="auto", cmap="cividis")
    axes[2].set_title("Overlay")
    axes[2].set_xlabel("Velocity x Time (scaled ~256)")
    axes[2].set_ylabel("Frames (C=55)")
    axes[2].set_xticks(x_ticks)
    axes[2].set_xticklabels(x_labels)
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label="Overlay Value")

    plt.tight_layout()

    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir, f"gradcam2d_plane_scaled_sample{sample_idx}_class{target_class}.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved 2D scaled Grad-CAM visualization to: {save_path}")
    return save_path


def main():
    # Configuration
    device = network_config.get("device", "cpu")

    # Load model
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

    # Load trained weights
    checkpoint = torch.load("./results/vit/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Initialize Grad-CAM with the last transformer block (you can change this)
    # You can also try: model.blocks[-2] for second-to-last block, etc.
    target_layer = model.blocks[-1]  # Last transformer block
    gradcam = TemporalMDSViTGradCAM(model, target_layer, device)

    # Load test data
    test_data = ProcessNewData(type_data="test")
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # Output directory
    save_dir = "./results/temporal_gradcam_3d"
    os.makedirs(save_dir, exist_ok=True)

    # Process samples
    class_names = ["pms", "bms", "cms"]  # Adjust based on your classes

    for idx, (data, label) in enumerate(tqdm(test_loader, desc="Processing samples")):
        # Move data to device
        data = data.to(device)  # Shape: (1, 55, H, W)

        # Get target class
        if isinstance(label, torch.Tensor) and label.numel() > 1:
            target_class = int(label.argmax(dim=1).item())
        else:
            target_class = (
                int(label.item()) if isinstance(label, torch.Tensor) else int(label)
            )

        try:
            # Compute Grad-CAM using attention layers
            # Try the more detailed spatial method first
            gradcam_maps = gradcam.compute_gradcam_with_spatial_detail(
                data, target_class
            )
        except Exception as e:
            print(f"Detailed method failed: {e}")
            # Fallback to simpler method
            gradcam_maps = gradcam.compute_gradcam(data, target_class)

        # Prepare data for visualization
        original_data = data.squeeze(0).detach().cpu().numpy()  # (55, H, W)

        # Normalize original data
        original_data_norm = np.zeros_like(original_data)
        for i in range(original_data.shape[0]):
            frame = original_data[i]
            if frame.max() > frame.min():
                original_data_norm[i] = (frame - frame.min()) / (
                    frame.max() - frame.min()
                )
            else:
                original_data_norm[i] = frame

        # Create overlay (weighted combination of original and Grad-CAM)
        overlay_data = np.zeros_like(original_data)
        for i in range(original_data.shape[0]):
            # Combine: 70% original + 30% Grad-CAM
            overlay_data[i] = 0.7 * original_data[i] + 0.3 * gradcam_maps[i]

        # Create 3D visualization
        save_path = create_2d_visualization(
            original_data=original_data,
            gradcam_maps=gradcam_maps,
            overlay_data=overlay_data,
            sample_idx=idx,
            target_class=target_class,
            save_dir=save_dir,
        )

        print(
            f"Processed sample {idx}: Class {target_class} ({class_names[target_class]})"
        )

    print(f"Processing complete! Results saved in {save_dir}")


if __name__ == "__main__":
    main()
