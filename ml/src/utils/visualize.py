import matplotlib.pyplot as plt

from src.config.classifier import test_sets, train_sets
from src.datasets.dataloader import ProcessNewData
from src.datasets.load_data import load_data
from src.utils.utils import fetch_batch, mini_batch

test_data = ProcessNewData(type_data="test")

datasets = test_data.datasets
labels = test_data.labels

print(labels[7])
# Select a few slices along the third dimension to visualize
slices_to_plot = [0, 10, 20, 30, 40, 50]
fig, axes = plt.subplots(1, len(slices_to_plot), figsize=(15, 5))

for i, slice_idx in enumerate(slices_to_plot):
    axes[i].imshow(
        datasets[7][:, :, slice_idx], cmap="viridis", interpolation="nearest"
    )
    axes[i].set_title(f"Slice {slice_idx}")
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("results/data_pedestrian.png")
plt.show()
