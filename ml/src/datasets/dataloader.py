import torch
import os
import numpy as np
from scipy.io import loadmat
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import train_test_split
from src.config.classifier import test_sets, train_sets
from src.datasets.load_data import load_data
from src.utils.utils import fetch_batch


class ProcessData(torch.utils.data.Dataset):
    def __init__(self, type_data: str = "train"):
        if type_data == "train":
            self.paths, self.labels = load_data(train_sets)
            self.datasets = fetch_batch(self.paths)
        elif type_data == "test":
            self.paths, self.labels = load_data(test_sets)
            self.datasets = fetch_batch(self.paths)

    def __getitem__(self, item):
        dataset = self.datasets[item]
        label = self.labels[item]

        dataset = torch.FloatTensor(dataset).permute(2, 0, 1)
        label = torch.FloatTensor(label)

        return dataset, label

    def __len__(self):
        return len(self.datasets)


class ProcessNewData(torch.utils.data.Dataset):
    def __init__(self, type_data: str = "train", is_saved: bool = True):
        if is_saved:
            # Load the arrays from .npy files
            train_paths = np.load(
                "./src/datasets/saved_data/train_paths.npy", allow_pickle=True
            )
            test_paths = np.load(
                "./src/datasets/saved_data/test_paths.npy", allow_pickle=True
            )
            train_labels = np.load(
                "./src/datasets/saved_data/train_labels.npy", allow_pickle=True
            )
            test_labels = np.load(
                "./src/datasets/saved_data/test_labels.npy", allow_pickle=True
            )
        else:
            paths, labels = self.get_datasets()
            train_paths, test_paths, train_labels, test_labels = train_test_split(
                paths,
                labels,
                test_size=0.2,
                stratify=labels,
                random_state=42,  # Use fixed random_state
            )
            # Save the arrays as .npy files
            np.save("./src/datasets/saved_data/train_paths.npy", train_paths)
            np.save("./src/datasets/saved_data/test_paths.npy", test_paths)
            np.save("./src/datasets/saved_data/train_labels.npy", train_labels)
            np.save("./src/datasets/saved_data/test_labels.npy", test_labels)

        if type_data == "train":
            self.datasets = self.read_mat(train_paths)
            self.labels = train_labels
            self.paths = train_paths
        elif type_data == "test":
            self.datasets = self.read_mat(test_paths)
            self.labels = test_labels
            self.paths = test_paths

    def get_datasets(self, data_dir="./src/datasets/mds_dataset/mds_data"):
        labels = ["pms", "bms", "cms"]
        num_labels = []
        file_paths = []

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                # Get the full path
                full_path = os.path.join(root, file)
                for i, label in enumerate(labels):
                    if label in full_path:
                        one_hot = np.zeros(len(labels))
                        one_hot[i] = 1
                        num_labels.append(one_hot)
                        break
                file_paths.append(full_path)

        return file_paths, num_labels

    def read_mat(self, file_paths):
        datasets = []
        for file_path in file_paths:
            dataset = loadmat(file_path)
            datasets.append(np.abs(dataset["STFT_data"]))
        return datasets

    def __getitem__(self, item):
        dataset = self.datasets[item]
        label = self.labels[item]

        dataset = torch.FloatTensor(dataset).permute(2, 0, 1)
        label = torch.FloatTensor(label)

        return dataset, label

    def __len__(self):
        return len(self.datasets)


class GetData(ProcessNewData):
    def __init__(self, type_data="train", is_saved=True):
        super().__init__(type_data, is_saved)

        paths, labels = self.get_datasets()

        self.labels = labels
        self.paths = paths
        self.datasets = self.read_mat(paths)


if __name__ == "__main__":
    train_data = ProcessNewData(type_data="train")
    train_loader = DataLoader(train_data, batch_size=5, shuffle=True)
    for data, label in train_loader:
        print(label)
        breakpoint()
