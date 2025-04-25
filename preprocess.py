import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as T
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import os

class OLIVESDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.transform = transform
        self.filtered_data = []
        for sample in hf_dataset:
            if any(sample.get(k) is None for k in ["B1", "B2", "B3", "B4", "B5", "B6"]):
                continue
            if sample.get("BCVA") is None or sample.get("CST") is None:
                continue
            self.filtered_data.append(sample)

    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, idx):
        sample = self.filtered_data[idx]
        image = sample["Image"].convert("L")
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor([sample[f"B{i}"] for i in range(1, 7)], dtype=torch.float32)
        extra_features = torch.tensor([sample["BCVA"], sample["CST"]], dtype=torch.float32)
        return image, extra_features, labels

def prepare_data_simple(sample_size=1000, batch_size=16):
    olives = load_dataset("gOLIVES/OLIVES_Dataset", "biomarker_detection")

    train_transform = T.Compose([
        T.Resize((256, 256)),
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor()
    ])

    test_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    small_train_data = olives["train"].select(range(sample_size))
    full_dataset = OLIVESDataset(hf_dataset=small_train_data, transform=None)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # wrap again with transform
    train_dataset = OLIVESDataset(hf_dataset=[full_dataset.filtered_data[i] for i in train_subset.indices], transform=train_transform)
    val_dataset = OLIVESDataset(hf_dataset=[full_dataset.filtered_data[i] for i in val_subset.indices], transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    test_dataset = OLIVESDataset(hf_dataset=olives["test"], transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, train_dataset