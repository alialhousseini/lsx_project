"""
Dataset loading and preprocessing utilities.
"""
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import medmnist
from medmnist import ChestMNIST
import os
import numpy as np
from PIL import Image


def get_mnist_dataloaders(batch_size=64, num_samples=None, critic_samples=None):
    """
    Load MNIST dataset and create dataloaders.

    Args:
        batch_size (int): Batch size for dataloaders
        num_samples (int, optional): Number of samples to use from training set
        critic_samples (int, optional): Number of samples to use for critic set

    Returns:
        tuple: (train_loader, critic_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load datasets
    train_dataset = datasets.MNIST(
        './data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(
        './data', train=False, download=True, transform=transform)

    # Limit training samples if specified
    if num_samples is not None:
        indices = torch.randperm(len(train_dataset))[:num_samples]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    # Limit testing samples to 0.2 * training samples
    if num_samples is not None:
        indices = torch.randperm(len(test_dataset))[:int(num_samples*0.2)]
        test_dataset = torch.utils.data.Subset(test_dataset, indices)

    # Create critic dataset
    if critic_samples is not None:
        if critic_samples <= len(train_dataset):
            train_indices = torch.randperm(len(train_dataset))
            critic_indices = train_indices[:critic_samples]
            train_indices = train_indices[critic_samples:]

            critic_dataset = torch.utils.data.Subset(
                train_dataset, critic_indices)
            train_dataset = torch.utils.data.Subset(
                train_dataset, train_indices)
        else:
            critic_dataset = train_dataset
    else:
        critic_dataset = train_dataset

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    critic_loader = DataLoader(
        critic_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, critic_loader, test_loader


def get_chestmnist_dataloaders(batch_size=64, num_samples=None, critic_samples=None):
    """
    Load ChestMNIST dataset and create dataloaders.

    Args:
        batch_size (int): Batch size for dataloaders
        num_samples (int, optional): Number of samples to use from training set
        critic_samples (int, optional): Number of samples to use for critic set

    Returns:
        tuple: (train_loader, critic_loader, test_loader)
    """
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # Load datasets
    train_dataset = ChestMNIST(
        split='train', transform=data_transform, download=True)
    test_dataset = ChestMNIST(
        split='test', transform=data_transform, download=True)

    # Limit testing sample to 0.2 * training_size
    if num_samples is not None:
        indices = torch.randperm(len(test_dataset))[:int(num_samples*0.2)]
        test_dataset = torch.utils.data.Subset(test_dataset, indices)

    # Limit training samples if specified
    if num_samples is not None:
        indices = torch.randperm(len(train_dataset))[:num_samples]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    # Create critic dataset
    if critic_samples is not None:
        if critic_samples <= len(train_dataset):
            train_indices = torch.randperm(len(train_dataset))
            critic_indices = train_indices[:critic_samples]
            train_indices = train_indices[critic_samples:]

            critic_dataset = torch.utils.data.Subset(
                train_dataset, critic_indices)
            train_dataset = torch.utils.data.Subset(
                train_dataset, train_indices)
        else:
            critic_dataset = train_dataset
    else:
        critic_dataset = train_dataset

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    critic_loader = DataLoader(
        critic_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, critic_loader, test_loader


class CUB10Dataset(Dataset):
    """
    CUB-10 dataset class (a subset of CUB-200-2011 with 10 classes).
    Expected directory structure:

        root_dir/
             class1/
                img1.jpg
                img2.jpg
                ...
             class2/
                imgX.jpg 
                imgY.jpg 
             ...

    Only the first 10 classes (alphabetically sorted) are used.
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with the CUB-10 dataset.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.transform = transform

        # List subdirectories in root_dir and take the first 10 classes (alphabetically sorted)
        classes = sorted([d for d in os.listdir(root_dir)
                         if os.path.isdir(os.path.join(root_dir, d))])[:10]
        self.class_to_idx = {cls_name: idx for idx,
                             cls_name in enumerate(classes)}

        self.image_paths = []
        self.targets = []

        for cls in classes:
            cls_folder = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_folder, fname))
                    self.targets.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        target = self.targets[idx]
        return image, target


def get_cub10_dataloaders(batch_size=64, train_split=0.8, num_samples=None, critic_samples=None, root_dir='./data/CUB_10'):
    """
    Load CUB-10 dataset and create dataloaders.

    Args:
        batch_size (int): Batch size for dataloaders.
        train_split (float): Fraction of data to use for training (rest is for testing).
        num_samples (int, optional): Number of training samples to use.
        critic_samples (int, optional): Number of samples to use for critic set from training data.
        root_dir (str): Root directory of the CUB-10 dataset.

    Returns:
        tuple: (train_loader, critic_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Load the entire dataset from root_dir
    dataset = CUB10Dataset(root_dir=root_dir, transform=transform)

    # Split into training and testing sets (e.g., 80% train, 20% test)
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size])

    # Optionally limit training samples
    if num_samples is not None and num_samples < len(train_dataset):
        indices = torch.randperm(len(train_dataset))[:num_samples]
        train_dataset = Subset(train_dataset, indices)

    # Optionally limit testing samples
    if num_samples is not None and num_samples < len(train_dataset):
        indices = torch.randperm(len(test_dataset))[:int(num_samples*0.2)]
        test_dataset = Subset(test_dataset, indices)

    # Create critic dataset from the training set if specified
    if critic_samples is not None and critic_samples <= len(train_dataset):
        indices = torch.randperm(len(train_dataset))
        critic_indices = indices[:critic_samples]
        train_indices = indices[critic_samples:]
        critic_dataset = Subset(train_dataset, critic_indices)
        train_dataset = Subset(train_dataset, train_indices)
    else:
        critic_dataset = train_dataset

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    critic_loader = DataLoader(
        critic_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, critic_loader, test_loader


if __name__ == "__main__":
    train_loader, critic_loader, test_loader = get_cub10_dataloaders()
    print(f"CUB-10: {len(train_loader.dataset)} training samples, {len(critic_loader.dataset)} critic samples, {len(test_loader.dataset)} test samples.")
