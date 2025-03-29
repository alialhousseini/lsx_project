"""
Dataset loading and preprocessing utilities.
"""
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
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
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
            
            critic_dataset = torch.utils.data.Subset(train_dataset, critic_indices)
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        else:
            critic_dataset = train_dataset
    else:
        critic_dataset = train_dataset
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    critic_loader = DataLoader(critic_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
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
    train_dataset = ChestMNIST(split='train', transform=data_transform, download=True)
    test_dataset = ChestMNIST(split='test', transform=data_transform, download=True)
    
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
            
            critic_dataset = torch.utils.data.Subset(train_dataset, critic_indices)
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        else:
            critic_dataset = train_dataset
    else:
        critic_dataset = train_dataset
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    critic_loader = DataLoader(critic_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, critic_loader, test_loader

class CUB10Dataset(Dataset):
    """
    CUB-10 dataset class (simplified version of CUB-200-2011 with 10 classes).
    """
    def __init__(self, root_dir, transform=None, train=True, num_classes=10):
        """
        Args:
            root_dir (string): Directory with the CUB-10 dataset
            transform (callable, optional): Optional transform to be applied on a sample
            train (bool): If True, creates dataset from training set, otherwise from test set
            num_classes (int): Number of classes to use (default: 10)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.num_classes = num_classes
        
        # This is a placeholder - in a real implementation, you would load the actual CUB dataset
        # and filter it to include only 10 classes
        self.data = []
        self.targets = []
        
        # Placeholder for loading the dataset
        # In a real implementation, you would:
        # 1. Load the CUB-200-2011 dataset
        # 2. Filter to keep only the first 10 classes
        # 3. Split into train/test sets
        
        # For now, we'll create dummy data for demonstration
        self._create_dummy_data()
    
    def _create_dummy_data(self):
        """Create dummy data for demonstration purposes."""
        # In a real implementation, this would be replaced with actual data loading
        num_samples = 300 if self.train else 100
        self.data = torch.randn(num_samples, 3, 224, 224)
        self.targets = torch.randint(0, self.num_classes, (num_samples,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

def get_cub10_dataloaders(batch_size=64, num_samples=None, critic_samples=None):
    """
    Load CUB-10 dataset and create dataloaders.
    
    Args:
        batch_size (int): Batch size for dataloaders
        num_samples (int, optional): Number of samples to use from training set
        critic_samples (int, optional): Number of samples to use for critic set
        
    Returns:
        tuple: (train_loader, critic_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = CUB10Dataset(root_dir='./data/CUB_10', transform=transform, train=True)
    test_dataset = CUB10Dataset(root_dir='./data/CUB_10', transform=transform, train=False)
    
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
            
            critic_dataset = torch.utils.data.Subset(train_dataset, critic_indices)
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        else:
            critic_dataset = train_dataset
    else:
        critic_dataset = train_dataset
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    critic_loader = DataLoader(critic_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, critic_loader, test_loader
