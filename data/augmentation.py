"""
Data augmentation utilities for the LSX project.

This module provides data augmentation techniques to enhance model training
and improve generalization performance.
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

class AugmentedDataset(Dataset):
    """
    Dataset wrapper that applies data augmentation on-the-fly.
    
    This class wraps an existing dataset and applies specified augmentations
    to the data during loading.
    """
    def __init__(self, dataset, transform=None):
        """
        Initialize the augmented dataset.
        
        Args:
            dataset: Base dataset to augment
            transform: Transformations to apply to the data
        """
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        """Return the length of the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset with augmentation applied.
        
        Args:
            idx: Index of the item to get
            
        Returns:
            tuple: (augmented_image, label)
        """
        image, label = self.dataset[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_mnist_augmentation():
    """
    Get data augmentation transforms for MNIST dataset.
    
    Returns:
        torchvision.transforms.Compose: Composition of transforms
    """
    return transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def get_chestmnist_augmentation():
    """
    Get data augmentation transforms for ChestMNIST dataset.
    
    Returns:
        torchvision.transforms.Compose: Composition of transforms
    """
    return transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

def get_cub10_augmentation():
    """
    Get data augmentation transforms for CUB-10 dataset.
    
    Returns:
        torchvision.transforms.Compose: Composition of transforms
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_augmented_dataloader(dataset, batch_size=64, augmentation=None, shuffle=True):
    """
    Create a dataloader with augmented data.
    
    Args:
        dataset: Base dataset
        batch_size: Batch size for the dataloader
        augmentation: Augmentation transforms to apply
        shuffle: Whether to shuffle the data
        
    Returns:
        torch.utils.data.DataLoader: Dataloader with augmented data
    """
    if augmentation:
        augmented_dataset = AugmentedDataset(dataset, augmentation)
    else:
        augmented_dataset = dataset
        
    return DataLoader(augmented_dataset, batch_size=batch_size, shuffle=shuffle)

def mixup_data(x, y, alpha=1.0, device='cuda'):
    """
    Applies Mixup augmentation to the batch.
    
    Args:
        x: Input batch
        y: Target batch
        alpha: Mixup interpolation coefficient
        device: Device to use
        
    Returns:
        tuple: (mixed_x, y_a, y_b, lam) - Mixed inputs, original targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup loss function.
    
    Args:
        criterion: Base loss function
        pred: Model predictions
        y_a: First set of targets
        y_b: Second set of targets
        lam: Mixup interpolation coefficient
        
    Returns:
        torch.Tensor: Mixup loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class CutMix:
    """
    CutMix data augmentation.
    
    This class implements the CutMix augmentation technique, which
    cuts and pastes patches from different images in a batch.
    """
    def __init__(self, alpha=1.0):
        """
        Initialize CutMix.
        
        Args:
            alpha: Alpha parameter for beta distribution
        """
        self.alpha = alpha
        
    def __call__(self, x, y, device='cuda'):
        """
        Apply CutMix to a batch of data.
        
        Args:
            x: Input batch
            y: Target batch
            device: Device to use
            
        Returns:
            tuple: (mixed_x, y_a, y_b, lam) - Mixed inputs, original targets, and lambda
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(device)
        
        # Get dimensions
        _, c, h, w = x.shape
        
        # Get random bounding box
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply CutMix
        mixed_x = x.clone()
        mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda to account for actual area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
        return mixed_x, y, y[index], lam
