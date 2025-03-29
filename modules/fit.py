"""
Fit module implementation for the LSX project.

This module implements the Fit phase of the LSX methodology, where the learner
is trained on the base task (e.g., image classification).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

class Fit:
    """
    Fit module for training the learner on the base task.
    
    This module implements the first phase of the LSX methodology, where
    the learner is optimized to solve a particular problem (e.g., supervised
    image classification).
    """
    def __init__(self, model, criterion=None, optimizer=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the Fit module.
        
        Args:
            model: The LSX model containing the learner
            criterion: Loss function (default: CrossEntropyLoss)
            optimizer: Optimizer (default: Adam)
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        
        # Default optimizer if none provided
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.learner.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer
            
        # Move model to device
        self.model.to(self.device)
        
    def train_epoch(self, train_loader):
        """
        Train the learner for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            float: Average loss for the epoch
        """
        self.model.learner.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(train_loader, desc="Training"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model.learner(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """
        Validate the learner on a validation set.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            tuple: (validation loss, validation accuracy)
        """
        self.model.learner.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model.learner(inputs)
                loss = self.criterion(outputs, targets)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / total
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, train_loader, val_loader=None, num_epochs=10, early_stopping=False, patience=5):
        """
        Train the learner for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            num_epochs: Number of epochs to train for
            early_stopping: Whether to use early stopping
            patience: Number of epochs to wait for improvement before stopping
            
        Returns:
            dict: Training history
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            
            # Validate if validation loader is provided
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Early stopping
                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        break
            
            print()
        
        return history
