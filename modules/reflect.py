"""
Reflect module implementation for the LSX project.

This module implements the Reflect phase of the LSX methodology, where the critic
assesses the quality of the learner's explanations.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class Reflect:
    """
    Reflect module for assessing the quality of explanations.
    
    This module implements the third phase of the LSX methodology, where
    the critic assesses the quality of the learner's explanations for performing
    the base task.
    """
    def __init__(self, model, criterion=None, optimizer=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the Reflect module.
        
        Args:
            model: The LSX model containing the critic
            criterion: Loss function (default: CrossEntropyLoss)
            optimizer: Optimizer (default: Adam)
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        
        # Default optimizer if none provided
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.critic.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer
            
        # Move model to device
        self.model.to(self.device)
        
    def train_critic(self, explanations, targets, num_epochs=1, batch_size=64):
        """
        Train the critic to classify explanations.
        
        Args:
            explanations: Tensor of explanations from the learner
            targets: Tensor of target classes
            num_epochs: Number of epochs to train for
            batch_size: Batch size for training
            
        Returns:
            float: Final loss value
        """
        # Create dataset and dataloader
        dataset = TensorDataset(explanations, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Train the critic
        self.model.critic.train()
        final_loss = 0.0
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_explanations, batch_targets in tqdm(dataloader, desc=f"Training critic (Epoch {epoch+1}/{num_epochs})"):
                batch_explanations = batch_explanations.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model.critic(batch_explanations)
                loss = self.criterion(outputs, batch_targets)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                running_loss += loss.item() * batch_explanations.size(0)
                _, predicted = outputs.max(1)
                total += batch_targets.size(0)
                correct += predicted.eq(batch_targets).sum().item()
            
            epoch_loss = running_loss / total
            epoch_acc = 100. * correct / total
            final_loss = epoch_loss
            
            print(f"Critic - Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
        
        return final_loss
    
    def evaluate_explanations(self, explanations, targets, batch_size=64):
        """
        Evaluate explanations using the critic.
        
        Args:
            explanations: Tensor of explanations from the learner
            targets: Tensor of target classes
            batch_size: Batch size for evaluation
            
        Returns:
            tuple: (loss, accuracy, class_probabilities)
        """
        # Create dataset and dataloader
        dataset = TensorDataset(explanations, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Evaluate explanations
        self.model.critic.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        
        with torch.no_grad():
            for batch_explanations, batch_targets in dataloader:
                batch_explanations = batch_explanations.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Forward pass
                outputs = self.model.critic(batch_explanations)
                loss = self.criterion(outputs, batch_targets)
                
                # Get probabilities
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu())
                
                # Statistics
                total_loss += loss.item() * batch_explanations.size(0)
                _, predicted = outputs.max(1)
                total += batch_targets.size(0)
                correct += predicted.eq(batch_targets).sum().item()
        
        # Calculate metrics
        avg_loss = total_loss / total
        accuracy = 100. * correct / total
        all_probs = torch.cat(all_probs, dim=0)
        
        return avg_loss, accuracy, all_probs
    
    def get_feedback(self, explanations, targets):
        """
        Get feedback from the critic on the quality of explanations.
        
        In CNN-LSX, the feedback is the cross-entropy loss of the critic
        classifying the explanations.
        
        Args:
            explanations: Tensor of explanations from the learner
            targets: Tensor of target classes
            
        Returns:
            float: Feedback value (loss)
        """
        # First, train the critic on the explanations
        loss = self.train_critic(explanations, targets)
        
        # Then, evaluate the explanations to get the feedback
        eval_loss, accuracy, _ = self.evaluate_explanations(explanations, targets)
        
        print(f"Critic Feedback - Loss: {eval_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return eval_loss
