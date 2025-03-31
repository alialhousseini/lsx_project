"""
Revise module implementation for the LSX project.

This module implements the Revise phase of the LSX methodology, where the learner
is updated based on the critic's feedback.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Revise:
    """
    Revise module for updating the learner based on the critic's feedback.

    This module implements the fourth phase of the LSX methodology, where
    the learner is refined based on the quality of its explanations.
    """

    def __init__(self, model, criterion=None, optimizer=None, lambda_expl=100, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the Revise module.

        Args:
            model: The LSX model containing the learner
            criterion: Loss function (default: CrossEntropyLoss)
            optimizer: Optimizer (default: Adam)
            lambda_expl: Scaling factor for explanation loss (default: 100)
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.lambda_expl = lambda_expl

        # Default optimizer if none provided
        if optimizer is None:
            self.optimizer = optim.Adam(
                self.model.learner.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer

        # Move model to device
        self.model.to(self.device)

    def revise_learner(self, train_loader, explanations, targets, num_epochs=1):
        """
        Revise the learner based on the critic's feedback.

        In CNN-LSX, the learner is finetuned based on a training signal from:
        1. The learner classifying the original input images (base task)
        2. The critic's classification loss over the provided explanations

        Args:
            train_loader: DataLoader for training data (original images)
            explanations: Tensor of explanations from the learner
            targets: Tensor of target classes
            num_epochs: Number of epochs to train for

        Returns:
            dict: Training history
        """
        history = {
            'base_loss': [],
            'expl_loss': [],
            'total_loss': [],
            'accuracy': []
        }

        # Store original explanations for reference
        original_explanations = explanations.clone().detach()

        for epoch in range(num_epochs):
            print(f"Revise Epoch {epoch+1}/{num_epochs}")

            self.model.learner.train()
            running_base_loss = 0.0
            running_expl_loss = 0.0
            running_total_loss = 0.0
            correct = 0
            total = 0

            # Create iterator for explanations dataset
            expl_dataset = TensorDataset(original_explanations, targets)
            expl_loader = DataLoader(
                expl_dataset, batch_size=train_loader.batch_size, shuffle=False)
            expl_iter = iter(expl_loader)

            for inputs, targets_batch in tqdm(train_loader, desc="Revising learner"):
                inputs, targets_batch = inputs.to(
                    self.device), targets_batch.to(self.device)

                # Get corresponding explanations batch
                try:
                    orig_explanations_batch, _ = next(expl_iter)
                except StopIteration:
                    expl_iter = iter(expl_loader)
                    orig_explanations_batch, _ = next(expl_iter)

                orig_explanations_batch = orig_explanations_batch.to(
                    self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass for base task
                outputs = self.model.learner(inputs)
                base_loss = self.criterion(outputs, targets_batch)

                # Generate new explanations
                inputs.requires_grad = True
                self.model.learner.zero_grad()

                # Forward pass again to get gradients
                outputs = self.model.learner(inputs)

                # Generate explanations for each sample
                new_explanations = []
                for i in range(inputs.size(0)):
                    # Get the score for the target class
                    score = outputs[i, targets_batch[i]]

                    # Compute gradients
                    if i > 0:
                        inputs.grad.zero_()
                    score.backward(retain_graph=(i < inputs.size(0) - 1))

                    # Compute input * gradient
                    explanation = inputs[i].detach().clone(
                    ) * inputs.grad[i].detach().clone()
                    new_explanations.append(explanation.unsqueeze(0))

                # Stack all explanations
                new_explanations = torch.cat(new_explanations, dim=0)

                # Forward pass through critic
                critic_outputs = self.model.critic(new_explanations)
                expl_loss = self.criterion(critic_outputs, targets_batch)

                # Combine losses
                total_loss = base_loss + self.lambda_expl * expl_loss

                # Backward pass and optimize
                total_loss.backward()
                self.optimizer.step()

                # Statistics
                running_base_loss += base_loss.item() * inputs.size(0)
                running_expl_loss += expl_loss.item() * inputs.size(0)
                running_total_loss += total_loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets_batch.size(0)
                correct += predicted.eq(targets_batch).sum().item()

            # Calculate epoch metrics
            epoch_base_loss = running_base_loss / total
            epoch_expl_loss = running_expl_loss / total
            epoch_total_loss = running_total_loss / total
            epoch_acc = 100. * correct / total

            # Update history
            history['base_loss'].append(epoch_base_loss)
            history['expl_loss'].append(epoch_expl_loss)
            history['total_loss'].append(epoch_total_loss)
            history['accuracy'].append(epoch_acc)

            print(f"Base Loss: {epoch_base_loss:.4f}, Expl Loss: {epoch_expl_loss:.4f}, "
                  f"Total Loss: {epoch_total_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        return history

    def final_finetuning(self, train_loader, explanations, num_epochs=1):
        """
        Perform final finetuning of the learner.

        As mentioned in the paper, as a final step, the learner produces explanations
        for all samples and is optimized for the base task while ensuring it doesn't
        diverge its explanations from the stored ones.

        Args:
            train_loader: DataLoader for training data
            explanations: Tensor of reference explanations
            num_epochs: Number of epochs to train for

        Returns:
            dict: Training history
        """
        history = {
            'base_loss': [],
            'expl_loss': [],
            'total_loss': [],
            'accuracy': []
        }

        # MSE loss for explanation consistency
        mse_loss = nn.MSELoss()

        for epoch in range(num_epochs):
            print(f"Final Finetuning Epoch {epoch+1}/{num_epochs}")

            self.model.learner.train()
            running_base_loss = 0.0
            running_expl_loss = 0.0
            running_total_loss = 0.0
            correct = 0
            total = 0

            # Create iterator for explanations dataset
            expl_dataset = TensorDataset(explanations)
            expl_loader = DataLoader(
                expl_dataset, batch_size=train_loader.batch_size, shuffle=False)
            expl_iter = iter(expl_loader)

            for inputs, targets_batch in tqdm(train_loader, desc="Final finetuning"):
                inputs, targets_batch = inputs.to(
                    self.device), targets_batch.to(self.device)

                # Get corresponding reference explanations batch
                try:
                    (ref_explanations_batch,) = next(expl_iter)
                except StopIteration:
                    expl_iter = iter(expl_loader)
                    (ref_explanations_batch,) = next(expl_iter)

                ref_explanations_batch = ref_explanations_batch.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass for base task
                outputs = self.model.learner(inputs)
                base_loss = self.criterion(outputs, targets_batch)

                # Generate new explanations
                inputs.requires_grad = True
                self.model.learner.zero_grad()

                # Forward pass again to get gradients
                outputs = self.model.learner(inputs)

                # Generate explanations for each sample
                new_explanations = []
                for i in range(inputs.size(0)):
                    # Get the score for the target class
                    score = outputs[i, targets_batch[i]]

                    # Compute gradients
                    if i > 0:
                        inputs.grad.zero_()
                    score.backward(retain_graph=(i < inputs.size(0) - 1))

                    # Compute input * gradient
                    explanation = inputs[i].detach().clone(
                    ) * inputs.grad[i].detach().clone()
                    new_explanations.append(explanation.unsqueeze(0))

                # Stack all explanations
                new_explanations = torch.cat(new_explanations, dim=0)

                # Compute explanation consistency loss
                expl_loss = mse_loss(new_explanations, ref_explanations_batch)

                # Combine losses
                total_loss = base_loss + self.lambda_expl * expl_loss

                # Backward pass and optimize
                total_loss.backward()
                self.optimizer.step()

                # Statistics
                running_base_loss += base_loss.item() * inputs.size(0)
                running_expl_loss += expl_loss.item() * inputs.size(0)
                running_total_loss += total_loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets_batch.size(0)
                correct += predicted.eq(targets_batch).sum().item()

            # Calculate epoch metrics
            epoch_base_loss = running_base_loss / total
            epoch_expl_loss = running_expl_loss / total
            epoch_total_loss = running_total_loss / total
            epoch_acc = 100. * correct / total

            # Update history
            history['base_loss'].append(epoch_base_loss)
            history['expl_loss'].append(epoch_expl_loss)
            history['total_loss'].append(epoch_total_loss)
            history['accuracy'].append(epoch_acc)

            print(f"Base Loss: {epoch_base_loss:.4f}, Expl Loss: {epoch_expl_loss:.4f}, "
                  f"Total Loss: {epoch_total_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        return history
