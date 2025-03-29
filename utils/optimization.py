"""
Optimization utilities for the LSX project.

This module provides tools for optimizing the LSX implementation to improve
performance, memory usage, and training speed.
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import time
from tqdm import tqdm

class ModelOptimizer:
    """
    Model optimizer for improving performance and reducing resource usage.
    
    This class provides methods for model pruning, quantization, and other
    optimization techniques to improve the efficiency of LSX models.
    """
    def __init__(self, model):
        """
        Initialize the model optimizer.
        
        Args:
            model: The LSX model to optimize
        """
        self.model = model
        
    def prune_model(self, amount=0.2, modules_to_prune=None):
        """
        Prune the model to reduce its size and potentially improve performance.
        
        Args:
            amount (float): Amount of parameters to prune (between 0 and 1)
            modules_to_prune (list): List of modules to prune (if None, prune all conv and linear layers)
            
        Returns:
            tuple: (model, sparsity) - Pruned model and its sparsity
        """
        # If no modules specified, prune all conv and linear layers
        if modules_to_prune is None:
            modules_to_prune = []
            for name, module in self.model.learner.named_modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    modules_to_prune.append((module, 'weight'))
        
        # Apply pruning
        for module, param_name in modules_to_prune:
            prune.l1_unstructured(module, name=param_name, amount=amount)
        
        # Calculate sparsity
        sparsity = self._calculate_sparsity()
        
        return self.model, sparsity
    
    def _calculate_sparsity(self):
        """
        Calculate the sparsity of the model.
        
        Returns:
            float: Sparsity as a percentage
        """
        total_params = 0
        zero_params = 0
        
        for name, param in self.model.learner.named_parameters():
            if 'weight' in name:
                total_params += param.numel()
                zero_params += (param == 0).sum().item()
        
        sparsity = 100. * zero_params / total_params
        
        return sparsity
    
    def quantize_model(self, dtype=torch.qint8):
        """
        Quantize the model to reduce its memory footprint.
        
        Args:
            dtype: Data type to quantize to
            
        Returns:
            torch.nn.Module: Quantized model
        """
        # Ensure model is in eval mode
        self.model.learner.eval()
        
        # Create quantized model
        quantized_model = torch.quantization.quantize_dynamic(
            self.model.learner,
            {nn.Linear, nn.Conv2d},
            dtype=dtype
        )
        
        # Replace learner with quantized model
        self.model.learner = quantized_model
        
        return self.model
    
    def benchmark_model(self, test_loader, device='cuda'):
        """
        Benchmark the model's inference speed.
        
        Args:
            test_loader: DataLoader for test data
            device: Device to use for benchmarking
            
        Returns:
            float: Average inference time per batch in milliseconds
        """
        # Ensure model is in eval mode
        self.model.learner.eval()
        self.model.learner.to(device)
        
        # Warm-up
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            with torch.no_grad():
                _ = self.model.learner(inputs)
            break
        
        # Benchmark
        total_time = 0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, _ in tqdm(test_loader, desc="Benchmarking"):
                inputs = inputs.to(device)
                
                start_time = time.time()
                _ = self.model.learner(inputs)
                end_time = time.time()
                
                total_time += (end_time - start_time) * 1000  # Convert to ms
                num_batches += 1
        
        avg_time = total_time / num_batches
        
        return avg_time
    
    def optimize_for_inference(self, test_loader, device='cuda'):
        """
        Optimize the model for inference by applying pruning and quantization.
        
        Args:
            test_loader: DataLoader for test data
            device: Device to use for optimization
            
        Returns:
            tuple: (model, results) - Optimized model and benchmark results
        """
        results = {
            'original': {},
            'pruned': {},
            'quantized': {}
        }
        
        # Benchmark original model
        print("Benchmarking original model...")
        orig_time = self.benchmark_model(test_loader, device)
        orig_size = self._get_model_size()
        results['original']['inference_time'] = orig_time
        results['original']['model_size'] = orig_size
        print(f"Original model - Inference time: {orig_time:.2f} ms, Size: {orig_size:.2f} MB")
        
        # Prune model
        print("Pruning model...")
        _, sparsity = self.prune_model(amount=0.2)
        results['pruned']['sparsity'] = sparsity
        
        # Benchmark pruned model
        pruned_time = self.benchmark_model(test_loader, device)
        pruned_size = self._get_model_size()
        results['pruned']['inference_time'] = pruned_time
        results['pruned']['model_size'] = pruned_size
        print(f"Pruned model - Inference time: {pruned_time:.2f} ms, Size: {pruned_size:.2f} MB, Sparsity: {sparsity:.2f}%")
        
        # Quantize model (only if on CPU)
        if device == 'cpu':
            print("Quantizing model...")
            self.quantize_model()
            
            # Benchmark quantized model
            quantized_time = self.benchmark_model(test_loader, device)
            quantized_size = self._get_model_size()
            results['quantized']['inference_time'] = quantized_time
            results['quantized']['model_size'] = quantized_size
            print(f"Quantized model - Inference time: {quantized_time:.2f} ms, Size: {quantized_size:.2f} MB")
        else:
            print("Skipping quantization (only supported on CPU)")
        
        return self.model, results
    
    def _get_model_size(self):
        """
        Get the size of the model in MB.
        
        Returns:
            float: Model size in MB
        """
        param_size = 0
        for param in self.model.learner.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.model.learner.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        
        return size_mb

class TrainingOptimizer:
    """
    Training optimizer for improving training speed and convergence.
    
    This class provides methods for optimizing the training process of LSX models,
    including learning rate scheduling, mixed precision training, and gradient accumulation.
    """
    def __init__(self, model, optimizer, criterion, device='cuda'):
        """
        Initialize the training optimizer.
        
        Args:
            model: The LSX model to optimize
            optimizer: The optimizer to use
            criterion: The loss function to use
            device: Device to use for training
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
    def train_with_lr_scheduler(self, train_loader, val_loader, num_epochs, scheduler_type='cosine'):
        """
        Train the model with learning rate scheduling.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs to train for
            scheduler_type: Type of scheduler to use ('step', 'cosine', or 'plateau')
            
        Returns:
            dict: Training history
        """
        # Initialize scheduler
        if scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        elif scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        elif scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train for one epoch
            train_loss, train_acc = self._train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validate
            val_loss, val_acc = self._validate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Update learning rate
            if scheduler_type == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # Record learning rate
            history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print()
        
        return history
    
    def _train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            tuple: (loss, accuracy)
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
    
    def _validate(self, val_loader):
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            tuple: (loss, accuracy)
        """
        self.model.learner.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model.learner(inputs)
                loss = self.criterion(outputs, targets)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = running_loss / total
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train_with_mixed_precision(self, train_loader, val_loader, num_epochs):
        """
        Train the model with mixed precision to improve training speed.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs to train for
            
        Returns:
            dict: Training history
        """
        # Initialize scaler for mixed precision training
        scaler = torch.cuda.amp.GradScaler()
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train for one epoch
            train_loss, train_acc = self._train_epoch_mixed_precision(train_loader, scaler)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validate
            val_loss, val_acc = self._validate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print()
        
        return history
    
    def _train_epoch_mixed_precision(self, train_loader, scaler):
        """
        Train for one epoch with mixed precision.
        
        Args:
            train_loader: DataLoader for training data
            scaler: GradScaler for mixed precision training
            
        Returns:
            tuple: (loss, accuracy)
        """
        self.model.learner.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(train_loader, desc="Training"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = self.model.learner(inputs)
                loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize with scaler
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
