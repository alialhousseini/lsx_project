"""
Performance benchmark tests for the LSX project.

This module contains benchmark tests to evaluate the performance of the LSX implementation
on different datasets and with different configurations.
"""
import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import get_mnist_dataloaders, get_chestmnist_dataloaders, get_cub10_dataloaders
from models.lsx import LSX
from models.enhanced_models import EnhancedCNN, PretrainedResNet
from modules.fit import Fit
from modules.explain import Explain
from modules.reflect import Reflect
from modules.revise import Revise
from utils.metrics import classification_accuracy, ridge_regression_accuracy, inter_vs_intraclass_explanation_similarity
from utils.explanation_methods import ExplanationMethods

def benchmark_model_architectures(dataset_name='mnist', batch_size=64, num_epochs=5):
    """
    Benchmark different model architectures on a specific dataset.
    
    Args:
        dataset_name (str): Name of the dataset to use
        batch_size (int): Batch size for dataloaders
        num_epochs (int): Number of epochs to train for
        
    Returns:
        dict: Dictionary of benchmark results
    """
    print(f"Benchmarking model architectures on {dataset_name} dataset")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    if dataset_name == 'mnist':
        train_loader, critic_loader, test_loader = get_mnist_dataloaders(
            batch_size=batch_size,
            num_samples=None,
            critic_samples=1000
        )
        input_channels = 1
        num_classes = 10
    elif dataset_name == 'chestmnist':
        train_loader, critic_loader, test_loader = get_chestmnist_dataloaders(
            batch_size=batch_size,
            num_samples=None,
            critic_samples=1000
        )
        input_channels = 1
        num_classes = 2
    elif dataset_name == 'cub10':
        train_loader, critic_loader, test_loader = get_cub10_dataloaders(
            batch_size=batch_size,
            num_samples=None,
            critic_samples=100
        )
        input_channels = 3
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Define model architectures to benchmark
    architectures = {
        'CNN': lambda: LSX(input_channels=input_channels, num_classes=num_classes),
        'EnhancedCNN': lambda: LSX(
            input_channels=input_channels, 
            num_classes=num_classes
        ).to(device)
    }
    
    # Add ResNet for RGB datasets
    if input_channels == 3:
        architectures['PretrainedResNet'] = lambda: LSX(
            input_channels=input_channels,
            num_classes=num_classes
        ).to(device)
        
        # Replace learner and critic with ResNet
        model = architectures['PretrainedResNet']()
        model.learner = PretrainedResNet(input_channels=input_channels, num_classes=num_classes)
        model.critic = PretrainedResNet(input_channels=input_channels, num_classes=num_classes)
    
    # Benchmark results
    results = {
        'training_time': {},
        'inference_time': {},
        'accuracy': {},
        'explanation_quality': {}
    }
    
    # Benchmark each architecture
    for name, model_fn in architectures.items():
        print(f"\nBenchmarking {name}...")
        
        # Initialize model
        model = model_fn().to(device)
        
        # Initialize modules
        fit_module = Fit(
            model=model,
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.learner.parameters(), lr=0.001),
            device=device
        )
        
        explain_module = Explain(
            model=model,
            device=device
        )
        
        # Measure training time
        start_time = time.time()
        fit_module.train(
            train_loader=train_loader,
            val_loader=test_loader,
            num_epochs=num_epochs,
            early_stopping=False
        )
        training_time = time.time() - start_time
        results['training_time'][name] = training_time
        print(f"Training time: {training_time:.2f} seconds")
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                _ = model.learner(inputs)
        inference_time = time.time() - start_time
        results['inference_time'][name] = inference_time
        print(f"Inference time: {inference_time:.2f} seconds")
        
        # Measure accuracy
        accuracy = classification_accuracy(model.learner, test_loader, device)
        results['accuracy'][name] = accuracy
        print(f"Accuracy: {accuracy:.2f}%")
        
        # Measure explanation quality
        explanations, _, targets = explain_module.generate_explanations(critic_loader)
        rr_acc = ridge_regression_accuracy(explanations, targets)
        iies = inter_vs_intraclass_explanation_similarity(explanations, targets)
        results['explanation_quality'][name] = {
            'ridge_regression_accuracy': rr_acc,
            'iies': iies
        }
        print(f"Explanation quality:")
        print(f"  Ridge Regression Accuracy: {rr_acc:.2f}%")
        print(f"  IIES: {iies:.4f}")
    
    # Plot results
    plot_benchmark_results(results, f"{dataset_name}_benchmark_results.png")
    
    return results

def benchmark_explanation_methods(dataset_name='mnist', batch_size=64):
    """
    Benchmark different explanation methods on a specific dataset.
    
    Args:
        dataset_name (str): Name of the dataset to use
        batch_size (int): Batch size for dataloaders
        
    Returns:
        dict: Dictionary of benchmark results
    """
    print(f"Benchmarking explanation methods on {dataset_name} dataset")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    if dataset_name == 'mnist':
        train_loader, critic_loader, test_loader = get_mnist_dataloaders(
            batch_size=batch_size,
            num_samples=None,
            critic_samples=1000
        )
        input_channels = 1
        num_classes = 10
    elif dataset_name == 'chestmnist':
        train_loader, critic_loader, test_loader = get_chestmnist_dataloaders(
            batch_size=batch_size,
            num_samples=None,
            critic_samples=1000
        )
        input_channels = 1
        num_classes = 2
    elif dataset_name == 'cub10':
        train_loader, critic_loader, test_loader = get_cub10_dataloaders(
            batch_size=batch_size,
            num_samples=None,
            critic_samples=100
        )
        input_channels = 3
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Initialize model
    model = LSX(input_channels=input_channels, num_classes=num_classes).to(device)
    
    # Initialize modules
    fit_module = Fit(
        model=model,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.learner.parameters(), lr=0.001),
        device=device
    )
    
    # Train model
    print("Training model...")
    fit_module.train(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=5,
        early_stopping=False
    )
    
    # Initialize explanation methods
    explanation_methods = ExplanationMethods(model.learner, device)
    
    # Define methods to benchmark
    methods = ['input_x_gradient', 'gradient_shap', 'integrated_gradients', 'saliency']
    
    # Benchmark results
    results = {
        'computation_time': {},
        'explanation_quality': {}
    }
    
    # Get a batch of data
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        break
    
    # Benchmark each method
    for method in methods:
        print(f"\nBenchmarking {method}...")
        
        # Measure computation time
        start_time = time.time()
        explanations = explanation_methods.get_explanation(inputs, targets, method=method)
        computation_time = time.time() - start_time
        results['computation_time'][method] = computation_time
        print(f"Computation time: {computation_time:.2f} seconds")
        
        # Measure explanation quality
        rr_acc = ridge_regression_accuracy(explanations, targets)
        results['explanation_quality'][method] = {
            'ridge_regression_accuracy': rr_acc
        }
        print(f"Explanation quality:")
        print(f"  Ridge Regression Accuracy: {rr_acc:.2f}%")
    
    # Plot results
    plot_explanation_benchmark_results(results, f"{dataset_name}_explanation_benchmark_results.png")
    
    return results

def plot_benchmark_results(results, save_path=None):
    """
    Plot benchmark results.
    
    Args:
        results (dict): Dictionary of benchmark results
        save_path (str, optional): Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training time
    axes[0, 0].bar(results['training_time'].keys(), results['training_time'].values())
    axes[0, 0].set_title('Training Time (seconds)')
    axes[0, 0].set_ylabel('Time (s)')
    axes[0, 0].grid(True, axis='y')
    
    # Plot inference time
    axes[0, 1].bar(results['inference_time'].keys(), results['inference_time'].values())
    axes[0, 1].set_title('Inference Time (seconds)')
    axes[0, 1].set_ylabel('Time (s)')
    axes[0, 1].grid(True, axis='y')
    
    # Plot accuracy
    axes[1, 0].bar(results['accuracy'].keys(), results['accuracy'].values())
    axes[1, 0].set_title('Accuracy (%)')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].grid(True, axis='y')
    
    # Plot explanation quality
    rr_acc = {k: v['ridge_regression_accuracy'] for k, v in results['explanation_quality'].items()}
    iies = {k: v['iies'] for k, v in results['explanation_quality'].items()}
    
    x = np.arange(len(rr_acc))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, rr_acc.values(), width, label='Ridge Regression Accuracy')
    axes[1, 1].bar(x + width/2, iies.values(), width, label='IIES')
    axes[1, 1].set_title('Explanation Quality')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(rr_acc.keys())
    axes[1, 1].legend()
    axes[1, 1].grid(True, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

def plot_explanation_benchmark_results(results, save_path=None):
    """
    Plot explanation benchmark results.
    
    Args:
        results (dict): Dictionary of benchmark results
        save_path (str, optional): Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot computation time
    axes[0].bar(results['computation_time'].keys(), results['computation_time'].values())
    axes[0].set_title('Computation Time (seconds)')
    axes[0].set_ylabel('Time (s)')
    axes[0].grid(True, axis='y')
    
    # Plot explanation quality
    rr_acc = {k: v['ridge_regression_accuracy'] for k, v in results['explanation_quality'].items()}
    
    axes[1].bar(rr_acc.keys(), rr_acc.values())
    axes[1].set_title('Ridge Regression Accuracy (%)')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].grid(True, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

if __name__ == '__main__':
    # Benchmark model architectures on MNIST
    benchmark_model_architectures('mnist')
    
    # Benchmark explanation methods on MNIST
    benchmark_explanation_methods('mnist')
