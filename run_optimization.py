"""
Run script for optimizing LSX models.

This script provides a command-line interface for optimizing LSX models
using various techniques to improve performance and training speed.
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import get_mnist_dataloaders, get_chestmnist_dataloaders, get_cub10_dataloaders
from models.lsx import LSX
from utils.optimization import ModelOptimizer, TrainingOptimizer
from utils.metrics import classification_accuracy

def main():
    """Main entry point for optimizing LSX models."""
    parser = argparse.ArgumentParser(description="Optimize LSX models")
    parser.add_argument("--dataset", type=str, choices=["mnist", "chestmnist", "cub10"], 
                        default="mnist", help="Dataset to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for dataloaders")
    parser.add_argument("--model_path", type=str, help="Path to pre-trained model (if not provided, will train a new one)")
    parser.add_argument("--optimization_type", type=str, choices=["model", "training", "all"],
                        default="all", help="Type of optimization to perform")
    parser.add_argument("--output_dir", type=str, default="optimization_results", help="Directory to save results")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "mnist":
        train_loader, critic_loader, test_loader = get_mnist_dataloaders(
            batch_size=args.batch_size,
            num_samples=None,
            critic_samples=1000
        )
        input_channels = 1
        num_classes = 10
    elif args.dataset == "chestmnist":
        train_loader, critic_loader, test_loader = get_chestmnist_dataloaders(
            batch_size=args.batch_size,
            num_samples=None,
            critic_samples=1000
        )
        input_channels = 1
        num_classes = 2
    elif args.dataset == "cub10":
        train_loader, critic_loader, test_loader = get_cub10_dataloaders(
            batch_size=args.batch_size,
            num_samples=None,
            critic_samples=100
        )
        input_channels = 3
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Initialize model
    model = LSX(input_channels=input_channels, num_classes=num_classes).to(device)
    
    # Load pre-trained model if provided
    if args.model_path:
        print(f"Loading pre-trained model from {args.model_path}...")
        model.load_models(args.model_path)
    else:
        print("Training a new model...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.learner.parameters(), lr=0.001)
        
        # Train model
        model.learner.train()
        for epoch in range(5):  # Train for a few epochs
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model.learner(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    
    # Evaluate baseline model
    baseline_acc = classification_accuracy(model.learner, test_loader, device)
    print(f"Baseline model accuracy: {baseline_acc:.2f}%")
    
    # Perform model optimization
    if args.optimization_type in ["model", "all"]:
        print("\n=== Model Optimization ===")
        model_optimizer = ModelOptimizer(model)
        
        # Optimize model for inference
        optimized_model, results = model_optimizer.optimize_for_inference(test_loader, device)
        
        # Save results
        with open(os.path.join(args.output_dir, "model_optimization_results.txt"), "w") as f:
            f.write("=== Model Optimization Results ===\n\n")
            
            f.write("Original Model:\n")
            f.write(f"  Inference Time: {results['original']['inference_time']:.2f} ms\n")
            f.write(f"  Model Size: {results['original']['model_size']:.2f} MB\n\n")
            
            f.write("Pruned Model:\n")
            f.write(f"  Inference Time: {results['pruned']['inference_time']:.2f} ms\n")
            f.write(f"  Model Size: {results['pruned']['model_size']:.2f} MB\n")
            f.write(f"  Sparsity: {results['pruned']['sparsity']:.2f}%\n\n")
            
            if 'quantized' in results and results['quantized']:
                f.write("Quantized Model:\n")
                f.write(f"  Inference Time: {results['quantized']['inference_time']:.2f} ms\n")
                f.write(f"  Model Size: {results['quantized']['model_size']:.2f} MB\n")
        
        # Save optimized model
        torch.save(optimized_model.learner.state_dict(), os.path.join(args.output_dir, "optimized_model.pth"))
    
    # Perform training optimization
    if args.optimization_type in ["training", "all"]:
        print("\n=== Training Optimization ===")
        
        # Reset model
        model = LSX(input_channels=input_channels, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.learner.parameters(), lr=0.001)
        
        # Initialize training optimizer
        training_optimizer = TrainingOptimizer(model, optimizer, criterion, device)
        
        # Train with learning rate scheduling
        print("Training with learning rate scheduling...")
        lr_history = training_optimizer.train_with_lr_scheduler(
            train_loader=train_loader,
            val_loader=test_loader,
            num_epochs=10,
            scheduler_type='cosine'
        )
        
        # Evaluate model
        lr_acc = classification_accuracy(model.learner, test_loader, device)
        print(f"LR scheduling model accuracy: {lr_acc:.2f}%")
        
        # Train with mixed precision if on CUDA
        if device.type == 'cuda':
            print("Training with mixed precision...")
            
            # Reset model
            model = LSX(input_channels=input_channels, num_classes=num_classes).to(device)
            optimizer = optim.Adam(model.learner.parameters(), lr=0.001)
            training_optimizer = TrainingOptimizer(model, optimizer, criterion, device)
            
            mp_history = training_optimizer.train_with_mixed_precision(
                train_loader=train_loader,
                val_loader=test_loader,
                num_epochs=10
            )
            
            # Evaluate model
            mp_acc = classification_accuracy(model.learner, test_loader, device)
            print(f"Mixed precision model accuracy: {mp_acc:.2f}%")
        else:
            print("Skipping mixed precision training (only supported on CUDA)")
            mp_acc = 0.0
        
        # Save results
        with open(os.path.join(args.output_dir, "training_optimization_results.txt"), "w") as f:
            f.write("=== Training Optimization Results ===\n\n")
            
            f.write(f"Baseline Accuracy: {baseline_acc:.2f}%\n\n")
            
            f.write("Learning Rate Scheduling:\n")
            f.write(f"  Final Accuracy: {lr_acc:.2f}%\n")
            f.write(f"  Improvement: {lr_acc - baseline_acc:.2f}%\n\n")
            
            if device.type == 'cuda':
                f.write("Mixed Precision Training:\n")
                f.write(f"  Final Accuracy: {mp_acc:.2f}%\n")
                f.write(f"  Improvement: {mp_acc - baseline_acc:.2f}%\n")
        
        # Save optimized model
        torch.save(model.learner.state_dict(), os.path.join(args.output_dir, "optimized_training_model.pth"))
    
    print(f"\n=== Optimization completed successfully ===")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
