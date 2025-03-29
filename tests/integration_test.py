"""
Integration test script for the LSX project.

This script tests the integration of all components of the LSX project
to ensure they work together seamlessly.
"""
import os
import sys
import torch
import argparse
import importlib.util

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import get_mnist_dataloaders, get_chestmnist_dataloaders, get_cub10_dataloaders
from utils.integration import LSXPipeline

def run_integration_test(config_path):
    """
    Run integration test with the given configuration.
    
    Args:
        config_path (str): Path to the configuration file
    """
    # Load configuration
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    
    # Extract configurations
    dataset_config = config.dataset_config
    
    # Set device
    device = torch.device(config.training_config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data based on dataset
    print("Loading data...")
    dataset_name = dataset_config['name']
    
    if dataset_name == 'mnist':
        train_loader, critic_loader, test_loader = get_mnist_dataloaders(
            batch_size=dataset_config['batch_size'],
            num_samples=dataset_config['num_samples'],
            critic_samples=dataset_config['critic_samples']
        )
    elif dataset_name == 'chestmnist':
        train_loader, critic_loader, test_loader = get_chestmnist_dataloaders(
            batch_size=dataset_config['batch_size'],
            num_samples=dataset_config['num_samples'],
            critic_samples=dataset_config['critic_samples']
        )
    elif dataset_name == 'cub10':
        train_loader, critic_loader, test_loader = get_cub10_dataloaders(
            batch_size=dataset_config['batch_size'],
            num_samples=dataset_config['num_samples'],
            critic_samples=dataset_config['critic_samples']
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Critic set size: {len(critic_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    
    # Create LSX pipeline
    pipeline = LSXPipeline(
        config=config,
        train_loader=train_loader,
        critic_loader=critic_loader,
        test_loader=test_loader,
        device=device
    )
    
    # Run the pipeline
    metrics = pipeline.run_pipeline()
    
    # Test alternative explanation methods
    pipeline.test_alternative_explanations()
    
    # Test data augmentation
    pipeline.test_data_augmentation()
    
    print("\n=== Integration test completed successfully ===")
    print(f"Results saved to {config.experiment_config['save_dir']}")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LSX integration test")
    parser.add_argument("--config", type=str, default="../configs/mnist_config.py", help="Path to configuration file")
    args = parser.parse_args()
    
    run_integration_test(args.config)
