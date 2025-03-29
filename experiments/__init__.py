"""
Experiment initialization module for the LSX project.

This module provides a common interface for initializing and running experiments
across different datasets.
"""
import os
import sys
import importlib.util
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def init_experiment():
    """
    Initialize experiment based on command line arguments.
    
    Returns:
        tuple: (args, config) - Parsed arguments and loaded configuration
    """
    parser = argparse.ArgumentParser(description="Run LSX experiment")
    parser.add_argument("--dataset", type=str, choices=["mnist", "chestmnist", "cub10"], 
                        default="mnist", help="Dataset to use")
    parser.add_argument("--config", type=str, help="Path to configuration file (overrides dataset)")
    args = parser.parse_args()
    
    # Determine config path
    if args.config:
        config_path = args.config
    else:
        config_path = f"../configs/{args.dataset}_config.py"
    
    # Load configuration
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    
    return args, config
