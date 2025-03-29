"""
Main entry point for the LSX project.

This script provides a unified interface for running LSX experiments
on different datasets.
"""
import os
import sys
import argparse
from experiments import mnist, chestmnist, cub10

def main():
    """
    Main entry point for running LSX experiments.
    """
    parser = argparse.ArgumentParser(description="Run LSX experiments")
    parser.add_argument("--dataset", type=str, choices=["mnist", "chestmnist", "cub10"], 
                        default="mnist", help="Dataset to use")
    parser.add_argument("--config", type=str, help="Path to configuration file (overrides dataset)")
    args = parser.parse_args()
    
    # Determine config path
    if args.config:
        config_path = args.config
    else:
        config_path = os.path.join("configs", f"{args.dataset}_config.py")
    
    # Run experiment based on dataset
    if args.dataset == "mnist" or (args.config and "mnist" in args.config and "chest" not in args.config):
        mnist.run_experiment(config_path)
    elif args.dataset == "chestmnist" or (args.config and "chestmnist" in args.config):
        chestmnist.run_experiment(config_path)
    elif args.dataset == "cub10" or (args.config and "cub10" in args.config):
        cub10.run_experiment(config_path)
    else:
        print(f"Unknown dataset: {args.dataset}")
        sys.exit(1)

if __name__ == "__main__":
    main()
