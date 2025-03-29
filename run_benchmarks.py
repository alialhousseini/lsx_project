"""
Main entry point for running benchmark tests.

This script provides a command-line interface for running benchmark tests
on different datasets and with different configurations.
"""
import os
import sys
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.benchmark_tests import benchmark_model_architectures, benchmark_explanation_methods

def main():
    """Main entry point for running benchmark tests."""
    parser = argparse.ArgumentParser(description="Run LSX benchmark tests")
    parser.add_argument("--dataset", type=str, choices=["mnist", "chestmnist", "cub10"], 
                        default="mnist", help="Dataset to use for benchmarking")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for dataloaders")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--benchmark_type", type=str, choices=["models", "explanations", "all"],
                        default="all", help="Type of benchmark to run")
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs("benchmark_results", exist_ok=True)
    
    # Run benchmarks
    if args.benchmark_type == "models" or args.benchmark_type == "all":
        print("\n=== Benchmarking Model Architectures ===")
        benchmark_model_architectures(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs
        )
    
    if args.benchmark_type == "explanations" or args.benchmark_type == "all":
        print("\n=== Benchmarking Explanation Methods ===")
        benchmark_explanation_methods(
            dataset_name=args.dataset,
            batch_size=args.batch_size
        )
    
    print("\n=== Benchmarking completed successfully ===")
    print("Results saved to benchmark_results directory")

if __name__ == "__main__":
    main()
