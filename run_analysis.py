"""
Run script for analyzing LSX experiment results.

This script provides a command-line interface for analyzing and visualizing
the results of LSX experiments.
"""
import os
import sys
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.results_analysis import ResultsAnalyzer

def main():
    """Main entry point for analyzing LSX experiment results."""
    parser = argparse.ArgumentParser(description="Analyze LSX experiment results")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing experiment results")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save analysis results")
    parser.add_argument("--generate_report", action="store_true", help="Generate comprehensive report")
    args = parser.parse_args()
    
    # Create output directory if not provided
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "analysis")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize results analyzer
    analyzer = ResultsAnalyzer(args.results_dir)
    
    # Generate plots
    print("Generating accuracy comparison plot...")
    analyzer.plot_accuracy_comparison(save_path=os.path.join(args.output_dir, "accuracy_comparison.png"))
    
    print("Generating explanation quality comparison plot...")
    analyzer.plot_explanation_quality_comparison(save_path=os.path.join(args.output_dir, "explanation_quality_comparison.png"))
    
    print("Generating optimization results plot...")
    analyzer.plot_optimization_results(save_path=os.path.join(args.output_dir, "optimization_results.png"))
    
    # Generate comprehensive report if requested
    if args.generate_report:
        print("Generating comprehensive report...")
        report_path = analyzer.generate_comprehensive_report(output_dir=os.path.join(args.output_dir, "report"))
        print(f"Report generated at: {report_path}")
    
    print(f"Analysis completed successfully. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
