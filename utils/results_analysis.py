"""
Enhanced results analysis module for the LSX project.

This module provides tools for analyzing and visualizing the results of LSX experiments,
including comparison between vanilla and LSX models, explanation quality metrics,
and performance improvements from optimization techniques.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

class ResultsAnalyzer:
    """
    Results analyzer for LSX experiments.
    
    This class provides methods for analyzing and visualizing the results of LSX experiments,
    including model performance, explanation quality, and optimization improvements.
    """
    def __init__(self, results_dir):
        """
        Initialize the results analyzer.
        
        Args:
            results_dir (str): Directory containing experiment results
        """
        self.results_dir = results_dir
        self.metrics = self._load_metrics()
        
    def _load_metrics(self):
        """
        Load metrics from the results directory.
        
        Returns:
            dict: Dictionary of metrics
        """
        metrics_path = os.path.join(self.results_dir, "metrics.txt")
        if not os.path.exists(metrics_path):
            return None
        
        metrics = {
            'vanilla': {},
            'lsx': {}
        }
        
        current_section = None
        
        with open(metrics_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith("=== Vanilla Model ==="):
                    current_section = 'vanilla'
                elif line.startswith("=== LSX Model ==="):
                    current_section = 'lsx'
                elif current_section and ":" in line:
                    key, value = line.split(":", 1)
                    metrics[current_section][key.strip()] = float(value.strip())
        
        return metrics
    
    def plot_accuracy_comparison(self, save_path=None):
        """
        Plot accuracy comparison between vanilla and LSX models.
        
        Args:
            save_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if self.metrics is None:
            print("No metrics found in results directory")
            return None
        
        # Extract accuracy metrics
        vanilla_acc = self.metrics['vanilla'].get('accuracy', 0)
        lsx_acc = self.metrics['lsx'].get('accuracy', 0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot bar chart
        models = ['Vanilla', 'LSX']
        accuracies = [vanilla_acc, lsx_acc]
        
        ax.bar(models, accuracies, color=['#1f77b4', '#ff7f0e'])
        
        # Add values on top of bars
        for i, acc in enumerate(accuracies):
            ax.text(i, acc + 1, f"{acc:.2f}%", ha='center')
        
        # Set labels and title
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Model Accuracy Comparison')
        ax.set_ylim(0, max(accuracies) * 1.2)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add improvement annotation
        improvement = lsx_acc - vanilla_acc
        if improvement > 0:
            ax.annotate(f"+{improvement:.2f}%", 
                        xy=(1, lsx_acc), 
                        xytext=(1.3, (vanilla_acc + lsx_acc) / 2),
                        arrowprops=dict(arrowstyle="->", color='green'),
                        color='green',
                        fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def plot_explanation_quality_comparison(self, save_path=None):
        """
        Plot explanation quality comparison between vanilla and LSX models.
        
        Args:
            save_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if self.metrics is None:
            print("No metrics found in results directory")
            return None
        
        # Extract explanation quality metrics
        metrics_to_plot = ['ridge_regression_accuracy', 'iies', 'comprehensiveness', 'sufficiency']
        available_metrics = []
        
        for metric in metrics_to_plot:
            if metric in self.metrics['vanilla'] and metric in self.metrics['lsx']:
                available_metrics.append(metric)
        
        if not available_metrics:
            print("No explanation quality metrics found")
            return None
        
        # Create figure
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(5 * len(available_metrics), 6))
        
        # Handle case with only one metric
        if len(available_metrics) == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(available_metrics):
            vanilla_value = self.metrics['vanilla'][metric]
            lsx_value = self.metrics['lsx'][metric]
            
            axes[i].bar(['Vanilla', 'LSX'], [vanilla_value, lsx_value], color=['#1f77b4', '#ff7f0e'])
            
            # Add values on top of bars
            axes[i].text(0, vanilla_value + 0.1, f"{vanilla_value:.2f}", ha='center')
            axes[i].text(1, lsx_value + 0.1, f"{lsx_value:.2f}", ha='center')
            
            # Set labels and title
            metric_name = ' '.join(word.capitalize() for word in metric.split('_'))
            axes[i].set_title(metric_name)
            axes[i].grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Add improvement annotation
            improvement = lsx_value - vanilla_value
            if abs(improvement) > 0.01:
                color = 'green' if (improvement > 0 and metric != 'iies') or (improvement < 0 and metric == 'iies') else 'red'
                sign = "+" if improvement > 0 else ""
                axes[i].annotate(f"{sign}{improvement:.2f}", 
                            xy=(1, lsx_value), 
                            xytext=(1.3, (vanilla_value + lsx_value) / 2),
                            arrowprops=dict(arrowstyle="->", color=color),
                            color=color,
                            fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def plot_optimization_results(self, save_path=None):
        """
        Plot optimization results if available.
        
        Args:
            save_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: Figure object or None if no optimization results found
        """
        model_opt_path = os.path.join(self.results_dir, "model_optimization_results.txt")
        training_opt_path = os.path.join(self.results_dir, "training_optimization_results.txt")
        
        if not os.path.exists(model_opt_path) and not os.path.exists(training_opt_path):
            print("No optimization results found")
            return None
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot model optimization results if available
        if os.path.exists(model_opt_path):
            model_results = self._parse_optimization_results(model_opt_path)
            
            if 'inference_time' in model_results:
                # Plot inference time comparison
                models = list(model_results['inference_time'].keys())
                times = list(model_results['inference_time'].values())
                
                axes[0].bar(models, times, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(models)])
                axes[0].set_title('Inference Time (ms)')
                axes[0].set_ylabel('Time (ms)')
                axes[0].grid(True, axis='y', linestyle='--', alpha=0.7)
                
                # Add values on top of bars
                for i, time in enumerate(times):
                    axes[0].text(i, time + 0.1, f"{time:.2f}", ha='center')
                
                # Add improvement annotations
                if 'Original' in model_results['inference_time'] and 'Pruned' in model_results['inference_time']:
                    orig_time = model_results['inference_time']['Original']
                    pruned_time = model_results['inference_time']['Pruned']
                    improvement = (orig_time - pruned_time) / orig_time * 100
                    
                    if improvement > 0:
                        axes[0].annotate(f"-{improvement:.1f}%", 
                                    xy=(1, pruned_time), 
                                    xytext=(1.3, (orig_time + pruned_time) / 2),
                                    arrowprops=dict(arrowstyle="->", color='green'),
                                    color='green',
                                    fontweight='bold')
        
        # Plot training optimization results if available
        if os.path.exists(training_opt_path):
            training_results = self._parse_optimization_results(training_opt_path)
            
            if 'accuracy' in training_results:
                # Plot accuracy comparison
                models = list(training_results['accuracy'].keys())
                accuracies = list(training_results['accuracy'].values())
                
                axes[1].bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(models)])
                axes[1].set_title('Model Accuracy (%)')
                axes[1].set_ylabel('Accuracy (%)')
                axes[1].grid(True, axis='y', linestyle='--', alpha=0.7)
                
                # Add values on top of bars
                for i, acc in enumerate(accuracies):
                    axes[1].text(i, acc + 0.1, f"{acc:.2f}%", ha='center')
                
                # Add improvement annotations
                if 'Baseline' in training_results['accuracy'] and 'LR Scheduling' in training_results['accuracy']:
                    baseline_acc = training_results['accuracy']['Baseline']
                    lr_acc = training_results['accuracy']['LR Scheduling']
                    improvement = lr_acc - baseline_acc
                    
                    if improvement > 0:
                        axes[1].annotate(f"+{improvement:.2f}%", 
                                    xy=(1, lr_acc), 
                                    xytext=(1.3, (baseline_acc + lr_acc) / 2),
                                    arrowprops=dict(arrowstyle="->", color='green'),
                                    color='green',
                                    fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def _parse_optimization_results(self, file_path):
        """
        Parse optimization results from a file.
        
        Args:
            file_path (str): Path to the optimization results file
            
        Returns:
            dict: Dictionary of parsed results
        """
        results = {
            'inference_time': {},
            'model_size': {},
            'accuracy': {}
        }
        
        current_section = None
        
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith("Original Model:"):
                    current_section = "Original"
                elif line.startswith("Pruned Model:"):
                    current_section = "Pruned"
                elif line.startswith("Quantized Model:"):
                    current_section = "Quantized"
                elif line.startswith("Baseline Accuracy:"):
                    _, value = line.split(":", 1)
                    results['accuracy']['Baseline'] = float(value.strip().rstrip('%'))
                elif line.startswith("Learning Rate Scheduling:"):
                    current_section = "LR Scheduling"
                elif line.startswith("Mixed Precision Training:"):
                    current_section = "Mixed Precision"
                elif current_section and ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if "Inference Time" in key:
                        results['inference_time'][current_section] = float(value.rstrip(' ms'))
                    elif "Model Size" in key:
                        results['model_size'][current_section] = float(value.rstrip(' MB'))
                    elif "Final Accuracy" in key:
                        results['accuracy'][current_section] = float(value.rstrip('%'))
        
        return results
    
    def visualize_explanations_tsne(self, explanations, targets, title="t-SNE Visualization of Explanations", save_path=None):
        """
        Visualize explanations using t-SNE dimensionality reduction.
        
        Args:
            explanations (torch.Tensor): Tensor of explanations
            targets (torch.Tensor): Tensor of target classes
            title (str): Title for the plot
            save_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Convert tensors to numpy arrays
        if isinstance(explanations, torch.Tensor):
            explanations = explanations.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        # Flatten explanations if needed
        if len(explanations.shape) > 2:
            explanations = explanations.reshape(explanations.shape[0], -1)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        explanations_tsne = tsne.fit_transform(explanations)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot t-SNE results
        scatter = ax.scatter(explanations_tsne[:, 0], explanations_tsne[:, 1], c=targets, cmap='tab10', alpha=0.7)
        
        # Add legend
        legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend1)
        
        # Set labels and title
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def visualize_explanations_pca(self, explanations, targets, title="PCA Visualization of Explanations", save_path=None):
        """
        Visualize explanations using PCA dimensionality reduction.
        
        Args:
            explanations (torch.Tensor): Tensor of explanations
            targets (torch.Tensor): Tensor of target classes
            title (str): Title for the plot
            save_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Convert tensors to numpy arrays
        if isinstance(explanations, torch.Tensor):
            explanations = explanations.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        # Flatten explanations if needed
        if len(explanations.shape) > 2:
            explanations = explanations.reshape(explanations.shape[0], -1)
        
        # Apply PCA
        pca = PCA(
(Content truncated due to size limit. Use line ranges to read in chunks)