"""
Visualization utilities for the LSX project.

This module provides functions for visualizing explanations, model performance,
and other aspects of the LSX methodology.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from torchvision.utils import make_grid

def visualize_explanation(image, explanation, title=None, save_path=None, cmap='coolwarm'):
    """
    Visualize an input image and its explanation.
    
    Args:
        image (torch.Tensor): Input image tensor of shape [C, H, W]
        explanation (torch.Tensor): Explanation tensor of shape [C, H, W]
        title (str, optional): Title for the plot
        save_path (str, optional): Path to save the visualization
        cmap (str): Colormap for the explanation visualization
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Convert tensors to numpy arrays
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(explanation, torch.Tensor):
        explanation = explanation.detach().cpu().numpy()
    
    # Transpose from [C, H, W] to [H, W, C]
    if image.shape[0] in [1, 3]:
        image = np.transpose(image, (1, 2, 0))
    if explanation.shape[0] in [1, 3]:
        explanation = np.transpose(explanation, (1, 2, 0))
    
    # Normalize image if needed
    if image.max() > 1:
        image = image / 255.0
    
    # For grayscale images, squeeze the channel dimension
    if image.shape[-1] == 1:
        image = np.squeeze(image, axis=-1)
    
    # Compute absolute values for explanation
    explanation_abs = np.abs(explanation)
    
    # For RGB explanations, convert to grayscale for visualization
    if explanation.shape[-1] == 3:
        explanation_abs = np.mean(explanation_abs, axis=-1)
    elif explanation.shape[-1] == 1:
        explanation_abs = np.squeeze(explanation_abs, axis=-1)
    
    # Normalize explanation for visualization
    explanation_norm = explanation_abs / (explanation_abs.max() + 1e-8)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot original image
    if len(image.shape) == 2 or image.shape[-1] == 1:
        axes[0].imshow(image, cmap='gray')
    else:
        axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot explanation
    im = axes[1].imshow(explanation_norm, cmap=cmap)
    axes[1].set_title('Explanation')
    axes[1].axis('off')
    
    # Add colorbar
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def visualize_explanation_grid(images, explanations, num_samples=10, title=None, save_path=None, cmap='coolwarm'):
    """
    Visualize a grid of input images and their explanations.
    
    Args:
        images (torch.Tensor): Batch of input images of shape [B, C, H, W]
        explanations (torch.Tensor): Batch of explanations of shape [B, C, H, W]
        num_samples (int): Number of samples to visualize
        title (str, optional): Title for the plot
        save_path (str, optional): Path to save the visualization
        cmap (str): Colormap for the explanation visualization
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Limit to num_samples
    if images.shape[0] > num_samples:
        indices = torch.randperm(images.shape[0])[:num_samples]
        images = images[indices]
        explanations = explanations[indices]
    
    # Convert tensors to numpy arrays
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    if isinstance(explanations, torch.Tensor):
        explanations = explanations.detach().cpu().numpy()
    
    # Compute absolute values for explanations
    explanations_abs = np.abs(explanations)
    
    # Normalize explanations for visualization
    explanations_norm = []
    for i in range(explanations_abs.shape[0]):
        expl = explanations_abs[i]
        expl_norm = expl / (expl.max() + 1e-8)
        explanations_norm.append(expl_norm)
    explanations_norm = np.stack(explanations_norm)
    
    # Create figure
    n_rows = num_samples
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, n_rows * 4))
    
    for i in range(n_rows):
        # Get image and explanation
        img = images[i]
        expl = explanations_norm[i]
        
        # Transpose from [C, H, W] to [H, W, C]
        if img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
        if expl.shape[0] in [1, 3]:
            expl = np.transpose(expl, (1, 2, 0))
        
        # Normalize image if needed
        if img.max() > 1:
            img = img / 255.0
        
        # For grayscale images, squeeze the channel dimension
        if img.shape[-1] == 1:
            img = np.squeeze(img, axis=-1)
        
        # For RGB explanations, convert to grayscale for visualization
        if expl.shape[-1] == 3:
            expl = np.mean(expl, axis=-1)
        elif expl.shape[-1] == 1:
            expl = np.squeeze(expl, axis=-1)
        
        # Plot original image
        if len(img.shape) == 2 or img.shape[-1] == 1:
            axes[i, 0].imshow(img, cmap='gray')
        else:
            axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Image {i+1}')
        axes[i, 0].axis('off')
        
        # Plot explanation
        im = axes[i, 1].imshow(expl, cmap=cmap)
        axes[i, 1].set_title(f'Explanation {i+1}')
        axes[i, 1].axis('off')
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def visualize_class_explanations(explanations, targets, num_classes=10, samples_per_class=4, title=None, save_path=None, cmap='coolwarm'):
    """
    Visualize explanations grouped by class.
    
    Args:
        explanations (torch.Tensor): Batch of explanations of shape [B, C, H, W]
        targets (torch.Tensor): Batch of target classes of shape [B]
        num_classes (int): Number of classes to visualize
        samples_per_class (int): Number of samples to visualize per class
        title (str, optional): Title for the plot
        save_path (str, optional): Path to save the visualization
        cmap (str): Colormap for the explanation visualization
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Convert tensors to numpy arrays
    if isinstance(explanations, torch.Tensor):
        explanations = explanations.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Compute absolute values for explanations
    explanations_abs = np.abs(explanations)
    
    # Create figure
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(samples_per_class * 3, num_classes * 3))
    
    for class_idx in range(num_classes):
        # Get explanations for this class
        class_mask = (targets == class_idx)
        class_explanations = explanations_abs[class_mask]
        
        # Skip if no explanations for this class
        if len(class_explanations) == 0:
            continue
        
        # Select random samples
        if len(class_explanations) > samples_per_class:
            indices = np.random.choice(len(class_explanations), samples_per_class, replace=False)
            class_explanations = class_explanations[indices]
        
        # Plot explanations
        for i in range(min(samples_per_class, len(class_explanations))):
            expl = class_explanations[i]
            
            # Transpose from [C, H, W] to [H, W, C]
            if expl.shape[0] in [1, 3]:
                expl = np.transpose(expl, (1, 2, 0))
            
            # For RGB explanations, convert to grayscale for visualization
            if len(expl.shape) == 3 and expl.shape[-1] == 3:
                expl = np.mean(expl, axis=-1)
            elif len(expl.shape) == 3 and expl.shape[-1] == 1:
                expl = np.squeeze(expl, axis=-1)
            
            # Normalize explanation
            expl_norm = expl / (expl.max() + 1e-8)
            
            # Plot explanation
            im = axes[class_idx, i].imshow(expl_norm, cmap=cmap)
            axes[class_idx, i].set_title(f'Class {class_idx}')
            axes[class_idx, i].axis('off')
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_training_history(history, title=None, save_path=None):
    """
    Plot training history.
    
    Args:
        history (dict): Dictionary containing training history
        title (str, optional): Title for the plot
        save_path (str, optional): Path to save the visualization
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss')
    if 'base_loss' in history:
        axes[0].plot(history['base_loss'], label='Base Loss')
    if 'expl_loss' in history:
        axes[0].plot(history['expl_loss'], label='Explanation Loss')
    if 'total_loss' in history:
        axes[0].plot(history['total_loss'], label='Total Loss')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Acc')
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], label='Val Acc')
    if 'accuracy' in history:
        axes[1].plot(history['accuracy'], label='Accuracy')
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_metrics_comparison(metrics_dict, title=None, save_path=None):
    """
    Plot comparison of metrics between different models.
    
    Args:
        metrics_dict (dict): Dictionary mapping model names to metric values
        title (str, optional): Title for the plot
        save_path (str, optional): Path to save the visualization
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Extract model names and metrics
    models = list(metrics_dict.keys())
    metrics = {}
    
    # Collect all metric names
    for model_metrics in metrics_dict.values():
        for metric_name in model_metrics:
            if metric_name not in metrics:
                metrics[metric_name] = []
    
    # Collect metric values for each model
    for metric_name in metrics:
        for model in models:
            if metric_name in metrics_dict[model]:
                metrics[metric_name].append(metrics_dict[model][metric_name])
            else:
                metrics[metric_name].append(0)
    
    # Create figure
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(num_metrics * 5, 5))
    
    # Handle case with only one metric
    if num_metrics == 1:
        axes = [axes]
    
    # Plot each metric
    for i, (metric_name, metric_values) in enumerate(metrics.items()):
        axes[i].bar(models, metric_values)
        axes[i].set_title(metric_name)
        axes[i].set_ylabel('Value')
        axes[i].set_xticklabels(models, rotation=45, ha='right')
        axes[i].grid(True, axis='y')
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig
