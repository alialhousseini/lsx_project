"""
Enhanced explanation methods for the LSX project.

This module provides additional explanation methods beyond the basic InputXGradient
method used in the core implementation. These methods can be used to generate
different types of explanations for the learner's predictions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from captum.attr import (
    InputXGradient,
    GradientShap,
    IntegratedGradients,
    Occlusion,
    GuidedBackprop,
    Saliency
)

class ExplanationMethods:
    """
    Class providing various explanation methods for neural networks.
    
    This class implements multiple explanation methods from the Captum library
    that can be used as alternatives to the default InputXGradient method.
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the explanation methods.
        
        Args:
            model: The model to explain
            device: Device to use for generating explanations
        """
        self.model = model
        self.device = device
        
        # Initialize explanation methods
        self.input_x_gradient = InputXGradient(self.model)
        self.gradient_shap = GradientShap(self.model)
        self.integrated_gradients = IntegratedGradients(self.model)
        self.occlusion = Occlusion(self.model)
        self.guided_backprop = GuidedBackprop(self.model)
        self.saliency = Saliency(self.model)
        
    def explain_input_x_gradient(self, inputs, targets):
        """
        Generate explanations using the InputXGradient method.
        
        Args:
            inputs: Input tensor
            targets: Target tensor
            
        Returns:
            torch.Tensor: Explanation tensor
        """
        return self.input_x_gradient.attribute(inputs, target=targets)
    
    def explain_gradient_shap(self, inputs, targets, n_samples=50, stdevs=0.0001):
        """
        Generate explanations using the GradientShap method.
        
        Args:
            inputs: Input tensor
            targets: Target tensor
            n_samples: Number of samples to use
            stdevs: Standard deviation of the noise
            
        Returns:
            torch.Tensor: Explanation tensor
        """
        # Create baseline (random noise)
        baseline_dist = torch.randn(n_samples, *inputs.shape[1:], device=self.device) * stdevs
        
        # Generate explanations
        return self.gradient_shap.attribute(inputs, baselines=baseline_dist, target=targets)
    
    def explain_integrated_gradients(self, inputs, targets, n_steps=50):
        """
        Generate explanations using the IntegratedGradients method.
        
        Args:
            inputs: Input tensor
            targets: Target tensor
            n_steps: Number of steps in the Riemann approximation
            
        Returns:
            torch.Tensor: Explanation tensor
        """
        # Create baseline (black image)
        baseline = torch.zeros_like(inputs)
        
        # Generate explanations
        return self.integrated_gradients.attribute(inputs, baselines=baseline, target=targets, n_steps=n_steps)
    
    def explain_occlusion(self, inputs, targets, sliding_window_shapes=(1, 5, 5), strides=(1, 2, 2)):
        """
        Generate explanations using the Occlusion method.
        
        Args:
            inputs: Input tensor
            targets: Target tensor
            sliding_window_shapes: Shape of the sliding window
            strides: Stride of the sliding window
            
        Returns:
            torch.Tensor: Explanation tensor
        """
        # Generate explanations
        return self.occlusion.attribute(
            inputs, 
            target=targets,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides
        )
    
    def explain_guided_backprop(self, inputs, targets):
        """
        Generate explanations using the GuidedBackprop method.
        
        Args:
            inputs: Input tensor
            targets: Target tensor
            
        Returns:
            torch.Tensor: Explanation tensor
        """
        return self.guided_backprop.attribute(inputs, target=targets)
    
    def explain_saliency(self, inputs, targets):
        """
        Generate explanations using the Saliency method.
        
        Args:
            inputs: Input tensor
            targets: Target tensor
            
        Returns:
            torch.Tensor: Explanation tensor
        """
        return self.saliency.attribute(inputs, target=targets)
    
    def get_explanation(self, inputs, targets, method='input_x_gradient', **kwargs):
        """
        Generate explanations using the specified method.
        
        Args:
            inputs: Input tensor
            targets: Target tensor
            method: Explanation method to use
            **kwargs: Additional arguments for the explanation method
            
        Returns:
            torch.Tensor: Explanation tensor
        """
        if method == 'input_x_gradient':
            return self.explain_input_x_gradient(inputs, targets)
        elif method == 'gradient_shap':
            return self.explain_gradient_shap(inputs, targets, **kwargs)
        elif method == 'integrated_gradients':
            return self.explain_integrated_gradients(inputs, targets, **kwargs)
        elif method == 'occlusion':
            return self.explain_occlusion(inputs, targets, **kwargs)
        elif method == 'guided_backprop':
            return self.explain_guided_backprop(inputs, targets)
        elif method == 'saliency':
            return self.explain_saliency(inputs, targets)
        else:
            raise ValueError(f"Unknown explanation method: {method}")
