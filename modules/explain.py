"""
Explain module implementation for the LSX project.

This module implements the Explain phase of the LSX methodology, where the learner
provides explanations for its predictions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import InputXGradient
from tqdm import tqdm


class Explain:
    """
    Explain module for generating explanations for the learner's predictions.

    This module implements the second phase of the LSX methodology, where
    the learner provides explanations to its predictions using an explanation
    method (InputXGradient in this implementation).
    """

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the Explain module.

        Args:
            model: The LSX model containing the learner
            device: Device to use for generating explanations ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device

        # Initialize the explanation method (InputXGradient)
        self.input_x_gradient = InputXGradient(self.model.learner)

    def generate_explanations(self, dataloader):
        """
        Generate explanations for a batch of inputs.

        Args:
            dataloader: DataLoader containing the inputs to explain

        Returns:
            tuple: (explanations, inputs, targets)
        """
        self.model.learner.eval()
        all_explanations = []
        all_inputs = []
        all_targets = []

        for inputs, targets in tqdm(dataloader, desc="Generating explanations"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            all_inputs.append(inputs.cpu())
            all_targets.append(targets.cpu())

            # Generate explanations using InputXGradient
            explanations = self._generate_input_x_gradient(inputs, targets)
            all_explanations.append(explanations.cpu())

        # Concatenate all batches
        all_explanations = torch.cat(all_explanations, dim=0)
        all_inputs = torch.cat(all_inputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        return all_explanations, all_inputs, all_targets

    def _generate_input_x_gradient(self, inputs, targets):
        """
        Generate explanations using the InputXGradient method.

        Following the notation in the paper, for an input sample x_i and the output
        of model f given the corresponding ground truth label y_i, InputXGradient is
        defined as:

        e_i = x_i * (∂f_yi(x_i) / ∂x_i)

        Args:
            inputs: Input tensor
            targets: Target tensor

        Returns:
            torch.Tensor: Explanation tensor
        """
        # Ensure inputs require gradients
        inputs.requires_grad = True

        # Forward pass
        outputs = self.model.learner(inputs)

        # Get gradients with respect to inputs
        explanations = []
        for i in range(inputs.size(0)):
            # Get the score for the target class
            score = outputs[i, targets[i]]

            # Compute gradients
            self.model.learner.zero_grad()
            score.backward(retain_graph=(i < inputs.size(0) - 1))

            # Compute input * gradient
            explanation = inputs[i].detach().clone() * \
                inputs.grad[i].detach().clone()
            explanations.append(explanation.unsqueeze(0))

            # Reset gradients for next iteration
            inputs.grad.zero_()

        # Stack all explanations
        explanations = torch.cat(explanations, dim=0)

        return explanations

    def explain_single_input(self, input_tensor, target=None):
        """
        Generate explanation for a single input.

        Args:
            input_tensor: Input tensor to explain
            target: Target class (if None, use the predicted class)

        Returns:
            torch.Tensor: Explanation tensor
        """
        self.model.learner.eval()
        input_tensor = input_tensor.to(self.device).unsqueeze(0)

        # If target is not provided, use the predicted class
        if target is None:
            with torch.no_grad():
                output = self.model.learner(input_tensor)
                target = output.argmax(dim=1)
        else:
            target = torch.tensor([target]).to(self.device)

        # Generate explanation
        explanation = self._generate_input_x_gradient(input_tensor, target)

        return explanation.squeeze(0)
