"""
CNN model architecture for the LSX project.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    Convolutional Neural Network model for image classification.

    This model follows the architecture described in the LSX paper:
    - Two convolutional layers
    - ReLU activation layers
    - One average pooling layer
    - Two linear layers
    """

    def __init__(self, input_channels=1, num_classes=10):
        """
        Initialize the CNN model.

        Args:
            input_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
            num_classes (int): Number of output classes
        """
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            input_channels, 32, kernel_size=3, stride=1)  # padding=0
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)  # padding=0

        # # Pooling layer
        # self.pool = nn.AvgPool2d(kernel_size=2)

        # Calculate the size of the flattened features after convolution and pooling
        # For MNIST (28x28): After conv1 -> 28x28, after pool -> 14x14, after conv2 -> 14x14, after pool -> 7x7
        # So flattened size is 64 * 7 * 7 = 3136
        # self.fc_input_size = 64 * 7 * 7

        # Linear layers
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_classes]
        """
        # First convolutional block
        x = self.conv1(x)
        x = F.relu(x)
        # x = self.pool(x)

        # Second convolutional block
        x = self.conv2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)  # Average pooling

        # Flatten the output for the fully connected layers
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def get_activations(self, x, layer_name=None):
        """
        Get activations from a specific layer.

        Args:
            x (torch.Tensor): Input tensor
            layer_name (str, optional): Name of the layer to get activations from

        Returns:
            torch.Tensor: Activations from the specified layer
        """
        activations = {}

        # First convolutional block
        x = self.conv1(x)
        activations['conv1'] = x

        x = F.relu(x)
        activations['relu1'] = x

        x = self.pool(x)
        activations['pool1'] = x

        # Second convolutional block
        x = self.conv2(x)
        activations['conv2'] = x

        x = F.relu(x)
        activations['relu2'] = x

        x = self.pool(x)
        activations['pool2'] = x

        # Flatten the output for the fully connected layers
        x = x.view(-1, self.fc_input_size)
        activations['flatten'] = x

        # Fully connected layers
        x = self.fc1(x)
        activations['fc1'] = x

        x = F.relu(x)
        activations['relu3'] = x

        x = self.fc2(x)
        activations['fc2'] = x

        if layer_name is not None:
            return activations[layer_name]

        return activations
