"""
Model architecture enhancements for the LSX project.

This module provides enhanced model architectures that can be used as alternatives
to the basic CNN model in the core implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EnhancedCNN(nn.Module):
    """
    Enhanced CNN model with more layers and batch normalization.
    
    This model provides a more sophisticated architecture than the basic CNN
    used in the core implementation, potentially leading to better performance.
    """
    def __init__(self, input_channels=1, num_classes=10):
        """
        Initialize the enhanced CNN model.
        
        Args:
            input_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
            num_classes (int): Number of output classes
        """
        super(EnhancedCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size of the flattened features after convolution and pooling
        # For MNIST (28x28): After conv1 -> 28x28, after pool -> 14x14, 
        # after conv2 -> 14x14, after pool -> 7x7, after conv3 -> 7x7, after pool -> 3x3
        # So flattened size is 128 * 3 * 3 = 1152
        self.fc_input_size = 128 * 3 * 3
        
        # Linear layers
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
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
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Third convolutional block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x
    
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
        
        x = self.bn1(x)
        activations['bn1'] = x
        
        x = F.relu(x)
        activations['relu1'] = x
        
        x = self.pool(x)
        activations['pool1'] = x
        
        # Second convolutional block
        x = self.conv2(x)
        activations['conv2'] = x
        
        x = self.bn2(x)
        activations['bn2'] = x
        
        x = F.relu(x)
        activations['relu2'] = x
        
        x = self.pool(x)
        activations['pool2'] = x
        
        # Third convolutional block
        x = self.conv3(x)
        activations['conv3'] = x
        
        x = self.bn3(x)
        activations['bn3'] = x
        
        x = F.relu(x)
        activations['relu3'] = x
        
        x = self.pool(x)
        activations['pool3'] = x
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, self.fc_input_size)
        activations['flatten'] = x
        
        # Fully connected layers
        x = self.fc1(x)
        activations['fc1'] = x
        
        x = F.relu(x)
        activations['relu4'] = x
        
        x = self.dropout1(x)
        activations['dropout1'] = x
        
        x = self.fc2(x)
        activations['fc2'] = x
        
        if layer_name is not None:
            return activations[layer_name]
        
        return activations

class PretrainedResNet(nn.Module):
    """
    Pretrained ResNet model adapted for the LSX project.
    
    This model uses a pretrained ResNet18 as a feature extractor and adds a
    custom classifier head for the specific task.
    """
    def __init__(self, input_channels=3, num_classes=10, pretrained=True):
        """
        Initialize the pretrained ResNet model.
        
        Args:
            input_channels (int): Number of input channels (3 for RGB)
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
        """
        super(PretrainedResNet, self).__init__()
        
        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Modify first layer if input channels != 3
        if input_channels != 3:
            self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace final fully connected layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_classes]
        """
        return self.resnet(x)
    
    def get_activations(self, x, layer_name=None):
        """
        Get activations from a specific layer.
        
        Args:
            x (torch.Tensor): Input tensor
            layer_name (str, optional): Name of the layer to get activations from
            
        Returns:
            torch.Tensor: Activations from the specified layer
        """
        # This is a simplified implementation that returns the features before the final classifier
        activations = {}
        
        # Get features from ResNet
        x = self.resnet.conv1(x)
        activations['conv1'] = x
        
        x = self.resnet.bn1(x)
        activations['bn1'] = x
        
        x = self.resnet.relu(x)
        activations['relu'] = x
        
        x = self.resnet.maxpool(x)
        activations['maxpool'] = x
        
        x = self.resnet.layer1(x)
        activations['layer1'] = x
        
        x = self.resnet.layer2(x)
        activations['layer2'] = x
        
        x = self.resnet.layer3(x)
        activations['layer3'] = x
        
        x = self.resnet.layer4(x)
        activations['layer4'] = x
        
        x = self.resnet.avgpool(x)
        activations['avgpool'] = x
        
        x = torch.flatten(x, 1)
        activations['flatten'] = x
        
        # Custom classifier
        fc_layers = list(self.resnet.fc.children())
        
        x = fc_layers[0](x)  # Linear
        activations['fc1'] = x
        
        x = fc_layers[1](x)  # ReLU
        activations['fc_relu'] = x
        
        x = fc_layers[2](x)  # Dropout
        activations['fc_dropout'] = x
        
        x = fc_layers[3](x)  # Linear
        activations['fc2'] = x
        
        if layer_name is not None:
            return activations[layer_name]
        
        return activations
