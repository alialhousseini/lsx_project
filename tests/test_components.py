"""
Unit tests for the LSX project.

This module contains unit tests for the individual components of the LSX project.
"""
import os
import sys
import unittest
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn import CNN
from models.lsx import LSX
from models.enhanced_models import EnhancedCNN, PretrainedResNet
from modules.fit import Fit
from modules.explain import Explain
from modules.reflect import Reflect
from modules.revise import Revise
from utils.metrics import classification_accuracy, ridge_regression_accuracy
from utils.explanation_methods import ExplanationMethods

class TestCNNModel(unittest.TestCase):
    """Test cases for the CNN model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNN(input_channels=1, num_classes=10).to(self.device)
        self.batch_size = 4
        self.input_tensor = torch.randn(self.batch_size, 1, 28, 28).to(self.device)
        
    def test_forward_pass(self):
        """Test forward pass through the CNN model."""
        output = self.model(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, 10))
        
    def test_get_activations(self):
        """Test getting activations from the CNN model."""
        activations = self.model.get_activations(self.input_tensor)
        self.assertIn('conv1', activations)
        self.assertIn('conv2', activations)
        self.assertIn('fc1', activations)
        self.assertIn('fc2', activations)
        
        # Test getting a specific layer
        conv1_activations = self.model.get_activations(self.input_tensor, 'conv1')
        self.assertEqual(conv1_activations.shape, (self.batch_size, 32, 28, 28))

class TestLSXModel(unittest.TestCase):
    """Test cases for the LSX model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSX(input_channels=1, num_classes=10).to(self.device)
        self.batch_size = 4
        self.input_tensor = torch.randn(self.batch_size, 1, 28, 28).to(self.device)
        self.target_tensor = torch.randint(0, 10, (self.batch_size,)).to(self.device)
        
    def test_forward_pass(self):
        """Test forward pass through the LSX model."""
        output = self.model(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, 10))
        
    def test_save_load_models(self):
        """Test saving and loading LSX models."""
        # Save model
        save_path = 'test_model.pth'
        self.model.save_models(save_path)
        
        # Create a new model
        new_model = LSX(input_channels=1, num_classes=10).to(self.device)
        
        # Load saved model
        new_model.load_models(save_path)
        
        # Check that parameters are the same
        for p1, p2 in zip(self.model.learner.parameters(), new_model.learner.parameters()):
            self.assertTrue(torch.allclose(p1, p2))
            
        for p1, p2 in zip(self.model.critic.parameters(), new_model.critic.parameters()):
            self.assertTrue(torch.allclose(p1, p2))
            
        # Clean up
        os.remove(save_path)

class TestEnhancedModels(unittest.TestCase):
    """Test cases for the enhanced model architectures."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4
        self.input_tensor_gray = torch.randn(self.batch_size, 1, 28, 28).to(self.device)
        self.input_tensor_rgb = torch.randn(self.batch_size, 3, 224, 224).to(self.device)
        
    def test_enhanced_cnn(self):
        """Test the EnhancedCNN model."""
        model = EnhancedCNN(input_channels=1, num_classes=10).to(self.device)
        output = model(self.input_tensor_gray)
        self.assertEqual(output.shape, (self.batch_size, 10))
        
        # Test activations
        activations = model.get_activations(self.input_tensor_gray)
        self.assertIn('conv1', activations)
        self.assertIn('conv2', activations)
        self.assertIn('conv3', activations)
        self.assertIn('fc1', activations)
        self.assertIn('fc2', activations)
        
    def test_pretrained_resnet(self):
        """Test the PretrainedResNet model."""
        model = PretrainedResNet(input_channels=3, num_classes=10, pretrained=False).to(self.device)
        output = model(self.input_tensor_rgb)
        self.assertEqual(output.shape, (self.batch_size, 10))
        
        # Test activations
        activations = model.get_activations(self.input_tensor_rgb)
        self.assertIn('conv1', activations)
        self.assertIn('layer1', activations)
        self.assertIn('layer2', activations)
        self.assertIn('layer3', activations)
        self.assertIn('layer4', activations)
        self.assertIn('fc1', activations)
        self.assertIn('fc2', activations)

class TestExplanationMethods(unittest.TestCase):
    """Test cases for the explanation methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNN(input_channels=1, num_classes=10).to(self.device)
        self.explanation_methods = ExplanationMethods(self.model, self.device)
        self.batch_size = 4
        self.input_tensor = torch.randn(self.batch_size, 1, 28, 28).to(self.device)
        self.target_tensor = torch.randint(0, 10, (self.batch_size,)).to(self.device)
        
    def test_input_x_gradient(self):
        """Test the InputXGradient explanation method."""
        explanations = self.explanation_methods.explain_input_x_gradient(self.input_tensor, self.target_tensor)
        self.assertEqual(explanations.shape, self.input_tensor.shape)
        
    def test_saliency(self):
        """Test the Saliency explanation method."""
        explanations = self.explanation_methods.explain_saliency(self.input_tensor, self.target_tensor)
        self.assertEqual(explanations.shape, self.input_tensor.shape)
        
    def test_get_explanation(self):
        """Test the get_explanation method with different methods."""
        methods = ['input_x_gradient', 'saliency']
        for method in methods:
            explanations = self.explanation_methods.get_explanation(
                self.input_tensor, self.target_tensor, method=method
            )
            self.assertEqual(explanations.shape, self.input_tensor.shape)

class TestMetrics(unittest.TestCase):
    """Test cases for the evaluation metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 10
        self.num_classes = 5
        self.explanations = torch.randn(self.batch_size, 1, 28, 28)
        self.targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
    def test_ridge_regression_accuracy(self):
        """Test the ridge regression accuracy metric."""
        accuracy = ridge_regression_accuracy(self.explanations, self.targets, test_size=0.5)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 100.0)

if __name__ == '__main__':
    unittest.main()
