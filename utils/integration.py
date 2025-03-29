"""
Integration and testing utilities for the LSX project.

This module provides tools for integrating and testing the various components
of the LSX implementation.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lsx import LSX
from models.enhanced_models import EnhancedCNN, PretrainedResNet
from modules.fit import Fit
from modules.explain import Explain
from modules.reflect import Reflect
from modules.revise import Revise
from utils.metrics import classification_accuracy, ridge_regression_accuracy, inter_vs_intraclass_explanation_similarity, explanation_faithfulness
from utils.visualization import visualize_explanation_grid, visualize_class_explanations, plot_training_history
from utils.explanation_methods import ExplanationMethods
from data.augmentation import get_mnist_augmentation, get_chestmnist_augmentation, get_cub10_augmentation, create_augmented_dataloader

class LSXPipeline:
    """
    End-to-end pipeline for training and evaluating LSX models.
    
    This class integrates all components of the LSX methodology into a single
    pipeline for easy experimentation and testing.
    """
    def __init__(self, config, train_loader, critic_loader, test_loader, device='cuda'):
        """
        Initialize the LSX pipeline.
        
        Args:
            config: Configuration dictionary
            train_loader: DataLoader for training data
            critic_loader: DataLoader for critic data
            test_loader: DataLoader for test data
            device: Device to use for training
        """
        self.config = config
        self.train_loader = train_loader
        self.critic_loader = critic_loader
        self.test_loader = test_loader
        self.device = device
        
        # Extract configurations
        self.dataset_config = config.get('dataset_config', {})
        self.model_config = config.get('model_config', {})
        self.training_config = config.get('training_config', {})
        self.lsx_config = config.get('lsx_config', {})
        self.experiment_config = config.get('experiment_config', {})
        
        # Create save directory
        self.save_dir = self.experiment_config.get('save_dir', './results')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize model
        self._init_model()
        
        # Initialize modules
        self._init_modules()
        
        # Initialize metrics
        self.metrics = {
            'vanilla': {},
            'lsx': {}
        }
        
    def _init_model(self):
        """Initialize the LSX model based on configuration."""
        model_type = self.model_config.get('model_type', 'cnn')
        input_channels = self.model_config.get('input_channels', 1)
        num_classes = self.model_config.get('num_classes', 10)
        
        if model_type == 'cnn':
            self.model = LSX(
                input_channels=input_channels,
                num_classes=num_classes
            )
        elif model_type == 'enhanced_cnn':
            learner = EnhancedCNN(
                input_channels=input_channels,
                num_classes=num_classes
            )
            critic = EnhancedCNN(
                input_channels=input_channels,
                num_classes=num_classes
            )
            self.model = LSX(
                input_channels=input_channels,
                num_classes=num_classes
            )
            self.model.learner = learner
            self.model.critic = critic
        elif model_type == 'resnet':
            if input_channels == 1:
                # For grayscale images, we need to adapt ResNet
                learner = PretrainedResNet(
                    input_channels=input_channels,
                    num_classes=num_classes
                )
                critic = PretrainedResNet(
                    input_channels=input_channels,
                    num_classes=num_classes
                )
                self.model = LSX(
                    input_channels=input_channels,
                    num_classes=num_classes
                )
                self.model.learner = learner
                self.model.critic = critic
            else:
                learner = PretrainedResNet(
                    input_channels=input_channels,
                    num_classes=num_classes
                )
                critic = PretrainedResNet(
                    input_channels=input_channels,
                    num_classes=num_classes
                )
                self.model = LSX(
                    input_channels=input_channels,
                    num_classes=num_classes
                )
                self.model.learner = learner
                self.model.critic = critic
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.to(self.device)
        
    def _init_modules(self):
        """Initialize the LSX modules based on configuration."""
        # Extract training parameters
        lr_fit = self.training_config.get('learning_rate_fit', 0.001)
        lr_reflect = self.training_config.get('learning_rate_reflect', 0.001)
        lr_revise = self.training_config.get('learning_rate_revise', 0.0005)
        lambda_expl = self.training_config.get('lambda_expl', 100)
        
        # Initialize modules
        self.fit_module = Fit(
            model=self.model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optim.Adam(self.model.learner.parameters(), lr=lr_fit),
            device=self.device
        )
        
        self.explain_module = Explain(
            model=self.model,
            device=self.device
        )
        
        self.reflect_module = Reflect(
            model=self.model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optim.Adam(self.model.critic.parameters(), lr=lr_reflect),
            device=self.device
        )
        
        self.revise_module = Revise(
            model=self.model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optim.Adam(self.model.learner.parameters(), lr=lr_revise),
            lambda_expl=lambda_expl,
            device=self.device
        )
        
        # Initialize explanation methods
        self.explanation_methods = ExplanationMethods(
            model=self.model.learner,
            device=self.device
        )
        
    def train_vanilla_model(self):
        """Train the vanilla model (Fit phase)."""
        print("\n=== Step 1: Training vanilla model (Fit) ===")
        
        # Extract training parameters
        num_epochs = self.training_config.get('num_epochs_fit', 10)
        early_stopping = self.training_config.get('early_stopping', True)
        patience = self.training_config.get('patience', 3)
        
        # Train vanilla model
        fit_history = self.fit_module.train(
            train_loader=self.train_loader,
            val_loader=self.test_loader,
            num_epochs=num_epochs,
            early_stopping=early_stopping,
            patience=patience
        )
        
        # Plot training history
        plot_training_history(
            history=fit_history,
            title="Vanilla Model Training",
            save_path=os.path.join(self.save_dir, "vanilla_training_history.png")
        )
        
        # Evaluate vanilla model
        vanilla_acc = classification_accuracy(self.model.learner, self.test_loader, self.device)
        print(f"Vanilla model test accuracy: {vanilla_acc:.2f}%")
        self.metrics['vanilla']['accuracy'] = vanilla_acc
        
        # Generate explanations for vanilla model
        print("\n=== Generating explanations for vanilla model ===")
        vanilla_explanations, vanilla_inputs, vanilla_targets = self.explain_module.generate_explanations(self.critic_loader)
        
        # Visualize vanilla explanations
        visualize_explanation_grid(
            images=vanilla_inputs,
            explanations=vanilla_explanations,
            num_samples=10,
            title="Vanilla Model Explanations",
            save_path=os.path.join(self.save_dir, "vanilla_explanations.png")
        )
        
        visualize_class_explanations(
            explanations=vanilla_explanations,
            targets=vanilla_targets,
            num_classes=self.model_config.get('num_classes', 10),
            samples_per_class=4,
            title="Vanilla Model Explanations by Class",
            save_path=os.path.join(self.save_dir, "vanilla_explanations_by_class.png")
        )
        
        # Evaluate explanation metrics for vanilla model
        vanilla_rr_acc = ridge_regression_accuracy(vanilla_explanations, vanilla_targets)
        vanilla_iies = inter_vs_intraclass_explanation_similarity(vanilla_explanations, vanilla_targets)
        vanilla_comp, vanilla_suff = explanation_faithfulness(
            self.model.learner, 
            self.test_loader, 
            lambda x, y: self.explain_module._generate_input_x_gradient(x, y),
            self.device
        )
        
        print(f"Vanilla model explanation metrics:")
        print(f"  Ridge Regression Accuracy: {vanilla_rr_acc:.2f}%")
        print(f"  IIES: {vanilla_iies:.4f}")
        print(f"  Comprehensiveness: {vanilla_comp:.2f}")
        print(f"  Sufficiency: {vanilla_suff:.2f}")
        
        self.metrics['vanilla']['ridge_regression_accuracy'] = vanilla_rr_acc
        self.metrics['vanilla']['iies'] = vanilla_iies
        self.metrics['vanilla']['comprehensiveness'] = vanilla_comp
        self.metrics['vanilla']['sufficiency'] = vanilla_suff
        
        # Save vanilla model
        if self.experiment_config.get('save_model', True):
            torch.save(self.model.learner.state_dict(), os.path.join(self.save_dir, "vanilla_model.pth"))
        
        return vanilla_explanations, vanilla_targets
    
    def train_lsx_model(self, explanations=None, targets=None):
        """
        Train the LSX model (Explain, Reflect, Revise phases).
        
        Args:
            explanations: Pre-generated explanations (if None, will generate new ones)
            targets: Targets for the explanations (if None, will generate new ones)
        """
        print("\n=== Step 2: LSX Training (Explain, Reflect, Revise) ===")
        
        # Extract training parameters
        iterations = self.lsx_config.get('iterations', 3)
        num_epochs_revise = self.training_config.get('num_epochs_revise', 5)
        num_epochs_finetune = self.training_config.get('num_epochs_finetune', 3)
        
        lsx_history = {
            'iterations': [],
            'accuracy': []
        }
        
        for iteration in range(iterations):
            print(f"\n--- LSX Iteration {iteration+1}/{iterations} ---")
            
            # Explain
            print("Explain: Generating explanations...")
            if explanations is None or targets is None or iteration > 0:
                explanations, inputs, targets = self.explain_module.generate_explanations(self.critic_loader)
            
            # Reflect
            print("Reflect: Training critic...")
            feedback = self.reflect_module.get_feedback(explanations, targets)
            print(f"Critic feedback (loss): {feedback:.4f}")
            
            # Revise
            print("Revise: Updating learner...")
            revise_history = self.revise_module.revise_learner(
                train_loader=self.train_loader,
                explanations=explanations,
                targets=targets,
                num_epochs=num_epochs_revise
            )
            
            # Evaluate after revision
            lsx_acc = classification_accuracy(self.model.learner, self.test_loader, self.device)
            print(f"LSX model test accuracy after iteration {iteration+1}: {lsx_acc:.2f}%")
            
            # Update history
            lsx_history['iterations'].append(iteration+1)
            lsx_history['accuracy'].append(lsx_acc)
            
            # Plot revision history
            plot_training_history(
                history=revise_history,
                title=f"LSX Revision History (Iteration {iteration+1})",
                save_path=os.path.join(self.save_dir, f"lsx_revision_history_iter{iteration+1}.png")
            )
        
        # Final finetuning
        print("\n=== Final Finetuning ===")
        final_explanations, _, _ = self.explain_module.generate_explanations(self.train_loader)
        finetune_history = self.revise_module.final_finetuning(
            train_loader=self.train_loader,
            explanations=final_explanations,
            num_epochs=num_epochs_finetune
        )
        
        # Plot finetuning history
        plot_training_history(
            history=finetune_history,
            title="LSX Final Finetuning",
            save_path=os.path.join(self.save_dir, "lsx_finetune_history.png")
        )
        
        # Evaluate final LSX model
        lsx_acc = classification_accuracy(self.model.learner, self.test_loader, self.device)
        print(f"Final LSX model test accuracy: {lsx_acc:.2f}%")
        self.metrics['lsx']['accuracy'] = lsx_acc
        
        # Generate explanations for LSX model
        print("\n=== Generating explanations for LSX model ===")
        lsx_explanations, lsx_inputs, lsx_targets = self.explain_module.generate_explanations(self.critic_loader)
        
        # Visualize LSX explanations
        visualize_explanation_grid(
            images=lsx_inputs,
            explanations=lsx_explanations,
            num_samples=10,
            title="LSX Model Explanations",
            save_path=os.path.join(self.save_dir, "lsx_explanations.png")
        )
        
        visualize_class_explanations(
            explanations=lsx_explanations,
            targets=lsx_targets,
            num_classes=self.model_config.get('num_classes', 10),
            samples_per_class=4,
            title="LSX Model Explanations by Class",
            save_path=os.path.join(self.save_dir, "lsx_explanations_by_class.png")
        )
        
        # Evaluate explanation metrics for LSX model
        lsx_rr_acc = ridge_regression_accuracy(lsx_explanations, lsx_targets)
        lsx_iies = inter_vs_intraclass_explanation_similarity(lsx_explanations, lsx_targets)
        lsx_comp, lsx_suff = explanation_faithfulness(
            self.model.learner, 
            self.test_loader, 
            lambda x, y: self.explain_module._generate_input_x_gradient(x, y),
            self.device
        )
        
        print(f"LSX model explanation metrics:")
        print(f"  Ridge Regression Accuracy: {lsx_rr_acc:.2f}%")
        print(f"  IIES: {lsx_iies:.4f}")
        print(f"  Comprehensiveness: {lsx_comp:.2f}")
        print(f"  Sufficiency: {lsx_suff:.2f}")
        
        self.metrics['lsx']['ridge_regression_accuracy'] = lsx_rr_acc
        self.metrics['lsx']['iies'] = lsx_iies
        self.metrics['lsx']['comprehensiveness'] = lsx_comp
        self.metrics['lsx']['sufficiency'] = lsx_suff
        
        # Save LSX model
        if self.experiment_config.get('save_model', True):
            self.model.save_models(os.path.join(self.save_dir, "lsx_model.pth"))
        
        return lsx_explanations, lsx_targets
    
    def run_pipeline(self):
        """Run the complete LSX pipeline."""
        # Train vanilla model
        explanations, targets = self.train_vanilla_model()
        
        # Train LSX model
        self.train_lsx_model(explanations, targets)
        
        # Save metrics
        with open(os.path.join(self.save_dir, "metrics.txt"), "w") as f:
            f.write("=== Vanilla Model ===\n")
            for metric, value in self.metrics['vanilla'].
(Content truncated due to size limit. Use line ranges to read in chunks)