"""
Hyperparameter optimization utilities for the LSX project.

This module provides tools for optimizing hyperparameters of the LSX models
to improve performance and explanation quality.
"""
import torch
import numpy as np
import optuna
from optuna.trial import Trial
import os
import sys
import copy
from functools import partial

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lsx import LSX
from modules.fit import Fit
from modules.explain import Explain
from modules.reflect import Reflect
from modules.revise import Revise
from utils.metrics import classification_accuracy, ridge_regression_accuracy

class LSXHyperparameterOptimizer:
    """
    Hyperparameter optimizer for LSX models.
    
    This class uses Optuna to find optimal hyperparameters for the LSX methodology,
    focusing on both classification performance and explanation quality.
    """
    def __init__(self, train_loader, critic_loader, test_loader, device='cuda'):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            train_loader: DataLoader for training data
            critic_loader: DataLoader for critic data
            test_loader: DataLoader for test data
            device: Device to use for training
        """
        self.train_loader = train_loader
        self.critic_loader = critic_loader
        self.test_loader = test_loader
        self.device = device
        
    def objective(self, trial: Trial):
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            float: Objective value to minimize/maximize
        """
        # Sample hyperparameters
        input_channels = trial.suggest_categorical('input_channels', [1, 3])
        num_classes = trial.suggest_categorical('num_classes', [2, 10])
        
        learning_rate_fit = trial.suggest_float('learning_rate_fit', 1e-4, 1e-2, log=True)
        learning_rate_reflect = trial.suggest_float('learning_rate_reflect', 1e-4, 1e-2, log=True)
        learning_rate_revise = trial.suggest_float('learning_rate_revise', 1e-4, 1e-2, log=True)
        
        lambda_expl = trial.suggest_float('lambda_expl', 10, 1000, log=True)
        num_epochs_fit = trial.suggest_int('num_epochs_fit', 5, 20)
        num_epochs_revise = trial.suggest_int('num_epochs_revise', 1, 10)
        
        # Initialize model
        model = LSX(
            input_channels=input_channels,
            num_classes=num_classes
        )
        model.to(self.device)
        
        # Initialize modules
        fit_module = Fit(
            model=model,
            optimizer=torch.optim.Adam(model.learner.parameters(), lr=learning_rate_fit),
            device=self.device
        )
        
        explain_module = Explain(
            model=model,
            device=self.device
        )
        
        reflect_module = Reflect(
            model=model,
            optimizer=torch.optim.Adam(model.critic.parameters(), lr=learning_rate_reflect),
            device=self.device
        )
        
        revise_module = Revise(
            model=model,
            optimizer=torch.optim.Adam(model.learner.parameters(), lr=learning_rate_revise),
            lambda_expl=lambda_expl,
            device=self.device
        )
        
        # Train vanilla model
        fit_module.train(
            train_loader=self.train_loader,
            val_loader=self.test_loader,
            num_epochs=num_epochs_fit,
            early_stopping=True,
            patience=3
        )
        
        # Evaluate vanilla model
        vanilla_acc = classification_accuracy(model.learner, self.test_loader, self.device)
        
        # Generate explanations
        explanations, _, targets = explain_module.generate_explanations(self.critic_loader)
        
        # Train critic
        reflect_module.get_feedback(explanations, targets)
        
        # Revise learner
        revise_module.revise_learner(
            train_loader=self.train_loader,
            explanations=explanations,
            targets=targets,
            num_epochs=num_epochs_revise
        )
        
        # Evaluate LSX model
        lsx_acc = classification_accuracy(model.learner, self.test_loader, self.device)
        
        # Generate explanations for LSX model
        lsx_explanations, _, lsx_targets = explain_module.generate_explanations(self.critic_loader)
        
        # Evaluate explanation quality
        lsx_rr_acc = ridge_regression_accuracy(lsx_explanations, lsx_targets)
        
        # Compute objective value (weighted combination of accuracy and explanation quality)
        objective_value = 0.7 * lsx_acc + 0.3 * lsx_rr_acc
        
        return objective_value
    
    def optimize(self, n_trials=100, study_name='lsx_optimization', direction='maximize'):
        """
        Run hyperparameter optimization.
        
        Args:
            n_trials: Number of trials to run
            study_name: Name of the study
            direction: Direction of optimization ('maximize' or 'minimize')
            
        Returns:
            dict: Best hyperparameters
        """
        study = optuna.create_study(study_name=study_name, direction=direction)
        study.optimize(self.objective, n_trials=n_trials)
        
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value: {study.best_value}")
        print("Best hyperparameters:")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")
        
        return study.best_params
    
    def train_with_best_params(self, best_params, save_path=None):
        """
        Train a model with the best hyperparameters.
        
        Args:
            best_params: Best hyperparameters from optimization
            save_path: Path to save the trained model
            
        Returns:
            LSX: Trained LSX model
        """
        # Extract hyperparameters
        input_channels = best_params['input_channels']
        num_classes = best_params['num_classes']
        
        learning_rate_fit = best_params['learning_rate_fit']
        learning_rate_reflect = best_params['learning_rate_reflect']
        learning_rate_revise = best_params['learning_rate_revise']
        
        lambda_expl = best_params['lambda_expl']
        num_epochs_fit = best_params['num_epochs_fit']
        num_epochs_revise = best_params['num_epochs_revise']
        
        # Initialize model
        model = LSX(
            input_channels=input_channels,
            num_classes=num_classes
        )
        model.to(self.device)
        
        # Initialize modules
        fit_module = Fit(
            model=model,
            optimizer=torch.optim.Adam(model.learner.parameters(), lr=learning_rate_fit),
            device=self.device
        )
        
        explain_module = Explain(
            model=model,
            device=self.device
        )
        
        reflect_module = Reflect(
            model=model,
            optimizer=torch.optim.Adam(model.critic.parameters(), lr=learning_rate_reflect),
            device=self.device
        )
        
        revise_module = Revise(
            model=model,
            optimizer=torch.optim.Adam(model.learner.parameters(), lr=learning_rate_revise),
            lambda_expl=lambda_expl,
            device=self.device
        )
        
        # Train vanilla model
        print("Training vanilla model...")
        fit_module.train(
            train_loader=self.train_loader,
            val_loader=self.test_loader,
            num_epochs=num_epochs_fit,
            early_stopping=True,
            patience=3
        )
        
        # Evaluate vanilla model
        vanilla_acc = classification_accuracy(model.learner, self.test_loader, self.device)
        print(f"Vanilla model accuracy: {vanilla_acc:.2f}%")
        
        # Generate explanations
        print("Generating explanations...")
        explanations, _, targets = explain_module.generate_explanations(self.critic_loader)
        
        # Train critic
        print("Training critic...")
        reflect_module.get_feedback(explanations, targets)
        
        # Revise learner
        print("Revising learner...")
        revise_module.revise_learner(
            train_loader=self.train_loader,
            explanations=explanations,
            targets=targets,
            num_epochs=num_epochs_revise
        )
        
        # Evaluate LSX model
        lsx_acc = classification_accuracy(model.learner, self.test_loader, self.device)
        print(f"LSX model accuracy: {lsx_acc:.2f}%")
        
        # Save model if path is provided
        if save_path:
            model.save_models(save_path)
            print(f"Model saved to {save_path}")
        
        return model
