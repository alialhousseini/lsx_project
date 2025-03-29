"""
Experiment script for ChestMNIST dataset.

This script runs the LSX experiment on the ChestMNIST dataset.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import argparse
import importlib.util

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import get_chestmnist_dataloaders
from models.lsx import LSX
from modules.fit import Fit
from modules.explain import Explain
from modules.reflect import Reflect
from modules.revise import Revise
from utils.metrics import classification_accuracy, ridge_regression_accuracy, inter_vs_intraclass_explanation_similarity, explanation_faithfulness
from utils.visualization import visualize_explanation_grid, visualize_class_explanations, plot_training_history, plot_metrics_comparison

def run_experiment(config_path):
    """
    Run the ChestMNIST experiment with the given configuration.
    
    Args:
        config_path (str): Path to the configuration file
    """
    # Load configuration
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    
    # Extract configurations
    dataset_config = config.dataset_config
    model_config = config.model_config
    training_config = config.training_config
    lsx_config = config.lsx_config
    experiment_config = config.experiment_config
    
    # Set random seed for reproducibility
    torch.manual_seed(training_config['random_seed'])
    np.random.seed(training_config['random_seed'])
    
    # Create save directory
    save_dir = experiment_config['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device = torch.device(training_config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_loader, critic_loader, test_loader = get_chestmnist_dataloaders(
        batch_size=dataset_config['batch_size'],
        num_samples=dataset_config['num_samples'],
        critic_samples=dataset_config['critic_samples']
    )
    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Critic set size: {len(critic_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    
    # Initialize model
    print("Initializing model...")
    model = LSX(
        input_channels=model_config['input_channels'],
        num_classes=model_config['num_classes']
    )
    model.to(device)
    
    # Initialize modules
    fit_module = Fit(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(model.learner.parameters(), lr=training_config['learning_rate_fit']),
        device=device
    )
    
    explain_module = Explain(
        model=model,
        device=device
    )
    
    reflect_module = Reflect(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(model.critic.parameters(), lr=training_config['learning_rate_reflect']),
        device=device
    )
    
    revise_module = Revise(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(model.learner.parameters(), lr=training_config['learning_rate_revise']),
        lambda_expl=training_config['lambda_expl'],
        device=device
    )
    
    # Training metrics
    metrics = {
        'vanilla': {},
        'lsx': {}
    }
    
    # Step 1: Train vanilla model (Fit)
    print("\n=== Step 1: Training vanilla model (Fit) ===")
    fit_history = fit_module.train(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=training_config['num_epochs_fit'],
        early_stopping=training_config['early_stopping'],
        patience=training_config['patience']
    )
    
    # Plot training history
    plot_training_history(
        history=fit_history,
        title="Vanilla Model Training",
        save_path=os.path.join(save_dir, "vanilla_training_history.png")
    )
    
    # Evaluate vanilla model
    vanilla_acc = classification_accuracy(model.learner, test_loader, device)
    print(f"Vanilla model test accuracy: {vanilla_acc:.2f}%")
    metrics['vanilla']['accuracy'] = vanilla_acc
    
    # Generate explanations for vanilla model
    print("\n=== Generating explanations for vanilla model ===")
    vanilla_explanations, vanilla_inputs, vanilla_targets = explain_module.generate_explanations(critic_loader)
    
    # Visualize vanilla explanations
    visualize_explanation_grid(
        images=vanilla_inputs,
        explanations=vanilla_explanations,
        num_samples=10,
        title="Vanilla Model Explanations",
        save_path=os.path.join(save_dir, "vanilla_explanations.png")
    )
    
    visualize_class_explanations(
        explanations=vanilla_explanations,
        targets=vanilla_targets,
        num_classes=model_config['num_classes'],
        samples_per_class=4,
        title="Vanilla Model Explanations by Class",
        save_path=os.path.join(save_dir, "vanilla_explanations_by_class.png")
    )
    
    # Evaluate explanation metrics for vanilla model
    vanilla_rr_acc = ridge_regression_accuracy(vanilla_explanations, vanilla_targets)
    vanilla_iies = inter_vs_intraclass_explanation_similarity(vanilla_explanations, vanilla_targets)
    vanilla_comp, vanilla_suff = explanation_faithfulness(
        model.learner, 
        test_loader, 
        lambda x, y: explain_module._generate_input_x_gradient(x, y),
        device
    )
    
    print(f"Vanilla model explanation metrics:")
    print(f"  Ridge Regression Accuracy: {vanilla_rr_acc:.2f}%")
    print(f"  IIES: {vanilla_iies:.4f}")
    print(f"  Comprehensiveness: {vanilla_comp:.2f}")
    print(f"  Sufficiency: {vanilla_suff:.2f}")
    
    metrics['vanilla']['ridge_regression_accuracy'] = vanilla_rr_acc
    metrics['vanilla']['iies'] = vanilla_iies
    metrics['vanilla']['comprehensiveness'] = vanilla_comp
    metrics['vanilla']['sufficiency'] = vanilla_suff
    
    # Save vanilla model
    if experiment_config['save_model']:
        torch.save(model.learner.state_dict(), os.path.join(save_dir, "vanilla_model.pth"))
    
    # Step 2: LSX Training (Explain, Reflect, Revise)
    print("\n=== Step 2: LSX Training (Explain, Reflect, Revise) ===")
    
    lsx_history = {
        'iterations': [],
        'accuracy': []
    }
    
    for iteration in range(lsx_config['iterations']):
        print(f"\n--- LSX Iteration {iteration+1}/{lsx_config['iterations']} ---")
        
        # Explain
        print("Explain: Generating explanations...")
        explanations, inputs, targets = explain_module.generate_explanations(critic_loader)
        
        # Reflect
        print("Reflect: Training critic...")
        feedback = reflect_module.get_feedback(explanations, targets)
        print(f"Critic feedback (loss): {feedback:.4f}")
        
        # Revise
        print("Revise: Updating learner...")
        revise_history = revise_module.revise_learner(
            train_loader=train_loader,
            explanations=explanations,
            targets=targets,
            num_epochs=training_config['num_epochs_revise']
        )
        
        # Evaluate after revision
        lsx_acc = classification_accuracy(model.learner, test_loader, device)
        print(f"LSX model test accuracy after iteration {iteration+1}: {lsx_acc:.2f}%")
        
        # Update history
        lsx_history['iterations'].append(iteration+1)
        lsx_history['accuracy'].append(lsx_acc)
        
        # Plot revision history
        plot_training_history(
            history=revise_history,
            title=f"LSX Revision History (Iteration {iteration+1})",
            save_path=os.path.join(save_dir, f"lsx_revision_history_iter{iteration+1}.png")
        )
    
    # Final finetuning
    print("\n=== Final Finetuning ===")
    final_explanations, _, _ = explain_module.generate_explanations(train_loader)
    finetune_history = revise_module.final_finetuning(
        train_loader=train_loader,
        explanations=final_explanations,
        num_epochs=training_config['num_epochs_finetune']
    )
    
    # Plot finetuning history
    plot_training_history(
        history=finetune_history,
        title="LSX Final Finetuning",
        save_path=os.path.join(save_dir, "lsx_finetune_history.png")
    )
    
    # Evaluate final LSX model
    lsx_acc = classification_accuracy(model.learner, test_loader, device)
    print(f"Final LSX model test accuracy: {lsx_acc:.2f}%")
    metrics['lsx']['accuracy'] = lsx_acc
    
    # Generate explanations for LSX model
    print("\n=== Generating explanations for LSX model ===")
    lsx_explanations, lsx_inputs, lsx_targets = explain_module.generate_explanations(critic_loader)
    
    # Visualize LSX explanations
    visualize_explanation_grid(
        images=lsx_inputs,
        explanations=lsx_explanations,
        num_samples=10,
        title="LSX Model Explanations",
        save_path=os.path.join(save_dir, "lsx_explanations.png")
    )
    
    visualize_class_explanations(
        explanations=lsx_explanations,
        targets=lsx_targets,
        num_classes=model_config['num_classes'],
        samples_per_class=4,
        title="LSX Model Explanations by Class",
        save_path=os.path.join(save_dir, "lsx_explanations_by_class.png")
    )
    
    # Evaluate explanation metrics for LSX model
    lsx_rr_acc = ridge_regression_accuracy(lsx_explanations, lsx_targets)
    lsx_iies = inter_vs_intraclass_explanation_similarity(lsx_explanations, lsx_targets)
    lsx_comp, lsx_suff = explanation_faithfulness(
        model.learner, 
        test_loader, 
        lambda x, y: explain_module._generate_input_x_gradient(x, y),
        device
    )
    
    print(f"LSX model explanation metrics:")
    print(f"  Ridge Regression Accuracy: {lsx_rr_acc:.2f}%")
    print(f"  IIES: {lsx_iies:.4f}")
    print(f"  Comprehensiveness: {lsx_comp:.2f}")
    print(f"  Sufficiency: {lsx_suff:.2f}")
    
    metrics['lsx']['ridge_regression_accuracy'] = lsx_rr_acc
    metrics['lsx']['iies'] = lsx_iies
    metrics['lsx']['comprehensiveness'] = lsx_comp
    metrics['lsx']['sufficiency'] = lsx_suff
    
    # Save LSX model
    if experiment_config['save_model']:
        model.save_models(os.path.join(save_dir, "lsx_model.pth"))
    
    # Plot metrics comparison
    plot_metrics_comparison(
        metrics_dict=metrics,
        title="Model Comparison",
        save_path=os.path.join(save_dir, "metrics_comparison.png")
    )
    
    # Save metrics
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write("=== Vanilla Model ===\n")
        for metric, value in metrics['vanilla'].items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\n=== LSX Model ===\n")
        for metric, value in metrics['lsx'].items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print("\n=== Experiment completed successfully ===")
    print(f"Results saved to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ChestMNIST experiment")
    parser.add_argument("--config", type=str, default="../configs/chestmnist_config.py", help="Path to configuration file")
    args = parser.parse_args()
    
    run_experiment(args.config)
