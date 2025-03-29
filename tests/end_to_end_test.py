"""
End-to-end test script for the LSX project.

This script runs an end-to-end test of the LSX workflow on a small subset of data
to verify that all components work together correctly.
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import get_mnist_dataloaders
from models.lsx import LSX
from modules.fit import Fit
from modules.explain import Explain
from modules.reflect import Reflect
from modules.revise import Revise
from utils.metrics import classification_accuracy, ridge_regression_accuracy
from utils.visualization import visualize_explanation_grid, plot_training_history

def run_end_to_end_test():
    """
    Run an end-to-end test of the LSX workflow on a small subset of MNIST data.
    """
    print("Running end-to-end test of LSX workflow...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test directory
    test_dir = "end_to_end_test_results"
    os.makedirs(test_dir, exist_ok=True)
    
    # Load a small subset of MNIST data
    train_loader, critic_loader, test_loader = get_mnist_dataloaders(
        batch_size=32,
        num_samples=1000,  # Small subset for quick testing
        critic_samples=200
    )
    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Critic set size: {len(critic_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    
    # Initialize model
    model = LSX(input_channels=1, num_classes=10).to(device)
    
    # Initialize modules
    fit_module = Fit(
        model=model,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.learner.parameters(), lr=0.001),
        device=device
    )
    
    explain_module = Explain(
        model=model,
        device=device
    )
    
    reflect_module = Reflect(
        model=model,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.critic.parameters(), lr=0.001),
        device=device
    )
    
    revise_module = Revise(
        model=model,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.learner.parameters(), lr=0.0005),
        lambda_expl=100,
        device=device
    )
    
    # Step 1: Train vanilla model (Fit)
    print("\n=== Step 1: Training vanilla model (Fit) ===")
    fit_history = fit_module.train(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=3,  # Few epochs for quick testing
        early_stopping=False
    )
    
    # Plot training history
    plot_training_history(
        history=fit_history,
        title="Vanilla Model Training",
        save_path=os.path.join(test_dir, "vanilla_training_history.png")
    )
    
    # Evaluate vanilla model
    vanilla_acc = classification_accuracy(model.learner, test_loader, device)
    print(f"Vanilla model test accuracy: {vanilla_acc:.2f}%")
    
    # Step 2: Generate explanations (Explain)
    print("\n=== Step 2: Generating explanations (Explain) ===")
    explanations, inputs, targets = explain_module.generate_explanations(critic_loader)
    
    # Visualize explanations
    visualize_explanation_grid(
        images=inputs,
        explanations=explanations,
        num_samples=10,
        title="Vanilla Model Explanations",
        save_path=os.path.join(test_dir, "vanilla_explanations.png")
    )
    
    # Step 3: Train critic (Reflect)
    print("\n=== Step 3: Training critic (Reflect) ===")
    feedback = reflect_module.get_feedback(explanations, targets)
    print(f"Critic feedback (loss): {feedback:.4f}")
    
    # Step 4: Update learner (Revise)
    print("\n=== Step 4: Updating learner (Revise) ===")
    revise_history = revise_module.revise_learner(
        train_loader=train_loader,
        explanations=explanations,
        targets=targets,
        num_epochs=2  # Few epochs for quick testing
    )
    
    # Plot revision history
    plot_training_history(
        history=revise_history,
        title="LSX Revision History",
        save_path=os.path.join(test_dir, "lsx_revision_history.png")
    )
    
    # Evaluate LSX model
    lsx_acc = classification_accuracy(model.learner, test_loader, device)
    print(f"LSX model test accuracy: {lsx_acc:.2f}%")
    
    # Generate explanations for LSX model
    print("\n=== Generating explanations for LSX model ===")
    lsx_explanations, lsx_inputs, lsx_targets = explain_module.generate_explanations(critic_loader)
    
    # Visualize LSX explanations
    visualize_explanation_grid(
        images=lsx_inputs,
        explanations=lsx_explanations,
        num_samples=10,
        title="LSX Model Explanations",
        save_path=os.path.join(test_dir, "lsx_explanations.png")
    )
    
    # Evaluate explanation quality
    vanilla_rr_acc = ridge_regression_accuracy(explanations, targets)
    lsx_rr_acc = ridge_regression_accuracy(lsx_explanations, lsx_targets)
    
    print(f"Vanilla model explanation quality (Ridge Regression Accuracy): {vanilla_rr_acc:.2f}%")
    print(f"LSX model explanation quality (Ridge Regression Accuracy): {lsx_rr_acc:.2f}%")
    
    # Save results
    with open(os.path.join(test_dir, "results.txt"), "w") as f:
        f.write("=== End-to-End Test Results ===\n\n")
        f.write(f"Vanilla model accuracy: {vanilla_acc:.2f}%\n")
        f.write(f"LSX model accuracy: {lsx_acc:.2f}%\n\n")
        f.write(f"Vanilla model explanation quality: {vanilla_rr_acc:.2f}%\n")
        f.write(f"LSX model explanation quality: {lsx_rr_acc:.2f}%\n")
    
    print("\n=== End-to-end test completed successfully ===")
    print(f"Results saved to {test_dir}")
    
    return {
        'vanilla_acc': vanilla_acc,
        'lsx_acc': lsx_acc,
        'vanilla_rr_acc': vanilla_rr_acc,
        'lsx_rr_acc': lsx_rr_acc
    }

if __name__ == '__main__':
    run_end_to_end_test()
