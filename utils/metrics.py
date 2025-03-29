"""
Evaluation metrics for the LSX project.

This module implements various metrics for evaluating the performance of LSX models,
including classification accuracy, explanation consolidation metrics, and explanation
faithfulness metrics.
"""
import torch
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def classification_accuracy(model, dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Calculate classification accuracy on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader containing the evaluation data
        device: Device to use for evaluation
        
    Returns:
        float: Accuracy as a percentage
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

def ridge_regression_accuracy(explanations, targets, test_size=0.2, random_state=42):
    """
    Calculate the accuracy of a ridge regression model trained on explanations.
    
    This metric is used to evaluate the separability of explanations.
    
    Args:
        explanations: Tensor of explanations
        targets: Tensor of target classes
        test_size: Proportion of the dataset to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        float: Accuracy as a percentage
    """
    # Convert tensors to numpy arrays
    if isinstance(explanations, torch.Tensor):
        explanations = explanations.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Flatten explanations if needed
    if len(explanations.shape) > 2:
        explanations = explanations.reshape(explanations.shape[0], -1)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        explanations, targets, test_size=test_size, random_state=random_state, stratify=targets
    )
    
    # Train ridge regression model
    clf = RidgeClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    
    return accuracy

def inter_vs_intraclass_explanation_similarity(explanations, targets, encoder_model=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Calculate the Inter- vs Intraclass Explanation Similarity (IIES) metric.
    
    This metric estimates how far the explanations from samples of one class are close to
    another compared to the explanations of samples from other classes.
    
    Args:
        explanations: Tensor of explanations
        targets: Tensor of target classes
        encoder_model: Model to encode explanations (if None, use raw explanations)
        device: Device to use for encoding
        
    Returns:
        float: IIES value (lower is better)
    """
    # Convert tensors to numpy arrays
    if isinstance(explanations, torch.Tensor):
        explanations_tensor = explanations
        explanations = explanations.cpu().numpy()
    else:
        explanations_tensor = torch.tensor(explanations)
    
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Get unique classes
    unique_classes = np.unique(targets)
    num_classes = len(unique_classes)
    
    # If encoder model is provided, encode explanations
    if encoder_model is not None:
        encoder_model = encoder_model.to(device)
        encoder_model.eval()
        
        # Process in batches to avoid memory issues
        batch_size = 32
        encoded_explanations = []
        
        with torch.no_grad():
            for i in range(0, len(explanations_tensor), batch_size):
                batch = explanations_tensor[i:i+batch_size].to(device)
                encoded_batch = encoder_model.get_activations(batch, 'flatten')
                encoded_explanations.append(encoded_batch.cpu().numpy())
        
        encoded_explanations = np.concatenate(encoded_explanations, axis=0)
    else:
        # Flatten explanations if needed
        if len(explanations.shape) > 2:
            encoded_explanations = explanations.reshape(explanations.shape[0], -1)
        else:
            encoded_explanations = explanations
    
    # Calculate class means
    class_means = []
    for class_idx in unique_classes:
        class_mask = (targets == class_idx)
        class_explanations = encoded_explanations[class_mask]
        class_mean = np.mean(class_explanations, axis=0)
        class_means.append(class_mean)
    
    # Calculate IIES
    iies = 0
    for k, class_idx in enumerate(unique_classes):
        class_mask = (targets == class_idx)
        class_explanations = encoded_explanations[class_mask]
        class_mean = class_means[k]
        
        # Calculate intraclass distance
        intraclass_distances = np.mean([np.linalg.norm(expl - class_mean) for expl in class_explanations])
        
        # Calculate interclass distance
        interclass_distances = np.mean([np.linalg.norm(class_mean - other_mean) 
                                       for j, other_mean in enumerate(class_means) if j != k])
        
        # Add to IIES
        iies += intraclass_distances / interclass_distances
    
    # Average over all classes
    iies /= num_classes
    
    return iies

def explanation_faithfulness(model, dataloader, explanation_fn, device='cuda' if torch.cuda.is_available() else 'cpu', percentages=[1, 5, 10, 20, 50]):
    """
    Calculate explanation faithfulness metrics: comprehensiveness and sufficiency.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader containing the evaluation data
        explanation_fn: Function to generate explanations
        device: Device to use for evaluation
        percentages: List of percentages to use for removing features
        
    Returns:
        tuple: (comprehensiveness, sufficiency)
    """
    model.eval()
    
    # Get median value for each channel
    all_inputs = []
    for inputs, _ in dataloader:
        all_inputs.append(inputs)
    all_inputs = torch.cat(all_inputs, dim=0)
    median_values = torch.median(all_inputs, dim=0).values
    
    # Calculate metrics
    comp_scores = []
    suff_scores = []
    
    for percentage in percentages:
        comp_score = _calculate_comprehensiveness(model, dataloader, explanation_fn, median_values, percentage, device)
        suff_score = _calculate_sufficiency(model, dataloader, explanation_fn, median_values, percentage, device)
        
        comp_scores.append(comp_score)
        suff_scores.append(suff_score)
    
    # Average over all percentages
    comprehensiveness = np.mean(comp_scores)
    sufficiency = np.mean(suff_scores)
    
    return comprehensiveness, sufficiency

def _calculate_comprehensiveness(model, dataloader, explanation_fn, median_values, percentage, device):
    """
    Calculate comprehensiveness score.
    
    Comprehensiveness measures the impact of removing important features on the model's performance.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader containing the evaluation data
        explanation_fn: Function to generate explanations
        median_values: Median values for each channel
        percentage: Percentage of features to remove
        device: Device to use for evaluation
        
    Returns:
        float: Comprehensiveness score
    """
    model.eval()
    original_acc = 0
    modified_acc = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        
        # Get original predictions
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            original_correct = predicted.eq(targets).sum().item()
        
        # Generate explanations
        explanations = explanation_fn(inputs, targets)
        
        # Create modified inputs by removing important features
        modified_inputs = inputs.clone()
        
        for i in range(batch_size):
            # Flatten explanation
            flat_expl = explanations[i].abs().view(-1)
            
            # Get indices of top features
            num_features = flat_expl.size(0)
            num_to_remove = int(num_features * percentage / 100)
            _, top_indices = torch.topk(flat_expl, num_to_remove)
            
            # Create mask
            mask = torch.ones_like(flat_expl)
            mask[top_indices] = 0
            mask = mask.view_as(explanations[i])
            
            # Apply mask
            modified_inputs[i] = inputs[i] * mask + median_values * (1 - mask)
        
        # Get predictions on modified inputs
        with torch.no_grad():
            outputs = model(modified_inputs)
            _, predicted = outputs.max(1)
            modified_correct = predicted.eq(targets).sum().item()
        
        # Update statistics
        original_acc += original_correct
        modified_acc += modified_correct
        total += batch_size
    
    # Calculate comprehensiveness
    original_acc = original_acc / total
    modified_acc = modified_acc / total
    comprehensiveness = original_acc - modified_acc
    
    return comprehensiveness * 100  # Convert to percentage

def _calculate_sufficiency(model, dataloader, explanation_fn, median_values, percentage, device):
    """
    Calculate sufficiency score.
    
    Sufficiency measures the impact of keeping only important features on the model's performance.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader containing the evaluation data
        explanation_fn: Function to generate explanations
        median_values: Median values for each channel
        percentage: Percentage of features to keep
        device: Device to use for evaluation
        
    Returns:
        float: Sufficiency score
    """
    model.eval()
    original_acc = 0
    modified_acc = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        
        # Get original predictions
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            original_correct = predicted.eq(targets).sum().item()
        
        # Generate explanations
        explanations = explanation_fn(inputs, targets)
        
        # Create modified inputs by keeping only important features
        modified_inputs = inputs.clone()
        
        for i in range(batch_size):
            # Flatten explanation
            flat_expl = explanations[i].abs().view(-1)
            
            # Get indices of top features
            num_features = flat_expl.size(0)
            num_to_keep = int(num_features * percentage / 100)
            _, top_indices = torch.topk(flat_expl, num_to_keep)
            
            # Create mask
            mask = torch.zeros_like(flat_expl)
            mask[top_indices] = 1
            mask = mask.view_as(explanations[i])
            
            # Apply mask
            modified_inputs[i] = inputs[i] * mask + median_values * (1 - mask)
        
        # Get predictions on modified inputs
        with torch.no_grad():
            outputs = model(modified_inputs)
            _, predicted = outputs.max(1)
            modified_correct = predicted.eq(targets).sum().item()
        
        # Update statistics
        original_acc += original_correct
        modified_acc += modified_correct
        total += batch_size
    
    # Calculate sufficiency
    original_acc = original_acc / total
    modified_acc = modified_acc / total
    sufficiency = original_acc - modified_acc
    
    return sufficiency * 100  # Convert to percentage
