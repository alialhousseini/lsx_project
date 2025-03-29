"""
Experiment configuration for CUB-10 dataset.
"""

# Dataset configuration
dataset_config = {
    'name': 'cub10',
    'batch_size': 32,
    'num_samples': None,  # Use full dataset
    'critic_samples': 100,  # Number of samples for critic
    'input_channels': 3,
    'num_classes': 10,
    'image_size': 224  # CUB images are typically resized to 224x224
}

# Model configuration
model_config = {
    'input_channels': dataset_config['input_channels'],
    'num_classes': dataset_config['num_classes']
}

# Training configuration
training_config = {
    'num_epochs_fit': 20,
    'num_epochs_reflect': 1,
    'num_epochs_revise': 5,
    'num_epochs_finetune': 3,
    'lambda_expl': 100,  # Scaling factor for explanation loss
    'learning_rate_fit': 0.0005,
    'learning_rate_reflect': 0.0005,
    'learning_rate_revise': 0.0001,
    'early_stopping': True,
    'patience': 5,
    'device': 'cuda',  # 'cuda' or 'cpu'
    'random_seed': 42
}

# LSX configuration
lsx_config = {
    'iterations': 3,  # Number of LSX iterations (Explain, Reflect, Revise loops)
}

# Experiment configuration
experiment_config = {
    'name': 'cub10_experiment',
    'save_dir': './results/cub10',
    'save_model': True,
    'save_explanations': True,
    'evaluate_metrics': True,
    'visualize_results': True
}
