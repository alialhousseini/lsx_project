"""
Experiment configuration for ChestMNIST dataset.
"""

# Dataset configuration
dataset_config = {
    'name': 'chestmnist',
    'batch_size': 64,
    'num_samples': None,  # Use full dataset
    'critic_samples': 1600,  # Number of samples for critic
    'input_channels': 1,
    'num_classes': 2  # Binary classification for ChestMNIST
}

# Model configuration
model_config = {
    'input_channels': dataset_config['input_channels'],
    'num_classes': dataset_config['num_classes']
}

# Training configuration
training_config = {
    'num_epochs_fit': 15,
    'num_epochs_reflect': 1,
    'num_epochs_revise': 5,
    'num_epochs_finetune': 3,
    'lambda_expl': 100,  # Scaling factor for explanation loss
    'learning_rate_fit': 0.001,
    'learning_rate_reflect': 0.001,
    'learning_rate_revise': 0.0005,
    'early_stopping': True,
    'patience': 3,
    'device': 'cuda',  # 'cuda' or 'cpu'
    'random_seed': 42
}

# LSX configuration
lsx_config = {
    'iterations': 3,  # Number of LSX iterations (Explain, Reflect, Revise loops)
}

# Experiment configuration
experiment_config = {
    'name': 'chestmnist_experiment',
    'save_dir': './results/chestmnist',
    'save_model': True,
    'save_explanations': True,
    'evaluate_metrics': True,
    'visualize_results': True
}
