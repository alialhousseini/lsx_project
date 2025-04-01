"""
Experiment configuration for MNIST dataset.
"""

# Dataset configuration
dataset_config = {
    'name': 'mnist',
    'batch_size': 128,
    'num_samples': None,  # Use full dataset - Default = None
    'critic_samples': 25000,  # Number of samples for critic - 1200
    'input_channels': 1,
    'num_classes': 10
}

# Model configuration
model_config = {
    'input_channels': dataset_config['input_channels'],
    'num_classes': dataset_config['num_classes']
}

# Training configuration
training_config = {
    'num_epochs_fit': 10,
    'num_epochs_reflect': 5,
    'num_epochs_revise': 5,
    'num_epochs_finetune': 10,
    'lambda_expl': 100,  # Scaling factor for explanation loss
    'learning_rate_fit': 0.01,
    'learning_rate_reflect': 0.1,
    'learning_rate_revise': 0.001,
    'early_stopping': True,
    'patience': 4,
    'device': 'cuda',  # 'cuda' or 'cpu'
    'random_seed': 42
}

# LSX configuration
lsx_config = {
    # Number of LSX iterations (Explain, Reflect, Revise loops)
    'iterations': 3,
}

# Experiment configuration
experiment_config = {
    'name': 'mnist_experiment',
    'save_dir': './results/mnist',
    'save_model': True,
    'save_explanations': True,
    'evaluate_metrics': True,
    'visualize_results': True
}
