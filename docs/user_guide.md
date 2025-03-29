# Learning by Self-Explaining (LSX) - User Guide

This document provides a step-by-step guide for using the LSX implementation.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lsx_project.git
cd lsx_project
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

### Running a Basic Experiment

To run a basic experiment on the MNIST dataset:

```bash
python main.py --dataset mnist
```

This will:
1. Load the MNIST dataset
2. Train a vanilla CNN model
3. Generate explanations for the model's predictions
4. Train a critic model to evaluate the explanations
5. Update the learner model based on the critic's feedback
6. Evaluate the final model and save the results

### Visualizing Results

To analyze the results of an experiment:

```bash
python run_analysis.py --results_dir ./results/mnist --generate_report
```

This will generate various plots comparing the vanilla and LSX models, as well as a comprehensive report.

## Step-by-Step Guide

### 1. Configuring an Experiment

The configuration files for experiments are located in the `configs` directory. You can modify these files to change the experiment parameters.

For example, to modify the MNIST experiment configuration:

1. Open `configs/mnist_config.py`
2. Adjust parameters such as batch size, learning rates, or model architecture
3. Save the file

### 2. Running an Experiment

To run an experiment with a custom configuration:

```bash
python main.py --config configs/custom_config.py
```

### 3. Understanding the LSX Workflow

The LSX methodology follows a four-step process:

1. **Fit**: Train the learner model on the training data
   - This is a standard supervised learning step
   - The model learns to make predictions on the input data

2. **Explain**: Generate explanations for the learner's predictions
   - Explanations are generated using gradient-based methods
   - The default method is InputXGradient

3. **Reflect**: Train the critic model to evaluate the quality of explanations
   - The critic learns to distinguish between explanations from different classes
   - This helps identify what makes a good explanation

4. **Revise**: Update the learner model based on the critic's feedback
   - The learner is fine-tuned to improve both prediction accuracy and explanation quality
   - This is done by adding an explanation loss term to the training objective

### 4. Optimizing Model Performance

To optimize model performance:

```bash
python run_optimization.py --dataset mnist --optimization_type all
```

This will:
1. Prune the model to reduce its size
2. Quantize the model (if on CPU)
3. Train with learning rate scheduling
4. Train with mixed precision (if on CUDA)

### 5. Benchmarking

To benchmark different model architectures and explanation methods:

```bash
python run_benchmarks.py --dataset mnist --benchmark_type all
```

This will compare:
1. Different model architectures (CNN, EnhancedCNN, PretrainedResNet)
2. Different explanation methods (InputXGradient, GradientShap, IntegratedGradients, etc.)

### 6. Testing

To run tests:

```bash
# Run unit tests
python -m unittest tests/test_components.py

# Run integration tests
python tests/integration_test.py --config configs/mnist_config.py

# Run end-to-end tests
python tests/end_to_end_test.py
```

## Advanced Usage

### Using Different Datasets

The implementation supports three datasets:
- MNIST: Handwritten digit classification
- ChestMNIST: Chest X-ray image classification
- CUB-10: Bird species classification

To use a different dataset:

```bash
python main.py --dataset chestmnist
```

### Using Enhanced Model Architectures

The implementation includes several model architectures:
- CNN: Basic convolutional neural network
- EnhancedCNN: More sophisticated CNN with batch normalization
- PretrainedResNet: ResNet18 with pretrained weights

To use a different architecture, modify the model_type in the configuration file:

```python
model_config = {
    'model_type': 'enhanced_cnn',  # Options: 'cnn', 'enhanced_cnn', 'resnet'
    'input_channels': 1,
    'num_classes': 10
}
```

### Using Advanced Explanation Methods

The implementation includes several explanation methods:
- InputXGradient: Default method used in the paper
- GradientShap: Gradient-based SHAP values
- IntegratedGradients: Path integral of gradients
- Occlusion: Perturbation-based method
- GuidedBackprop: Guided backpropagation
- Saliency: Simple gradient-based saliency

To use a different explanation method, modify the LSXPipeline class or use the ExplanationMethods class directly.

### Hyperparameter Optimization

To optimize hyperparameters using Optuna:

```python
from utils.hyperparameter_optimization import LSXHyperparameterOptimizer

# Initialize optimizer
optimizer = LSXHyperparameterOptimizer(train_loader, critic_loader, test_loader, device)

# Run optimization
best_params = optimizer.optimize(n_trials=50)

# Train with best parameters
model = optimizer.train_with_best_params(best_params, save_path="best_model.pth")
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size in the configuration file
   - Use a smaller model architecture
   - Use mixed precision training

2. **Slow training**
   - Use a smaller subset of the data for initial experiments
   - Enable mixed precision training if using CUDA
   - Use learning rate scheduling

3. **Poor explanation quality**
   - Try different explanation methods
   - Adjust the lambda_expl parameter to increase the weight of the explanation loss
   - Increase the number of LSX iterations

### Getting Help

If you encounter any issues or have questions, please:
1. Check the documentation in the `docs` directory
2. Look at the example scripts in the `experiments` directory
3. Refer to the original paper for theoretical background

## Next Steps

After getting familiar with the basic functionality, you can:
1. Implement your own datasets by extending the data loading utilities
2. Add new model architectures to the `models` directory
3. Implement additional explanation methods in the `utils/explanation_methods.py` file
4. Develop new metrics for evaluating explanation quality in the `utils/metrics.py` file
