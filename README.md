# Learning by Self-Explaining (LSX)

This repository contains an implementation of the Learning by Self-Explaining (LSX) methodology as described in the paper "Learning by Self-Explaining" (https://arxiv.org/abs/2309.08395).

## Overview

Learning by Self-Explaining (LSX) is a novel approach that combines self-refining ML and explanatory interactive learning. The LSX model consists of two submodels (learner and critic) that work together to improve model performance through self-explanation.

The LSX methodology follows a four-step process:
1. **Fit**: Train the learner model on the training data
2. **Explain**: Generate explanations for the learner's predictions
3. **Reflect**: Train the critic model to evaluate the quality of explanations
4. **Revise**: Update the learner model based on the critic's feedback

This implementation focuses on the CNN-LSX instantiation and supports the MNIST, ChestMNIST, and CUB-10 datasets.

## Repository Structure

```
lsx_project/
├── configs/                  # Configuration files for different datasets
│   ├── mnist_config.py
│   ├── chestmnist_config.py
│   └── cub10_config.py
├── data/                     # Data handling utilities
│   ├── datasets.py           # Dataset loaders
│   ├── utils.py              # Data utility functions
│   └── augmentation.py       # Data augmentation techniques
├── models/                   # Model definitions
│   ├── cnn.py                # Basic CNN model
│   ├── lsx.py                # LSX model implementation
│   └── enhanced_models.py    # Enhanced model architectures
├── modules/                  # LSX methodology modules
│   ├── fit.py                # Fit module for training the learner
│   ├── explain.py            # Explain module for generating explanations
│   ├── reflect.py            # Reflect module for training the critic
│   └── revise.py             # Revise module for updating the learner
├── utils/                    # Utility functions
│   ├── metrics.py            # Evaluation metrics
│   ├── visualization.py      # Visualization utilities
│   ├── explanation_methods.py # Enhanced explanation methods
│   ├── hyperparameter_optimization.py # Hyperparameter optimization
│   ├── optimization.py       # Model and training optimization
│   ├── integration.py        # Integration utilities
│   └── results_analysis.py   # Results analysis and visualization
├── experiments/              # Experiment scripts for different datasets
│   ├── mnist.py
│   ├── chestmnist.py
│   └── cub10.py
├── tests/                    # Test scripts
│   ├── test_components.py    # Unit tests for individual components
│   ├── integration_test.py   # Integration tests
│   ├── benchmark_tests.py    # Performance benchmark tests
│   └── end_to_end_test.py    # End-to-end test of the LSX workflow
├── main.py                   # Main entry point for running experiments
├── run_benchmarks.py         # Script for running benchmark tests
├── run_optimization.py       # Script for optimizing models
├── run_analysis.py           # Script for analyzing results
└── requirements.txt          # Project dependencies
```

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

## Usage

### Running Experiments

To run an experiment on a specific dataset:

```bash
python main.py --dataset mnist
```

Available datasets: `mnist`, `chestmnist`, `cub10`

You can also specify a custom configuration file:

```bash
python main.py --config configs/custom_config.py
```

### Running Tests

To run unit tests:

```bash
python -m unittest tests/test_components.py
```

To run integration tests:

```bash
python tests/integration_test.py --config configs/mnist_config.py
```

To run end-to-end tests:

```bash
python tests/end_to_end_test.py
```

### Running Benchmarks

To benchmark model architectures and explanation methods:

```bash
python run_benchmarks.py --dataset mnist --benchmark_type all
```

Options for `benchmark_type`: `models`, `explanations`, `all`

### Optimizing Models

To optimize model performance and training:

```bash
python run_optimization.py --dataset mnist --optimization_type all
```

Options for `optimization_type`: `model`, `training`, `all`

### Analyzing Results

To analyze experiment results:

```bash
python run_analysis.py --results_dir /path/to/results --generate_report
```

## Implementation Details

### CNN-LSX Implementation

The CNN-LSX implementation consists of:

1. **Learner Model**: A convolutional neural network that makes predictions on the input data.
2. **Critic Model**: A separate convolutional neural network that evaluates the quality of explanations.
3. **Explanation Method**: The InputXGradient method is used as the default explanation technique, with additional methods available in the `explanation_methods.py` module.

### LSX Modules

The LSX methodology is implemented as four separate modules:

1. **Fit Module**: Trains the learner model on the training data using standard supervised learning.
2. **Explain Module**: Generates explanations for the learner's predictions using gradient-based methods.
3. **Reflect Module**: Trains the critic model to evaluate the quality of explanations based on their class-discriminative power.
4. **Revise Module**: Updates the learner model based on the critic's feedback to improve both prediction accuracy and explanation quality.

### Enhanced Features

This implementation includes several enhancements beyond the basic LSX methodology:

1. **Enhanced Model Architectures**: Additional model architectures like EnhancedCNN and PretrainedResNet.
2. **Advanced Explanation Methods**: Multiple explanation techniques including GradientShap, IntegratedGradients, and Saliency.
3. **Data Augmentation**: Techniques like random affine transformations, horizontal flips, and mixup.
4. **Hyperparameter Optimization**: Automated hyperparameter tuning using Optuna.
5. **Model Optimization**: Techniques like pruning and quantization to improve inference speed.
6. **Training Optimization**: Learning rate scheduling and mixed precision training.
7. **Results Analysis**: Comprehensive analysis and visualization of experiment results.

## Extending the Implementation

### Adding New Datasets

To add a new dataset:

1. Create a new dataset loader in `data/datasets.py`
2. Create a configuration file in the `configs` directory
3. Update the `main.py` script to include the new dataset

### Adding New Model Architectures

To add a new model architecture:

1. Create a new model class in `models/enhanced_models.py`
2. Ensure it implements the required methods (forward, get_activations)
3. Update the LSX model to use the new architecture

### Adding New Explanation Methods

To add a new explanation method:

1. Add the method to the `ExplanationMethods` class in `utils/explanation_methods.py`
2. Implement the method using the Captum library or custom code
3. Update the `get_explanation` method to include the new method

## Citation

If you use this implementation in your research, please cite the original paper:

```
@article{lertvittayakumjorn2023learning,
  title={Learning by Self-Explaining},
  author={Lertvittayakumjorn, Piyawat and Specia, Lucia and Toni, Francesca},
  journal={arXiv preprint arXiv:2309.08395},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
