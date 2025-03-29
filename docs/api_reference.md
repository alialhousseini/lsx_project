# Learning by Self-Explaining (LSX) - API Reference

This document provides a detailed API reference for the LSX implementation.

## Models

### LSX

```python
class LSX(nn.Module):
    def __init__(self, input_channels=1, num_classes=10)
    def forward(self, x)
    def save_models(self, path)
    def load_models(self, path)
```

### CNN

```python
class CNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10)
    def forward(self, x)
    def get_activations(self, x, layer_name=None)
```

### EnhancedCNN

```python
class EnhancedCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10)
    def forward(self, x)
    def get_activations(self, x, layer_name=None)
```

### PretrainedResNet

```python
class PretrainedResNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, pretrained=True)
    def forward(self, x)
    def get_activations(self, x, layer_name=None)
```

## Modules

### Fit

```python
class Fit:
    def __init__(self, model, criterion=None, optimizer=None, device='cuda')
    def train(self, train_loader, val_loader=None, num_epochs=10, early_stopping=False, patience=3)
    def evaluate(self, data_loader)
```

### Explain

```python
class Explain:
    def __init__(self, model, device='cuda')
    def generate_explanations(self, data_loader)
    def _generate_input_x_gradient(self, inputs, targets)
```

### Reflect

```python
class Reflect:
    def __init__(self, model, criterion=None, optimizer=None, device='cuda')
    def get_feedback(self, explanations, targets, num_epochs=10)
```

### Revise

```python
class Revise:
    def __init__(self, model, criterion=None, optimizer=None, lambda_expl=100, device='cuda')
    def revise_learner(self, train_loader, explanations, targets, num_epochs=5)
    def final_finetuning(self, train_loader, explanations, num_epochs=3)
```

## Data

### Datasets

```python
def get_mnist_dataloaders(batch_size=64, num_samples=None, critic_samples=1000)
def get_chestmnist_dataloaders(batch_size=64, num_samples=None, critic_samples=1000)
def get_cub10_dataloaders(batch_size=64, num_samples=None, critic_samples=100)
```

### Augmentation

```python
class AugmentedDataset(Dataset):
    def __init__(self, dataset, transform=None)
    def __len__()
    def __getitem__(self, idx)

def get_mnist_augmentation()
def get_chestmnist_augmentation()
def get_cub10_augmentation()
def create_augmented_dataloader(dataset, batch_size=64, augmentation=None, shuffle=True)
def mixup_data(x, y, alpha=1.0, device='cuda')
def mixup_criterion(criterion, pred, y_a, y_b, lam)

class CutMix:
    def __init__(self, alpha=1.0)
    def __call__(self, x, y, device='cuda')
```

## Utils

### Metrics

```python
def classification_accuracy(model, data_loader, device='cuda')
def ridge_regression_accuracy(explanations, targets, test_size=0.2, random_state=42)
def inter_vs_intraclass_explanation_similarity(explanations, targets)
def explanation_faithfulness(model, data_loader, explanation_fn, device='cuda', num_samples=100)
```

### Visualization

```python
def visualize_explanation(image, explanation, title=None, save_path=None)
def visualize_explanation_grid(images, explanations, num_samples=10, title=None, save_path=None)
def visualize_class_explanations(explanations, targets, num_classes=10, samples_per_class=5, title=None, save_path=None)
def plot_training_history(history, title=None, save_path=None)
```

### Explanation Methods

```python
class ExplanationMethods:
    def __init__(self, model, device='cuda')
    def explain_input_x_gradient(self, inputs, targets)
    def explain_gradient_shap(self, inputs, targets, n_samples=50, stdevs=0.0001)
    def explain_integrated_gradients(self, inputs, targets, n_steps=50)
    def explain_occlusion(self, inputs, targets, sliding_window_shapes=(1, 5, 5), strides=(1, 2, 2))
    def explain_guided_backprop(self, inputs, targets)
    def explain_saliency(self, inputs, targets)
    def get_explanation(self, inputs, targets, method='input_x_gradient', **kwargs)
```

### Hyperparameter Optimization

```python
class LSXHyperparameterOptimizer:
    def __init__(self, train_loader, critic_loader, test_loader, device='cuda')
    def objective(self, trial)
    def optimize(self, n_trials=100, study_name='lsx_optimization', direction='maximize')
    def train_with_best_params(self, best_params, save_path=None)
```

### Optimization

```python
class ModelOptimizer:
    def __init__(self, model)
    def prune_model(self, amount=0.2, modules_to_prune=None)
    def quantize_model(self, dtype=torch.qint8)
    def benchmark_model(self, test_loader, device='cuda')
    def optimize_for_inference(self, test_loader, device='cuda')

class TrainingOptimizer:
    def __init__(self, model, optimizer, criterion, device='cuda')
    def train_with_lr_scheduler(self, train_loader, val_loader, num_epochs, scheduler_type='cosine')
    def train_with_mixed_precision(self, train_loader, val_loader, num_epochs)
```

### Integration

```python
class LSXPipeline:
    def __init__(self, config, train_loader, critic_loader, test_loader, device='cuda')
    def train_vanilla_model()
    def train_lsx_model(self, explanations=None, targets=None)
    def run_pipeline()
    def test_alternative_explanations()
    def test_data_augmentation()
```

### Results Analysis

```python
class ResultsAnalyzer:
    def __init__(self, results_dir)
    def plot_accuracy_comparison(self, save_path=None)
    def plot_explanation_quality_comparison(self, save_path=None)
    def plot_optimization_results(self, save_path=None)
    def visualize_explanations_tsne(self, explanations, targets, title="t-SNE Visualization of Explanations", save_path=None)
    def visualize_explanations_pca(self, explanations, targets, title="PCA Visualization of Explanations", save_path=None)
    def generate_comprehensive_report(self, output_dir=None)
```

## Experiments

### MNIST

```python
def run_experiment(config_path)
```

### ChestMNIST

```python
def run_experiment(config_path)
```

### CUB-10

```python
def run_experiment(config_path)
```

## Main Scripts

### main.py

```python
def main()
```

### run_benchmarks.py

```python
def main()
```

### run_optimization.py

```python
def main()
```

### run_analysis.py

```python
def main()
```
