# Laplace Approximation for Bayesian Neural Networks

[![Julia](https://img.shields.io/badge/Julia-1.9+-blue.svg)](https://julialang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](.)


A comprehensive Bayesian neural networks framework with uncertainty quantification, out-of-distribution detection, and multiple inference methods comparison.

![Laplace Logo](wide_logo.png)

---

## Table of Contents
- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [Bayesian Methods](#-bayesian-methods)
- [OOD Detection](#-ood-detection)
- [Visualization](#-visualization)
- [Testing](#-testing)
- [Project Structure](#-project-structure)
- [Examples](#-examples)
- [Advanced Usage](#️-advanced-usage)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

---

## Key Features

### **Neural Network Classification**
- **MNIST Digit Recognition**: 3-layer MLP with ReLU activation for handwritten digit classification
- **German Credit Risk Assessment**: 2-layer network with sigmoid activation for credit risk evaluation
- **Laplace Approximation**: Bayesian uncertainty quantification using LaplaceRedux.jl

### **Multiple Bayesian Inference Methods**
- **HMC (Hamiltonian Monte Carlo)**: Gold standard for posterior sampling with theoretical guarantees
- **SWAG (Stochastic Weight Averaging-Gaussian)**: Practical posterior approximation balancing speed and quality
- **MAP (Maximum A Posteriori)**: Fast deterministic baseline method
- **Laplace Approximation**: Fast inference built on MAP with excellent uncertainty quantification

### **Out-of-Distribution (OOD) Detection**
- **Multiple OOD Datasets**: FashionMNIST, CIFAR-10, synthetic noise, custom distributions
- **Uncertainty Thresholding**: Automatic threshold calibration based on entropy, max probability, variance
- **Comprehensive Evaluation**: AUROC, AUPR, FPR@95TPR, calibration curves

### **Advanced Visualizations**
- **ROC and Precision-Recall Curves**: Complete detection performance analysis
- **Calibration Plots**: Confidence calibration quality assessment
- **Uncertainty Distributions**: Comparison of correct vs incorrect prediction uncertainties
- **Decision Boundaries**: Visual boundaries for German credit classification
- **Method Comparison Charts**: Side-by-side evaluation of different Bayesian inference methods

### **Graphical User Interface**
- **Interactive GUI**: Web-based user-friendly interface
- **Real-time Training**: Visualize training progress and results
- **Method Comparison**: Interactive Bayesian method performance comparison
- **One-click Export**: Complete export functionality for results and charts

---

## Quick Start

### 30-Second Demo
```julia
# 1. Activate the project
using Pkg; Pkg.activate("."); Pkg.instantiate()

# 2. Run comprehensive demo
include("start_demo.jl")
# Follow the interactive menu to explore different features

# 3. Or launch GUI
include("launch_gui.py")  # For Python GUI
# Or
include("laplace_gui.py")
```

### First Example - MNIST Classification
```julia
# Load and run MNIST example
include("examples/mnist_example.jl")
# Output: Training progress, accuracy analysis, uncertainty visualization
```

---

## Installation

### Prerequisites
- **Julia 1.9+** (recommended: latest stable version)
- **Python 3.8+** (for GUI features, optional)

### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/laplace.git
cd laplace
```

### Step 2: Install Julia Dependencies
```julia
# Start Julia in the project directory
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### Step 3: Install Python Dependencies (Optional, for GUI)
```bash
pip install -r requirements.txt
```

### Step 4: Test Installation
```julia
include("test/runtests.jl")
```

**Expected output**: All tests should pass with green checkmarks ✅

---

## Usage Guide

### Option 1: Interactive GUI 🖥️
```julia
# Launch web-based GUI
include("launch_gui.py")
# Navigate to http://localhost:8050 in your browser
```

### Option 2: Command Line Interface 💻
```julia
# Start interactive demo
include("start_demo.jl")
# Choose from menu options:
# [1] MNIST Classification Demo
# [2] German Credit Classification  
# [3] Bayesian Methods Comparison
# [4] OOD Detection Demo
# [5] Comprehensive Analysis
```

### Option 3: Direct Script Execution 📜
```julia
# Run specific examples
include("examples/mnist_example.jl")              # MNIST classification
include("examples/german_credit_example.jl")     # German credit classification
include("examples/bayesian_methods_comparison.jl") # Methods comparison
include("examples/comprehensive_ood_demo.jl")    # Complete OOD demo
```

---

## Bayesian Methods

This library implements four major Bayesian inference methods. Here's how to use each:

### MAP (Maximum A Posteriori)
**Best for**: Quick prototyping, baseline comparisons
```julia
include("src/BayesianMethods.jl")
using .BayesianMethods

# Create MAP model
nn = Chain(Dense(784, 64, relu), Dense(64, 10))
map_model = BayesianModel(nn, MAPMethod(weight_decay=0.01))

# Train
train_bayesian_model!(map_model, train_data, 50)

# Predict
predictions = predict_bayesian(map_model, test_data)
```

### Laplace Approximation
**Best for**: Production applications, balanced performance
```julia
# Create Laplace model
laplace_model = BayesianModel(deepcopy(nn), LaplaceMethod())

# Train (requires trained MAP model as starting point)
train_bayesian_model!(laplace_model, train_data, 50)

# Predict with uncertainty
predictions = predict_bayesian(laplace_model, test_data)
uncertainty = get_bayesian_uncertainty(laplace_model, test_data; method=:entropy)
```

### SWAG (Stochastic Weight Averaging)
**Best for**: Balanced speed and quality
```julia
# Create SWAG model
swag_model = BayesianModel(deepcopy(nn), SWAGMethod(n_models=20, start_epoch=30))

# Train (collects models during training)
train_bayesian_model!(swag_model, train_data, 100)

# Sample predictions
predictions = predict_bayesian(swag_model, test_data; n_samples=50)
```

### HMC (Hamiltonian Monte Carlo)
**Best for**: Research, highest uncertainty quality
```julia
# Create HMC model
hmc_model = BayesianModel(deepcopy(nn), HMCMethod(n_samples=1000, step_size=0.001))

# Train (MCMC sampling)
train_bayesian_model!(hmc_model, train_data, 100)

# Get posterior samples
predictions = predict_bayesian(hmc_model, test_data; n_samples=100)
```

### Method Comparison
```julia
# Compare all methods on same data
results = compare_bayesian_methods(
    nn, train_data, test_data, test_labels;
    methods=[:map, :laplace, :swag, :hmc],
    epochs=50
)

# Plot comparison
plot_bayesian_methods_comparison(results)
```

---

## OOD Detection

Out-of-distribution detection helps identify when your model encounters data different from its training distribution.

### Quick OOD Detection
```julia
include("src/OODDetection.jl")
using .OODDetection

# Create OOD detector
detector = OODDetector(trained_model; method=:entropy)

# Fit threshold on validation data
fit_ood_threshold!(detector, validation_data; percentile=95.0)

# Detect OOD samples
ood_predictions, uncertainty_scores = detect_ood(detector, test_data)
```

### Available OOD Datasets
```julia
include("src/OODDatasets.jl")
using .OODDatasets

# Load different OOD datasets for MNIST
fashion_x, fashion_y, _ = load_ood_mnist(:fashionmnist, 1000)  # Similar domain
cifar_x, cifar_y, _ = load_ood_mnist(:cifar10, 1000)          # Different domain  
noise_x, noise_y, _ = load_ood_mnist(:uniform_noise, 1000)    # Pure noise

# Load OOD datasets for German Credit
shifted_x, shifted_y, _ = load_ood_german_credit(:shifted_distribution, 100)
noise_x, noise_y, _ = load_ood_german_credit(:uniform_noise, 100)
```

### Uncertainty Methods
- **`:entropy`**: Information-theoretic uncertainty (most effective)
- **`:max_prob`**: Based on maximum prediction probability
- **`:variance`**: Predictive variance (for Bayesian methods)
- **`:mutual_info`**: Mutual information between parameters and predictions

### Evaluation Metrics
```julia
# Evaluate OOD detection performance
metrics = evaluate_ood_detection(detector, in_dist_data, ood_data)

println("AUROC: $(metrics["auroc"])")           # Area under ROC curve
println("AUPR: $(metrics["aupr"])")             # Area under PR curve  
println("FPR@95TPR: $(metrics["fpr_at_95"])")  # False positive rate at 95% TPR
```

### Complete OOD Example
```julia
# Run comprehensive OOD detection demo
include("examples/comprehensive_ood_demo.jl")
# Generates: ROC curves, calibration plots, performance across multiple datasets
```

---

## Visualization

Rich visualization capabilities for understanding model behavior and uncertainty.

### Basic Visualizations
```julia
include("src/Visualizations.jl")
using .Visualizations

# Plot training progress
plot_training_progress(losses, accuracies)

# Visualize predictions on sample data
plot_mnist_samples(trained_model, test_x, test_y, 6)

# Decision boundaries (for 2D data)
plot_german_credit_decision_boundary(trained_model, X, y, classes)
```

### Uncertainty Analysis
```julia
# Uncertainty distribution comparison
plot_uncertainty_histogram(entropies, correct_mask)

# Calibration assessment
plot_calibration_curve(confidences, accuracies)

# Method comparison
plot_laplace_vs_map_comparison(model, test_x)
```

### OOD Detection Visualizations
```julia
# ROC curve analysis
plot_roc_curve(metrics)

# Precision-recall curve
plot_precision_recall_curve(metrics)

# Complete OOD detection summary
plot_ood_detection_summary(metrics, scores_in, scores_ood)
```

---

## Testing

### Run All Tests
```julia
using Pkg; Pkg.activate(".")
include("test/runtests.jl")
```

### Run Specific Test Modules
```julia
include("test/test_mnist.jl")                    # MNIST classification tests
include("test/test_german_credit.jl")            # German credit tests  
include("test/test_bayesian_methods.jl")         # Bayesian methods tests
include("test/test_ood_datasets.jl")             # OOD dataset tests
include("test/test_ood_detection.jl")            # OOD detection tests
include("test/test_visualizations.jl")           # Visualization tests
```

### Test Coverage
- Neural network training and prediction
- All Bayesian inference methods
- OOD dataset loading and generation
- OOD detection algorithms
- Visualization functions
- GUI integration tests

---

## Project Structure

```
laplace/
├── Project.toml                     # Julia project dependencies
├── README.md                       # This documentation
├── PROJECT_SUMMARY.md              # Detailed project summary
├── start_demo.jl                   # Interactive demo launcher
├── launch_gui.py                    # GUI launcher
├── laplace_gui.py                   # Main GUI application
├── requirements.txt                # Python dependencies for GUI
├── src/                            # Core source code
│   ├── MNISTClassifier.jl             # MNIST classification 
│   ├── GermanCreditClassifier.jl      # German credit classification 
│   ├── BayesianMethods.jl             # Multiple Bayesian methods
│   ├── OODDatasets.jl                 # OOD dataset loading 
│   ├── OODDetection.jl                # OOD detection algorithms 
│   ├── Visualizations.jl              # Visualization functions
│   └── GUIInterface.jl                # GUI backend interface
├── test/                           # Comprehensive test suite
│   ├── runtests.jl                    # Main test runner
│   ├── test_mnist.jl                  # MNIST tests 
│   ├── test_german_credit.jl          # German credit tests 
│   ├── test_bayesian_methods.jl       # Bayesian methods tests 
│   ├── test_ood_datasets.jl           # OOD dataset tests 
│   ├── test_ood_detection.jl          # OOD detection tests 
│   └── test_visualizations.jl         # Visualization tests 
├── examples/                       # Usage examples and tutorials
│   ├── mnist_example.jl               # MNIST complete example 
│   ├── german_credit_example.jl       # German credit example 
│   ├── bayesian_methods_comparison.jl # Methods comparison
│   ├── comprehensive_ood_demo.jl      # Complete OOD demo 
│   ├── mnist_ood_example.jl           # MNIST OOD detection
│   └── german_credit_ood_example.jl   # German credit OOD detection
│
└── Multi-Class-Julia-*.ipynb       # Original reference notebooks
```

---

## Examples

### Example 1: MNIST Classification with Uncertainty
```julia
# Complete MNIST classification workflow
include("examples/mnist_example.jl")

# What it does:
# 1. Loads MNIST dataset (5000 training samples)
# 2. Creates 3-layer neural network (784→64→64→10)
# 3. Trains using MAP method
# 4. Applies Laplace approximation
# 5. Evaluates on test set with uncertainty quantification
# 6. Visualizes results with uncertainty analysis
```

### Example 2: Bayesian Methods Comparison
```julia
# Compare all four Bayesian methods
include("examples/bayesian_methods_comparison.jl")

# What it does:
# 1. Creates identical network architectures
# 2. Trains MAP, Laplace, SWAG, and HMC models
# 3. Compares accuracy and uncertainty quality
# 4. Generates comprehensive comparison plots
# 5. Provides timing and performance metrics
```

### Example 3: OOD Detection Analysis
```julia
# Comprehensive OOD detection demo
include("examples/comprehensive_ood_demo.jl")

# What it does:
# 1. Trains MNIST classifier
# 2. Tests against FashionMNIST, CIFAR-10, and noise
# 3. Evaluates multiple uncertainty methods
# 4. Generates ROC curves and calibration plots
# 5. Provides detailed performance analysis
```

### Example 4: German Credit Risk Assessment
```julia
# German credit classification with decision boundaries
include("examples/german_credit_example.jl")

# What it does:
# 1. Generates synthetic German credit data
# 2. Trains 2-layer classification network
# 3. Visualizes decision boundaries
# 4. Analyzes uncertainty in decision regions
# 5. Compares different uncertainty methods
```

---

## Advanced Usage

### Custom Neural Network Architectures
```julia
# Define custom architecture
custom_nn = Chain(
    Dense(784, 128, relu),
    Dropout(0.2),
    Dense(128, 64, relu), 
    Dropout(0.2),
    Dense(64, 10)
)

# Use with any Bayesian method
model = BayesianModel(custom_nn, LaplaceMethod())
```

### Hyperparameter Tuning
```julia
# Experiment with different configurations
configurations = [
    (hidden_size=32, weight_decay=0.001),
    (hidden_size=64, weight_decay=0.01), 
    (hidden_size=128, weight_decay=0.1)
]

results = []
for config in configurations
    nn = Chain(Dense(784, config.hidden_size, relu), Dense(config.hidden_size, 10))
    model = BayesianModel(nn, MAPMethod(weight_decay=config.weight_decay))
    train_bayesian_model!(model, train_data, 50)
    push!(results, evaluate_bayesian_model(model, test_data, test_labels))
end
```

### Custom OOD Detection
```julia
# Create custom OOD dataset
function create_custom_ood(n_samples)
    # Your custom data generation logic
    return custom_x, custom_y, "Custom OOD"
end

# Use custom uncertainty function
function custom_uncertainty(predictions)
    # Your custom uncertainty calculation
    return uncertainty_scores
end

# Integrate into detection pipeline
detector = OODDetector(model; uncertainty_fn=custom_uncertainty)
```

### Batch Processing
```julia
# Process multiple datasets
datasets = ["mnist", "fashionmnist", "cifar10"]
results = Dict()

for dataset in datasets
    # Load data
    x, y = load_dataset(dataset)
    
    # Create and train model
    model = create_model_for_dataset(dataset)
    train_bayesian_model!(model, (x, y), 50)
    
    # Store results
    results[dataset] = evaluate_model(model, test_data)
end
```

---

## Troubleshooting

### Common Issues and Solutions

#### Installation Problems
**Problem**: `Pkg.instantiate()` fails with dependency conflicts
```julia
# Solution: Clean environment and reinstall
using Pkg
Pkg.rm.(keys(Pkg.project().dependencies))
Pkg.instantiate()
```

**Problem**: LaplaceRedux.jl installation fails
```julia
# Solution: Install specific compatible version
Pkg.add(PackageSpec(name="LaplaceRedux", version="0.4"))
```

#### Runtime Errors
**Problem**: "OutOfMemory" error during HMC sampling
```julia
# Solution: Reduce sample size or use smaller model
hmc_model = BayesianModel(nn, HMCMethod(n_samples=100))  # Instead of 1000
```

**Problem**: GPU out of memory
```julia
# Solution: Use CPU instead of GPU
using Flux
Flux.gpu_backend = "CPU"
```

#### Performance Issues
**Problem**: Training is very slow
```julia
# Solutions:
# 1. Reduce dataset size for initial testing
train_x = train_x[:, 1:1000]  # Use first 1000 samples

# 2. Use smaller network architecture
nn = Chain(Dense(784, 32, relu), Dense(32, 10))  # Smaller hidden layer

# 3. Reduce training epochs
train_bayesian_model!(model, train_data, 20)  # Instead of 100
```

#### Visualization Problems
**Problem**: Plots not displaying
```julia
# Solution: Ensure plotting backend is available
using Plots
gr()  # or plotlyjs()
plot([1,2,3])  # Test plot
```

### Getting Help
1. **Check Examples**: Look at `examples/` directory for working code
2. **Run Tests**: Execute `test/runtests.jl` to verify installation
3. **Enable Verbose Mode**: Add `verbose=true` to function calls
4. **Check Logs**: Look for error messages in Julia REPL

### Performance Optimization Tips
1. **Use appropriate batch sizes**: Not too large (memory) or too small (inefficient)
2. **Choose right method for use case**: MAP for speed, HMC for quality
3. **Monitor memory usage**: Use `@time` and `@allocated` macros
4. **Use compiled functions**: Avoid global variables in tight loops

## Performance Benchmarks

### MNIST Classification Performance
- **Architecture**: 3-layer MLP (784→64→64→10)
- **Expected Accuracy**: ~90% on test set
- **Uncertainty Quality**: Higher entropy for incorrect predictions
- **Training Time**: 2-3 minutes for 5000 samples, 50 epochs

### German Credit Classification Performance  
- **Architecture**: 2-layer MLP (2→16→4)
- **Expected Accuracy**: Varies with synthetic data
- **Visualization**: Clear decision boundaries between classes
- **Training Time**: <1 minute for 100 samples, 200 epochs

### OOD Detection Performance
- **MNIST vs FashionMNIST**: AUROC ~0.95+ (excellent separation)
- **MNIST vs CIFAR-10**: AUROC ~0.90+ (good separation)
- **MNIST vs Noise**: AUROC ~0.99+ (near-perfect separation)
- **German Credit Shifted**: AUROC ~0.80-0.95 (depends on shift magnitude)

---

## Dependencies

### Core Dependencies
- **Flux.jl**: Neural network framework
- **LaplaceRedux.jl**: Laplace approximation implementation  
- **MLDatasets.jl**: MNIST data loading
- **Plots.jl**: Visualization
- **CSV.jl & DataFrames.jl**: Data handling
- **Test.jl**: Testing framework
- **Statistics.jl**: Statistical functions
- **LinearAlgebra.jl**: Linear algebra operations

### Optional GUI Dependencies
- **Blink.jl**: Web interface
- **WebIO.jl**: Interactive components
- **Interact.jl**: UI widgets
- **PyCall.jl**: Python interoperability

---

## License

This project is provided for educational purposes. Please check the licenses of individual dependencies.

---

## Acknowledgments

- **LaplaceRedux.jl** team for excellent Laplace approximation implementation
- **Flux.jl** community for powerful neural network framework  
- Julia ecosystem contributors

