# 🔬 Python GUI for Laplace Approximation Bayesian Neural Networks

A comprehensive Python GUI interface for the Julia-based Bayesian neural network framework, featuring all core functionalities with an intuitive graphical interface.

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** with pip
2. **Julia 1.6+** installed and accessible from command line
3. **Git** (for cloning repositories)

### Installation

1. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Julia Installation**
   ```bash
   julia --version
   ```

3. **Install Julia Dependencies**
   ```bash
   julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
   ```

4. **Run the GUI**
   ```bash
   python laplace_gui.py
   ```

## 📋 Features Overview

### 🔢 MNIST Classification Tab
- **Interactive Training**: Configure hidden units, samples, and epochs via sliders
- **Real-time Monitoring**: View training progress and results
- **Uncertainty Visualization**: Generate and display uncertainty plots
- **Model Testing**: Test predictions on MNIST test set

**Key Functions:**
- Train MNIST models with Laplace approximation
- Visualize digit classification uncertainty
- Evaluate model performance with test data

### 💳 German Credit Classification Tab
- **Flexible Configuration**: Adjust model architecture and parameters
- **Data Management**: Create sample data or load custom CSV files
- **Decision Boundary Visualization**: Generate decision boundary plots
- **Multi-class Support**: Handle 2-10 class classification problems

**Key Functions:**
- Create synthetic credit risk datasets
- Train credit classification models
- Visualize decision boundaries with uncertainty

### 🔬 Bayesian Methods Comparison Tab
- **Method Selection**: Choose from MAP, Laplace, SWAG, and HMC methods
- **Performance Comparison**: Compare accuracy and uncertainty quality
- **Uncertainty Analysis**: Generate uncertainty distribution plots
- **Results Export**: Save comparison results to JSON

**Available Methods:**
- **MAP**: Maximum A Posteriori (fastest baseline)
- **Laplace**: Laplace approximation (recommended balance)
- **SWAG**: Stochastic Weight Averaging Gaussian
- **HMC**: Hamiltonian Monte Carlo (highest quality)

### 🚨 Out-of-Distribution (OOD) Detection Tab
- **Dataset Selection**: Choose from FashionMNIST, CIFAR-10, noise, or shifted distributions
- **Detection Methods**: Multiple uncertainty quantification approaches
- **Threshold Calibration**: Automatic threshold fitting with percentile control
- **Performance Evaluation**: ROC curves and comprehensive metrics

**Detection Methods:**
- Entropy-based detection
- Maximum probability
- Predictive variance
- Mutual information

### ⚙️ Settings & Configuration Tab
- **Julia Path Management**: Configure Julia project directory
- **Visualization Settings**: Control plot quality and display options
- **System Monitoring**: Real-time log display and system status
- **Advanced Controls**: Reinitialize Julia, save logs, and more

## 🎯 Usage Guide

### Getting Started

1. **Launch the GUI**
   ```bash
   python laplace_gui.py
   ```

2. **Wait for Julia Initialization**
   - The status bar will show "Julia initialized successfully!" when ready
   - This may take 1-2 minutes on first run

3. **Start with MNIST**
   - Go to the "🔢 MNIST Classification" tab
   - Click "Train MNIST Model" with default parameters
   - Once training completes, try "Test Predictions"

### Typical Workflow

#### 1. MNIST Classification
```
1. Set parameters (Hidden Units: 50, Samples: 1000, Epochs: 20)
2. Click "Train MNIST Model"
3. Wait for training completion
4. Click "Test Predictions" to evaluate
5. Click "Show Uncertainty" for visualization
```

#### 2. Bayesian Methods Comparison
```
1. Select methods (e.g., MAP + Laplace)
2. Set samples (500) and epochs (30)
3. Click "Compare Methods"
4. View results in text area
5. Click "Plot Uncertainty" for visualization
```

#### 3. OOD Detection
```
1. First train a model in MNIST tab
2. Go to OOD Detection tab
3. Click "Load Trained Model"
4. Select OOD dataset (e.g., FashionMNIST)
5. Click "Run OOD Detection"
6. View ROC curves and performance metrics
```

## 🔧 Technical Details

### Architecture
- **Frontend**: Python with Tkinter for cross-platform GUI
- **Backend**: Julia computational engine via PyJulia
- **Threading**: Background Julia execution prevents GUI freezing
- **Visualization**: Matplotlib integration for plots and charts

### Key Components
- **LaplaceGUI Class**: Main application controller
- **Julia Integration**: Seamless Python-Julia communication
- **Tab-based Interface**: Organized functionality by domain
- **Real-time Logging**: System status and error monitoring

### File Structure
```
laplace/
├── laplace_gui.py          # Main Python GUI application
├── requirements.txt        # Python dependencies
├── GUI_README.md          # This documentation
├── wide_logo.png          # Application logo
├── src/                   # Julia source modules
├── examples/              # Julia example scripts
└── test/                  # Julia test suite
```

## 🎛️ Configuration Options

### Model Parameters
- **Hidden Units**: 2-200 (adjustable via sliders)
- **Training Samples**: 100-10,000 for MNIST
- **Epochs**: 5-500 depending on method
- **Learning Rates**: Automatically optimized per method

### Visualization Settings
- **Figure DPI**: 50-200 for plot quality control
- **Interactive Plots**: Toggle plot display
- **Real-time Updates**: Live training progress

### Advanced Settings
- **Julia Project Path**: Custom project directory
- **Log Management**: Save and clear system logs
- **Memory Management**: Automatic cleanup of large objects

## 🚨 Troubleshooting

### Common Issues

**1. "Julia not initialized yet"**
```
Solution: Wait for Julia initialization to complete (status bar indicator)
Alternative: Click "Reinitialize Julia" in Settings tab
```

**2. "Module not found" errors**
```
Solution: Ensure all Julia dependencies are installed:
julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
```

**3. PyJulia installation issues**
```
Solution: Install PyJulia with:
pip install julia
python -c "import julia; julia.install()"
```

**4. Logo not displaying**
```
Solution: Ensure wide_logo.png is in the same directory as laplace_gui.py
Install Pillow: pip install Pillow
```

**5. Slow performance**
```
Solution: Reduce sample sizes and epochs for faster execution
Use MAP method for quickest results
Enable GPU if available (automatically detected)
```

### Performance Tips

1. **Start Small**: Use default parameters before scaling up
2. **Method Selection**: 
   - MAP for quick prototyping
   - Laplace for production use
   - HMC for research quality
3. **Memory Management**: Close and restart GUI for long sessions
4. **Parallel Execution**: GUI remains responsive during computation

## 📊 Expected Results

### MNIST Performance
- **Accuracy**: 85-89% depending on method
- **Training Time**: 2-3 minutes for 1000 samples
- **OOD Detection**: AUROC > 0.90 for most datasets

### German Credit Performance  
- **Training Time**: < 1 minute for 100 samples
- **Visualization**: Clear decision boundaries
- **Uncertainty**: High uncertainty at class boundaries

### Method Comparison
- **MAP**: Fastest, basic uncertainty
- **Laplace**: Good speed/quality balance  
- **SWAG**: Medium speed, good approximation
- **HMC**: Slowest, highest quality

## 🎓 Educational Use

This GUI is designed for:
- **Learning Bayesian Neural Networks**: Interactive exploration of concepts
- **Research**: Comparing different inference methods
- **Teaching**: Demonstrating uncertainty quantification
- **Prototyping**: Quick testing of Bayesian approaches

## 🔗 Integration with Julia Framework

The GUI seamlessly integrates with all Julia modules:
- `MNISTClassifier.jl`: MNIST digit classification
- `GermanCreditClassifier.jl`: Credit risk assessment
- `BayesianMethods.jl`: Multiple inference methods
- `OODDetection.jl`: Out-of-distribution detection
- `Visualizations.jl`: Advanced plotting capabilities

## 📞 Support

For issues with:
- **GUI Problems**: Check GUI_README.md and troubleshooting section
- **Julia Errors**: See main README.md and Julia documentation
- **Method Questions**: Refer to academic papers and Julia module documentation

## 🎯 Next Steps

After familiarizing yourself with the GUI:
1. **Explore Parameters**: Try different configurations
2. **Custom Data**: Load your own datasets
3. **Method Comparison**: Run comprehensive benchmarks  
4. **Research Applications**: Apply to your specific use cases
5. **Extend Functionality**: Add custom methods or visualizations

The GUI provides a user-friendly entry point to the powerful Julia-based Bayesian neural network framework, making advanced uncertainty quantification accessible to users of all levels.