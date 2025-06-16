#!/usr/bin/env python3
"""
🔬 Laplace Approximation Bayesian Neural Networks - Python GUI
GUI for Julia-based Bayesian neural network framework
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import tkinter.font as tkFont
import threading
import queue
import os
import sys
from PIL import Image, ImageTk
import julia
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class LaplaceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🔬 Laplace Approximation Bayesian Neural Networks")
        self.root.geometry("1200x800")
        
        # Initialize Julia
        self.julia_initialized = False
        self.julia_queue = queue.Queue()
        
        # Load and display logo
        self.setup_logo()
        
        # Create GUI elements
        self.setup_gui()
        
        # Initialize Julia in background
        self.init_julia_background()
    
    def setup_logo(self):
        """Load and setup the logo"""
        try:
            logo_path = os.path.join(os.path.dirname(__file__), "wide_logo.png")
            if os.path.exists(logo_path):
                image = Image.open(logo_path)
                # Resize logo to fit header
                image = image.resize((300, 80), Image.Resampling.LANCZOS)
                self.logo = ImageTk.PhotoImage(image)
            else:
                self.logo = None
        except Exception as e:
            print(f"Could not load logo: {e}")
            self.logo = None
    
    def setup_gui(self):
        """Setup the main GUI elements"""
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Header frame with logo
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=10, pady=5)
        
        if self.logo:
            logo_label = ttk.Label(header_frame, image=self.logo)
            logo_label.pack(side=tk.LEFT)
        
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        title_font = tkFont.Font(size=16, weight="bold")
        ttk.Label(title_frame, text="Laplace Approximation Bayesian Neural Networks", 
                 font=title_font).pack(anchor=tk.E)
        ttk.Label(title_frame, text="Python GUI for Julia Framework", 
                 font=tkFont.Font(size=10)).pack(anchor=tk.E)
        
        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create tabs
        self.create_mnist_tab()
        self.create_german_credit_tab()
        self.create_bayesian_methods_tab()
        self.create_ood_detection_tab()
        self.create_settings_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Initializing Julia...")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_mnist_tab(self):
        """Create MNIST classification tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🔢 MNIST Classification")
        
        # Left panel for controls
        left_frame = ttk.LabelFrame(frame, text="MNIST Configuration")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Parameters
        ttk.Label(left_frame, text="Hidden Units:").pack(anchor=tk.W)
        self.mnist_hidden = tk.IntVar(value=50)
        ttk.Scale(left_frame, from_=10, to=200, variable=self.mnist_hidden, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        ttk.Label(left_frame, textvariable=self.mnist_hidden).pack()
        
        ttk.Label(left_frame, text="Training Samples:").pack(anchor=tk.W, pady=(10,0))
        self.mnist_samples = tk.IntVar(value=1000)
        ttk.Scale(left_frame, from_=100, to=10000, variable=self.mnist_samples, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        ttk.Label(left_frame, textvariable=self.mnist_samples).pack()
        
        ttk.Label(left_frame, text="Epochs:").pack(anchor=tk.W, pady=(10,0))
        self.mnist_epochs = tk.IntVar(value=20)
        ttk.Scale(left_frame, from_=5, to=100, variable=self.mnist_epochs, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        ttk.Label(left_frame, textvariable=self.mnist_epochs).pack()
        
        # Buttons
        ttk.Button(left_frame, text="Train MNIST Model", 
                  command=self.train_mnist_model).pack(fill=tk.X, pady=10)
        ttk.Button(left_frame, text="Test Predictions", 
                  command=self.test_mnist_predictions).pack(fill=tk.X, pady=2)
        ttk.Button(left_frame, text="Show Uncertainty", 
                  command=self.show_mnist_uncertainty).pack(fill=tk.X, pady=2)
        
        # Right panel for results
        right_frame = ttk.LabelFrame(frame, text="Results & Visualization")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results text area
        self.mnist_results = scrolledtext.ScrolledText(right_frame, height=15)
        self.mnist_results.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Matplotlib figure for visualization
        self.mnist_fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.mnist_canvas = FigureCanvasTkAgg(self.mnist_fig, right_frame)
        self.mnist_canvas.draw()
        self.mnist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_german_credit_tab(self):
        """Create German Credit classification tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="💳 German Credit")
        
        # Left panel for controls
        left_frame = ttk.LabelFrame(frame, text="German Credit Configuration")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Parameters
        ttk.Label(left_frame, text="Hidden Units:").pack(anchor=tk.W)
        self.gc_hidden = tk.IntVar(value=3)
        ttk.Scale(left_frame, from_=2, to=20, variable=self.gc_hidden, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        ttk.Label(left_frame, textvariable=self.gc_hidden).pack()
        
        ttk.Label(left_frame, text="Classes:").pack(anchor=tk.W, pady=(10,0))
        self.gc_classes = tk.IntVar(value=4)
        ttk.Scale(left_frame, from_=2, to=10, variable=self.gc_classes, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        ttk.Label(left_frame, textvariable=self.gc_classes).pack()
        
        ttk.Label(left_frame, text="Epochs:").pack(anchor=tk.W, pady=(10,0))
        self.gc_epochs = tk.IntVar(value=100)
        ttk.Scale(left_frame, from_=50, to=500, variable=self.gc_epochs, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        ttk.Label(left_frame, textvariable=self.gc_epochs).pack()
        
        # Data file selection
        ttk.Label(left_frame, text="Data File:").pack(anchor=tk.W, pady=(10,0))
        self.gc_file = tk.StringVar(value="data1.csv")
        ttk.Entry(left_frame, textvariable=self.gc_file).pack(fill=tk.X, pady=2)
        ttk.Button(left_frame, text="Browse", 
                  command=self.browse_gc_file).pack(fill=tk.X, pady=2)
        
        # Buttons
        ttk.Button(left_frame, text="Create Sample Data", 
                  command=self.create_gc_sample_data).pack(fill=tk.X, pady=10)
        ttk.Button(left_frame, text="Train Model", 
                  command=self.train_gc_model).pack(fill=tk.X, pady=2)
        ttk.Button(left_frame, text="Visualize Decision Boundary", 
                  command=self.visualize_gc_boundary).pack(fill=tk.X, pady=2)
        
        # Right panel for results
        right_frame = ttk.LabelFrame(frame, text="Results & Visualization")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results text area
        self.gc_results = scrolledtext.ScrolledText(right_frame, height=15)
        self.gc_results.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Matplotlib figure for visualization
        self.gc_fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.gc_canvas = FigureCanvasTkAgg(self.gc_fig, right_frame)
        self.gc_canvas.draw()
        self.gc_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_bayesian_methods_tab(self):
        """Create Bayesian Methods comparison tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🔬 Bayesian Methods")
        
        # Left panel for controls
        left_frame = ttk.LabelFrame(frame, text="Method Configuration")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Method selection
        ttk.Label(left_frame, text="Select Methods:").pack(anchor=tk.W)
        self.methods = {
            "MAP": tk.BooleanVar(value=True),
            "Laplace": tk.BooleanVar(value=True),
            "SWAG": tk.BooleanVar(value=False),
            "HMC": tk.BooleanVar(value=False)
        }
        
        for method, var in self.methods.items():
            ttk.Checkbutton(left_frame, text=method, variable=var).pack(anchor=tk.W)
        
        # Parameters
        ttk.Label(left_frame, text="Training Samples:").pack(anchor=tk.W, pady=(10,0))
        self.bm_samples = tk.IntVar(value=500)
        ttk.Scale(left_frame, from_=100, to=2000, variable=self.bm_samples, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        ttk.Label(left_frame, textvariable=self.bm_samples).pack()
        
        ttk.Label(left_frame, text="Epochs:").pack(anchor=tk.W, pady=(10,0))
        self.bm_epochs = tk.IntVar(value=30)
        ttk.Scale(left_frame, from_=10, to=100, variable=self.bm_epochs, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        ttk.Label(left_frame, textvariable=self.bm_epochs).pack()
        
        # Buttons
        ttk.Button(left_frame, text="Compare Methods", 
                  command=self.compare_bayesian_methods).pack(fill=tk.X, pady=10)
        ttk.Button(left_frame, text="Plot Uncertainty", 
                  command=self.plot_bayesian_uncertainty).pack(fill=tk.X, pady=2)
        ttk.Button(left_frame, text="Save Results", 
                  command=self.save_bayesian_results).pack(fill=tk.X, pady=2)
        
        # Right panel for results
        right_frame = ttk.LabelFrame(frame, text="Comparison Results")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results text area
        self.bm_results = scrolledtext.ScrolledText(right_frame, height=15)
        self.bm_results.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Matplotlib figure for visualization
        self.bm_fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.bm_canvas = FigureCanvasTkAgg(self.bm_fig, right_frame)
        self.bm_canvas.draw()
        self.bm_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_ood_detection_tab(self):
        """Create Out-of-Distribution detection tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🚨 OOD Detection")
        
        # Left panel for controls
        left_frame = ttk.LabelFrame(frame, text="OOD Detection Configuration")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # OOD dataset selection
        ttk.Label(left_frame, text="OOD Dataset:").pack(anchor=tk.W)
        self.ood_dataset = tk.StringVar(value="FashionMNIST")
        ood_combo = ttk.Combobox(left_frame, textvariable=self.ood_dataset, 
                                values=["FashionMNIST", "CIFAR10", "Noise", "Shifted"])
        ood_combo.pack(fill=tk.X, pady=2)
        
        # Detection method
        ttk.Label(left_frame, text="Detection Method:").pack(anchor=tk.W, pady=(10,0))
        self.ood_method = tk.StringVar(value="entropy")
        method_combo = ttk.Combobox(left_frame, textvariable=self.ood_method, 
                                   values=["entropy", "max_prob", "variance", "mutual_info"])
        method_combo.pack(fill=tk.X, pady=2)
        
        # Threshold percentile
        ttk.Label(left_frame, text="Threshold Percentile:").pack(anchor=tk.W, pady=(10,0))
        self.ood_threshold = tk.DoubleVar(value=95.0)
        ttk.Scale(left_frame, from_=80.0, to=99.0, variable=self.ood_threshold, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        ttk.Label(left_frame, textvariable=self.ood_threshold).pack()
        
        # Buttons
        ttk.Button(left_frame, text="Load Trained Model", 
                  command=self.load_ood_model).pack(fill=tk.X, pady=10)
        ttk.Button(left_frame, text="Run OOD Detection", 
                  command=self.run_ood_detection).pack(fill=tk.X, pady=2)
        ttk.Button(left_frame, text="Plot ROC Curve", 
                  command=self.plot_ood_roc).pack(fill=tk.X, pady=2)
        ttk.Button(left_frame, text="Evaluate Performance", 
                  command=self.evaluate_ood_performance).pack(fill=tk.X, pady=2)
        
        # Right panel for results
        right_frame = ttk.LabelFrame(frame, text="OOD Detection Results")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results text area
        self.ood_results = scrolledtext.ScrolledText(right_frame, height=15)
        self.ood_results.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Matplotlib figure for visualization
        self.ood_fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.ood_canvas = FigureCanvasTkAgg(self.ood_fig, right_frame)
        self.ood_canvas.draw()
        self.ood_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_settings_tab(self):
        """Create settings and configuration tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="⚙️ Settings")
        
        # Julia configuration
        julia_frame = ttk.LabelFrame(frame, text="Julia Configuration")
        julia_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(julia_frame, text="Julia Project Path:").pack(anchor=tk.W)
        self.julia_path = tk.StringVar(value=os.path.dirname(__file__))
        ttk.Entry(julia_frame, textvariable=self.julia_path).pack(fill=tk.X, pady=2)
        ttk.Button(julia_frame, text="Browse", 
                  command=self.browse_julia_path).pack(anchor=tk.W, pady=2)
        ttk.Button(julia_frame, text="Reinitialize Julia", 
                  command=self.reinit_julia).pack(anchor=tk.W, pady=2)
        
        # Visualization settings
        viz_frame = ttk.LabelFrame(frame, text="Visualization Settings")
        viz_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(viz_frame, text="Figure DPI:").pack(anchor=tk.W)
        self.figure_dpi = tk.IntVar(value=100)
        ttk.Scale(viz_frame, from_=50, to=200, variable=self.figure_dpi, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        
        self.show_plots = tk.BooleanVar(value=True)
        ttk.Checkbutton(viz_frame, text="Show interactive plots", 
                       variable=self.show_plots).pack(anchor=tk.W)
        
        # Log display
        log_frame = ttk.LabelFrame(frame, text="System Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Control buttons
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text="Clear Log", 
                  command=self.clear_log).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Save Log", 
                  command=self.save_log).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="About", 
                  command=self.show_about).pack(side=tk.RIGHT, padx=2)
    
    def init_julia_background(self):
        """Initialize Julia in background thread"""
        def init_julia():
            try:
                self.log("Initializing Julia...")
                
                # First, ensure Python's julia package is properly installed
                import julia
                try:
                    julia.install()
                except Exception as e:
                    self.log(f"Warning: Julia installation check failed: {e}")
                
                # Set environment variables for Julia
                os.environ["JULIA_NUM_THREADS"] = "1"
                os.environ["JULIA_SSL_NO_VERIFY"] = "1"
                
                # Initialize Julia with specific options
                self.j = julia.Julia(
                    compiled_modules=False,
                    runtime="julia",
                    debug=False,
                    sysimage=None,
                    optimization_level=0
                )
                
                # Run initialization script
                init_script = os.path.join(os.path.dirname(__file__), "init_julia.jl")
                if not os.path.exists(init_script):
                    raise Exception(f"Initialization script not found: {init_script}")
                
                self.log("Running Julia initialization script...")
                self.j.eval(f'include("{init_script}")')
                
                # Activate Julia project
                julia_path = self.julia_path.get()
                self.j.eval(f'using Pkg; Pkg.activate("{julia_path}")')
                
                # Import necessary modules with error handling
                modules = [
                    "src/MNISTClassifier.jl",
                    "src/GermanCreditClassifier.jl",
                    "src/BayesianMethods.jl",
                    "src/OODDetection.jl",
                    "src/Visualizations.jl",
                    "src/GUIInterface.jl"
                ]
                
                for module in modules:
                    try:
                        self.j.eval(f'include("{module}")')
                    except Exception as e:
                        self.log(f"Warning: Failed to load module {module}: {e}")
                
                # Import modules with error handling
                module_names = [
                    "MNISTClassifier",
                    "GermanCreditClassifier",
                    "BayesianMethods",
                    "OODDetection",
                    "Visualizations",
                    "GUIInterface"
                ]
                
                for name in module_names:
                    try:
                        self.j.eval(f'using .{name}')
                    except Exception as e:
                        self.log(f"Warning: Failed to import module {name}: {e}")
                
                self.julia_initialized = True
                self.julia_queue.put(("status", "Julia initialized successfully!"))
                self.log("Julia initialization complete.")
                
            except Exception as e:
                error_msg = f"Julia initialization failed: {str(e)}"
                self.julia_queue.put(("error", error_msg))
                self.log(error_msg)
                self.julia_initialized = False
        
        # Start Julia initialization in background
        julia_thread = threading.Thread(target=init_julia, daemon=True)
        julia_thread.start()
        
        # Check for Julia initialization completion
        self.root.after(1000, self.check_julia_queue)
    
    def check_julia_queue(self):
        """Check Julia initialization queue"""
        try:
            while True:
                msg_type, message = self.julia_queue.get_nowait()
                if msg_type == "status":
                    self.status_var.set(message)
                elif msg_type == "error":
                    self.status_var.set(message)
                    messagebox.showerror("Julia Error", message)
        except queue.Empty:
            pass
        
        if not self.julia_initialized:
            self.root.after(1000, self.check_julia_queue)
    
    def log(self, message):
        """Add message to log"""
        if hasattr(self, 'log_text'):
            self.log_text.insert(tk.END, f"{message}\n")
            self.log_text.see(tk.END)
        print(message)  # Also print to console
    
    def run_julia_command(self, command, result_widget=None):
        """Run Julia command in background thread"""
        def run_command():
            try:
                if not self.julia_initialized:
                    raise Exception("Julia not initialized yet")
                
                self.log(f"Running: {command}")
                result = self.j.eval(command)
                
                if result_widget:
                    result_widget.insert(tk.END, f"Command: {command}\n")
                    result_widget.insert(tk.END, f"Result: {result}\n\n")
                    result_widget.see(tk.END)
                
                return result
                
            except Exception as e:
                error_msg = f"Error executing Julia command: {e}"
                self.log(error_msg)
                if result_widget:
                    result_widget.insert(tk.END, f"Error: {error_msg}\n\n")
                    result_widget.see(tk.END)
                messagebox.showerror("Julia Error", error_msg)
        
        # Run in background thread
        command_thread = threading.Thread(target=run_command, daemon=True)
        command_thread.start()
    
    # MNIST Tab Methods
    def train_mnist_model(self):
        """Train MNIST model"""
        if not self.julia_initialized:
            messagebox.showwarning("Warning", "Julia not initialized yet")
            return
        
        hidden = self.mnist_hidden.get()
        samples = self.mnist_samples.get()
        epochs = self.mnist_epochs.get()
        
        command = f'train_mnist!({hidden}, {epochs}, 0.001, {samples})'
        
        self.run_julia_command(command, self.mnist_results)
    
    def test_mnist_predictions(self):
        """Test MNIST predictions"""
        command = 'predict_mnist_gui(100)'
        
        self.run_julia_command(command, self.mnist_results)
    
    def show_mnist_uncertainty(self):
        """Show MNIST uncertainty visualization"""
        command = 'plot_mnist_uncertainty(10)'
        
        self.run_julia_command(command, self.mnist_results)
    
    # German Credit Tab Methods
    def browse_gc_file(self):
        """Browse for German Credit data file"""
        filename = filedialog.askopenfilename(
            title="Select German Credit Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.gc_file.set(filename)
    
    def create_gc_sample_data(self):
        """Create sample German Credit data"""
        filename = self.gc_file.get()
        command = f'create_sample_data("{filename}")'
        self.run_julia_command(command, self.gc_results)
    
    def train_gc_model(self):
        """Train German Credit model"""
        if not self.julia_initialized:
            messagebox.showwarning("Warning", "Julia not initialized yet")
            return
        
        hidden = self.gc_hidden.get()
        classes = self.gc_classes.get()
        epochs = self.gc_epochs.get()
        filename = self.gc_file.get()
        
        command = f'train_german_credit!({hidden}, 2, {classes}, "{filename}", {epochs})'
        
        self.run_julia_command(command, self.gc_results)
    
    def visualize_gc_boundary(self):
        """Visualize German Credit decision boundary"""
        command = f'visualize_decision_boundary("{self.gc_file.get()}")'
        
        self.run_julia_command(command, self.gc_results)
    
    # Bayesian Methods Tab Methods
    def compare_bayesian_methods(self):
        """Compare Bayesian methods"""
        if not self.julia_initialized:
            messagebox.showwarning("Warning", "Julia not initialized yet")
            return
        
        selected_methods = [method for method, var in self.methods.items() if var.get()]
        samples = self.bm_samples.get()
        epochs = self.bm_epochs.get()
        
        if not selected_methods:
            messagebox.showwarning("Warning", "Please select at least one method")
            return
        
        methods_str = '["' + '", "'.join(selected_methods) + '"]'
        command = f'compare_bayesian_methods_gui({methods_str}, {samples}, {epochs})'
        
        self.run_julia_command(command, self.bm_results)
    
    def plot_bayesian_uncertainty(self):
        """Plot Bayesian uncertainty"""
        command = 'plot_uncertainty_distributions_gui()'
        
        self.run_julia_command(command, self.bm_results)
    
    def save_bayesian_results(self):
        """Save Bayesian results"""
        filename = filedialog.asksaveasfilename(
            title="Save Bayesian Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            command = f'save_bayesian_results_gui("{filename}")'
            self.run_julia_command(command, self.bm_results)
    
    # OOD Detection Tab Methods
    def load_ood_model(self):
        """Load trained model for OOD detection"""
        command = 'create_ood_detector("model", "entropy")'
        
        self.run_julia_command(command, self.ood_results)
    
    def run_ood_detection(self):
        """Run OOD detection"""
        if not self.julia_initialized:
            messagebox.showwarning("Warning", "Julia not initialized yet")
            return
        
        dataset = self.ood_dataset.get()
        method = self.ood_method.get()
        threshold = self.ood_threshold.get()
        
        command = f'run_comprehensive_ood_demo("detector", "{dataset}", {threshold})'
        
        self.run_julia_command(command, self.ood_results)
    
    def plot_ood_roc(self):
        """Plot OOD ROC curve"""
        command = 'plot_ood_roc_curve("results")'
        
        self.run_julia_command(command, self.ood_results)
    
    def evaluate_ood_performance(self):
        """Evaluate OOD detection performance"""
        command = 'evaluate_ood_performance("results")'
        
        self.run_julia_command(command, self.ood_results)
    
    # Settings Tab Methods
    def browse_julia_path(self):
        """Browse for Julia project path"""
        directory = filedialog.askdirectory(title="Select Julia Project Directory")
        if directory:
            self.julia_path.set(directory)
    
    def reinit_julia(self):
        """Reinitialize Julia"""
        self.julia_initialized = False
        self.status_var.set("Reinitializing Julia...")
        self.init_julia_background()
    
    def clear_log(self):
        """Clear log text"""
        self.log_text.delete(1.0, tk.END)
    
    def save_log(self):
        """Save log to file"""
        filename = filedialog.asksaveasfilename(
            title="Save Log",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w') as f:
                f.write(self.log_text.get(1.0, tk.END))
            messagebox.showinfo("Success", f"Log saved to {filename}")
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
🔬 Laplace Approximation Bayesian Neural Networks
Python GUI for Julia Framework

This GUI provides an interface to the Julia-based Bayesian neural network
framework featuring:

• MNIST digit classification with Laplace approximation
• German Credit risk classification
• Multiple Bayesian inference methods (MAP, Laplace, SWAG, HMC)
• Out-of-distribution detection
• Advanced uncertainty quantification
• Interactive visualizations

Developed with Python/Tkinter frontend and Julia computational backend.
        """
        messagebox.showinfo("About", about_text)

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = LaplaceGUI(root)
    
    # Handle window closing
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()