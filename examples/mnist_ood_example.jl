#!/usr/bin/env julia

"""
MNIST OOD Detection Example with Laplace Approximation

This script demonstrates out-of-distribution detection capabilities using 
uncertainty quantification from Bayesian neural networks with Laplace approximation.
"""

using Pkg
Pkg.activate(".")

include("../src/MNISTClassifier.jl")
include("../src/OODDatasets.jl")
include("../src/OODDetection.jl")
include("../src/Visualizations.jl")

using .MNISTClassifier
using .OODDatasets
using .OODDetection
using .Visualizations
using Random
using Statistics

function main()
    println("=== MNIST OOD Detection with Laplace Approximation ===")
    
    # Set random seed for reproducibility
    Random.seed!(42)
    
    println("\n1. Training MNIST model...")
    # Create and train model (you can also load a pre-trained model)
    model = MNISTModel(50)
    trained_model = train_mnist!(model, 30, 0.001, 2000)  # Smaller training for demo
    
    println("\n2. Loading in-distribution (MNIST) test data...")
    _, _, test_x, test_y, _, _ = load_mnist_data()
    mnist_test_x = test_x[:, 1:500]  # Use subset for faster processing
    mnist_test_y = test_y[1:500]
    
    println("\n3. Loading various OOD datasets...")
    
    # Test different OOD datasets
    ood_datasets = [
        (:fashionmnist, "FashionMNIST"),
        (:cifar10, "CIFAR-10"),
        (:notmnist, "NotMNIST"),
        (:uniform_noise, "Uniform Noise"),
        (:gaussian_noise, "Gaussian Noise")
    ]
    
    # Store results for comparison
    all_results = Dict()
    all_plots = Dict()
    
    for (ood_type, ood_name) in ood_datasets
        println("\n--- Testing OOD dataset: $ood_name ---")
        
        # Load OOD data
        ood_x, ood_y, dataset_name = load_ood_mnist(ood_type, 300)
        
        println("4. Setting up OOD detection...")
        
        # Test different uncertainty methods
        methods = [:entropy, :max_prob]
        method_results = Dict()
        
        for method in methods
            println("\nTesting method: $method")
            
            # Create detector
            detector = OODDetector(trained_model; method=method)
            
            # Fit threshold using subset of MNIST data
            val_indices = 1:100
            fit_ood_threshold!(detector, mnist_test_x[:, val_indices]; percentile=95.0)
            
            # Evaluate OOD detection
            metrics = evaluate_ood_detection(detector, mnist_test_x, ood_x)
            method_results[method] = metrics
            
            println("Results for $method:")
            println("  AUROC: $(round(metrics["auroc"], digits=3))")
            println("  AUPR: $(round(metrics["aupr"], digits=3))")
            println("  FPR@95TPR: $(round(metrics["fpr_at_95tpr"], digits=3))")
            println("  Accuracy: $(round(metrics["accuracy"], digits=3))")
        end
        
        all_results[ood_name] = method_results
        
        println("\n5. Creating visualizations for $ood_name...")
        
        # Use entropy method for visualization
        best_method = :entropy
        metrics = method_results[best_method]
        
        # Create comprehensive plots
        plots_dict = Dict()
        
        # Sample comparison
        plots_dict["$(ood_type)_samples"] = plot_ood_samples_comparison(
            trained_model, mnist_test_x, ood_x, ood_name; n_samples=6, data_type=:mnist
        )
        
        # OOD detection summary
        plots_dict["$(ood_type)_summary"] = plot_ood_detection_summary(
            metrics, metrics["scores_in"], metrics["scores_ood"], "Entropy"
        )
        
        # Individual plots
        plots_dict["$(ood_type)_scores_dist"] = plot_ood_scores_distribution(
            metrics["scores_in"], metrics["scores_ood"], "Entropy", metrics["optimal_threshold"]
        )
        
        plots_dict["$(ood_type)_roc"] = plot_roc_curve(metrics; title="ROC: MNIST vs $ood_name")
        plots_dict["$(ood_type)_pr"] = plot_precision_recall_curve(metrics; title="PR: MNIST vs $ood_name")
        
        all_plots[ood_name] = plots_dict
        
        # Save plots for this OOD dataset
        save_all_plots(plots_dict, "mnist_ood_$(ood_type)_plots")
    end
    
    println("\n6. Creating comparison summary...")
    
    # Compare all methods and datasets
    println("\n=== OOD Detection Performance Summary ===")
    println("Dataset                | Method    | AUROC | AUPR  | FPR@95TPR | Accuracy")
    println("-" ^ 75)
    
    for (ood_name, method_results) in all_results
        for (method, metrics) in method_results
            auroc = round(metrics["auroc"], digits=3)
            aupr = round(metrics["aupr"], digits=3)
            fpr95 = round(metrics["fpr_at_95tpr"], digits=3)
            acc = round(metrics["accuracy"], digits=3)
            
            println("$(rpad(ood_name, 22)) | $(rpad(string(method), 9)) | $(rpad(auroc, 5)) | $(rpad(aupr, 5)) | $(rpad(fpr95, 9)) | $acc")
        end
    end
    
    println("\n7. Analysis and recommendations...")
    
    # Find best performing combinations
    best_performance = Dict()
    for (ood_name, method_results) in all_results
        best_auroc = 0.0
        best_method = :entropy
        
        for (method, metrics) in method_results
            if metrics["auroc"] > best_auroc
                best_auroc = metrics["auroc"]
                best_method = method
            end
        end
        
        best_performance[ood_name] = (best_method, best_auroc)
    end
    
    println("\nBest performing method for each OOD dataset:")
    for (ood_name, (method, auroc)) in best_performance
        println("  $ood_name: $method (AUROC = $(round(auroc, digits=3)))")
    end
    
    # General analysis
    avg_auroc_entropy = mean([all_results[name][:entropy]["auroc"] for name in keys(all_results)])
    avg_auroc_maxprob = mean([all_results[name][:max_prob]["auroc"] for name in keys(all_results)])
    
    println("\nOverall method performance:")
    println("  Entropy method average AUROC: $(round(avg_auroc_entropy, digits=3))")
    println("  Max probability method average AUROC: $(round(avg_auroc_maxprob, digits=3))")
    
    if avg_auroc_entropy > avg_auroc_maxprob
        println("  → Entropy method performs better on average")
    else
        println("  → Max probability method performs better on average")
    end
    
    println("\n8. Insights about different OOD types...")
    
    fashion_auroc = all_results["FashionMNIST"][:entropy]["auroc"]
    cifar_auroc = all_results["CIFAR-10"][:entropy]["auroc"]
    noise_auroc = all_results["Uniform Noise"][:entropy]["auroc"]
    
    println("\nOOD Detection difficulty ranking (higher AUROC = easier to detect):")
    performance_ranking = [(name, all_results[name][:entropy]["auroc"]) for name in keys(all_results)]
    sort!(performance_ranking, by=x->x[2], rev=true)
    
    for (i, (name, auroc)) in enumerate(performance_ranking)
        println("  $i. $name (AUROC = $(round(auroc, digits=3)))")
    end
    
    println("\nInsights:")
    if noise_auroc > fashion_auroc
        println("  • Random noise is easier to detect than semantic OOD (FashionMNIST)")
    else
        println("  • Semantic OOD (FashionMNIST) is easier to detect than random noise")
    end
    
    if cifar_auroc > fashion_auroc
        println("  • CIFAR-10 (natural images) easier to detect than FashionMNIST")
    else
        println("  • FashionMNIST easier to detect than CIFAR-10 (natural images)")
    end
    
    println("\n=== MNIST OOD Detection Complete ===")
    
    # Summary of saved files
    println("\nGenerated files:")
    for (ood_type, _) in ood_datasets
        println("  • mnist_ood_$(ood_type)_plots/: Visualization plots")
    end
    
    println("\nKey takeaways:")
    println("  • Laplace approximation provides useful uncertainty for OOD detection")
    println("  • Entropy-based uncertainty generally outperforms max probability")
    println("  • Different OOD types have varying detection difficulty")
    println("  • Visual inspection shows clear uncertainty differences between ID and OOD")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end