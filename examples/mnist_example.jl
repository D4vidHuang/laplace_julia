#!/usr/bin/env julia

"""
MNIST Classification Example with Laplace Approximation

This script demonstrates how to use the MNIST classifier with Laplace approximation
for uncertainty quantification in neural network predictions.
"""

using Pkg
Pkg.activate(".")

include("../src/MNISTClassifier.jl")
include("../src/Visualizations.jl")
using .MNISTClassifier
using .Visualizations
using Random
using Statistics

function main()
    println("=== MNIST Classification with Laplace Approximation ===")
    
    # Set random seed for reproducibility
    Random.seed!(42)
    
    # Create and train model
    println("\n1. Creating MNIST model...")
    model = MNISTModel(50)  # 50 hidden units as in notebook
    
    println("\n2. Training model (this may take a few minutes)...")
    trained_model = train_mnist!(model, 50, 0.001, 5000)  # 50 epochs, lr=0.001, 5000 samples
    
    println("\n3. Evaluating model...")
    results = evaluate_mnist(trained_model, 1000)  # Evaluate on 1000 test samples
    
    println("\n=== Results ===")
    println("Test Accuracy: $(round(results["accuracy"] * 100, digits=2))%")
    println("Average Entropy: $(round(results["avg_entropy"], digits=3))")
    println("Entropy (Correct): $(round(results["correct_avg_entropy"], digits=3))")
    println("Entropy (Incorrect): $(round(results["incorrect_avg_entropy"], digits=3))")
    println("Uncertainty Difference: $(round(results["incorrect_avg_entropy"] - results["correct_avg_entropy"], digits=3))")
    
    # Class-wise accuracy
    println("\n=== Per-Class Accuracy ===")
    for digit in 0:9
        mask = results["true_classes"] .== digit
        if sum(mask) > 0
            digit_accuracy = mean(results["predicted_classes"][mask] .== digit)
            println("Digit $digit: $(round(digit_accuracy * 100, digits=1))% ($(sum(mask)) samples)")
        end
    end
    
    println("\n4. Creating visualizations...")
    
    # Load test data for visualization
    _, _, test_x, test_y, _, _ = load_mnist_data()
    
    # Create plots
    plots_dict = Dict()
    
    # Sample predictions plot
    plots_dict["mnist_samples"] = plot_mnist_samples(trained_model, test_x, test_y, 6)
    
    # Uncertainty histogram
    correct_mask = results["predicted_classes"] .== results["true_classes"]
    plots_dict["uncertainty_hist"] = plot_uncertainty_histogram(
        results["entropies"], correct_mask, "MNIST Uncertainty Distribution"
    )
    
    # Prediction method comparison
    plots_dict["method_comparison"] = plot_prediction_comparison(trained_model, test_x, 100)
    
    # Save all plots
    save_all_plots(plots_dict, "mnist_plots")
    
    println("\n5. Comparing Laplace vs Plugin predictions...")
    
    # Compare prediction methods
    test_subset = test_x[:, 1:100]
    laplace_preds = predict_mnist(trained_model, test_subset; link_approx=:probit)
    plugin_preds = predict_mnist(trained_model, test_subset; link_approx=:plugin)
    
    laplace_entropies = [-sum(p .* log.(p .+ 1e-8)) for p in laplace_preds]
    plugin_entropies = [-sum(p .* log.(p .+ 1e-8)) for p in plugin_preds]
    
    println("Laplace method - Average entropy: $(round(mean(laplace_entropies), digits=3))")
    println("Plugin method - Average entropy: $(round(mean(plugin_entropies), digits=3))")
    println("Method difference: $(round(mean(laplace_entropies) - mean(plugin_entropies), digits=3))")
    
    # Find samples where methods disagree most
    differences = abs.(laplace_entropies .- plugin_entropies)
    top_diff_indices = sortperm(differences, rev=true)[1:5]
    
    println("\nTop 5 samples with largest method differences:")
    for (i, idx) in enumerate(top_diff_indices)
        laplace_pred = argmax(laplace_preds[idx]) - 1
        plugin_pred = argmax(plugin_preds[idx]) - 1
        true_label = test_y[idx]
        println("Sample $idx: True=$true_label, Laplace=$laplace_pred, Plugin=$plugin_pred, Diff=$(round(differences[idx], digits=3))")
    end
    
    println("\n=== MNIST Classification Complete ===")
    println("Plots saved in: mnist_plots/")
    println("- mnist_samples.png: Sample digit predictions")
    println("- uncertainty_hist.png: Uncertainty distribution")
    println("- method_comparison.png: Laplace vs Plugin comparison")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end