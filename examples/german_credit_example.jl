#!/usr/bin/env julia

"""
German Credit Classification Example with Laplace Approximation

This script demonstrates how to use the German Credit classifier with Laplace approximation
for uncertainty quantification in neural network predictions.
"""

using Pkg
Pkg.activate(".")

include("../src/GermanCreditClassifier.jl")
include("../src/Visualizations.jl")
using .GermanCreditClassifier
using .Visualizations
using Random
using Statistics

function main()
    println("=== German Credit Classification with Laplace Approximation ===")
    
    # Set random seed for reproducibility
    Random.seed!(42)
    
    # Create sample data if it doesn't exist
    data_file = "german_credit_data.csv"
    if !isfile(data_file)
        println("\n1. Creating sample German Credit dataset...")
        create_sample_data(data_file)
    else
        println("\n1. Using existing data file: $data_file")
    end
    
    println("\n2. Creating German Credit model...")
    model = GermanCreditModel(3, 2, 4)  # 3 hidden units, 2 input features, 4 classes
    
    println("\n3. Training model...")
    trained_model = train_german_credit!(model, data_file, 200)  # 200 epochs as in notebook
    
    println("\n4. Evaluating model...")
    results = evaluate_german_credit(trained_model, data_file)
    
    println("\n=== Results ===")
    println("Training Accuracy: $(round(results["accuracy"] * 100, digits=2))%")
    println("Average Entropy: $(round(results["avg_entropy"], digits=3))")
    println("Entropy (Correct): $(round(results["correct_avg_entropy"], digits=3))")
    println("Entropy (Incorrect): $(round(results["incorrect_avg_entropy"], digits=3))")
    println("Uncertainty Difference: $(round(results["incorrect_avg_entropy"] - results["correct_avg_entropy"], digits=3))")
    
    # Class-wise accuracy
    println("\n=== Per-Class Accuracy ===")
    for class_label in results["unique_labels"]
        mask = results["true_labels"] .== class_label
        if sum(mask) > 0
            class_accuracy = mean(results["predicted_labels"][mask] .== class_label)
            println("Class $class_label: $(round(class_accuracy * 100, digits=1))% ($(sum(mask)) samples)")
        end
    end
    
    println("\n5. Creating visualizations...")
    
    # Load data for visualization
    X, y_labels, _, _, unique_labels = load_german_credit_data(data_file)
    
    # Create plots
    plots_dict = Dict()
    
    # Decision boundary plots for each class
    for class_label in unique_labels
        plots_dict["decision_boundary_class_$class_label"] = plot_german_credit_decision_boundary(
            trained_model, X, y_labels, unique_labels; target_class=class_label
        )
    end
    
    # Overall decision boundary
    plots_dict["decision_boundary_all"] = plot_german_credit_decision_boundary(
        trained_model, X, y_labels, unique_labels
    )
    
    # Uncertainty histogram
    correct_mask = results["predicted_labels"] .== results["true_labels"]
    plots_dict["uncertainty_hist"] = plot_uncertainty_histogram(
        results["entropies"], correct_mask, "German Credit Uncertainty Distribution"
    )
    
    # Save all plots
    save_all_plots(plots_dict, "german_credit_plots")
    
    println("\n6. Comparing Laplace vs Plugin predictions...")
    
    # Compare prediction methods
    laplace_preds = predict_german_credit(trained_model, X; link_approx=:probit)
    plugin_preds = predict_german_credit(trained_model, X; link_approx=:plugin)
    
    laplace_entropies = [-sum(p .* log.(p .+ 1e-8)) for p in laplace_preds]
    plugin_entropies = [-sum(p .* log.(p .+ 1e-8)) for p in plugin_preds]
    
    println("Laplace method - Average entropy: $(round(mean(laplace_entropies), digits=3))")
    println("Plugin method - Average entropy: $(round(mean(plugin_entropies), digits=3))")
    println("Method difference: $(round(mean(laplace_entropies) - mean(plugin_entropies), digits=3))")
    
    # Find samples where methods disagree most
    differences = abs.(laplace_entropies .- plugin_entropies)
    top_diff_indices = sortperm(differences, rev=true)[1:min(5, length(differences))]
    
    println("\nTop $(length(top_diff_indices)) samples with largest method differences:")
    for (i, idx) in enumerate(top_diff_indices)
        laplace_pred = argmax(laplace_preds[idx])
        plugin_pred = argmax(plugin_preds[idx])
        true_label = y_labels[idx]
        laplace_label = unique_labels[laplace_pred]
        plugin_label = unique_labels[plugin_pred]
        println("Sample $idx: True=$true_label, Laplace=$laplace_label, Plugin=$plugin_label, Diff=$(round(differences[idx], digits=3))")
    end
    
    println("\n7. Analyzing prediction uncertainty by class...")
    
    # Group entropy by true class
    for class_label in unique_labels
        mask = results["true_labels"] .== class_label
        if sum(mask) > 0
            class_entropies = results["entropies"][mask]
            println("Class $class_label entropy: $(round(mean(class_entropies), digits=3)) ± $(round(std(class_entropies), digits=3))")
        end
    end
    
    println("\n=== German Credit Classification Complete ===")
    println("Plots saved in: german_credit_plots/")
    println("- decision_boundary_*.png: Decision boundaries for each class")
    println("- decision_boundary_all.png: Overall decision boundary")
    println("- uncertainty_hist.png: Uncertainty distribution")
    
    # Show data file info
    println("\nData file: $data_file")
    println("Features: 2D (feature1, feature2)")
    println("Classes: $(length(unique_labels)) ($(join(unique_labels, ", ")))")
    println("Total samples: $(length(y_labels))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end