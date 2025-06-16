#!/usr/bin/env julia

"""
Comprehensive Bayesian Methods Comparison

This script demonstrates and compares multiple Bayesian inference methods:
- HMC (Hamiltonian Monte Carlo)
- SWAG (Stochastic Weight Averaging - Gaussian)
- MAP (Maximum A Posteriori)
- Laplace Approximation

Applied to both MNIST and German Credit classification tasks.
"""

using Pkg
Pkg.activate(".")

include("../src/MNISTClassifier.jl")
include("../src/GermanCreditClassifier.jl")
include("../src/BayesianMethods.jl")
include("../src/Visualizations.jl")

using .MNISTClassifier
using .GermanCreditClassifier
using .BayesianMethods
using .Visualizations
using Random
using Statistics
using Plots

function create_neural_network(input_dim::Int, hidden_dim::Int, output_dim::Int)
    """Create a simple neural network architecture"""
    using Flux
    return Chain(
        Dense(input_dim, hidden_dim, relu),
        Dense(hidden_dim, hidden_dim, relu),
        Dense(hidden_dim, output_dim)
    )
end

function main()
    println("=" * 70)
    println("    Comprehensive Bayesian Methods Comparison")
    println("    HMC | SWAG | MAP | Laplace Approximation")
    println("=" * 70)
    
    Random.seed!(42)
    
    # ===== MNIST Analysis =====
    println("\n🔢 MNIST Classification with Bayesian Methods")
    println("-" * 50)
    
    # Load MNIST data
    println("Loading MNIST data...")
    train_x, train_y, test_x, test_y, classes, _ = load_mnist_data()
    
    # Use smaller subset for faster computation
    n_train = 1000
    n_test = 200
    
    mnist_train_x = train_x[:, 1:n_train]
    mnist_train_y = train_y[1:n_train]
    mnist_test_x = test_x[:, 1:n_test]
    mnist_test_y = test_y[1:n_test]
    
    # Convert to format expected by BayesianMethods
    mnist_data_train = (mnist_train_x, mnist_train_y)
    
    println("Training data shape: $(size(mnist_train_x))")
    println("Test data shape: $(size(mnist_test_x))")
    
    # Create neural network architecture
    mnist_nn = create_neural_network(784, 32, 10)
    
    # Compare all Bayesian methods
    println("\nComparing Bayesian methods on MNIST...")
    mnist_results = compare_bayesian_methods(
        mnist_nn, mnist_data_train, mnist_test_x, mnist_test_y;
        methods=[:map, :hmc, :swag, :laplace],
        epochs=30,
        verbose=true
    )
    
    # Create visualizations
    println("\nGenerating MNIST visualizations...")
    mnist_comparison_plot = plot_bayesian_methods_comparison(mnist_results; save_plots=false)
    mnist_uncertainty_plot = plot_uncertainty_distributions(mnist_results, mnist_test_x, "MNIST")
    
    # Save MNIST results
    save_bayesian_results(mnist_results, "mnist_bayesian_results.txt")
    
    # ===== German Credit Analysis =====
    println("\n💳 German Credit Classification with Bayesian Methods")
    println("-" * 55)
    
    # Create German Credit data
    data_file = "bayesian_german_credit.csv"
    create_sample_data(data_file)
    
    # Load and prepare data
    X_train, y_train, _, _, unique_labels = load_german_credit_data(data_file)
    
    # Use subset for faster computation
    n_samples = min(200, size(X_train, 2))
    gc_train_x = X_train[:, 1:n_samples]
    gc_train_y = y_train[1:n_samples]
    
    # Test data (use different subset)
    n_test_gc = min(50, size(X_train, 2) - n_samples)
    if n_test_gc > 0
        gc_test_x = X_train[:, (n_samples+1):(n_samples+n_test_gc)]
        gc_test_y = y_train[(n_samples+1):(n_samples+n_test_gc)]
    else
        # Use part of training data as test
        test_indices = 1:min(50, n_samples)
        gc_test_x = gc_train_x[:, test_indices]
        gc_test_y = gc_train_y[test_indices]
    end
    
    # Convert to format expected by BayesianMethods
    gc_data_train = (gc_train_x, gc_train_y)
    
    println("Training data shape: $(size(gc_train_x))")
    println("Test data shape: $(size(gc_test_x))")
    println("Number of classes: $(length(unique_labels))")
    
    # Create neural network architecture for German Credit
    gc_nn = create_neural_network(2, 16, length(unique_labels))
    
    # Compare all Bayesian methods
    println("\nComparing Bayesian methods on German Credit...")
    gc_results = compare_bayesian_methods(
        gc_nn, gc_data_train, gc_test_x, gc_test_y;
        methods=[:map, :hmc, :swag, :laplace],
        epochs=50,
        verbose=true
    )
    
    # Create visualizations
    println("\nGenerating German Credit visualizations...")
    gc_comparison_plot = plot_bayesian_methods_comparison(gc_results; save_plots=false)
    gc_uncertainty_plot = plot_uncertainty_distributions(gc_results, gc_test_x, "German Credit")
    
    # Save German Credit results
    save_bayesian_results(gc_results, "german_credit_bayesian_results.txt")
    
    # ===== Combined Analysis =====
    println("\n📊 Combined Analysis & Insights")
    println("-" * 35)
    
    # Print comprehensive comparison
    print_comprehensive_comparison(mnist_results, gc_results)
    
    # Create combined visualizations
    combined_plots = Dict(
        "mnist_comparison" => mnist_comparison_plot,
        "mnist_uncertainty" => mnist_uncertainty_plot,
        "gc_comparison" => gc_comparison_plot,
        "gc_uncertainty" => gc_uncertainty_plot
    )
    
    save_all_plots(combined_plots, "bayesian_methods_analysis")
    
    # Clean up
    rm(data_file, force=true)
    
    println("\n" * "=" * 70)
    println("    Analysis Complete!")
    println("=" * 70)
    
    println("\n📁 Generated Files:")
    println("• mnist_bayesian_results.txt: Detailed MNIST results")
    println("• german_credit_bayesian_results.txt: Detailed German Credit results")
    println("• bayesian_methods_analysis/: Visualization plots")
    
    println("\n🎯 Key Findings:")
    analyze_method_performance(mnist_results, gc_results)
    
    return Dict(
        "mnist_results" => mnist_results,
        "gc_results" => gc_results,
        "combined_plots" => combined_plots
    )
end

function print_comprehensive_comparison(mnist_results::Dict, gc_results::Dict)
    println("\n" * "=" * 80)
    println("COMPREHENSIVE BAYESIAN METHODS COMPARISON")
    println("=" * 80)
    
    # MNIST Results
    println("\n📊 MNIST CLASSIFICATION RESULTS")
    println("-" * 40)
    println("Method    | Status     | Accuracy | Avg Entropy | Avg Confidence")
    println("-" * 65)
    
    for method in [:map, :hmc, :swag, :laplace]
        if haskey(mnist_results, method)
            result = mnist_results[method]
            status = result["trained_successfully"] ? "Success" : "Failed"
            
            if result["trained_successfully"]
                acc = round(result["accuracy"] * 100, digits=1)
                ent = round(result["avg_entropy"], digits=3)
                conf = round(result["avg_confidence"], digits=3)
                println("$(rpad(string(method), 9)) | $(rpad(status, 10)) | $(rpad(acc, 8))% | $(rpad(ent, 11)) | $(conf)")
            else
                println("$(rpad(string(method), 9)) | $(rpad(status, 10)) | N/A      | N/A         | N/A")
            end
        end
    end
    
    # German Credit Results
    println("\n💳 GERMAN CREDIT CLASSIFICATION RESULTS")
    println("-" * 45)
    println("Method    | Status     | Accuracy | Avg Entropy | Avg Confidence")
    println("-" * 65)
    
    for method in [:map, :hmc, :swag, :laplace]
        if haskey(gc_results, method)
            result = gc_results[method]
            status = result["trained_successfully"] ? "Success" : "Failed"
            
            if result["trained_successfully"]
                acc = round(result["accuracy"] * 100, digits=1)
                ent = round(result["avg_entropy"], digits=3)
                conf = round(result["avg_confidence"], digits=3)
                println("$(rpad(string(method), 9)) | $(rpad(status, 10)) | $(rpad(acc, 8))% | $(rpad(ent, 11)) | $(conf)")
            else
                println("$(rpad(string(method), 9)) | $(rpad(status, 10)) | N/A      | N/A         | N/A")
            end
        end
    end
end

function analyze_method_performance(mnist_results::Dict, gc_results::Dict)
    # Find best performing methods
    mnist_successful = [k for (k, v) in mnist_results if v["trained_successfully"]]
    gc_successful = [k for (k, v) in gc_results if v["trained_successfully"]]
    
    if !isempty(mnist_successful)
        mnist_best = maximum([mnist_results[k]["accuracy"] for k in mnist_successful])
        mnist_best_method = [k for k in mnist_successful if mnist_results[k]["accuracy"] == mnist_best][1]
        println("• Best MNIST method: $mnist_best_method ($(round(mnist_best*100, digits=1))% accuracy)")
    end
    
    if !isempty(gc_successful)
        gc_best = maximum([gc_results[k]["accuracy"] for k in gc_successful])
        gc_best_method = [k for k in gc_successful if gc_results[k]["accuracy"] == gc_best][1]
        println("• Best German Credit method: $gc_best_method ($(round(gc_best*100, digits=1))% accuracy)")
    end
    
    # Method stability analysis
    all_successful = intersect(mnist_successful, gc_successful)
    if length(all_successful) > 1
        println("• Most stable methods (worked on both datasets): $(join(all_successful, ", "))")
    elseif length(all_successful) == 1
        println("• Only $([all_successful][1]) method worked consistently across both datasets")
    else
        println("• No method worked consistently across both datasets")
    end
    
    # Uncertainty analysis
    println("\n🔍 UNCERTAINTY ANALYSIS:")
    for dataset_name in ["MNIST", "German Credit"]
        results = dataset_name == "MNIST" ? mnist_results : gc_results
        successful = [k for (k, v) in results if v["trained_successfully"]]
        
        if !isempty(successful)
            entropies = [results[k]["avg_entropy"] for k in successful]
            min_entropy_method = successful[argmin(entropies)]
            max_entropy_method = successful[argmax(entropies)]
            
            println("• $dataset_name: Most confident method = $min_entropy_method, Most uncertain = $max_entropy_method")
        end
    end
    
    # Computational efficiency insights
    println("\n⚡ COMPUTATIONAL INSIGHTS:")
    println("• MAP: Fastest, deterministic, good baseline")
    println("• Laplace: Fast, good uncertainty quantification, builds on MAP")
    println("• SWAG: Moderate cost, good approximation to posterior")
    println("• HMC: Slowest, most principled, highest quality uncertainty")
    
    # Recommendations
    println("\n💡 RECOMMENDATIONS:")
    
    successful_count = Dict()
    for method in [:map, :hmc, :swag, :laplace]
        count = 0
        if haskey(mnist_results, method) && mnist_results[method]["trained_successfully"]
            count += 1
        end
        if haskey(gc_results, method) && gc_results[method]["trained_successfully"]
            count += 1
        end
        successful_count[method] = count
    end
    
    most_reliable = maximum(values(successful_count))
    most_reliable_methods = [k for (k, v) in successful_count if v == most_reliable]
    
    if most_reliable >= 2
        println("• For reliability: Use $(join(most_reliable_methods, " or ")) (worked on $most_reliable/2 datasets)")
    end
    
    println("• For speed: Use MAP for quick prototyping")
    println("• For uncertainty: Use Laplace approximation for good speed/quality trade-off")
    println("• For research: Use HMC for principled Bayesian inference")
    println("• For practical applications: Consider SWAG for robust uncertainty")
end

if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end