#!/usr/bin/env julia

"""
German Credit OOD Detection Example with Laplace Approximation

This script demonstrates out-of-distribution detection for the German Credit dataset
using uncertainty quantification from Bayesian neural networks.
"""

using Pkg
Pkg.activate(".")

include("../src/GermanCreditClassifier.jl")
include("../src/OODDatasets.jl") 
include("../src/OODDetection.jl")
include("../src/Visualizations.jl")

using .GermanCreditClassifier
using .OODDatasets
using .OODDetection
using .Visualizations
using Random
using Statistics

function main()
    println("=== German Credit OOD Detection with Laplace Approximation ===")
    
    # Set random seed for reproducibility
    Random.seed!(42)
    
    println("\n1. Setting up German Credit dataset...")
    
    # Create or load training data
    data_file = "german_credit_train.csv"
    if !isfile(data_file)
        create_sample_data(data_file)
    end
    
    println("\n2. Training German Credit model...")
    model = GermanCreditModel(3, 2, 4)
    trained_model = train_german_credit!(model, data_file, 100)  # Faster training for demo
    
    println("\n3. Loading in-distribution data...")
    X_train, y_train, _, _, unique_labels = load_german_credit_data(data_file)
    
    println("\n4. Creating various OOD datasets...")
    
    # Test different types of OOD data
    ood_datasets = [
        (:shifted_distribution, "Shifted Distribution"),
        (:different_scale, "Different Scale"),
        (:uniform_random, "Uniform Random"),
        (:outlier_clusters, "Outlier Clusters"),
        (:high_noise, "High Noise")
    ]
    
    # Store results for comparison
    all_results = Dict()
    all_plots = Dict()
    
    for (ood_type, ood_name) in ood_datasets
        println("\n--- Testing OOD dataset: $ood_name ---")
        
        # Load OOD data
        X_ood, y_ood, dataset_name = load_ood_german_credit(ood_type, 100)
        
        println("5. Setting up OOD detection for $ood_name...")
        
        # Test different uncertainty methods
        methods = [:entropy, :max_prob, :variance]
        method_results = Dict()
        
        for method in methods
            println("  Testing method: $method")
            
            # Create detector
            detector = OODDetector(trained_model; method=method)
            
            # Fit threshold using training data (as validation)
            fit_ood_threshold!(detector, X_train; percentile=90.0)  # Lower percentile for smaller dataset
            
            # Evaluate OOD detection
            metrics = evaluate_ood_detection(detector, X_train, X_ood)
            method_results[method] = metrics
            
            println("    AUROC: $(round(metrics["auroc"], digits=3))")
            println("    AUPR: $(round(metrics["aupr"], digits=3))")
            println("    FPR@95TPR: $(round(metrics["fpr_at_95tpr"], digits=3))")
        end
        
        all_results[ood_name] = method_results
        
        println("\n6. Creating visualizations for $ood_name...")
        
        # Use best performing method for visualization
        best_method = :entropy
        best_auroc = 0.0
        for (method, metrics) in method_results
            if metrics["auroc"] > best_auroc
                best_auroc = metrics["auroc"]
                best_method = method
            end
        end
        
        metrics = method_results[best_method]
        
        # Create plots
        plots_dict = Dict()
        
        # Data distribution comparison
        plots_dict["$(ood_type)_data_comparison"] = plot_ood_samples_comparison(
            trained_model, X_train, X_ood, ood_name; data_type=:german_credit
        )
        
        # Decision boundary with OOD data
        # Combine data for boundary plotting
        X_combined = hcat(X_train, X_ood)
        y_combined = vcat(y_train, fill(5, size(X_ood, 2)))  # Use label 5 for OOD
        unique_combined = vcat(unique_labels, [5])
        
        plots_dict["$(ood_type)_boundary"] = plot_german_credit_decision_boundary(
            trained_model, X_combined', y_combined, unique_combined
        )
        
        # OOD detection summary
        plots_dict["$(ood_type)_summary"] = plot_ood_detection_summary(
            metrics, metrics["scores_in"], metrics["scores_ood"], string(best_method)
        )
        
        # Uncertainty distribution
        plots_dict["$(ood_type)_uncertainty"] = plot_uncertainty_vs_ood(
            metrics["scores_in"], metrics["scores_ood"], metrics["optimal_threshold"]
        )
        
        all_plots[ood_name] = plots_dict
        
        # Save plots for this OOD dataset
        save_all_plots(plots_dict, "german_credit_ood_$(ood_type)_plots")
    end
    
    println("\n7. Creating comprehensive comparison...")
    
    # Performance comparison table
    println("\n=== OOD Detection Performance Summary ===")
    println("Dataset                | Method    | AUROC | AUPR  | FPR@95TPR")
    println("-" ^ 65)
    
    for (ood_name, method_results) in all_results
        for (method, metrics) in method_results
            auroc = round(metrics["auroc"], digits=3)
            aupr = round(metrics["aupr"], digits=3)
            fpr95 = round(metrics["fpr_at_95tpr"], digits=3)
            
            println("$(rpad(ood_name, 22)) | $(rpad(string(method), 9)) | $(rpad(auroc, 5)) | $(rpad(aupr, 5)) | $fpr95")
        end
    end
    
    println("\n8. Method comparison across all OOD types...")
    
    # Calculate average performance by method
    method_performance = Dict()
    for method in [:entropy, :max_prob, :variance]
        aurocs = [all_results[name][method]["auroc"] for name in keys(all_results)]
        auprs = [all_results[name][method]["aupr"] for name in keys(all_results)]
        fpr95s = [all_results[name][method]["fpr_at_95tpr"] for name in keys(all_results)]
        
        method_performance[method] = Dict(
            "avg_auroc" => mean(aurocs),
            "avg_aupr" => mean(auprs),
            "avg_fpr95" => mean(fpr95s),
            "std_auroc" => std(aurocs)
        )
    end
    
    println("\nAverage performance by method:")
    for (method, perf) in method_performance
        println("  $method:")
        println("    AUROC: $(round(perf["avg_auroc"], digits=3)) ± $(round(perf["std_auroc"], digits=3))")
        println("    AUPR:  $(round(perf["avg_aupr"], digits=3))")
        println("    FPR@95TPR: $(round(perf["avg_fpr95"], digits=3))")
    end
    
    # Find best overall method
    best_method_overall = nothing
    best_avg_auroc = 0.0
    for (method, perf) in method_performance
        if perf["avg_auroc"] > best_avg_auroc
            best_avg_auroc = perf["avg_auroc"]
            best_method_overall = method
        end
    end
    
    println("\nBest overall method: $best_method_overall (Avg AUROC = $(round(best_avg_auroc, digits=3)))")
    
    println("\n9. Analyzing OOD type difficulty...")
    
    # Rank OOD types by detection difficulty
    ood_difficulty = Dict()
    for (ood_name, method_results) in all_results
        # Use entropy method for ranking
        ood_difficulty[ood_name] = method_results[:entropy]["auroc"]
    end
    
    sorted_difficulty = sort(collect(ood_difficulty), by=x->x[2], rev=true)
    
    println("\nOOD Detection difficulty ranking (AUROC with entropy method):")
    for (i, (name, auroc)) in enumerate(sorted_difficulty)
        difficulty = auroc > 0.9 ? "Easy" : auroc > 0.8 ? "Medium" : "Hard"
        println("  $i. $name: $(round(auroc, digits=3)) ($difficulty)")
    end
    
    println("\n10. Insights and recommendations...")
    
    # Analyze which types of OOD are hardest/easiest
    easiest_ood = sorted_difficulty[1]
    hardest_ood = sorted_difficulty[end]
    
    println("\nKey insights:")
    println("  • Easiest to detect: $(easiest_ood[1]) (AUROC = $(round(easiest_ood[2], digits=3)))")
    println("  • Hardest to detect: $(hardest_ood[1]) (AUROC = $(round(hardest_ood[2], digits=3)))")
    
    # Check if certain patterns emerge
    high_performance = [name for (name, auroc) in sorted_difficulty if auroc > 0.85]
    low_performance = [name for (name, auroc) in sorted_difficulty if auroc < 0.75]
    
    if length(high_performance) > 0
        println("  • High-performance detection (AUROC > 0.85): $(join(high_performance, ", "))")
    end
    if length(low_performance) > 0
        println("  • Challenging detection (AUROC < 0.75): $(join(low_performance, ", "))")
    end
    
    # Method-specific insights
    if method_performance[:entropy]["avg_auroc"] > method_performance[:max_prob]["avg_auroc"]
        println("  • Entropy-based uncertainty outperforms max probability on average")
    else
        println("  • Max probability uncertainty outperforms entropy on average")
    end
    
    if method_performance[:variance]["avg_auroc"] > 0.8
        println("  • Variance-based uncertainty shows strong performance")
    else
        println("  • Variance-based uncertainty shows limited performance")
    end
    
    println("\n11. Creating combined analysis plots...")
    
    # Create method comparison plot
    method_names = collect(keys(method_performance))
    avg_aurocs = [method_performance[m]["avg_auroc"] for m in method_names]
    std_aurocs = [method_performance[m]["std_auroc"] for m in method_names]
    
    comparison_plot = bar(string.(method_names), avg_aurocs, 
                         yerr=std_aurocs, alpha=0.7, color=:blues,
                         title="Method Comparison (Average AUROC)",
                         ylabel="AUROC", legend=false)
    
    # Save comparison plot
    save_all_plots(Dict("method_comparison" => comparison_plot), "german_credit_ood_comparison")
    
    println("\n=== German Credit OOD Detection Complete ===")
    
    # Summary
    println("\nGenerated files:")
    for (ood_type, _) in ood_datasets
        println("  • german_credit_ood_$(ood_type)_plots/: Plots for $ood_type")
    end
    println("  • german_credit_ood_comparison/: Method comparison plots")
    
    println("\nKey findings:")
    println("  • Best uncertainty method: $best_method_overall")
    println("  • Most detectable OOD: $(easiest_ood[1])")
    println("  • Most challenging OOD: $(hardest_ood[1])")
    println("  • Average detection performance: $(round(mean([d[2] for d in sorted_difficulty]), digits=3)) AUROC")
    
    println("\nRecommendations:")
    println("  • Use $(best_method_overall) for general OOD detection")
    println("  • Focus on $(hardest_ood[1])-type shifts in production monitoring")
    println("  • Consider ensemble methods for robust detection")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end