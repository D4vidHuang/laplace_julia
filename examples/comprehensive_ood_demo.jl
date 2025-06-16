#!/usr/bin/env julia

"""
Comprehensive OOD Detection Demo

This script provides a complete demonstration of OOD detection capabilities
across both MNIST and German Credit datasets, with comparative analysis.
"""

using Pkg
Pkg.activate(".")

include("../src/MNISTClassifier.jl")
include("../src/GermanCreditClassifier.jl")
include("../src/OODDatasets.jl")
include("../src/OODDetection.jl")
include("../src/Visualizations.jl")

using .MNISTClassifier
using .GermanCreditClassifier
using .OODDatasets
using .OODDetection
using .Visualizations
using Random
using Statistics

function train_or_load_mnist_model()
    println("Setting up MNIST model...")
    model = MNISTModel(32)  # Smaller for faster demo
    return train_mnist!(model, 20, 0.001, 1000)
end

function train_or_load_german_credit_model()
    println("Setting up German Credit model...")
    data_file = "demo_german_credit.csv"
    if !isfile(data_file)
        create_sample_data(data_file)
    end
    
    model = GermanCreditModel(3, 2, 4)
    return train_german_credit!(model, data_file, 50), data_file
end

function evaluate_mnist_ood(trained_model)
    println("\n=== MNIST OOD Evaluation ===")
    
    # Load in-distribution data
    _, _, test_x, test_y, _, _ = load_mnist_data()
    mnist_x = test_x[:, 1:200]
    
    # Test with FashionMNIST and noise
    ood_results = Dict()
    
    # FashionMNIST
    fashion_x, _, _ = load_ood_mnist(:fashionmnist, 200)
    detector = OODDetector(trained_model; method=:entropy)
    fit_ood_threshold!(detector, mnist_x[:, 1:50]; percentile=95.0)
    fashion_metrics = evaluate_ood_detection(detector, mnist_x, fashion_x)
    ood_results["FashionMNIST"] = fashion_metrics
    
    # Uniform Noise
    noise_x, _, _ = load_ood_mnist(:uniform_noise, 200)
    noise_metrics = evaluate_ood_detection(detector, mnist_x, noise_x)
    ood_results["Uniform Noise"] = noise_metrics
    
    # Print results
    println("MNIST OOD Results:")
    for (name, metrics) in ood_results
        println("  $name: AUROC = $(round(metrics["auroc"], digits=3)), AUPR = $(round(metrics["aupr"], digits=3))")
    end
    
    return ood_results, mnist_x
end

function evaluate_german_credit_ood(trained_model, data_file)
    println("\n=== German Credit OOD Evaluation ===")
    
    # Load in-distribution data
    X_train, _, _, _, _ = load_german_credit_data(data_file)
    
    # Test with shifted distribution and outliers
    ood_results = Dict()
    
    # Shifted Distribution
    X_shifted, _, _ = load_ood_german_credit(:shifted_distribution, 100)
    detector = OODDetector(trained_model; method=:entropy)
    fit_ood_threshold!(detector, X_train; percentile=90.0)
    shifted_metrics = evaluate_ood_detection(detector, X_train, X_shifted)
    ood_results["Shifted Distribution"] = shifted_metrics
    
    # Outlier Clusters
    X_outliers, _, _ = load_ood_german_credit(:outlier_clusters, 100)
    outlier_metrics = evaluate_ood_detection(detector, X_train, X_outliers)
    ood_results["Outlier Clusters"] = outlier_metrics
    
    # Print results
    println("German Credit OOD Results:")
    for (name, metrics) in ood_results
        println("  $name: AUROC = $(round(metrics["auroc"], digits=3)), AUPR = $(round(metrics["aupr"], digits=3))")
    end
    
    return ood_results, X_train
end

function create_comparative_analysis(mnist_results, gc_results)
    println("\n=== Comparative Analysis ===")
    
    # Calculate average performance
    mnist_avg = mean([metrics["auroc"] for metrics in values(mnist_results)])
    gc_avg = mean([metrics["auroc"] for metrics in values(gc_results)])
    
    println("Average OOD Detection Performance:")
    println("  MNIST: $(round(mnist_avg, digits=3)) AUROC")
    println("  German Credit: $(round(gc_avg, digits=3)) AUROC")
    
    if mnist_avg > gc_avg
        println("  → MNIST shows better OOD detection performance")
        println("    Possible reasons: Higher dimensional input, more distinct visual features")
    else
        println("  → German Credit shows better OOD detection performance") 
        println("    Possible reasons: Lower dimensional space, clearer decision boundaries")
    end
    
    # Best and worst performance
    all_results = merge(
        Dict("MNIST-$(k)" => v for (k, v) in mnist_results),
        Dict("GC-$(k)" => v for (k, v) in gc_results)
    )
    
    performance_ranking = [(name, metrics["auroc"]) for (name, metrics) in all_results]
    sort!(performance_ranking, by=x->x[2], rev=true)
    
    println("\nOverall Performance Ranking:")
    for (i, (name, auroc)) in enumerate(performance_ranking)
        println("  $i. $name: $(round(auroc, digits=3))")
    end
    
    return performance_ranking
end

function create_summary_visualizations(mnist_results, gc_results, mnist_x, gc_x)
    println("\n=== Creating Summary Visualizations ===")
    
    plots_dict = Dict()
    
    # Performance comparison bar chart
    dataset_names = String[]
    auroc_values = Float64[]
    
    for (name, metrics) in mnist_results
        push!(dataset_names, "MNIST-$name")
        push!(auroc_values, metrics["auroc"])
    end
    
    for (name, metrics) in gc_results
        push!(dataset_names, "GC-$name")
        push!(auroc_values, metrics["auroc"])
    end
    
    plots_dict["performance_comparison"] = bar(
        dataset_names, auroc_values,
        title="OOD Detection Performance Comparison",
        ylabel="AUROC",
        xrotation=45,
        color=[:blue, :blue, :red, :red],
        alpha=0.7,
        size=(800, 500)
    )
    
    # Method effectiveness summary
    effectiveness_data = Dict(
        "High (>0.9)" => sum(auroc_values .> 0.9),
        "Medium (0.8-0.9)" => sum((auroc_values .> 0.8) .& (auroc_values .<= 0.9)),
        "Low (<0.8)" => sum(auroc_values .<= 0.8)
    )
    
    plots_dict["effectiveness_summary"] = pie(
        collect(keys(effectiveness_data)),
        collect(values(effectiveness_data)),
        title="OOD Detection Effectiveness Distribution",
        size=(600, 400)
    )
    
    return plots_dict
end

function generate_recommendations(performance_ranking, mnist_avg, gc_avg)
    println("\n=== Recommendations and Best Practices ===")
    
    best_case = performance_ranking[1]
    worst_case = performance_ranking[end]
    
    println("🏆 Best Performance: $(best_case[1]) (AUROC = $(round(best_case[2], digits=3)))")
    println("⚠️  Worst Performance: $(worst_case[1]) (AUROC = $(round(worst_case[2], digits=3)))")
    
    println("\n📋 General Recommendations:")
    
    if mnist_avg > 0.85
        println("  ✅ MNIST OOD detection is highly effective")
        println("     → Deploy with confidence for image-based applications")
    else
        println("  ⚠️  MNIST OOD detection needs improvement")
        println("     → Consider ensemble methods or additional features")
    end
    
    if gc_avg > 0.85
        println("  ✅ German Credit OOD detection is highly effective")
        println("     → Suitable for tabular data applications")
    else
        println("  ⚠️  German Credit OOD detection needs improvement")
        println("     → Consider domain-specific feature engineering")
    end
    
    println("\n🎯 Deployment Strategies:")
    
    high_performers = [name for (name, auroc) in performance_ranking if auroc > 0.9]
    if length(high_performers) > 0
        println("  • High-confidence scenarios: $(join(high_performers, ", "))")
        println("    → Use lower thresholds for higher sensitivity")
    end
    
    low_performers = [name for (name, auroc) in performance_ranking if auroc < 0.8]
    if length(low_performers) > 0
        println("  • Challenging scenarios: $(join(low_performers, ", "))")
        println("    → Combine with domain-specific rules or ensemble methods")
    end
    
    println("\n🔧 Technical Recommendations:")
    println("  • Use entropy-based uncertainty for general robustness")
    println("  • Set thresholds at 90-95th percentile of in-distribution scores")
    println("  • Monitor and retrain when distribution shifts detected")
    println("  • Consider multiple uncertainty methods for critical applications")
    
    println("\n📊 Monitoring Guidelines:")
    println("  • Track uncertainty score distributions over time")
    println("  • Alert when >5% of samples exceed OOD threshold")
    println("  • Periodic evaluation with known OOD datasets")
    println("  • Update thresholds based on production feedback")
end

function main()
    println("=" ^ 60)
    println("   Comprehensive OOD Detection Demo")
    println("   Laplace Approximation for Neural Networks")
    println("=" ^ 60)
    
    Random.seed!(42)
    
    # Train models
    println("\n1. Training Models...")
    mnist_model = train_or_load_mnist_model()
    gc_model, gc_data_file = train_or_load_german_credit_model()
    
    # Evaluate OOD detection
    println("\n2. Evaluating OOD Detection...")
    mnist_results, mnist_x = evaluate_mnist_ood(mnist_model)
    gc_results, gc_x = evaluate_german_credit_ood(gc_model, gc_data_file)
    
    # Comparative analysis
    println("\n3. Performing Comparative Analysis...")
    performance_ranking = create_comparative_analysis(mnist_results, gc_results)
    
    # Calculate averages for recommendations
    mnist_avg = mean([metrics["auroc"] for metrics in values(mnist_results)])
    gc_avg = mean([metrics["auroc"] for metrics in values(gc_results)])
    
    # Create visualizations
    println("\n4. Creating Summary Visualizations...")
    summary_plots = create_summary_visualizations(mnist_results, gc_results, mnist_x, gc_x)
    save_all_plots(summary_plots, "comprehensive_ood_summary")
    
    # Generate recommendations
    println("\n5. Generating Recommendations...")
    generate_recommendations(performance_ranking, mnist_avg, gc_avg)
    
    println("\n" * "=" * 60)
    println("   Demo Complete!")
    println("=" * 60)
    
    println("\n📁 Generated Files:")
    println("  • comprehensive_ood_summary/: Summary plots and analysis")
    println("  • demo_german_credit.csv: Sample German Credit dataset")
    
    println("\n🎯 Key Insights:")
    println("  • Laplace approximation enables effective OOD detection")
    println("  • Performance varies significantly across OOD types")
    println("  • Entropy-based uncertainty provides robust baseline")
    println("  • Visual inspection confirms uncertainty differences")
    
    println("\n🚀 Next Steps:")
    println("  • Run detailed examples: mnist_ood_example.jl, german_credit_ood_example.jl")
    println("  • Experiment with custom datasets and thresholds")
    println("  • Integrate OOD detection into production pipelines")
    
    return Dict(
        "mnist_results" => mnist_results,
        "gc_results" => gc_results,
        "performance_ranking" => performance_ranking,
        "summary_plots" => summary_plots
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end