#!/usr/bin/env julia

"""
Laplace vs MAP Comparison Example

This script demonstrates the differences between Laplace approximation and MAP estimation
for both confidence calibration and OOD detection performance.
"""

using Pkg
Pkg.activate(".")

include("../src/MNISTClassifier.jl")
include("../src/GermanCreditClassifier.jl")
include("../src/OODDatasets.jl")
include("../src/Visualizations.jl")

using .MNISTClassifier
using .GermanCreditClassifier
using .OODDatasets
using .Visualizations
using Random
using Statistics

function main()
    println("=" * 60)
    println("   Laplace vs MAP Comparison Analysis")
    println("   Confidence Calibration & OOD Detection")
    println("=" * 60)
    
    Random.seed!(42)
    
    # ===== MNIST Analysis =====
    println("\n🔢 MNIST Classification Analysis")
    println("-" * 40)
    
    # Train MNIST model
    println("1. Training MNIST model...")
    mnist_model = MNISTModel(32)  # Smaller for faster demo
    trained_mnist = train_mnist!(mnist_model, 20, 0.001, 1500)
    
    # Load test data
    _, _, test_x, test_y, _, _ = load_mnist_data()
    mnist_test_x = test_x[:, 1:300]
    mnist_test_y = test_y[1:300]
    
    println("\n2. Analyzing confidence differences...")
    
    # Basic Laplace vs MAP comparison
    plot_basic, stats_basic = plot_laplace_vs_map_comparison(trained_mnist, mnist_test_x; n_samples=200)
    
    # Confidence calibration analysis
    plot_calib, stats_calib = plot_confidence_comparison(trained_mnist, mnist_test_x, mnist_test_y; confidence_bins=10)
    
    println("\n3. OOD detection comparison...")
    
    # Load OOD datasets for MNIST
    fashion_x, _, _ = load_ood_mnist(:fashionmnist, 200)
    noise_x, _, _ = load_ood_mnist(:uniform_noise, 200)
    
    # Compare AUROC for different OOD types
    plot_fashion, stats_fashion = plot_auroc_comparison(trained_mnist, mnist_test_x, fashion_x, "FashionMNIST")
    plot_noise, stats_noise = plot_auroc_comparison(trained_mnist, mnist_test_x, noise_x, "Uniform Noise")
    
    # Comprehensive improvement analysis
    ood_datasets_mnist = [fashion_x, noise_x]
    ood_names_mnist = ["FashionMNIST", "Uniform Noise"]
    
    plot_improvement_mnist, results_mnist = plot_laplace_improvement_analysis(
        trained_mnist, mnist_test_x, ood_datasets_mnist, ood_names_mnist
    )
    
    # Save MNIST plots
    mnist_plots = Dict(
        "mnist_basic_comparison" => plot_basic,
        "mnist_confidence_calibration" => plot_calib,
        "mnist_ood_fashion" => plot_fashion,
        "mnist_ood_noise" => plot_noise,
        "mnist_improvement_analysis" => plot_improvement_mnist
    )
    save_all_plots(mnist_plots, "mnist_laplace_vs_map_analysis")
    
    # ===== German Credit Analysis =====
    println("\n💳 German Credit Classification Analysis")
    println("-" * 45)
    
    # Train German Credit model
    println("1. Training German Credit model...")
    data_file = "comparison_german_credit.csv"
    if !isfile(data_file)
        create_sample_data(data_file)
    end
    
    gc_model = GermanCreditModel(3, 2, 4)
    trained_gc = train_german_credit!(gc_model, data_file, 80)
    
    # Load in-distribution data
    X_train, y_train, _, _, _ = load_german_credit_data(data_file)
    
    println("\n2. Analyzing confidence differences...")
    
    # Basic comparison (note: for German Credit we need to handle data format)
    gc_test_x = X_train[:, 1:min(100, size(X_train, 2))]
    gc_test_y = y_train[1:min(100, length(y_train))]
    
    plot_gc_basic, stats_gc_basic = plot_laplace_vs_map_comparison(trained_gc, gc_test_x; n_samples=80)
    plot_gc_calib, stats_gc_calib = plot_confidence_comparison(trained_gc, gc_test_x, gc_test_y; confidence_bins=8)
    
    println("\n3. OOD detection comparison...")
    
    # Load OOD datasets for German Credit
    shifted_x, _, _ = load_ood_german_credit(:shifted_distribution, 80)
    outlier_x, _, _ = load_ood_german_credit(:outlier_clusters, 80)
    
    # Compare AUROC
    plot_gc_shifted, stats_gc_shifted = plot_auroc_comparison(trained_gc, X_train, shifted_x, "Shifted Distribution")
    plot_gc_outlier, stats_gc_outlier = plot_auroc_comparison(trained_gc, X_train, outlier_x, "Outlier Clusters")
    
    # Comprehensive improvement analysis
    ood_datasets_gc = [shifted_x, outlier_x]
    ood_names_gc = ["Shifted Distribution", "Outlier Clusters"]
    
    plot_improvement_gc, results_gc = plot_laplace_improvement_analysis(
        trained_gc, X_train, ood_datasets_gc, ood_names_gc
    )
    
    # Save German Credit plots
    gc_plots = Dict(
        "gc_basic_comparison" => plot_gc_basic,
        "gc_confidence_calibration" => plot_gc_calib,
        "gc_ood_shifted" => plot_gc_shifted,
        "gc_ood_outlier" => plot_gc_outlier,
        "gc_improvement_analysis" => plot_improvement_gc
    )
    save_all_plots(gc_plots, "german_credit_laplace_vs_map_analysis")
    
    # ===== Combined Analysis =====
    println("\n📊 Combined Analysis & Summary")
    println("-" * 35)
    
    # Summary statistics
    println("\n=== CONFIDENCE CALIBRATION SUMMARY ===")
    println("Dataset      | Laplace ECE | MAP ECE | Better Method")
    println("-" * 50)
    println("MNIST        | $(rpad(round(stats_calib["laplace_ece"], digits=4), 11)) | $(rpad(round(stats_calib["map_ece"], digits=4), 7)) | $(stats_calib["laplace_ece"] < stats_calib["map_ece"] ? "Laplace" : "MAP")")
    println("German Credit| $(rpad(round(stats_gc_calib["laplace_ece"], digits=4), 11)) | $(rpad(round(stats_gc_calib["map_ece"], digits=4), 7)) | $(stats_gc_calib["laplace_ece"] < stats_gc_calib["map_ece"] ? "Laplace" : "MAP")")
    
    println("\n=== OOD DETECTION SUMMARY ===")
    println("Dataset & OOD Type       | Laplace AUROC | MAP AUROC | Improvement")
    println("-" * 65)
    println("MNIST vs FashionMNIST    | $(rpad(round(stats_fashion["laplace_auroc"], digits=3), 13)) | $(rpad(round(stats_fashion["map_auroc"], digits=3), 9)) | $(round(stats_fashion["improvement"], digits=4))")
    println("MNIST vs Uniform Noise   | $(rpad(round(stats_noise["laplace_auroc"], digits=3), 13)) | $(rpad(round(stats_noise["map_auroc"], digits=3), 9)) | $(round(stats_noise["improvement"], digits=4))")
    println("GC vs Shifted Dist       | $(rpad(round(stats_gc_shifted["laplace_auroc"], digits=3), 13)) | $(rpad(round(stats_gc_shifted["map_auroc"], digits=3), 9)) | $(round(stats_gc_shifted["improvement"], digits=4))")
    println("GC vs Outlier Clusters   | $(rpad(round(stats_gc_outlier["laplace_auroc"], digits=3), 13)) | $(rpad(round(stats_gc_outlier["map_auroc"], digits=3), 9)) | $(round(stats_gc_outlier["improvement"], digits=4))")
    
    # Overall metrics
    all_improvements = [
        stats_fashion["improvement"],
        stats_noise["improvement"], 
        stats_gc_shifted["improvement"],
        stats_gc_outlier["improvement"]
    ]
    
    avg_improvement = mean(all_improvements)
    positive_improvements = sum(all_improvements .> 0)
    
    println("\n=== OVERALL LAPLACE APPROXIMATION IMPACT ===")
    println("Average AUROC improvement: $(round(avg_improvement, digits=4))")
    println("Cases where Laplace wins: $positive_improvements/$(length(all_improvements))")
    println("Best improvement: $(round(maximum(all_improvements), digits=4))")
    println("Worst case: $(round(minimum(all_improvements), digits=4))")
    
    # Confidence analysis
    all_conf_diffs = [
        stats_basic["conf_diff"],
        stats_gc_basic["conf_diff"]
    ]
    
    avg_conf_diff = mean(all_conf_diffs)
    
    println("\nConfidence behavior:")
    println("Average confidence difference (Laplace - MAP): $(round(avg_conf_diff, digits=4))")
    println("Laplace generally provides $(avg_conf_diff > 0 ? "higher" : "lower") confidence")
    
    # Key insights
    println("\n🔍 KEY INSIGHTS:")
    
    if avg_improvement > 0
        println("✅ Laplace approximation improves OOD detection on average")
    else
        println("⚠️  MAP estimation performs better for OOD detection on average")
    end
    
    if stats_calib["laplace_ece"] < stats_calib["map_ece"] && stats_gc_calib["laplace_ece"] < stats_gc_calib["map_ece"]
        println("✅ Laplace provides better calibration for both datasets")
    elseif stats_calib["laplace_ece"] < stats_calib["map_ece"] || stats_gc_calib["laplace_ece"] < stats_gc_calib["map_ece"]
        println("🔄 Laplace provides better calibration for some datasets")
    else
        println("⚠️  MAP provides better calibration overall")
    end
    
    if positive_improvements >= 3
        println("🎯 Strong evidence for Laplace approximation benefits")
    elseif positive_improvements >= 2
        println("📈 Moderate evidence for Laplace approximation benefits")
    else
        println("❓ Mixed results - context-dependent performance")
    end
    
    println("\n💡 RECOMMENDATIONS:")
    
    if avg_improvement > 0.02
        println("• Use Laplace approximation for OOD detection (significant improvement)")
    elseif avg_improvement > 0
        println("• Consider Laplace approximation for OOD detection (modest improvement)")
    else
        println("• Evaluate case-by-case; MAP may be sufficient")
    end
    
    if avg_conf_diff > 0.05
        println("• Laplace provides notably higher confidence - good for decision-making")
    elseif avg_conf_diff < -0.05
        println("• Laplace provides more conservative confidence - good for safety-critical applications")
    else
        println("• Confidence differences are minimal between methods")
    end
    
    println("• Always check calibration on your specific dataset")
    println("• Consider computational cost vs. performance trade-offs")
    
    # Clean up
    rm(data_file, force=true)
    
    println("\n" * "=" * 60)
    println("   Analysis Complete!")
    println("=" * 60)
    
    println("\n📁 Generated Files:")
    println("• mnist_laplace_vs_map_analysis/: MNIST comparison plots")
    println("• german_credit_laplace_vs_map_analysis/: German Credit comparison plots")
    
    println("\n🎯 Key Takeaways:")
    println("• Laplace approximation effect varies by dataset and OOD type")
    println("• Confidence calibration generally improves with Laplace")
    println("• OOD detection benefits depend on the specific distribution shift")
    println("• Visual analysis provides insights beyond aggregate metrics")
    
    return Dict(
        "mnist_results" => Dict(
            "basic" => stats_basic,
            "calibration" => stats_calib,
            "ood_fashion" => stats_fashion,
            "ood_noise" => stats_noise,
            "improvement" => results_mnist
        ),
        "gc_results" => Dict(
            "basic" => stats_gc_basic,
            "calibration" => stats_gc_calib,
            "ood_shifted" => stats_gc_shifted,
            "ood_outlier" => stats_gc_outlier,
            "improvement" => results_gc
        ),
        "summary" => Dict(
            "avg_auroc_improvement" => avg_improvement,
            "positive_improvements" => positive_improvements,
            "avg_confidence_diff" => avg_conf_diff
        )
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end