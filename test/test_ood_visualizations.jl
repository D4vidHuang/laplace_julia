using Test
using Random
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

@testset "OOD Visualization Tests" begin
    Random.seed!(42)
    
    @testset "OOD Score Distribution Plot" begin
        # Create sample data
        scores_in = [0.1, 0.2, 0.15, 0.25, 0.18]
        scores_ood = [0.8, 0.9, 0.85, 0.95, 0.88]
        threshold = 0.5
        
        @testset "Basic Distribution Plot" begin
            plot_obj = plot_ood_scores_distribution(scores_in, scores_ood, "Entropy")
            @test plot_obj !== nothing
        end
        
        @testset "Distribution Plot with Threshold" begin
            plot_obj = plot_ood_scores_distribution(scores_in, scores_ood, "Entropy", threshold)
            @test plot_obj !== nothing
        end
        
        @testset "Custom Parameters" begin
            plot_obj = plot_ood_scores_distribution(scores_in, scores_ood, "Max Probability"; 
                                                  threshold=threshold, bins=20)
            @test plot_obj !== nothing
        end
    end
    
    @testset "ROC Curve Plot" begin
        # Create sample metrics
        metrics = Dict(
            "tpr_values" => [0.0, 0.25, 0.5, 0.75, 1.0],
            "fpr_values" => [0.0, 0.1, 0.2, 0.4, 1.0],
            "auroc" => 0.85
        )
        
        @testset "Basic ROC Plot" begin
            plot_obj = plot_roc_curve(metrics)
            @test plot_obj !== nothing
        end
        
        @testset "ROC Plot with Custom Title" begin
            plot_obj = plot_roc_curve(metrics; title="Custom ROC Curve")
            @test plot_obj !== nothing
        end
    end
    
    @testset "Precision-Recall Curve Plot" begin
        # Create sample metrics
        metrics = Dict(
            "precision_values" => [1.0, 0.8, 0.6, 0.4, 0.2],
            "recall_values" => [0.0, 0.25, 0.5, 0.75, 1.0],
            "aupr" => 0.75,
            "n_ood" => 10,
            "n_in" => 20
        )
        
        @testset "Basic PR Plot" begin
            plot_obj = plot_precision_recall_curve(metrics)
            @test plot_obj !== nothing
        end
        
        @testset "PR Plot with Custom Title" begin
            plot_obj = plot_precision_recall_curve(metrics; title="Custom PR Curve")
            @test plot_obj !== nothing
        end
    end
    
    @testset "OOD Samples Comparison" begin
        # Create and train small models for testing
        mnist_model = MNISTModel(4)
        trained_mnist = train_mnist!(mnist_model, 1, 0.01, 5)
        
        test_file = "test_ood_viz_data.csv"
        create_sample_data(test_file)
        gc_model = GermanCreditModel(3, 2, 4)
        trained_gc = train_german_credit!(gc_model, test_file, 3)
        
        @testset "MNIST OOD Comparison" begin
            x_in = rand(Float32, 784, 6)
            x_ood = rand(Float32, 784, 6)
            
            plot_obj = plot_ood_samples_comparison(trained_mnist, x_in, x_ood, "Test OOD"; 
                                                 n_samples=4, data_type=:mnist)
            @test plot_obj !== nothing
        end
        
        @testset "German Credit OOD Comparison" begin
            x_in = rand(Float32, 2, 20)
            x_ood = rand(Float32, 2, 15) .+ 5.0f0  # Shifted data
            
            plot_obj = plot_ood_samples_comparison(trained_gc, x_in, x_ood, "Shifted Data"; 
                                                 data_type=:german_credit)
            @test plot_obj !== nothing
        end
        
        @testset "Invalid Data Type" begin
            x_in = rand(Float32, 10, 5)
            x_ood = rand(Float32, 10, 5)
            
            @test_throws ErrorException plot_ood_samples_comparison(trained_mnist, x_in, x_ood, "Test"; 
                                                                  data_type=:invalid)
        end
        
        # Clean up
        rm(test_file, force=true)
    end
    
    @testset "Uncertainty vs OOD Plot" begin
        scores_in = [0.1, 0.15, 0.2, 0.12, 0.18]
        scores_ood = [0.8, 0.9, 0.85, 0.95, 0.82]
        threshold = 0.5
        
        @testset "Basic Uncertainty Plot" begin
            plot_obj = plot_uncertainty_vs_ood(scores_in, scores_ood)
            @test plot_obj !== nothing
        end
        
        @testset "Uncertainty Plot with Threshold" begin
            plot_obj = plot_uncertainty_vs_ood(scores_in, scores_ood, threshold)
            @test plot_obj !== nothing
        end
    end
    
    @testset "Calibration Curve Plot" begin
        # Create sample calibration data
        confidences = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0]
        accuracies = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]  # Binary outcomes
        
        @testset "Basic Calibration Plot" begin
            plot_obj = plot_calibration_curve(confidences, accuracies)
            @test plot_obj !== nothing
        end
        
        @testset "Calibration Plot with Custom Bins" begin
            plot_obj = plot_calibration_curve(confidences, accuracies; n_bins=5)
            @test plot_obj !== nothing
        end
        
        @testset "Edge Cases" begin
            # All same confidence
            conf_same = fill(0.5, 10)
            acc_mixed = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            plot_obj = plot_calibration_curve(conf_same, acc_mixed)
            @test plot_obj !== nothing
            
            # Perfect calibration
            conf_perfect = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            acc_perfect = [0, 0, 0, 1, 1, 1]
            plot_obj = plot_calibration_curve(conf_perfect, acc_perfect; n_bins=3)
            @test plot_obj !== nothing
        end
    end
    
    @testset "OOD Detection Summary Plot" begin
        # Create comprehensive sample metrics
        metrics = Dict(
            "tpr_values" => [0.0, 0.5, 1.0],
            "fpr_values" => [0.0, 0.2, 1.0],
            "auroc" => 0.9,
            "precision_values" => [1.0, 0.8, 0.6],
            "recall_values" => [0.0, 0.5, 1.0],
            "aupr" => 0.8,
            "n_ood" => 10,
            "n_in" => 20,
            "optimal_threshold" => 0.5
        )
        
        scores_in = [0.1, 0.2, 0.15, 0.25, 0.18]
        scores_ood = [0.8, 0.9, 0.85, 0.95, 0.88]
        
        @testset "Basic Summary Plot" begin
            plot_obj = plot_ood_detection_summary(metrics, scores_in, scores_ood)
            @test plot_obj !== nothing
        end
        
        @testset "Summary Plot with Custom Method Name" begin
            plot_obj = plot_ood_detection_summary(metrics, scores_in, scores_ood, "Max Probability")
            @test plot_obj !== nothing
        end
    end
    
    @testset "Error Handling for Visualizations" begin
        # Test with untrained models
        untrained_mnist = MNISTModel(4)
        untrained_gc = GermanCreditModel(3, 2, 4)
        
        @testset "Untrained Model Errors" begin
            x_dummy = rand(Float32, 784, 6)
            
            @test_throws ErrorException plot_ood_samples_comparison(untrained_mnist, x_dummy, x_dummy, "Test"; 
                                                                  data_type=:mnist)
            
            x_dummy_gc = rand(Float32, 2, 10)
            @test_throws ErrorException plot_ood_samples_comparison(untrained_gc, x_dummy_gc, x_dummy_gc, "Test"; 
                                                                  data_type=:german_credit)
        end
        
        @testset "Invalid Input Data" begin
            # Test with empty arrays
            @test_nowarn plot_ood_scores_distribution(Float64[], Float64[], "Test")
            
            # Test with mismatched arrays
            scores_short = [0.1, 0.2]
            scores_long = [0.8, 0.9, 0.85, 0.95]
            @test_nowarn plot_uncertainty_vs_ood(scores_short, scores_long)
        end
    end
    
    @testset "Integration with Real OOD Detection" begin
        # Create small integrated test
        model = MNISTModel(4)
        trained_model = train_mnist!(model, 1, 0.01, 8)
        
        # Create detector and fit threshold
        detector = OODDetector(trained_model; method=:entropy)
        x_in = rand(Float32, 784, 6)
        fit_ood_threshold!(detector, x_in; percentile=90.0)
        
        # Create OOD data
        x_ood = rand(Float32, 784, 4) .+ 1.0f0
        
        # Evaluate
        metrics = evaluate_ood_detection(detector, x_in, x_ood)
        
        @testset "Real Integration Plots" begin
            # Test all plot types with real data
            summary_plot = plot_ood_detection_summary(metrics, metrics["scores_in"], metrics["scores_ood"])
            @test summary_plot !== nothing
            
            roc_plot = plot_roc_curve(metrics)
            @test roc_plot !== nothing
            
            pr_plot = plot_precision_recall_curve(metrics)
            @test pr_plot !== nothing
            
            dist_plot = plot_ood_scores_distribution(metrics["scores_in"], metrics["scores_ood"], "Entropy")
            @test dist_plot !== nothing
            
            comparison_plot = plot_ood_samples_comparison(trained_model, x_in, x_ood, "Test OOD"; 
                                                        n_samples=4, data_type=:mnist)
            @test comparison_plot !== nothing
        end
    end
    
    @testset "Laplace vs MAP Comparison Plots" begin
        # Create and train small model
        model = MNISTModel(4)
        trained_model = train_mnist!(model, 1, 0.01, 8)
        
        test_x = rand(Float32, 784, 10)
        test_y = rand(0:9, 10)
        
        @testset "Basic Laplace vs MAP Comparison" begin
            plot_obj, stats = plot_laplace_vs_map_comparison(trained_model, test_x; n_samples=8)
            @test plot_obj !== nothing
            @test haskey(stats, "laplace_confidence")
            @test haskey(stats, "map_confidence")
            @test haskey(stats, "conf_diff")
            @test haskey(stats, "entropy_diff")
            @test length(stats["laplace_confidence"]) == 8
            @test length(stats["map_confidence"]) == 8
        end
        
        @testset "Confidence Comparison with Calibration" begin
            plot_obj, stats = plot_confidence_comparison(trained_model, test_x, test_y; confidence_bins=5)
            @test plot_obj !== nothing
            @test haskey(stats, "laplace_ece")
            @test haskey(stats, "map_ece")
            @test haskey(stats, "laplace_accuracy")
            @test haskey(stats, "map_accuracy")
            @test 0 ≤ stats["laplace_ece"] ≤ 1
            @test 0 ≤ stats["map_ece"] ≤ 1
        end
        
        @testset "AUROC Comparison" begin
            x_in = rand(Float32, 784, 8)
            x_ood = rand(Float32, 784, 6) .+ 0.5f0
            
            plot_obj, stats = plot_auroc_comparison(trained_model, x_in, x_ood, "Test OOD")
            @test plot_obj !== nothing
            @test haskey(stats, "laplace_auroc")
            @test haskey(stats, "map_auroc")
            @test haskey(stats, "improvement")
            @test 0 ≤ stats["laplace_auroc"] ≤ 1
            @test 0 ≤ stats["map_auroc"] ≤ 1
        end
        
        @testset "Comprehensive Improvement Analysis" begin
            x_in = rand(Float32, 784, 8)
            x_ood1 = rand(Float32, 784, 6) .+ 0.3f0
            x_ood2 = rand(Float32, 784, 6) .+ 0.7f0
            
            ood_list = [x_ood1, x_ood2]
            ood_names = ["Test OOD 1", "Test OOD 2"]
            
            plot_obj, results = plot_laplace_improvement_analysis(trained_model, x_in, ood_list, ood_names)
            @test plot_obj !== nothing
            @test haskey(results, "Test OOD 1")
            @test haskey(results, "Test OOD 2")
            
            for name in ood_names
                @test haskey(results[name], "laplace_auroc")
                @test haskey(results[name], "map_auroc")
                @test haskey(results[name], "improvement")
            end
        end
        
        @testset "Utility Functions" begin
            x_test = rand(Float32, 784, 5)
            
            # Test uncertainty score functions
            laplace_scores = get_laplace_uncertainty_scores(trained_model, x_test)
            map_scores = get_map_uncertainty_scores(trained_model, x_test)
            
            @test length(laplace_scores) == 5
            @test length(map_scores) == 5
            @test all(score ≥ 0 for score in laplace_scores)
            @test all(score ≥ 0 for score in map_scores)
            
            # Test AUROC calculation
            scores_in = [0.1, 0.2, 0.15]
            scores_ood = [0.8, 0.9, 0.85]
            auroc = calculate_auroc_from_scores(scores_in, scores_ood)
            @test 0 ≤ auroc ≤ 1
            
            # Test ROC curve calculation
            fpr, tpr = calculate_roc_curve(scores_in, scores_ood)
            @test length(fpr) == length(tpr)
            @test fpr[1] == 0.0 && tpr[1] == 0.0  # Should start at origin
            
            # Test ECE calculation
            confidences = [0.6, 0.8, 0.9, 0.7]
            correct = [true, true, false, true]
            ece = calculate_ece(confidences, correct, 2)
            @test 0 ≤ ece ≤ 1
        end
        
        @testset "Error Handling for Laplace vs MAP" begin
            # Test with untrained model
            untrained_model = MNISTModel(4)
            test_data = rand(Float32, 784, 5)
            
            @test_throws ErrorException plot_laplace_vs_map_comparison(untrained_model, test_data)
            @test_throws ErrorException plot_confidence_comparison(untrained_model, test_data, rand(0:9, 5))
            @test_throws ErrorException plot_auroc_comparison(untrained_model, test_data, test_data)
            @test_throws ErrorException plot_laplace_improvement_analysis(untrained_model, test_data, [test_data], ["Test"])
        end
    end
end