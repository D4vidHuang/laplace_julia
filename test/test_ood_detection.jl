using Test
using Random
include("../src/MNISTClassifier.jl")
include("../src/OODDatasets.jl")
include("../src/OODDetection.jl")
using .MNISTClassifier
using .OODDatasets
using .OODDetection

@testset "OOD Detection Tests" begin
    Random.seed!(42)
    
    @testset "OODDetector Creation" begin
        # Create a simple trained model for testing
        model = MNISTModel(8)
        trained_model = train_mnist!(model, 1, 0.01, 20)
        
        @testset "Valid Methods" begin
            for method in [:entropy, :max_prob, :variance, :mutual_info]
                detector = OODDetector(trained_model; method=method)
                @test detector.method == method
                @test !detector.calibrated
                @test detector.threshold === nothing
                @test detector.in_distribution_scores === nothing
            end
        end
        
        @testset "Invalid Method" begin
            @test_throws ErrorException OODDetector(trained_model; method=:invalid_method)
        end
    end
    
    @testset "Uncertainty Score Calculation" begin
        # Create and train a small model
        model = MNISTModel(4)
        trained_model = train_mnist!(model, 1, 0.01, 10)
        
        # Create test data
        test_x = rand(Float32, 784, 5)
        
        @testset "Entropy Method" begin
            detector = OODDetector(trained_model; method=:entropy)
            scores = get_uncertainty_scores(detector, test_x)
            
            @test length(scores) == 5
            @test all(score ≥ 0 for score in scores)  # Entropy is non-negative
            @test all(isfinite(score) for score in scores)  # Should be finite
        end
        
        @testset "Max Probability Method" begin
            detector = OODDetector(trained_model; method=:max_prob)
            scores = get_uncertainty_scores(detector, test_x)
            
            @test length(scores) == 5
            @test all(0 ≤ score ≤ 1 for score in scores)  # 1 - max_prob ∈ [0,1]
        end
        
        @testset "Variance Method" begin
            detector = OODDetector(trained_model; method=:variance)
            scores = get_uncertainty_scores(detector, test_x)
            
            @test length(scores) == 5
            @test all(score ≥ 0 for score in scores)  # Variance is non-negative
        end
        
        @testset "Single Sample Input" begin
            detector = OODDetector(trained_model; method=:entropy)
            single_sample = rand(Float32, 784)
            scores = get_uncertainty_scores(detector, single_sample)
            
            @test length(scores) == 1
            @test scores[1] ≥ 0
        end
    end
    
    @testset "Threshold Fitting" begin
        # Create and train model
        model = MNISTModel(4)
        trained_model = train_mnist!(model, 1, 0.01, 15)
        
        detector = OODDetector(trained_model; method=:entropy)
        
        # Create validation data
        val_x = rand(Float32, 784, 10)
        
        @testset "Basic Threshold Fitting" begin
            threshold = fit_ood_threshold!(detector, val_x; percentile=95.0)
            
            @test detector.calibrated
            @test detector.threshold !== nothing
            @test detector.threshold == threshold
            @test detector.in_distribution_scores !== nothing
            @test length(detector.in_distribution_scores) == 10
        end
        
        @testset "Different Percentiles" begin
            for percentile in [90.0, 95.0, 99.0]
                detector_test = OODDetector(trained_model; method=:entropy)
                threshold = fit_ood_threshold!(detector_test, val_x; percentile=percentile)
                
                @test threshold ≥ 0
                @test detector_test.calibrated
            end
        end
        
        @testset "Untrained Model Error" begin
            untrained_model = MNISTModel(4)
            detector_untrained = OODDetector(untrained_model; method=:entropy)
            
            @test_throws ErrorException fit_ood_threshold!(detector_untrained, val_x)
        end
    end
    
    @testset "OOD Detection" begin
        # Create and train model
        model = MNISTModel(4)
        trained_model = train_mnist!(model, 1, 0.01, 15)
        
        detector = OODDetector(trained_model; method=:entropy)
        
        # Fit threshold
        val_x = rand(Float32, 784, 10)
        fit_ood_threshold!(detector, val_x; percentile=95.0)
        
        @testset "Basic OOD Detection" begin
            test_x = rand(Float32, 784, 8)
            ood_predictions, scores = detect_ood(detector, test_x)
            
            @test length(ood_predictions) == 8
            @test length(scores) == 8
            @test all(pred isa Bool for pred in ood_predictions)
            @test all(score ≥ 0 for score in scores)
        end
        
        @testset "Uncalibrated Detector Error" begin
            uncalibrated_detector = OODDetector(trained_model; method=:entropy)
            test_x = rand(Float32, 784, 5)
            
            @test_throws ErrorException detect_ood(uncalibrated_detector, test_x)
        end
    end
    
    @testset "OOD Evaluation" begin
        # Create and train model
        model = MNISTModel(4)
        trained_model = train_mnist!(model, 1, 0.01, 15)
        
        detector = OODDetector(trained_model; method=:entropy)
        
        # Create test data
        x_in = rand(Float32, 784, 10)  # In-distribution
        x_ood = rand(Float32, 784, 8) .+ 1.0f0  # Slightly different for OOD
        
        @testset "Basic Evaluation" begin
            metrics = evaluate_ood_detection(detector, x_in, x_ood)
            
            @test haskey(metrics, "auroc")
            @test haskey(metrics, "aupr")
            @test haskey(metrics, "fpr_at_95tpr")
            @test haskey(metrics, "optimal_threshold")
            @test haskey(metrics, "accuracy")
            @test haskey(metrics, "precision")
            @test haskey(metrics, "recall")
            @test haskey(metrics, "f1_score")
            
            @test 0 ≤ metrics["auroc"] ≤ 1
            @test 0 ≤ metrics["aupr"] ≤ 1
            @test 0 ≤ metrics["fpr_at_95tpr"] ≤ 1
            @test 0 ≤ metrics["accuracy"] ≤ 1
        end
        
        @testset "Evaluation with Scores" begin
            metrics = evaluate_ood_detection(detector, x_in, x_ood; return_scores=true)
            
            @test haskey(metrics, "scores_in")
            @test haskey(metrics, "scores_ood")
            @test haskey(metrics, "scores_all")
            @test haskey(metrics, "y_true")
            
            @test length(metrics["scores_in"]) == 10
            @test length(metrics["scores_ood"]) == 8
            @test length(metrics["scores_all"]) == 18
            @test length(metrics["y_true"]) == 18
        end
    end
    
    @testset "Metrics Calculation" begin
        # Create synthetic test data
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]  # 4 in-dist, 4 OOD
        scores = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]  # Increasing scores
        
        @testset "Basic Metrics" begin
            metrics = calculate_ood_metrics(y_true, scores)
            
            @test haskey(metrics, "auroc")
            @test haskey(metrics, "aupr")
            @test haskey(metrics, "fpr_at_95tpr")
            @test haskey(metrics, "optimal_threshold")
            
            # For perfect separation, AUROC should be 1.0
            @test metrics["auroc"] ≈ 1.0 atol=0.01
        end
        
        @testset "ROC Curve Data" begin
            metrics = calculate_ood_metrics(y_true, scores)
            
            @test haskey(metrics, "tpr_values")
            @test haskey(metrics, "fpr_values")
            @test haskey(metrics, "thresholds")
            
            @test length(metrics["tpr_values"]) == length(metrics["fpr_values"])
            @test length(metrics["tpr_values"]) == length(metrics["thresholds"])
            
            # Check ROC curve properties
            @test metrics["tpr_values"][1] == 0.0  # Starts at (0,0)
            @test metrics["fpr_values"][1] == 0.0
            @test metrics["tpr_values"][end] == 1.0  # Ends at (1,1)
        end
        
        @testset "Perfect vs Random Performance" begin
            # Perfect separation
            y_perfect = [0, 0, 1, 1]
            scores_perfect = [0.1, 0.2, 0.8, 0.9]
            metrics_perfect = calculate_ood_metrics(y_perfect, scores_perfect)
            
            # Random performance
            y_random = [0, 1, 0, 1]
            scores_random = [0.1, 0.2, 0.8, 0.9]
            metrics_random = calculate_ood_metrics(y_random, scores_random)
            
            @test metrics_perfect["auroc"] > metrics_random["auroc"]
        end
    end
    
    @testset "Method Comparison" begin
        # Create and train model
        model = MNISTModel(4)
        trained_model = train_mnist!(model, 1, 0.01, 10)
        
        x_in = rand(Float32, 784, 8)
        x_ood = rand(Float32, 784, 6) .+ 0.5f0
        
        @testset "Multiple Methods" begin
            methods = [:entropy, :max_prob]
            results = compare_ood_methods(trained_model, x_in, x_ood; methods=methods)
            
            @test haskey(results, :entropy)
            @test haskey(results, :max_prob)
            
            for method in methods
                @test haskey(results[method], "auroc")
                @test haskey(results[method], "aupr")
                @test 0 ≤ results[method]["auroc"] ≤ 1
            end
        end
    end
end