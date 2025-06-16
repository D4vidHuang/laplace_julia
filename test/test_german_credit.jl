using Test
using Random
include("../src/GermanCreditClassifier.jl")
using .GermanCreditClassifier

@testset "German Credit Classifier Tests" begin
    Random.seed!(42)
    
    @testset "Model Creation" begin
        model = GermanCreditModel(5, 2, 4)
        @test !model.trained
        @test model.la === nothing
        @test model.n_classes == 4
        @test length(model.nn) == 2  # 2 layers
    end
    
    @testset "Data Creation and Loading" begin
        # Test sample data creation
        test_file = "test_data.csv"
        create_sample_data(test_file)
        @test isfile(test_file)
        
        # Test data loading
        X, y_labels, x, y_train, unique_labels = load_german_credit_data(test_file)
        
        @test size(X, 2) == 2  # 2 features
        @test length(y_labels) == size(X, 1)
        @test length(x) == length(y_labels)
        @test length(y_train) == length(y_labels)
        @test length(unique_labels) == 4  # 4 classes
        
        # Check that all labels are in unique_labels
        @test all(label in unique_labels for label in y_labels)
        
        # Check one-hot encoding
        @test all(sum(y) == 1 for y in y_train)  # Each one-hot vector sums to 1
        @test all(all(val in [0, 1] for val in y) for y in y_train)  # Binary values
        
        # Clean up
        rm(test_file, force=true)
    end
    
    @testset "Model Training" begin
        test_file = "test_train_data.csv"
        create_sample_data(test_file)
        
        model = GermanCreditModel(3, 2, 4)
        
        # Train with few epochs for testing
        trained_model = train_german_credit!(model, test_file, 5)
        
        @test trained_model.trained
        @test trained_model.la !== nothing
        
        # Test prediction functionality
        X, _, _, _, _ = load_german_credit_data(test_file)
        predictions = predict_german_credit(trained_model, X[1:5, :])
        
        @test length(predictions) == 5
        @test all(length(p) == 4 for p in predictions)  # 4 classes
        @test all(sum(p) ≈ 1.0 for p in predictions)  # Probabilities sum to 1
        @test all(all(0 ≤ prob ≤ 1 for prob in p) for p in predictions)  # Valid probabilities
        
        # Clean up
        rm(test_file, force=true)
    end
    
    @testset "Model Evaluation" begin
        test_file = "test_eval_data.csv"
        create_sample_data(test_file)
        
        model = GermanCreditModel(3, 2, 4)
        trained_model = train_german_credit!(model, test_file, 3)
        
        results = evaluate_german_credit(trained_model, test_file)
        
        @test haskey(results, "accuracy")
        @test haskey(results, "avg_entropy")
        @test haskey(results, "predictions")
        @test haskey(results, "predicted_classes")
        @test haskey(results, "predicted_labels")
        @test haskey(results, "true_labels")
        @test haskey(results, "unique_labels")
        
        @test 0 ≤ results["accuracy"] ≤ 1
        @test results["avg_entropy"] ≥ 0
        @test length(results["predictions"]) == length(results["true_labels"])
        @test length(results["predicted_classes"]) == length(results["true_labels"])
        @test length(results["predicted_labels"]) == length(results["true_labels"])
        @test all(label in results["unique_labels"] for label in results["predicted_labels"])
        
        # Clean up
        rm(test_file, force=true)
    end
    
    @testset "Error Handling" begin
        model = GermanCreditModel()
        
        # Test prediction on untrained model
        @test_throws ErrorException predict_german_credit(model, rand(5, 2))
        
        # Test evaluation on untrained model
        @test_throws ErrorException evaluate_german_credit(model)
    end
end