using Test
using Random
include("../src/MNISTClassifier.jl")
using .MNISTClassifier

@testset "MNIST Classifier Tests" begin
    Random.seed!(42)
    
    @testset "Model Creation" begin
        model = MNISTModel(32)
        @test !model.trained
        @test model.la === nothing
        @test length(model.nn) == 3  # 3 layers
    end
    
    @testset "Data Loading" begin
        train_x, train_y, test_x, test_y, x, y_train = load_mnist_data(100)
        
        @test size(train_x) == (784, 100)  # 28*28 = 784
        @test length(train_y) == 100
        @test size(test_x, 1) == 784
        @test length(x) == 100
        @test length(y_train) == 100
        
        # Check data normalization
        @test all(0 ≤ pixel ≤ 1 for pixel in train_x)
        @test all(0 ≤ pixel ≤ 1 for pixel in test_x)
        
        # Check labels are in correct range
        @test all(0 ≤ label ≤ 9 for label in train_y)
        @test all(0 ≤ label ≤ 9 for label in test_y)
    end
    
    @testset "Model Training" begin
        model = MNISTModel(16)  # Smaller model for faster testing
        
        # Train with very few epochs and samples for testing
        trained_model = train_mnist!(model, 2, 0.01, 50)
        
        @test trained_model.trained
        @test trained_model.la !== nothing
        
        # Test prediction functionality
        _, _, test_x, test_y, _, _ = load_mnist_data(50)
        predictions = predict_mnist(trained_model, test_x[:, 1:5])
        
        @test length(predictions) == 5
        @test all(length(p) == 10 for p in predictions)  # 10 classes
        @test all(sum(p) ≈ 1.0 for p in predictions)  # Probabilities sum to 1
        @test all(all(0 ≤ prob ≤ 1 for prob in p) for p in predictions)  # Valid probabilities
    end
    
    @testset "Model Evaluation" begin
        model = MNISTModel(8)  # Even smaller model
        trained_model = train_mnist!(model, 1, 0.01, 25)
        
        results = evaluate_mnist(trained_model, 10)
        
        @test haskey(results, "accuracy")
        @test haskey(results, "avg_entropy")
        @test haskey(results, "predictions")
        @test haskey(results, "predicted_classes")
        @test haskey(results, "true_classes")
        
        @test 0 ≤ results["accuracy"] ≤ 1
        @test results["avg_entropy"] ≥ 0
        @test length(results["predictions"]) == 10
        @test length(results["predicted_classes"]) == 10
        @test length(results["true_classes"]) == 10
    end
    
    @testset "Error Handling" begin
        model = MNISTModel()
        
        # Test prediction on untrained model
        @test_throws ErrorException predict_mnist(model, rand(784, 5))
        
        # Test evaluation on untrained model
        @test_throws ErrorException evaluate_mnist(model)
    end
end