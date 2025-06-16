using Test
using Random
include("../src/Visualizations.jl")
include("../src/MNISTClassifier.jl") 
include("../src/GermanCreditClassifier.jl")
using .Visualizations
using .MNISTClassifier
using .GermanCreditClassifier

@testset "Visualization Tests" begin
    Random.seed!(42)
    
    @testset "MNIST Visualization" begin
        # Create and train a small MNIST model
        model = MNISTModel(8)
        trained_model = train_mnist!(model, 1, 0.01, 20)
        
        _, _, test_x, test_y, _, _ = load_mnist_data(20)
        
        # Test MNIST samples plot
        plot_obj = plot_mnist_samples(trained_model, test_x, test_y, 4)
        @test plot_obj !== nothing
        
        # Test prediction comparison
        comp_plot = plot_prediction_comparison(trained_model, test_x, 10)
        @test comp_plot !== nothing
    end
    
    @testset "German Credit Visualization" begin
        # Create sample data and train model
        test_file = "test_viz_data.csv"
        create_sample_data(test_file)
        
        model = GermanCreditModel(3, 2, 4)
        trained_model = train_german_credit!(model, test_file, 3)
        
        X, y_labels, _, _, unique_labels = load_german_credit_data(test_file)
        
        # Test decision boundary plot
        boundary_plot = plot_german_credit_decision_boundary(trained_model, X, y_labels, unique_labels)
        @test boundary_plot !== nothing
        
        # Test with specific target class
        target_plot = plot_german_credit_decision_boundary(trained_model, X, y_labels, unique_labels; target_class=1)
        @test target_plot !== nothing
        
        # Clean up
        rm(test_file, force=true)
    end
    
    @testset "Uncertainty Visualization" begin
        # Create sample data
        entropies = [0.1, 0.3, 0.8, 1.2, 0.2, 0.9, 1.5, 0.4]
        correct_mask = [true, true, false, false, true, false, false, true]
        
        # Test uncertainty histogram
        hist_plot = plot_uncertainty_histogram(entropies, correct_mask)
        @test hist_plot !== nothing
    end
    
    @testset "Training Progress Visualization" begin
        # Sample training data
        losses = [2.3, 1.8, 1.2, 0.9, 0.7, 0.5]
        accuracies = [0.1, 0.3, 0.5, 0.7, 0.8, 0.85]
        
        # Test loss only
        loss_plot = plot_training_progress(losses)
        @test loss_plot !== nothing
        
        # Test with accuracies
        full_plot = plot_training_progress(losses, accuracies)
        @test full_plot !== nothing
    end
    
    @testset "Plot Saving" begin
        # Create a simple plot
        test_plots = Dict(
            "test_plot" => plot([1, 2, 3], [1, 4, 9], title="Test Plot")
        )
        
        test_dir = "test_plots"
        save_all_plots(test_plots, test_dir)
        
        @test isdir(test_dir)
        @test isfile(joinpath(test_dir, "test_plot.png"))
        
        # Clean up
        rm(test_dir, recursive=true, force=true)
    end
    
    @testset "Error Handling" begin
        # Test with untrained model
        model = MNISTModel()
        
        @test_throws ErrorException plot_mnist_samples(model, rand(784, 6), [1, 2, 3, 4, 5, 6])
        @test_throws ErrorException plot_prediction_comparison(model, rand(784, 10))
        
        # Test German credit with untrained model
        gc_model = GermanCreditModel()
        @test_throws ErrorException plot_german_credit_decision_boundary(gc_model, rand(10, 2), [1, 1, 2, 2, 3, 3, 4, 4, 1, 2], [1, 2, 3, 4])
    end
end