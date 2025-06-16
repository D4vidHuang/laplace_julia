using Test
using Random
include("../src/OODDatasets.jl")
using .OODDatasets

@testset "OOD Datasets Tests" begin
    Random.seed!(42)
    
    @testset "MNIST OOD Datasets" begin
        @testset "FashionMNIST OOD" begin
            x, y, name = load_ood_mnist(:fashionmnist, 50)
            
            @test size(x, 1) == 784  # 28*28
            @test size(x, 2) == 50
            @test length(y) == 50
            @test name == "FashionMNIST"
            @test all(0 ≤ pixel ≤ 1 for pixel in x)  # Normalized
        end
        
        @testset "CIFAR-10 OOD" begin
            x, y, name = load_ood_mnist(:cifar10, 30)
            
            @test size(x, 1) == 784  # Resized to 28*28
            @test size(x, 2) == 30
            @test length(y) == 30
            @test name == "CIFAR-10"
            @test all(0 ≤ pixel ≤ 1 for pixel in x)
        end
        
        @testset "NotMNIST OOD" begin
            x, y, name = load_ood_mnist(:notmnist, 25)
            
            @test size(x, 1) == 784
            @test size(x, 2) == 25
            @test length(y) == 25
            @test name == "NotMNIST"
            @test all(0 ≤ pixel ≤ 1 for pixel in x)
            @test all(0 ≤ label ≤ 25 for label in y)  # 26 letters
        end
        
        @testset "Corrupted MNIST OOD" begin
            x, y, name = load_ood_mnist(:corrupted_mnist, 20)
            
            @test size(x, 1) == 784
            @test size(x, 2) == 20
            @test length(y) == 20
            @test name == "Corrupted MNIST"
            @test all(0 ≤ pixel ≤ 1 for pixel in x)
        end
        
        @testset "Noise OOD Datasets" begin
            # Uniform noise
            x_uniform, y_uniform, name_uniform = load_ood_mnist(:uniform_noise, 30)
            @test size(x_uniform) == (784, 30)
            @test name_uniform == "Uniform Noise"
            @test all(0 ≤ pixel ≤ 1 for pixel in x_uniform)
            
            # Gaussian noise
            x_gaussian, y_gaussian, name_gaussian = load_ood_mnist(:gaussian_noise, 30)
            @test size(x_gaussian) == (784, 30)
            @test name_gaussian == "Gaussian Noise"
            @test all(0 ≤ pixel ≤ 1 for pixel in x_gaussian)
        end
        
        @testset "Invalid OOD Dataset" begin
            @test_throws ErrorException load_ood_mnist(:invalid_dataset, 10)
        end
    end
    
    @testset "German Credit OOD Datasets" begin
        @testset "Shifted Distribution OOD" begin
            x, y, name = load_ood_german_credit(:shifted_distribution, 50)
            
            @test size(x, 1) == 2  # 2 features
            @test size(x, 2) == 50
            @test length(y) == 50
            @test name == "Shifted Distribution"
            @test all(1 ≤ label ≤ 4 for label in y)
        end
        
        @testset "Different Scale OOD" begin
            x, y, name = load_ood_german_credit(:different_scale, 40)
            
            @test size(x, 1) == 2
            @test size(x, 2) == 40
            @test length(y) == 40
            @test name == "Different Scale"
            
            # Should have larger variance than typical data
            @test std(x[1, :]) > 2  # Larger scale
        end
        
        @testset "Uniform Random OOD" begin
            x, y, name = load_ood_german_credit(:uniform_random, 35)
            
            @test size(x, 1) == 2
            @test size(x, 2) == 35
            @test length(y) == 35
            @test name == "Uniform Random"
            
            # Should be roughly uniformly distributed
            @test all(-10 ≤ val ≤ 10 for val in x)
        end
        
        @testset "Outlier Clusters OOD" begin
            x, y, name = load_ood_german_credit(:outlier_clusters, 30)
            
            @test size(x, 1) == 2
            @test size(x, 2) == 30
            @test length(y) == 30
            @test name == "Outlier Clusters"
            
            # Should have some extreme values (outliers)
            @test any(abs.(x[1, :]) .> 10)  # Some outliers
        end
        
        @testset "High Noise OOD" begin
            x, y, name = load_ood_german_credit(:high_noise, 25)
            
            @test size(x, 1) == 2
            @test size(x, 2) == 25
            @test length(y) == 25
            @test name == "High Noise"
            
            # Should have high variance
            @test std(x[1, :]) > 5  # High noise
        end
        
        @testset "Invalid German Credit OOD Dataset" begin
            @test_throws ErrorException load_ood_german_credit(:invalid_dataset, 10)
        end
    end
    
    @testset "Custom OOD Creation" begin
        @testset "Shifted Distribution" begin
            x, y, name = create_ood_german_credit(n_samples=20, distribution_type=:shifted, shift_factor=3.0)
            
            @test size(x, 1) == 2
            @test size(x, 2) == 20
            @test length(y) == 20
            @test occursin("Synthetic OOD", name)
            
            # Check that data is shifted
            @test abs(mean(x[1, :]) - 3.0) < 1.0  # Approximately shifted by shift_factor
        end
        
        @testset "Scaled Distribution" begin
            x, y, name = create_ood_german_credit(n_samples=15, distribution_type=:scaled, shift_factor=2.0)
            
            @test size(x, 1) == 2
            @test size(x, 2) == 15
            @test length(y) == 15
            
            # Check that data has different scale
            @test std(x[1, :]) > 1.5  # Should have larger variance
        end
        
        @testset "Uniform Distribution" begin
            x, y, name = create_ood_german_credit(n_samples=25, distribution_type=:uniform, shift_factor=4.0)
            
            @test size(x, 1) == 2
            @test size(x, 2) == 25
            @test length(y) == 25
            
            # Check uniform distribution properties
            @test all(-4.0 ≤ val ≤ 4.0 for val in x)  # Within expected range
        end
    end
end