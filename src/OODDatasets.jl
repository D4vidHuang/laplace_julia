module OODDatasets

using MLDatasets
using Random
using Statistics
using LinearAlgebra

export load_ood_mnist, load_ood_german_credit, create_ood_german_credit

"""
Load out-of-distribution datasets for MNIST
Supported datasets: :fashionmnist, :cifar10, :notmnist, :corrupted_mnist
"""
function load_ood_mnist(dataset_type::Symbol, n_samples::Int=1000; seed::Int=42)
    Random.seed!(seed)
    
    if dataset_type == :fashionmnist
        return load_fashionmnist_ood(n_samples)
    elseif dataset_type == :cifar10
        return load_cifar10_ood(n_samples)
    elseif dataset_type == :notmnist
        return load_notmnist_ood(n_samples)
    elseif dataset_type == :corrupted_mnist
        return load_corrupted_mnist_ood(n_samples)
    elseif dataset_type == :uniform_noise
        return load_uniform_noise_ood(n_samples)
    elseif dataset_type == :gaussian_noise
        return load_gaussian_noise_ood(n_samples)
    else
        error("Unsupported OOD dataset type: $dataset_type. Supported: :fashionmnist, :cifar10, :notmnist, :corrupted_mnist, :uniform_noise, :gaussian_noise")
    end
end

function load_fashionmnist_ood(n_samples::Int)
    println("Loading FashionMNIST as OOD dataset...")
    
    # Load FashionMNIST
    test_x, test_y = MLDatasets.FashionMNIST(split=:test)[:]
    
    # Reshape and normalize like MNIST
    test_x = reshape(Float32.(test_x), 28*28, :) ./ 255.0f0
    
    # Sample subset
    if n_samples > 0 && n_samples < size(test_x, 2)
        indices = randperm(size(test_x, 2))[1:n_samples]
        test_x = test_x[:, indices]
        test_y = test_y[indices]
    end
    
    return test_x, test_y, "FashionMNIST"
end

function load_cifar10_ood(n_samples::Int)
    println("Loading CIFAR-10 as OOD dataset...")
    
    # Load CIFAR-10
    test_x, test_y = MLDatasets.CIFAR10(split=:test)[:]
    
    # Convert to grayscale and resize to 28x28
    # CIFAR-10 is 32x32x3, we'll convert to grayscale and resize
    test_x_gray = zeros(Float32, 28, 28, size(test_x, 4))
    
    for i in 1:size(test_x, 4)
        # Convert RGB to grayscale
        img = test_x[:, :, :, i]
        gray_img = 0.299f0 * img[:, :, 1] + 0.587f0 * img[:, :, 2] + 0.114f0 * img[:, :, 3]
        
        # Simple resize by taking every 32/28 ≈ 1.14th pixel
        resize_indices = round.(Int, range(1, 32, length=28))
        resized_img = gray_img[resize_indices, resize_indices]
        test_x_gray[:, :, i] = resized_img
    end
    
    # Reshape to MNIST format and normalize
    test_x = reshape(test_x_gray, 28*28, :) ./ 255.0f0
    
    # Sample subset
    if n_samples > 0 && n_samples < size(test_x, 2)
        indices = randperm(size(test_x, 2))[1:n_samples]
        test_x = test_x[:, indices]
        test_y = test_y[indices]
    end
    
    return test_x, test_y, "CIFAR-10"
end

function load_notmnist_ood(n_samples::Int)
    println("Creating NotMNIST-like dataset...")
    
    # Create synthetic "NotMNIST" data - letters instead of digits
    # We'll create simple geometric patterns that look letter-like
    test_x = zeros(Float32, 28*28, n_samples)
    test_y = rand(0:25, n_samples)  # 26 letters
    
    for i in 1:n_samples
        img = zeros(Float32, 28, 28)
        
        # Create letter-like patterns based on label
        letter_idx = test_y[i]
        
        if letter_idx < 5  # A-E: vertical lines
            img[5:23, 10:12] .= 1.0f0
            img[5:7, 8:15] .= 1.0f0
            img[13:15, 8:15] .= 1.0f0
        elseif letter_idx < 10  # F-J: horizontal lines
            img[10:12, 5:23] .= 1.0f0
            img[8:15, 5:7] .= 1.0f0
            img[8:15, 13:15] .= 1.0f0
        elseif letter_idx < 15  # K-O: diagonal lines
            for j in 1:20
                if j+7 <= 28 && 15-j >= 1
                    img[j+7, 15-j] = 1.0f0
                    img[j+7, 15+j] = 1.0f0
                end
            end
        elseif letter_idx < 20  # P-T: circles
            center_x, center_y = 14, 14
            for x in 1:28, y in 1:28
                if 8 < sqrt((x-center_x)^2 + (y-center_y)^2) < 10
                    img[x, y] = 1.0f0
                end
            end
        else  # U-Z: random patterns
            img[rand(1:28, 50), rand(1:28, 50)] .= 1.0f0
        end
        
        # Add noise
        img += 0.1f0 * randn(Float32, 28, 28)
        img = clamp.(img, 0.0f0, 1.0f0)
        
        test_x[:, i] = reshape(img, 28*28)
    end
    
    return test_x, test_y, "NotMNIST"
end

function load_corrupted_mnist_ood(n_samples::Int)
    println("Creating corrupted MNIST dataset...")
    
    # Load actual MNIST test data
    test_x, test_y = MLDatasets.MNIST(split=:test)[:]
    test_x = reshape(Float32.(test_x), 28*28, :) ./ 255.0f0
    
    # Sample subset
    if n_samples > 0 && n_samples < size(test_x, 2)
        indices = randperm(size(test_x, 2))[1:n_samples]
        test_x = test_x[:, indices]
        test_y = test_y[indices]
    else
        n_samples = size(test_x, 2)
    end
    
    # Apply various corruptions
    for i in 1:n_samples
        corruption_type = rand(1:5)
        img = reshape(test_x[:, i], 28, 28)
        
        if corruption_type == 1  # Salt and pepper noise
            noise_mask = rand(28, 28) .< 0.3
            img[noise_mask] = rand(Float32, sum(noise_mask))
        elseif corruption_type == 2  # Gaussian blur (simple)
            kernel = ones(Float32, 3, 3) ./ 9
            blurred = zeros(Float32, 28, 28)
            for x in 2:27, y in 2:27
                blurred[x, y] = sum(img[x-1:x+1, y-1:y+1] .* kernel)
            end
            img = blurred
        elseif corruption_type == 3  # Rotation simulation
            img = img'  # Simple transpose as rotation
        elseif corruption_type == 4  # Inversion
            img = 1.0f0 .- img
        else  # Heavy noise
            img += 0.5f0 * randn(Float32, 28, 28)
            img = clamp.(img, 0.0f0, 1.0f0)
        end
        
        test_x[:, i] = reshape(img, 28*28)
    end
    
    return test_x, test_y, "Corrupted MNIST"
end

function load_uniform_noise_ood(n_samples::Int)
    println("Creating uniform noise dataset...")
    
    test_x = rand(Float32, 28*28, n_samples)
    test_y = rand(0:9, n_samples)  # Random labels
    
    return test_x, test_y, "Uniform Noise"
end

function load_gaussian_noise_ood(n_samples::Int)
    println("Creating Gaussian noise dataset...")
    
    test_x = clamp.(0.5f0 .+ 0.2f0 * randn(Float32, 28*28, n_samples), 0.0f0, 1.0f0)
    test_y = rand(0:9, n_samples)  # Random labels
    
    return test_x, test_y, "Gaussian Noise"
end

"""
Load out-of-distribution datasets for German Credit
"""
function load_ood_german_credit(dataset_type::Symbol, n_samples::Int=100; seed::Int=42)
    Random.seed!(seed)
    
    if dataset_type == :shifted_distribution
        return create_shifted_distribution_ood(n_samples)
    elseif dataset_type == :different_scale
        return create_different_scale_ood(n_samples)
    elseif dataset_type == :uniform_random
        return create_uniform_random_ood(n_samples)
    elseif dataset_type == :outlier_clusters
        return create_outlier_clusters_ood(n_samples)
    elseif dataset_type == :high_noise
        return create_high_noise_ood(n_samples)
    else
        error("Unsupported OOD dataset type: $dataset_type. Supported: :shifted_distribution, :different_scale, :uniform_random, :outlier_clusters, :high_noise")
    end
end

function create_shifted_distribution_ood(n_samples::Int)
    println("Creating shifted distribution OOD dataset...")
    
    # Create data with shifted means
    X = zeros(Float32, 2, n_samples)
    y = zeros(Int, n_samples)
    
    for i in 1:n_samples
        # Shift the distribution significantly from original
        X[1, i] = randn() + 10  # Shift by +10
        X[2, i] = randn() - 8   # Shift by -8
        y[i] = rand(1:4)  # Random class
    end
    
    return X, y, "Shifted Distribution"
end

function create_different_scale_ood(n_samples::Int)
    println("Creating different scale OOD dataset...")
    
    # Create data with much larger variance
    X = zeros(Float32, 2, n_samples)
    y = zeros(Int, n_samples)
    
    for i in 1:n_samples
        # Much larger scale than training data
        X[1, i] = 5 * randn()  # 5x larger scale
        X[2, i] = 5 * randn()  # 5x larger scale
        y[i] = rand(1:4)
    end
    
    return X, y, "Different Scale"
end

function create_uniform_random_ood(n_samples::Int)
    println("Creating uniform random OOD dataset...")
    
    # Uniform distribution in a wide range
    X = zeros(Float32, 2, n_samples)
    y = zeros(Int, n_samples)
    
    for i in 1:n_samples
        X[1, i] = rand() * 20 - 10  # Uniform in [-10, 10]
        X[2, i] = rand() * 20 - 10  # Uniform in [-10, 10]
        y[i] = rand(1:4)
    end
    
    return X, y, "Uniform Random"
end

function create_outlier_clusters_ood(n_samples::Int)
    println("Creating outlier clusters OOD dataset...")
    
    # Create clusters far from training data
    X = zeros(Float32, 2, n_samples)
    y = zeros(Int, n_samples)
    
    # Define outlier cluster centers
    centers = [[-15.0, -15.0], [15.0, 15.0], [-15.0, 15.0], [15.0, -15.0]]
    
    for i in 1:n_samples
        center_idx = rand(1:4)
        center = centers[center_idx]
        
        X[1, i] = center[1] + randn()
        X[2, i] = center[2] + randn()
        y[i] = center_idx
    end
    
    return X, y, "Outlier Clusters"
end

function create_high_noise_ood(n_samples::Int)
    println("Creating high noise OOD dataset...")
    
    # Create data with extremely high noise
    X = zeros(Float32, 2, n_samples)
    y = zeros(Int, n_samples)
    
    for i in 1:n_samples
        # High noise around zero
        X[1, i] = 10 * randn()  # Very high variance
        X[2, i] = 10 * randn()  # Very high variance
        y[i] = rand(1:4)
    end
    
    return X, y, "High Noise"
end

"""
Create synthetic OOD data for German Credit with specified properties
"""
function create_ood_german_credit(;n_samples::Int=100, 
                                 distribution_type::Symbol=:shifted,
                                 shift_factor::Float64=5.0,
                                 noise_factor::Float64=1.0,
                                 seed::Int=42)
    Random.seed!(seed)
    
    X = zeros(Float32, 2, n_samples)
    y = zeros(Int, n_samples)
    
    if distribution_type == :shifted
        # Shifted Gaussian
        for i in 1:n_samples
            X[1, i] = randn() * noise_factor + shift_factor
            X[2, i] = randn() * noise_factor - shift_factor
            y[i] = rand(1:4)
        end
    elseif distribution_type == :scaled
        # Different scale
        for i in 1:n_samples
            X[1, i] = randn() * shift_factor * noise_factor
            X[2, i] = randn() * shift_factor * noise_factor
            y[i] = rand(1:4)
        end
    elseif distribution_type == :uniform
        # Uniform distribution
        for i in 1:n_samples
            X[1, i] = (rand() - 0.5) * 2 * shift_factor
            X[2, i] = (rand() - 0.5) * 2 * shift_factor
            y[i] = rand(1:4)
        end
    end
    
    return X, y, "Synthetic OOD ($distribution_type)"
end

end # module