using Flux
using Flux: onehotbatch, onecold
using MLDatasets
using Optimisers
using Optimisers: Adam
using Random
using Statistics
using CUDA
using JLD2
using ProgressMeter

# Set random seed for reproducibility
Random.seed!(42)

# LeNet architecture matching the Python implementation
function LeNet()
    Chain(
        # First convolution block: Conv -> ReLU -> MaxPool
        Conv((5, 5), 1=>6, pad=0),  # explicit padding=0
        x -> relu.(x),
        MaxPool((2, 2)),
        
        # Second convolution block: Conv -> ReLU -> MaxPool
        Conv((5, 5), 6=>16, pad=0),
        x -> relu.(x),
        MaxPool((2, 2)),
        
        # Flatten layer
        Flux.flatten,
        
        # Fully connected layers with explicit initialization
        Dense(16 * 4 * 4, 120, relu; init=Flux.glorot_uniform),
        Dense(120, 84, relu; init=Flux.glorot_uniform),
        Dense(84, 10; init=Flux.glorot_uniform)
    )
end

# Cosine annealing learning rate schedule
function cosine_annealing_lr(epoch, total_epochs, initial_lr)
    return initial_lr * 0.5 * (1 + cos(π * epoch / total_epochs))
end

# Calculate accuracy
function calculate_accuracy(model, data_loader)
    correct = 0
    total = 0
    for (x, y) in data_loader
        ŷ = model(x)
        correct += sum(onecold(ŷ, 0:9) .== onecold(y, 0:9))
        total += size(y, 2)
    end
    return (correct / total) * 100
end

# Training function with optimizations
function train_map(; epochs=5, initial_lr=0.0001, weight_decay=5e-4, batch_size=128)
    # Load and preprocess MNIST data
    println("Loading and preprocessing MNIST dataset...")
    train_x, train_y = MNIST(:train)[:]
    test_x, test_y = MNIST(:test)[:]
    
    # Convert to Float32 and normalize to [0,1]
    train_x = Float32.(reshape(train_x, 28, 28, 1, :)) ./ 255f0
    test_x = Float32.(reshape(test_x, 28, 28, 1, :)) ./ 255f0
    
    # Convert labels to one-hot format
    train_y = Flux.onehotbatch(train_y, 0:9)
    test_y = Flux.onehotbatch(test_y, 0:9)
    
    # Create data loaders
    train_data = Flux.DataLoader((train_x, train_y), batchsize=batch_size, shuffle=true)
    test_data = Flux.DataLoader((test_x, test_y), batchsize=batch_size)
    
    # Initialize model and optimizer
    model = LeNet()
    
    # Initialize optimizer with weight decay
    opt = Optimisers.setup(Adam(initial_lr), model)
    
    println("Starting training for $epochs epochs...")
    println("Initial learning rate: $initial_lr, Weight decay: $weight_decay")
    
    # Training loop
    best_acc = 0.0
    best_model = nothing
    patience = 3
    no_improve = 0
    
    for epoch in 1:epochs
        # Training phase
        model = Flux.trainmode!(model)  # Ensure training mode
        total_loss = 0.0
        num_batches = 0
        progress = Progress(length(train_data), "Epoch $epoch/$epochs:")
        
        for (x, y) in train_data
            # Forward pass and loss computation
            loss, grads = Flux.withgradient(model) do m
                ŷ = m(x)
                ce_loss = Flux.logitcrossentropy(ŷ, y)
                l2_loss = weight_decay * sum(sum(abs2, p) for p in Flux.params(m))
                ce_loss + l2_loss
            end
            
            # Update parameters
            Optimisers.update!(opt, model, grads[1])
            
            total_loss += loss
            num_batches += 1
            ProgressMeter.next!(progress, showvalues=[(:loss, round(loss, digits=4))])
        end
        
        # Evaluation phase
        model = Flux.testmode!(model)  # Ensure evaluation mode
        train_acc = 0.0
        test_acc = 0.0
        
        # Calculate training accuracy
        for (x, y) in train_data
            ŷ = model(x)
            train_acc += sum(onecold(ŷ, 0:9) .== onecold(y, 0:9)) / size(y, 2)
        end
        train_acc = train_acc / length(train_data) * 100
        
        # Calculate test accuracy
        for (x, y) in test_data
            ŷ = model(x)
            test_acc += sum(onecold(ŷ, 0:9) .== onecold(y, 0:9)) / size(y, 2)
        end
        test_acc = test_acc / length(test_data) * 100
        
        # Save best model and check for early stopping
        if test_acc > best_acc
            best_acc = test_acc
            best_model = deepcopy(model)
            no_improve = 0
        else
            no_improve += 1
            if no_improve >= patience
                println("\nEarly stopping triggered! No improvement for $patience epochs.")
                break
            end
        end
        
        avg_loss = total_loss / num_batches
        println("\nEpoch [$epoch/$epochs] - Loss: $(round(avg_loss, digits=4)) - Train Acc: $(round(train_acc, digits=2))% - Test Acc: $(round(test_acc, digits=2))%")
    end
    
    # Final results
    println("\nTraining completed!")
    println("Best Test Accuracy: $(round(best_acc, digits=2))%")
    
    # Save model
    mkpath("models")
    @save "models/MNIST_map.jld2" best_model
    println("Model saved to models/MNIST_map.jld2")
    
    return best_model
end

# Run training
model = train_map()