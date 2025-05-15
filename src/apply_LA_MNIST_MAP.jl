using Flux
using LaplaceRedux
using JLD2
using MLDatasets

# Load trained model
function load_map_model()
    model = LeNet()
    model_state = JLD2.load("mnist_map_model.jld2", "model_state")
    Flux.loadmodel!(model, model_state)
    return model
end

# LeNet architecture (same as training)
function LeNet()
    Chain(
        Conv((5, 5), 1=>6, relu, pad=2),
        MaxPool((2, 2)),
        Conv((5, 5), 6=>16, relu),
        MaxPool((2, 2)),
        flatten,
        Dense(400, 120, relu),
        Dense(120, 84, relu),
        Dense(84, 10)
    )
end

# Apply Laplace approximation
function apply_laplace()
    # Load data
    train_x, train_y = MNIST(:train)[:]
    train_x = reshape(train_x, 28, 28, 1, :)
    train_y = Flux.onehotbatch(train_y, 0:9)
    
    # Load model
    model = load_map_model()
    
    # Apply standard Laplace approximation (LA)
    la = Laplace(model, likelihood=:classification)
    fit!(la, (train_x, train_y))
    
    # Apply Laplace* (with optimized hyperparameters, assuming subset selection)
    la_star = Laplace(model, likelihood=:classification, subset_of_weights=:last_layer)
    fit!(la_star, (train_x, train_y))
    
    # Save Laplace models
    JLD2.save("mnist_la.jld2", "la", la)
    JLD2.save("mnist_la_star.jld2", "la_star", la_star)
    
    return la, la_star
end

