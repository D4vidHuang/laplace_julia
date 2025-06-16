module MNISTClassifier

using Flux
using MLDatasets
using LaplaceRedux
using Random
using Statistics

export MNISTModel, train_mnist!, predict_mnist, evaluate_mnist

"""
MNIST Classification Model with Laplace Approximation
"""
struct MNISTModel
    nn::Chain
    la::Union{Nothing, Laplace}
    trained::Bool
end

function MNISTModel(n_hidden::Int=50)
    D = 28 * 28  # MNIST input size
    out_dim = 10  # 10 classes (0-9)
    
    nn = Chain(
        Dense(D, n_hidden, relu),
        Dense(n_hidden, n_hidden, relu),
        Dense(n_hidden, out_dim)
    )
    
    return MNISTModel(nn, nothing, false)
end

function load_mnist_data(n_samples::Int=5000)
    # Load MNIST dataset
    train_x, train_y = MLDatasets.MNIST(split=:train)[:]
    test_x, test_y = MLDatasets.MNIST(split=:test)[:]
    
    # Reshape and normalize data
    train_x = reshape(Float32.(train_x), 28*28, :) ./ 255.0f0
    test_x = reshape(Float32.(test_x), 28*28, :) ./ 255.0f0
    
    # Take subset if specified
    if n_samples > 0 && n_samples < size(train_x, 2)
        train_x = train_x[:, 1:n_samples]
        train_y = train_y[1:n_samples]
    end
    
    # Convert to vectors for LaplaceRedux
    x = [train_x[:, i] for i in 1:size(train_x, 2)]
    y_train = Flux.onehotbatch(train_y, 0:9)
    
    return (train_x, train_y, test_x, test_y, x, y_train)
end

function train_mnist!(model::MNISTModel, epochs::Int=50, lr::Float64=0.001, n_samples::Int=5000)
    # Load data
    train_x, train_y, _, _, x, y_train = load_mnist_data(n_samples)
    
    # Prepare data
    data = zip(x, y_train)
    loss(x, y) = Flux.Losses.logitcrossentropy(model.nn(x), y)
    
    # Training
    opt = Flux.Adam(lr)
    show_every = max(1, epochs ÷ 10)
    
    for epoch = 1:epochs
        loss_sum = 0
        
        # Mini-batch training
        for batch in Flux.DataLoader((train_x, y_train), batchsize=64, shuffle=true)
            x_batch, y_batch = batch
            gs = gradient(Flux.params(model.nn)) do
                loss(x_batch, y_batch)
            end
            Flux.update!(opt, Flux.params(model.nn), gs)
            loss_sum += loss(x_batch, y_batch)
        end
        
        if epoch % show_every == 0
            println("Epoch $epoch")
            println("Average Loss: $(loss_sum / length(Flux.DataLoader((train_x, y_train), batchsize=64)))")
            
            # Calculate accuracy
            predictions = model.nn(train_x)
            predicted_classes = Flux.onecold(predictions, 0:9)
            accuracy = mean(predicted_classes .== train_y)
            println("Accuracy: $(round(accuracy * 100, digits=2))%")
        end
    end
    
    # Fit Laplace approximation
    la = Laplace(model.nn; likelihood=:classification)
    fit!(la, data)
    optimize_prior!(la; verbosity=1, n_steps=100)
    
    return MNISTModel(model.nn, la, true)
end

function predict_mnist(model::MNISTModel, x_test; link_approx=:probit)
    if !model.trained || model.la === nothing
        error("Model must be trained first")
    end
    
    return predict(model.la, x_test; link_approx=link_approx)
end

function evaluate_mnist(model::MNISTModel, test_size::Int=1000)
    if !model.trained || model.la === nothing
        error("Model must be trained first")
    end
    
    _, _, test_x, test_y, _, _ = load_mnist_data()
    
    # Use subset of test data
    test_x_subset = test_x[:, 1:min(test_size, size(test_x, 2))]
    test_y_subset = test_y[1:min(test_size, length(test_y))]
    
    # Get predictions
    test_predictions = predict_mnist(model, test_x_subset)
    test_pred_classes = [argmax(p) - 1 for p in test_predictions]
    
    # Calculate accuracy
    accuracy = mean(test_pred_classes .== test_y_subset)
    
    # Calculate uncertainty (entropy)
    entropies = [-sum(p .* log.(p .+ 1e-8)) for p in test_predictions]
    
    # Separate correct and incorrect predictions
    correct_mask = test_pred_classes .== test_y_subset
    correct_entropies = entropies[correct_mask]
    incorrect_entropies = entropies[.!correct_mask]
    
    results = Dict(
        "accuracy" => accuracy,
        "avg_entropy" => mean(entropies),
        "correct_avg_entropy" => isempty(correct_entropies) ? 0.0 : mean(correct_entropies),
        "incorrect_avg_entropy" => isempty(incorrect_entropies) ? 0.0 : mean(incorrect_entropies),
        "predictions" => test_predictions,
        "predicted_classes" => test_pred_classes,
        "true_classes" => test_y_subset,
        "entropies" => entropies
    )
    
    return results
end

end # module