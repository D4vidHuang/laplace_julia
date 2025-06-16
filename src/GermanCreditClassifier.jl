module GermanCreditClassifier

using Flux
using CSV
using DataFrames
using LaplaceRedux
using Random
using Statistics

export GermanCreditModel, load_german_credit_data, train_german_credit!, predict_german_credit, evaluate_german_credit, create_sample_data

"""
German Credit Classification Model with Laplace Approximation
"""
struct GermanCreditModel
    nn::Chain
    la::Union{Nothing, Laplace}
    trained::Bool
    n_classes::Int
end

function GermanCreditModel(n_hidden::Int=3, input_dim::Int=2, n_classes::Int=4)
    nn = Chain(
        Dense(input_dim, n_hidden, σ),
        Dense(n_hidden, n_classes)
    )
    
    return GermanCreditModel(nn, nothing, false, n_classes)
end

function load_german_credit_data(file_path::String="data1.csv")
    if !isfile(file_path)
        # Create sample data if file doesn't exist
        println("Creating sample German credit data...")
        create_sample_data(file_path)
    end
    
    df = CSV.read(file_path, DataFrame)
    
    # Extract features and labels
    x_matrix = Matrix(df[:, 1:2])  # First 2 columns as features
    y_labels = df[:, 3]  # Third column as labels
    
    # Convert to format for Flux
    x = [x_matrix[i, :] for i in 1:size(x_matrix, 1)]
    X = x_matrix'  # Transpose to (features, samples) format
    
    # One-hot encode labels
    unique_labels = sort(unique(y_labels))
    y_train = Flux.onehotbatch(y_labels, unique_labels)
    y_train = Flux.unstack(y_train', 1)
    
    return (X, y_labels, x, y_train, unique_labels)
end

function create_sample_data(file_path::String)
    # Create sample German credit dataset
    Random.seed!(42)
    n_samples = 100
    
    # Generate synthetic data with 4 classes
    data = []
    for class in 1:4
        n_class = n_samples ÷ 4
        for i in 1:n_class
            x1 = randn() + (class - 1) * 2
            x2 = randn() + (class - 1) * 1.5
            push!(data, [x1, x2, class])
        end
    end
    
    # Create DataFrame and save
    df = DataFrame(feature1=Float64[], feature2=Float64[], class=Int[])
    for row in data
        push!(df, (row[1], row[2], row[3]))
    end
    
    CSV.write(file_path, df)
    println("Sample data created at $file_path")
end

function train_german_credit!(model::GermanCreditModel, data_file::String="data1.csv", epochs::Int=200)
    # Load data
    X, y_labels, x, y_train, unique_labels = load_german_credit_data(data_file)
    
    # Prepare data
    data = zip(x, y_train)
    loss(x, y) = Flux.Losses.logitcrossentropy(model.nn(x), y)
    
    # Training
    opt = Flux.Adam()
    avg_loss(data) = mean(map(d -> loss(d[1], d[2]), data))
    show_every = max(1, epochs ÷ 10)
    
    for epoch = 1:epochs
        for d in data
            gs = gradient(Flux.params(model.nn)) do
                loss(d...)
            end
            Flux.update!(opt, Flux.params(model.nn), gs)
        end
        
        if epoch % show_every == 0
            println("Epoch $epoch")
            println("Average Loss: $(avg_loss(data))")
        end
    end
    
    # Fit Laplace approximation
    la = Laplace(model.nn; likelihood=:classification)
    fit!(la, data)
    optimize_prior!(la; verbosity=1, n_steps=100)
    
    return GermanCreditModel(model.nn, la, true, model.n_classes)
end

function predict_german_credit(model::GermanCreditModel, x_test; link_approx=:probit)
    if !model.trained || model.la === nothing
        error("Model must be trained first")
    end
    
    return predict(model.la, x_test; link_approx=link_approx)
end

function evaluate_german_credit(model::GermanCreditModel, data_file::String="data1.csv")
    if !model.trained || model.la === nothing
        error("Model must be trained first")
    end
    
    # Load data
    X, y_labels, x, y_train, unique_labels = load_german_credit_data(data_file)
    
    # Get predictions by processing each sample individually
    predictions = []
    for i in 1:size(X, 2)
        sample = X[:, i:i]  # Get single column as matrix
        pred = predict_german_credit(model, sample)
        push!(predictions, pred[1])  # Take the first (and only) prediction
    end
    
    pred_classes = [argmax(p) for p in predictions]
    
    # Map back to original labels
    pred_labels = [unique_labels[i] for i in pred_classes]
    
    # Calculate accuracy
    accuracy = mean(pred_labels .== y_labels)
    
    # Calculate uncertainty (entropy)
    entropies = [-sum(p .* log.(p .+ 1e-8)) for p in predictions]
    
    # Separate correct and incorrect predictions
    correct_mask = pred_labels .== y_labels
    correct_entropies = entropies[correct_mask]
    incorrect_entropies = entropies[.!correct_mask]
    
    results = Dict(
        "accuracy" => accuracy,
        "avg_entropy" => mean(entropies),
        "correct_avg_entropy" => isempty(correct_entropies) ? 0.0 : mean(correct_entropies),
        "incorrect_avg_entropy" => isempty(incorrect_entropies) ? 0.0 : mean(incorrect_entropies),
        "predictions" => predictions,
        "predicted_classes" => pred_classes,
        "predicted_labels" => pred_labels,
        "true_labels" => y_labels,
        "entropies" => entropies,
        "unique_labels" => unique_labels
    )
    
    return results
end

end # module