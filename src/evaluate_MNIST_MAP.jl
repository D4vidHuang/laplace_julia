using Flux
using LaplaceRedux
using MLDatasets
using JLD2
using ROCAnalysis
using Statistics
using JSON

# LeNet architecture
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

# Load models
function load_models()
    # Load MAP model
    map_model = LeNet()
    model_state = JLD2.load("mnist_map_model.jld2", "model_state")
    Flux.loadmodel!(map_model, model_state)
    
    # Load LA and LA* models
    la = JLD2.load("mnist_la.jld2", "la")
    la_star = JLD2.load("mnist_la_star.jld2", "la_star")
    
    return map_model, la, la_star
end

# Evaluation function
function evaluate_models()
    # Load test data
    test_x, test_y = MNIST(:test)[:]
    test_x = reshape(test_x, 28, 28, 1, :)
    test_y = Flux.onehotbatch(test_y, 0:9)
    test_labels = Flux.onecold(test_y)
    
    # Load models
    map_model, la, la_star = load_models()
    
    # Function to compute metrics
    function compute_metrics(model, model_type, is_laplace=false)
        # Get predictions
        if is_laplace
            ŷ = predict(la, test_x)
        else
            ŷ = softmax(map_model(test_x))
        end
        
        # Accuracy
        preds = Flux.onecold(ŷ)
        accuracy = mean(preds .== test_labels)
        
        # Confidence (max predictive probability)
        confidence = mean(maximum(ŷ, dims=1))
        
        # AUROC for OOD detection (using confidence as score)
        # For simplicity, assume in-distribution vs. random noise as OOD
        noise_x = rand(Float32, 28, 28, 1, 1000)
        if is_laplace
            noise_ŷ = predict(la, noise_x)
        else
            noise_ŷ = softmax(map_model(noise_x))
        end
        scores = [maximum(ŷ, dims=1)[:]; maximum(noise_ŷ, dims=1)[:]]
        labels = [ones(Int, size(test_x, 4)); zeros(Int, 1000)]
        auroc = roc(scores, labels).auc
        
        return Dict(
            "model" => model_type,
            "accuracy" => accuracy,
            "confidence" => confidence,
            "auroc" => auroc
        )
    end
    
    # Compute metrics for all models
    results = []
    push!(results, compute_metrics(map_model, "MAP", false))
    push!(results, compute_metrics(la, "LA", true))
    push!(results, compute_metrics(la_star, "LA*", true))
    
    # Save results
    open("mnist_results.json", "w") do f
        JSON.print(f, results, 2)
    end
    
    return results
end

# Run evaluation
results = evaluate_models()
println("Evaluation Results:")
for res in results
    println("$(res["model"]):")
    println("  Accuracy: $(res["accuracy"])")
    println("  Confidence: $(res["confidence"])")
    println("  AUROC: $(res["auroc"])")
end