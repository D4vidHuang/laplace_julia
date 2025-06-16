module BayesianMethods

using Flux
using Random
using Statistics
using LinearAlgebra
using Distributions
# Note: LaplaceRedux and Plots should be loaded by user when needed

export BayesianModel, HMCMethod, SWAGMethod, MAPMethod, LaplaceMethod,
       train_bayesian_model!, predict_bayesian, evaluate_bayesian_model,
       get_bayesian_uncertainty, compare_bayesian_methods,
       plot_bayesian_methods_comparison, plot_uncertainty_distributions,
       save_bayesian_results, get_parameter_vector, set_parameter_vector!,
       vector_to_param_dict, get_parameter_vector_from_dict, set_parameters_from_dict!

"""
Abstract type for Bayesian inference methods
"""
abstract type BayesianInferenceMethod end

"""
Hamiltonian Monte Carlo method
"""
struct HMCMethod <: BayesianInferenceMethod
    n_samples::Int
    n_warmup::Int
    step_size::Float64
    n_leapfrog::Int
end

HMCMethod(; n_samples=1000, n_warmup=200, step_size=0.01, n_leapfrog=10) = 
    HMCMethod(n_samples, n_warmup, step_size, n_leapfrog)

"""
Stochastic Weight Averaging - Gaussian (SWAG) method
"""
struct SWAGMethod <: BayesianInferenceMethod
    n_models::Int
    start_epoch::Int
    update_freq::Int
    max_rank::Int
end

SWAGMethod(; n_models=20, start_epoch=50, update_freq=5, max_rank=20) = 
    SWAGMethod(n_models, start_epoch, update_freq, max_rank)

"""
Maximum A Posteriori (MAP) method
"""
struct MAPMethod <: BayesianInferenceMethod
    weight_decay::Float64
end

MAPMethod(; weight_decay=0.01) = MAPMethod(weight_decay)

"""
Laplace approximation method (for comparison)
"""
struct LaplaceMethod <: BayesianInferenceMethod
    # No additional parameters needed as it uses LaplaceRedux
end

"""
Bayesian model container
"""
mutable struct BayesianModel
    nn::Chain
    method::BayesianInferenceMethod
    trained::Bool
    posterior_samples::Union{Nothing, Vector{Dict}}
    mean_weights::Union{Nothing, Dict}
    cov_weights::Union{Nothing, Matrix{Float64}}
    laplace_model::Union{Nothing, Any}
end

function BayesianModel(nn::Chain, method::BayesianInferenceMethod)
    return BayesianModel(nn, method, false, nothing, nothing, nothing, nothing)
end

"""
Train a Bayesian model using the specified inference method
"""
function train_bayesian_model!(model::BayesianModel, data, epochs::Int=100; 
                              lr::Float64=0.001, verbose::Bool=true)
    
    if model.method isa MAPMethod
        return train_map!(model, data, epochs; lr=lr, verbose=verbose)
    elseif model.method isa HMCMethod
        return train_hmc!(model, data; verbose=verbose)
    elseif model.method isa SWAGMethod
        return train_swag!(model, data, epochs; lr=lr, verbose=verbose)
    elseif model.method isa LaplaceMethod
        return train_laplace!(model, data, epochs; lr=lr, verbose=verbose)
    else
        error("Unknown Bayesian inference method: $(typeof(model.method))")
    end
end

"""
Train using MAP estimation with L2 regularization
"""
function train_map!(model::BayesianModel, data, epochs::Int; lr::Float64=0.001, verbose::Bool=true)
    opt = Flux.Adam(lr)
    weight_decay = model.method.weight_decay
    
    # Extract data
    X, y_train = if data isa Tuple
        data
    else
        # Convert from zip format
        xs = [d[1] for d in data]
        ys = [d[2] for d in data]
        (hcat(xs...), ys)
    end
    
    # Convert labels to one-hot if needed
    if y_train isa Vector{Int}
        n_classes = maximum(y_train)
        y_onehot = Flux.onehotbatch(y_train, 1:n_classes)
    else
        y_onehot = y_train
    end
    
    loss_fn(x, y) = Flux.Losses.logitcrossentropy(model.nn(x), y) + 
                    weight_decay * sum(sum(abs2, p) for p in Flux.params(model.nn))
    
    show_every = max(1, epochs ÷ 10)
    
    for epoch in 1:epochs
        total_loss = 0.0
        
        # Mini-batch training
        for batch in Flux.DataLoader((X, y_onehot), batchsize=64, shuffle=true)
            x_batch, y_batch = batch
            
            gs = gradient(Flux.params(model.nn)) do
                loss_fn(x_batch, y_batch)
            end
            
            Flux.update!(opt, Flux.params(model.nn), gs)
            total_loss += loss_fn(x_batch, y_batch)
        end
        
        if verbose && epoch % show_every == 0
            avg_loss = total_loss / length(Flux.DataLoader((X, y_onehot), batchsize=64))
            println("MAP Epoch $epoch: Loss = $(round(avg_loss, digits=4))")
        end
    end
    
    # Store the MAP estimate
    model.mean_weights = Dict()
    for (name, param) in zip(["W1", "b1", "W2", "b2", "W3", "b3"], Flux.params(model.nn))
        model.mean_weights[name] = copy(param)
    end
    
    model.trained = true
    return model
end

"""
Train using Hamiltonian Monte Carlo
"""
function train_hmc!(model::BayesianModel, data; verbose::Bool=true)
    # Convert data format
    X, y_train = if data isa Tuple
        data
    else
        xs = [d[1] for d in data]
        ys = [d[2] for d in data]
        (hcat(xs...), ys)
    end
    
    # Convert labels to one-hot if needed
    if y_train isa Vector{Int}
        n_classes = maximum(y_train)
        y_onehot = Flux.onehotbatch(y_train, 1:n_classes)
    else
        y_onehot = y_train
    end
    
    hmc_params = model.method
    
    # Get initial parameters
    initial_params = get_parameter_vector(model.nn)
    n_params = length(initial_params)
    
    # Define log posterior
    function log_posterior(params)
        set_parameter_vector!(model.nn, params)
        
        # Log likelihood
        try
            predictions = model.nn(X)
            log_lik = -Flux.Losses.logitcrossentropy(predictions, y_onehot)
            
            # Log prior (Gaussian with variance = 1)
            log_prior = -0.5 * sum(params.^2)
            
            return log_lik + log_prior
        catch
            return -Inf
        end
    end
    
    # Gradient of log posterior
    function grad_log_posterior(params)
        set_parameter_vector!(model.nn, params)
        
        gs = gradient(Flux.params(model.nn)) do
            -Flux.Losses.logitcrossentropy(model.nn(X), y_onehot)
        end
        
        grad_vec = get_gradient_vector(gs, model.nn)
        return grad_vec .- params  # Add prior gradient
    end
    
    # HMC sampling
    samples = hmc_sample(log_posterior, grad_log_posterior, initial_params, 
                        hmc_params.n_samples, hmc_params.n_warmup, 
                        hmc_params.step_size, hmc_params.n_leapfrog; verbose=verbose)
    
    # Store samples
    model.posterior_samples = []
    for sample in samples
        param_dict = vector_to_param_dict(sample, model.nn)
        push!(model.posterior_samples, param_dict)
    end
    
    # Compute posterior mean
    model.mean_weights = Dict()
    param_names = ["W1", "b1", "W2", "b2", "W3", "b3"]
    for name in param_names
        if haskey(model.posterior_samples[1], name)
            model.mean_weights[name] = mean([s[name] for s in model.posterior_samples])
        end
    end
    
    model.trained = true
    return model
end

"""
Train using SWAG (Stochastic Weight Averaging - Gaussian)
"""
function train_swag!(model::BayesianModel, data, epochs::Int; lr::Float64=0.001, verbose::Bool=true)
    swag_params = model.method
    opt = Flux.Adam(lr)
    
    # Convert data format
    X, y_train = if data isa Tuple
        data
    else
        xs = [d[1] for d in data]
        ys = [d[2] for d in data]
        (hcat(xs...), ys)
    end
    
    # Convert labels to one-hot if needed
    if y_train isa Vector{Int}
        n_classes = maximum(y_train)
        y_onehot = Flux.onehotbatch(y_train, 1:n_classes)
    else
        y_onehot = y_train
    end
    
    loss_fn(x, y) = Flux.Losses.logitcrossentropy(model.nn(x), y)
    
    # Storage for SWAG
    collected_models = []
    weight_sum = nothing
    weight_sq_sum = nothing
    n_collected = 0
    
    show_every = max(1, epochs ÷ 10)
    
    for epoch in 1:epochs
        total_loss = 0.0
        
        # Standard SGD training
        for batch in Flux.DataLoader((X, y_onehot), batchsize=64, shuffle=true)
            x_batch, y_batch = batch
            
            gs = gradient(Flux.params(model.nn)) do
                loss_fn(x_batch, y_batch)
            end
            
            Flux.update!(opt, Flux.params(model.nn), gs)
            total_loss += loss_fn(x_batch, y_batch)
        end
        
        # Collect models for SWAG after start_epoch
        if epoch >= swag_params.start_epoch && (epoch - swag_params.start_epoch) % swag_params.update_freq == 0
            current_weights = get_parameter_vector(model.nn)
            
            if weight_sum === nothing
                weight_sum = copy(current_weights)
                weight_sq_sum = current_weights.^2
            else
                weight_sum .+= current_weights
                weight_sq_sum .+= current_weights.^2
            end
            
            push!(collected_models, copy(current_weights))
            n_collected += 1
            
            # Limit the number of collected models
            if length(collected_models) > swag_params.n_models
                popfirst!(collected_models)
                # Recompute statistics
                weight_sum = sum(collected_models)
                weight_sq_sum = sum(w.^2 for w in collected_models)
                n_collected = length(collected_models)
            end
        end
        
        if verbose && epoch % show_every == 0
            avg_loss = total_loss / length(Flux.DataLoader((X, y_onehot), batchsize=64))
            println("SWAG Epoch $epoch: Loss = $(round(avg_loss, digits=4)), Collected = $n_collected")
        end
    end
    
    if n_collected > 1
        # Compute SWAG statistics
        mean_weights = weight_sum / n_collected
        var_weights = weight_sq_sum / n_collected - mean_weights.^2
        
        # Low-rank component (simplified)
        deviations = hcat([w - mean_weights for w in collected_models]...)
        U, S, V = svd(deviations)
        rank = min(swag_params.max_rank, size(U, 2), n_collected - 1)
        
        # Store results
        model.mean_weights = vector_to_param_dict(mean_weights, model.nn)
        
        # Approximate covariance (diagonal + low-rank)
        model.cov_weights = Diagonal(var_weights) + U[:, 1:rank] * Diagonal(S[1:rank].^2 / (n_collected - 1)) * U[:, 1:rank]'
        
        # Store collected samples for prediction
        model.posterior_samples = [vector_to_param_dict(w, model.nn) for w in collected_models]
    else
        error("Not enough models collected for SWAG. Increase epochs or decrease start_epoch.")
    end
    
    model.trained = true
    return model
end

"""
Train using Laplace approximation (via LaplaceRedux)
"""
function train_laplace!(model::BayesianModel, data, epochs::Int; lr::Float64=0.001, verbose::Bool=true)
    # First train with MAP
    map_method = MAPMethod(weight_decay=0.01)
    temp_model = BayesianModel(deepcopy(model.nn), map_method)
    train_map!(temp_model, data, epochs; lr=lr, verbose=verbose)
    
    # Copy trained weights back
    Flux.loadparams!(model.nn, Flux.params(temp_model.nn))
    
    # Apply Laplace approximation
    # Note: LaplaceRedux should be loaded before calling this function
    try
        LaplaceRedux  # Check if LaplaceRedux is available
        
        # Convert data to LaplaceRedux format
        data_laplace = if data isa Tuple
            X, y_train = data
            x_vec = [X[:, i] for i in 1:size(X, 2)]
            zip(x_vec, y_train)
        else
            data
        end
        
        la = LaplaceRedux.Laplace(model.nn; likelihood=:classification)
        LaplaceRedux.fit!(la, data_laplace)
        LaplaceRedux.optimize_prior!(la; verbosity=verbose ? 1 : 0, n_steps=50)
        
        model.laplace_model = la
    catch e
        if e isa UndefVarError
            error("LaplaceRedux not loaded. Please run 'using LaplaceRedux' before using Laplace method.")
        else
            rethrow(e)
        end
    end
    model.trained = true
    return model
end

"""
Make predictions with uncertainty using the trained Bayesian model
"""
function predict_bayesian(model::BayesianModel, x_test; n_samples::Int=100)
    if !model.trained
        error("Model must be trained first")
    end
    
    if model.method isa LaplaceMethod
        return predict(model.laplace_model, x_test; link_approx=:probit)
    elseif model.method isa MAPMethod
        return [softmax(model.nn(x_test[:, i])) for i in 1:size(x_test, 2)]
    elseif model.method isa HMCMethod
        return predict_with_samples(model, x_test, model.posterior_samples; n_samples=n_samples)
    elseif model.method isa SWAGMethod
        return predict_swag(model, x_test; n_samples=n_samples)
    else
        error("Unknown method for prediction")
    end
end

"""
Predict using posterior samples
"""
function predict_with_samples(model::BayesianModel, x_test, samples; n_samples::Int=100)
    n_test = size(x_test, 2)
    n_use = min(n_samples, length(samples))
    sample_indices = randperm(length(samples))[1:n_use]
    
    predictions = []
    
    for i in 1:n_test
        x_i = x_test[:, i]
        sample_preds = []
        
        for idx in sample_indices
            set_parameters_from_dict!(model.nn, samples[idx])
            pred = softmax(model.nn(x_i))
            push!(sample_preds, pred)
        end
        
        # Average predictions
        mean_pred = mean(sample_preds)
        push!(predictions, mean_pred)
    end
    
    return predictions
end

"""
Predict using SWAG by sampling from the Gaussian posterior
"""
function predict_swag(model::BayesianModel, x_test; n_samples::Int=100)
    if model.cov_weights === nothing
        error("SWAG covariance not computed")
    end
    
    n_test = size(x_test, 2)
    predictions = []
    
    mean_vec = get_parameter_vector_from_dict(model.mean_weights, model.nn)
    
    for i in 1:n_test
        x_i = x_test[:, i]
        sample_preds = []
        
        for _ in 1:n_samples
            # Sample from Gaussian posterior
            if size(model.cov_weights, 1) == length(mean_vec)
                sampled_params = rand(MvNormal(mean_vec, Hermitian(model.cov_weights + 1e-6*I)))
            else
                # Fallback to diagonal approximation
                sampled_params = mean_vec + randn(length(mean_vec)) * 0.1
            end
            
            set_parameter_vector!(model.nn, sampled_params)
            pred = softmax(model.nn(x_i))
            push!(sample_preds, pred)
        end
        
        # Average predictions
        mean_pred = mean(sample_preds)
        push!(predictions, mean_pred)
    end
    
    return predictions
end

"""
Get uncertainty scores for Bayesian methods
"""
function get_bayesian_uncertainty(model::BayesianModel, x_test; method::Symbol=:entropy, n_samples::Int=100)
    predictions = predict_bayesian(model, x_test; n_samples=n_samples)
    
    if method == :entropy
        return [-sum(p .* log.(p .+ 1e-8)) for p in predictions]
    elseif method == :max_prob
        return [1.0 - maximum(p) for p in predictions]
    elseif method == :variance
        if model.method isa MAPMethod
            # For MAP, return entropy as proxy for variance
            return [-sum(p .* log.(p .+ 1e-8)) for p in predictions]
        else
            # For other methods, compute predictive variance
            return compute_predictive_variance(model, x_test; n_samples=n_samples)
        end
    else
        error("Unknown uncertainty method: $method")
    end
end

"""
Compute predictive variance for uncertainty quantification
"""
function compute_predictive_variance(model::BayesianModel, x_test; n_samples::Int=100)
    if model.method isa LaplaceMethod
        # Use Laplace predictive variance
        predictions = predict_bayesian(model, x_test)
        return [sum(p .* (1 .- p)) for p in predictions]  # Approximate variance
    else
        # Use sample-based variance
        n_test = size(x_test, 2)
        variances = []
        
        for i in 1:n_test
            x_i = x_test[:, i]
            sample_preds = []
            
            if model.method isa HMCMethod
                sample_indices = randperm(length(model.posterior_samples))[1:min(n_samples, length(model.posterior_samples))]
                for idx in sample_indices
                    set_parameters_from_dict!(model.nn, model.posterior_samples[idx])
                    pred = softmax(model.nn(x_i))
                    push!(sample_preds, pred)
                end
            elseif model.method isa SWAGMethod
                mean_vec = get_parameter_vector_from_dict(model.mean_weights, model.nn)
                for _ in 1:n_samples
                    if size(model.cov_weights, 1) == length(mean_vec)
                        sampled_params = rand(MvNormal(mean_vec, Hermitian(model.cov_weights + 1e-6*I)))
                    else
                        sampled_params = mean_vec + randn(length(mean_vec)) * 0.1
                    end
                    set_parameter_vector!(model.nn, sampled_params)
                    pred = softmax(model.nn(x_i))
                    push!(sample_preds, pred)
                end
            end
            
            # Compute variance across samples
            mean_pred = mean(sample_preds)
            variance = mean([sum((p - mean_pred).^2) for p in sample_preds])
            push!(variances, variance)
        end
        
        return variances
    end
end

"""
Evaluate Bayesian model performance
"""
function evaluate_bayesian_model(model::BayesianModel, x_test, y_test; n_samples::Int=100)
    predictions = predict_bayesian(model, x_test; n_samples=n_samples)
    
    # Calculate accuracy
    pred_classes = [argmax(p) for p in predictions]
    if minimum(y_test) == 0  # 0-indexed labels
        pred_classes = pred_classes .- 1
    end
    accuracy = mean(pred_classes .== y_test)
    
    # Calculate uncertainty metrics
    entropies = [-sum(p .* log.(p .+ 1e-8)) for p in predictions]
    confidences = [maximum(p) for p in predictions]
    
    # Separate correct and incorrect predictions
    correct_mask = pred_classes .== y_test
    correct_entropies = entropies[correct_mask]
    incorrect_entropies = entropies[.!correct_mask]
    
    return Dict(
        "accuracy" => accuracy,
        "predictions" => predictions,
        "predicted_classes" => pred_classes,
        "avg_entropy" => mean(entropies),
        "avg_confidence" => mean(confidences),
        "correct_avg_entropy" => isempty(correct_entropies) ? 0.0 : mean(correct_entropies),
        "incorrect_avg_entropy" => isempty(incorrect_entropies) ? 0.0 : mean(incorrect_entropies),
        "entropies" => entropies,
        "confidences" => confidences
    )
end

"""
Compare multiple Bayesian methods
"""
function compare_bayesian_methods(nn_architecture, data_train, data_test, y_test; 
                                methods=[:map, :hmc, :swag, :laplace], 
                                epochs=50, verbose=true)
    
    results = Dict()
    
    for method_name in methods
        println("\n" * "="^50)
        println("Training with method: $method_name")
        println("="^50)
        
        # Create method instance
        if method_name == :map
            method = MAPMethod(weight_decay=0.01)
        elseif method_name == :hmc
            method = HMCMethod(n_samples=500, n_warmup=100, step_size=0.001, n_leapfrog=5)
        elseif method_name == :swag
            method = SWAGMethod(n_models=15, start_epoch=30, update_freq=3)
        elseif method_name == :laplace
            method = LaplaceMethod()
        else
            error("Unknown method: $method_name")
        end
        
        # Create model
        nn = deepcopy(nn_architecture)
        model = BayesianModel(nn, method)
        
        # Train model
        try
            train_bayesian_model!(model, data_train, epochs; verbose=verbose)
            
            # Evaluate model
            eval_results = evaluate_bayesian_model(model, data_test, y_test)
            eval_results["method"] = method_name
            eval_results["trained_successfully"] = true
            
            results[method_name] = eval_results
            
            println("Results for $method_name:")
            println("  Accuracy: $(round(eval_results["accuracy"] * 100, digits=2))%")
            println("  Avg Entropy: $(round(eval_results["avg_entropy"], digits=4))")
            println("  Avg Confidence: $(round(eval_results["avg_confidence"], digits=4))")
            
        catch e
            println("Failed to train $method_name: $e")
            results[method_name] = Dict("method" => method_name, "trained_successfully" => false, "error" => string(e))
        end
    end
    
    return results
end

# Utility functions

function get_parameter_vector(nn::Chain)
    params = Flux.params(nn)
    return vcat([vec(p) for p in params]...)
end

function set_parameter_vector!(nn::Chain, param_vec::Vector)
    params = Flux.params(nn)
    idx = 1
    for p in params
        len = length(p)
        p .= reshape(param_vec[idx:idx+len-1], size(p))
        idx += len
    end
end

function get_gradient_vector(grads, nn::Chain)
    params = Flux.params(nn)
    return vcat([vec(grads[p]) for p in params]...)
end

function vector_to_param_dict(param_vec::Vector, nn::Chain)
    params = Flux.params(nn)
    param_dict = Dict()
    param_names = ["W1", "b1", "W2", "b2", "W3", "b3"]
    
    idx = 1
    for (i, p) in enumerate(params)
        len = length(p)
        if i <= length(param_names)
            param_dict[param_names[i]] = reshape(param_vec[idx:idx+len-1], size(p))
        end
        idx += len
    end
    
    return param_dict
end

function get_parameter_vector_from_dict(param_dict::Dict, nn::Chain)
    param_names = ["W1", "b1", "W2", "b2", "W3", "b3"]
    param_vec = Float64[]
    
    for name in param_names
        if haskey(param_dict, name)
            append!(param_vec, vec(param_dict[name]))
        end
    end
    
    return param_vec
end

function set_parameters_from_dict!(nn::Chain, param_dict::Dict)
    params = Flux.params(nn)
    param_names = ["W1", "b1", "W2", "b2", "W3", "b3"]
    
    for (i, p) in enumerate(params)
        if i <= length(param_names) && haskey(param_dict, param_names[i])
            p .= param_dict[param_names[i]]
        end
    end
end

"""
Simple HMC sampler implementation
"""
function hmc_sample(log_posterior, grad_log_posterior, initial_params, n_samples, n_warmup, step_size, n_leapfrog; verbose=true)
    samples = []
    current_params = copy(initial_params)
    current_log_p = log_posterior(current_params)
    
    n_accept = 0
    
    for i in 1:(n_samples + n_warmup)
        # Sample momentum
        momentum = randn(length(current_params))
        
        # Leapfrog integration
        new_params = copy(current_params)
        new_momentum = copy(momentum)
        
        # Half step for momentum
        grad = grad_log_posterior(new_params)
        new_momentum .+= 0.5 * step_size * grad
        
        # Full steps
        for _ in 1:n_leapfrog
            new_params .+= step_size * new_momentum
            if new_params != new_params  # Check for NaN
                break
            end
            grad = grad_log_posterior(new_params)
            new_momentum .+= step_size * grad
        end
        
        # Half step for momentum
        grad = grad_log_posterior(new_params)
        new_momentum .-= 0.5 * step_size * grad
        
        # Metropolis acceptance
        new_log_p = log_posterior(new_params)
        
        if !isfinite(new_log_p)
            new_log_p = -Inf
        end
        
        # Kinetic energy difference
        old_kinetic = 0.5 * sum(momentum.^2)
        new_kinetic = 0.5 * sum(new_momentum.^2)
        
        log_alpha = new_log_p - current_log_p - new_kinetic + old_kinetic
        
        if log(rand()) < log_alpha
            current_params = new_params
            current_log_p = new_log_p
            n_accept += 1
        end
        
        # Store sample after warmup
        if i > n_warmup
            push!(samples, copy(current_params))
        end
        
        if verbose && i % max(1, (n_samples + n_warmup) ÷ 10) == 0
            accept_rate = n_accept / i
            println("HMC Step $i: Accept rate = $(round(accept_rate * 100, digits=1))%, Log P = $(round(current_log_p, digits=2))")
        end
    end
    
    final_accept_rate = n_accept / (n_samples + n_warmup)
    if verbose
        println("Final HMC acceptance rate: $(round(final_accept_rate * 100, digits=1))%")
    end
    
    return samples
end

"""
Visualization functions for comparing Bayesian methods
"""
function plot_bayesian_methods_comparison(results::Dict; save_plots=true)
    # Note: Plots should be loaded before calling this function
    
    methods = collect(keys(results))
    
    # Extract metrics
    accuracies = [results[m]["accuracy"] for m in methods if results[m]["trained_successfully"]]
    entropies = [results[m]["avg_entropy"] for m in methods if results[m]["trained_successfully"]]
    confidences = [results[m]["avg_confidence"] for m in methods if results[m]["trained_successfully"]]
    successful_methods = [m for m in methods if results[m]["trained_successfully"]]
    
    # Accuracy comparison
    p1 = bar(string.(successful_methods), accuracies, 
             title="Accuracy Comparison", ylabel="Accuracy",
             color=[:blue, :red, :green, :orange][1:length(successful_methods)],
             alpha=0.7)
    ylims!(p1, (0, 1))
    
    # Uncertainty comparison
    p2 = bar(string.(successful_methods), entropies,
             title="Average Uncertainty", ylabel="Entropy",
             color=[:blue, :red, :green, :orange][1:length(successful_methods)],
             alpha=0.7)
    
    # Confidence comparison
    p3 = bar(string.(successful_methods), confidences,
             title="Average Confidence", ylabel="Confidence",
             color=[:blue, :red, :green, :orange][1:length(successful_methods)],
             alpha=0.7)
    ylims!(p3, (0, 1))
    
    # Combined plot
    summary_plot = plot(p1, p2, p3, layout=(1, 3), size=(900, 300))
    
    if save_plots
        savefig(summary_plot, "bayesian_methods_comparison.png")
        println("Saved comparison plot: bayesian_methods_comparison.png")
    end
    
    return summary_plot
end

function plot_uncertainty_distributions(results::Dict, x_test, model_name="Model")
    # Note: Plots should be loaded before calling this function
    
    plt_list = []
    
    for (method_name, result) in results
        if result["trained_successfully"]
            entropies = result["entropies"]
            
            p = histogram(entropies, alpha=0.7, bins=20, normalize=:probability,
                         title="$method_name Uncertainty", 
                         xlabel="Entropy", ylabel="Density",
                         label="$method_name")
            
            push!(plt_list, p)
        end
    end
    
    if length(plt_list) > 0
        combined_plot = plot(plt_list..., 
                           layout=(2, 2), size=(800, 600))
        return combined_plot
    else
        error("No successful training results to plot")
    end
end

function save_bayesian_results(results::Dict, filename="bayesian_results.txt")
    open(filename, "w") do f
        write(f, "Bayesian Methods Comparison Results\n")
        write(f, "=" ^ 40 * "\n\n")
        
        for (method, result) in results
            write(f, "Method: $method\n")
            if result["trained_successfully"]
                write(f, "  Status: Successfully trained\n")
                write(f, "  Accuracy: $(round(result["accuracy"] * 100, digits=2))%\n")
                write(f, "  Average Entropy: $(round(result["avg_entropy"], digits=4))\n")
                write(f, "  Average Confidence: $(round(result["avg_confidence"], digits=4))\n")
                
                if haskey(result, "correct_avg_entropy") && haskey(result, "incorrect_avg_entropy")
                    write(f, "  Correct Predictions Entropy: $(round(result["correct_avg_entropy"], digits=4))\n")
                    write(f, "  Incorrect Predictions Entropy: $(round(result["incorrect_avg_entropy"], digits=4))\n")
                end
            else
                write(f, "  Status: Training failed\n")
                write(f, "  Error: $(result["error"])\n")
            end
            write(f, "\n")
        end
        
        # Summary statistics
        successful_results = [r for r in values(results) if r["trained_successfully"]]
        if length(successful_results) > 0
            accuracies = [r["accuracy"] for r in successful_results]
            best_accuracy = maximum(accuracies)
            best_method = [k for (k,v) in results if v["trained_successfully"] && v["accuracy"] == best_accuracy][1]
            
            write(f, "Summary:\n")
            write(f, "  Best performing method: $best_method ($(round(best_accuracy*100, digits=2))% accuracy)\n")
            write(f, "  Successfully trained methods: $(sum([r["trained_successfully"] for r in values(results)]))\n")
        end
    end
    
    println("Results saved to: $filename")
end

end # module