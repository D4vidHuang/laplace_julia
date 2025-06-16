module OODDetection

using Statistics
using LaplaceRedux
using LinearAlgebra

export OODDetector, fit_ood_threshold!, detect_ood, evaluate_ood_detection, 
       calculate_ood_metrics, get_uncertainty_scores

"""
Out-of-Distribution Detection using uncertainty quantification
"""
mutable struct OODDetector
    model::Any  # Trained model with Laplace approximation
    threshold::Union{Nothing, Float64}
    method::Symbol  # :entropy, :max_prob, :variance, :mutual_info
    calibrated::Bool
    in_distribution_scores::Union{Nothing, Vector{Float64}}
end

function OODDetector(model; method::Symbol=:entropy)
    if !(method in [:entropy, :max_prob, :variance, :mutual_info])
        error("Unsupported method: $method. Supported: :entropy, :max_prob, :variance, :mutual_info")
    end
    
    return OODDetector(model, nothing, method, false, nothing)
end

"""
Fit OOD detection threshold using in-distribution validation data
"""
function fit_ood_threshold!(detector::OODDetector, x_val; 
                           percentile::Float64=95.0, 
                           link_approx::Symbol=:probit)
    println("Fitting OOD threshold using $(detector.method) method...")
    
    if detector.model.la === nothing
        error("Model must have fitted Laplace approximation")
    end
    
    # Get uncertainty scores for in-distribution data
    scores = get_uncertainty_scores(detector, x_val; link_approx=link_approx)
    detector.in_distribution_scores = scores
    
    # Set threshold at specified percentile
    detector.threshold = quantile(scores, percentile / 100.0)
    detector.calibrated = true
    
    println("Threshold set to: $(round(detector.threshold, digits=4)) ($(percentile)th percentile)")
    
    return detector.threshold
end

"""
Get uncertainty scores based on the specified method
"""
function get_uncertainty_scores(detector::OODDetector, x_data; link_approx::Symbol=:probit)
    if detector.model.la === nothing
        error("Model must have fitted Laplace approximation")
    end
    
    # Handle different input formats
    if ndims(x_data) == 1
        x_data = reshape(x_data, :, 1)
    end
    
    predictions = predict(detector.model.la, x_data; link_approx=link_approx)
    
    scores = zeros(Float64, length(predictions))
    
    for (i, pred) in enumerate(predictions)
        if detector.method == :entropy
            # Predictive entropy (higher = more uncertain)
            scores[i] = -sum(pred .* log.(pred .+ 1e-8))
        elseif detector.method == :max_prob
            # Maximum probability (lower = more uncertain, so we use 1 - max_prob)
            scores[i] = 1.0 - maximum(pred)
        elseif detector.method == :variance
            # Variance of predictions (higher = more uncertain)
            mean_pred = mean(pred)
            scores[i] = sum((pred .- mean_pred).^2) / length(pred)
        elseif detector.method == :mutual_info
            # Mutual information approximation
            entropy_mean = -sum(pred .* log.(pred .+ 1e-8))
            # For single prediction, this approximates to entropy
            scores[i] = entropy_mean
        end
    end
    
    return scores
end

"""
Detect OOD samples based on fitted threshold
"""
function detect_ood(detector::OODDetector, x_test; link_approx::Symbol=:probit)
    if !detector.calibrated
        error("Detector must be calibrated first using fit_ood_threshold!")
    end
    
    scores = get_uncertainty_scores(detector, x_test; link_approx=link_approx)
    ood_predictions = scores .> detector.threshold
    
    return ood_predictions, scores
end

"""
Evaluate OOD detection performance
"""
function evaluate_ood_detection(detector::OODDetector, x_in, x_ood; 
                               link_approx::Symbol=:probit,
                               return_scores::Bool=true)
    # Get scores for both datasets
    scores_in = get_uncertainty_scores(detector, x_in; link_approx=link_approx)
    scores_ood = get_uncertainty_scores(detector, x_ood; link_approx=link_approx)
    
    # Create true labels (0 = in-distribution, 1 = out-of-distribution)
    y_true = vcat(zeros(Int, length(scores_in)), ones(Int, length(scores_ood)))
    scores_all = vcat(scores_in, scores_ood)
    
    # Calculate metrics
    metrics = calculate_ood_metrics(y_true, scores_all)
    
    if return_scores
        metrics["scores_in"] = scores_in
        metrics["scores_ood"] = scores_ood
        metrics["scores_all"] = scores_all
        metrics["y_true"] = y_true
    end
    
    return metrics
end

"""
Calculate comprehensive OOD detection metrics
"""
function calculate_ood_metrics(y_true::Vector{Int}, scores::Vector{Float64})
    # Sort by scores for threshold sweep
    sorted_indices = sortperm(scores, rev=true)
    y_sorted = y_true[sorted_indices]
    scores_sorted = scores[sorted_indices]
    
    n_total = length(y_true)
    n_ood = sum(y_true)
    n_in = n_total - n_ood
    
    # Calculate ROC curve points
    tpr_values = Float64[]
    fpr_values = Float64[]
    thresholds = Float64[]
    
    # Add starting point
    push!(tpr_values, 0.0)
    push!(fpr_values, 0.0)
    push!(thresholds, Inf)
    
    tp = 0
    fp = 0
    
    for i in 1:n_total
        if y_sorted[i] == 1  # OOD sample
            tp += 1
        else  # In-distribution sample
            fp += 1
        end
        
        tpr = tp / n_ood
        fpr = fp / n_in
        
        push!(tpr_values, tpr)
        push!(fpr_values, fpr)
        push!(thresholds, scores_sorted[i])
    end
    
    # Calculate AUC using trapezoidal rule
    auc = 0.0
    for i in 2:length(fpr_values)
        auc += (fpr_values[i] - fpr_values[i-1]) * (tpr_values[i] + tpr_values[i-1]) / 2
    end
    
    # Calculate AUROC (area under ROC curve)
    auroc = auc
    
    # Calculate AUPR (area under precision-recall curve)
    precision_values = Float64[]
    recall_values = Float64[]
    
    tp = 0
    fp = 0
    
    for i in 1:n_total
        if y_sorted[i] == 1
            tp += 1
        else
            fp += 1
        end
        
        precision = tp / (tp + fp)
        recall = tp / n_ood
        
        push!(precision_values, precision)
        push!(recall_values, recall)
    end
    
    # Calculate AUPR
    aupr = 0.0
    for i in 2:length(recall_values)
        aupr += (recall_values[i] - recall_values[i-1]) * (precision_values[i] + precision_values[i-1]) / 2
    end
    
    # Find optimal threshold (Youden's index)
    youden_scores = tpr_values .- fpr_values
    optimal_idx = argmax(youden_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Calculate metrics at optimal threshold
    y_pred = scores .>= optimal_threshold
    tp_opt = sum(y_true .== 1 .&& y_pred)
    fp_opt = sum(y_true .== 0 .&& y_pred)
    tn_opt = sum(y_true .== 0 .&& .!y_pred)
    fn_opt = sum(y_true .== 1 .&& .!y_pred)
    
    accuracy = (tp_opt + tn_opt) / n_total
    precision_opt = tp_opt / (tp_opt + fp_opt)
    recall_opt = tp_opt / (tp_opt + fn_opt)
    f1_score = 2 * precision_opt * recall_opt / (precision_opt + recall_opt)
    
    # Calculate FPR@95TPR (common OOD metric)
    target_tpr = 0.95
    fpr_at_95tpr = nothing
    for i in 1:length(tpr_values)
        if tpr_values[i] >= target_tpr
            fpr_at_95tpr = fpr_values[i]
            break
        end
    end
    if fpr_at_95tpr === nothing
        fpr_at_95tpr = 1.0
    end
    
    return Dict(
        "auroc" => auroc,
        "aupr" => aupr,
        "fpr_at_95tpr" => fpr_at_95tpr,
        "optimal_threshold" => optimal_threshold,
        "accuracy" => accuracy,
        "precision" => precision_opt,
        "recall" => recall_opt,
        "f1_score" => f1_score,
        "tpr_values" => tpr_values,
        "fpr_values" => fpr_values,
        "thresholds" => thresholds,
        "precision_values" => precision_values,
        "recall_values" => recall_values,
        "n_in" => n_in,
        "n_ood" => n_ood
    )
end

"""
Compare multiple OOD detection methods
"""
function compare_ood_methods(model, x_in, x_ood; methods::Vector{Symbol}=[:entropy, :max_prob], link_approx::Symbol=:probit)
    results = Dict()
    
    for method in methods
        println("Evaluating method: $method")
        detector = OODDetector(model; method=method)
        
        # Use a subset for threshold fitting
        n_val = min(100, size(x_in, 2))
        indices = randperm(size(x_in, 2))[1:n_val]
        x_val = x_in[:, indices]
        
        fit_ood_threshold!(detector, x_val; percentile=95.0, link_approx=link_approx)
        metrics = evaluate_ood_detection(detector, x_in, x_ood; link_approx=link_approx)
        
        results[method] = metrics
    end
    
    return results
end

end # module