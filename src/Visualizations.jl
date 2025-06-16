module Visualizations

using Plots
using Statistics
using LaplaceRedux

export plot_mnist_samples, plot_uncertainty_histogram, plot_german_credit_decision_boundary, 
       plot_training_progress, plot_prediction_comparison, save_all_plots,
       plot_ood_scores_distribution, plot_roc_curve, plot_precision_recall_curve,
       plot_ood_samples_comparison, plot_uncertainty_vs_ood, plot_calibration_curve,
       plot_ood_detection_summary, plot_laplace_vs_map_comparison, plot_confidence_comparison,
       plot_auroc_comparison, plot_laplace_improvement_analysis

"""
Plot MNIST digit samples with predictions and confidence
"""
function plot_mnist_samples(model, test_x, test_y, n_samples::Int=6)
    if model.la === nothing
        error("Model must have Laplace approximation fitted")
    end
    
    plt_list = []
    for i in 1:min(n_samples, size(test_x, 2))
        digit = test_x[:, i]
        true_label = test_y[i]
        pred_probs = predict(model.la, digit)
        pred_label = argmax(pred_probs[1]) - 1
        confidence = maximum(pred_probs[1])
        
        title_str = "True: $true_label, Pred: $pred_label\nConf: $(round(confidence, digits=2))"
        
        # Reshape and plot digit
        digit_img = reshape(digit, 28, 28)
        plt = heatmap(digit_img, color=:grays, aspect_ratio=:equal, 
                     title=title_str, showaxis=false, grid=false, 
                     titlefontsize=8)
        push!(plt_list, plt)
    end
    
    final_plot = plot(plt_list..., layout=(2, 3), size=(600, 400))
    return final_plot
end

"""
Plot uncertainty histogram comparing correct vs incorrect predictions
"""
function plot_uncertainty_histogram(entropies, correct_mask; title="Uncertainty Distribution")
    correct_entropies = entropies[correct_mask]
    incorrect_entropies = entropies[.!correct_mask]
    
    p = histogram(correct_entropies, alpha=0.6, label="Correct", 
                 color=:green, bins=20, normalize=:probability)
    histogram!(p, incorrect_entropies, alpha=0.6, label="Incorrect", 
              color=:red, bins=20, normalize=:probability)
    
    xlabel!(p, "Prediction Entropy")
    ylabel!(p, "Probability Density")
    title!(p, title)
    
    return p
end

"""
Plot decision boundary for German Credit classifier
"""
function plot_german_credit_decision_boundary(model, X, y_labels, unique_labels; 
                                            target_class=nothing, resolution=100)
    if model.la === nothing
        error("Model must have Laplace approximation fitted")
    end
    
    # Create grid
    x1_min, x1_max = extrema(X[:, 1])
    x2_min, x2_max = extrema(X[:, 2])
    
    x1_range = range(x1_min - 1, x1_max + 1, length=resolution)
    x2_range = range(x2_min - 1, x2_max + 1, length=resolution)
    
    # Create mesh grid
    grid_points = [[x1, x2] for x1 in x1_range, x2 in x2_range]
    grid_matrix = hcat(vec(grid_points)...)'
    
    # Get predictions for grid
    predictions = predict(model.la, grid_matrix)
    
    if target_class !== nothing
        # Plot probability for specific class
        class_idx = findfirst(==(target_class), unique_labels)
        if class_idx === nothing
            error("Target class $target_class not found in unique labels")
        end
        
        probs = [p[class_idx] for p in predictions]
        prob_matrix = reshape(probs, resolution, resolution)
        
        p = contourf(x1_range, x2_range, prob_matrix', 
                    levels=20, color=:viridis, alpha=0.7)
        title!(p, "P(Class = $target_class)")
    else
        # Plot predicted classes
        pred_classes = [argmax(p) for p in predictions]
        class_matrix = reshape(pred_classes, resolution, resolution)
        
        p = contourf(x1_range, x2_range, class_matrix', 
                    levels=length(unique_labels), color=:Set1, alpha=0.7)
        title!(p, "Predicted Classes")
    end
    
    # Overlay actual data points
    colors = [:red, :blue, :green, :orange, :purple]
    for (i, label) in enumerate(unique_labels)
        mask = y_labels .== label
        if sum(mask) > 0
            scatter!(p, X[mask, 1], X[mask, 2], 
                    label="Class $label", color=colors[i], 
                    markersize=4, markerstrokewidth=1)
        end
    end
    
    xlabel!(p, "Feature 1")
    ylabel!(p, "Feature 2")
    
    return p
end

"""
Plot training progress (loss over epochs)
"""
function plot_training_progress(losses, accuracies=nothing)
    p1 = plot(losses, label="Training Loss", color=:blue, linewidth=2)
    xlabel!(p1, "Epoch")
    ylabel!(p1, "Loss")
    title!(p1, "Training Progress")
    
    if accuracies !== nothing
        p2 = plot(accuracies, label="Training Accuracy", color=:red, linewidth=2)
        xlabel!(p2, "Epoch")
        ylabel!(p2, "Accuracy")
        title!(p2, "Training Accuracy")
        
        final_plot = plot(p1, p2, layout=(2, 1), size=(600, 400))
        return final_plot
    end
    
    return p1
end

"""
Compare Laplace vs Plugin predictions
"""
function plot_prediction_comparison(model, test_x, n_samples::Int=100)
    if model.la === nothing
        error("Model must have Laplace approximation fitted")
    end
    
    # Get predictions with both methods
    laplace_preds = predict(model.la, test_x[:, 1:n_samples]; link_approx=:probit)
    plugin_preds = predict(model.la, test_x[:, 1:n_samples]; link_approx=:plugin)
    
    # Calculate entropies
    laplace_entropies = [-sum(p .* log.(p .+ 1e-8)) for p in laplace_preds]
    plugin_entropies = [-sum(p .* log.(p .+ 1e-8)) for p in plugin_preds]
    
    # Plot comparison
    p = scatter(plugin_entropies, laplace_entropies, 
               alpha=0.6, markersize=3, color=:blue)
    
    # Add diagonal line
    min_val = min(minimum(plugin_entropies), minimum(laplace_entropies))
    max_val = max(maximum(plugin_entropies), maximum(laplace_entropies))
    plot!(p, [min_val, max_val], [min_val, max_val], 
          color=:red, linestyle=:dash, linewidth=2, label="y=x")
    
    xlabel!(p, "Plugin Method Entropy")
    ylabel!(p, "Laplace Method Entropy")
    title!(p, "Uncertainty Comparison: Laplace vs Plugin")
    
    return p
end

"""
Save all generated plots to files
"""
function save_all_plots(plots_dict::Dict, output_dir::String="plots")
    if !isdir(output_dir)
        mkdir(output_dir)
    end
    
    for (name, plot_obj) in plots_dict
        filename = joinpath(output_dir, "$(name).png")
        savefig(plot_obj, filename)
        println("Saved plot: $filename")
    end
end

"""
Plot distribution of OOD scores for in-distribution vs out-of-distribution data
"""
function plot_ood_scores_distribution(scores_in, scores_ood, method_name::String="Uncertainty"; 
                                     threshold=nothing, bins=30)
    p = histogram(scores_in, alpha=0.6, label="In-Distribution", 
                 color=:blue, bins=bins, normalize=:probability)
    histogram!(p, scores_ood, alpha=0.6, label="Out-of-Distribution", 
              color=:red, bins=bins, normalize=:probability)
    
    # Add threshold line if provided
    if threshold !== nothing
        vline!(p, [threshold], color=:black, linestyle=:dash, linewidth=2, 
               label="Threshold")
    end
    
    xlabel!(p, "$method_name Score")
    ylabel!(p, "Probability Density")
    title!(p, "OOD Detection: $method_name Distribution")
    
    return p
end

"""
Plot ROC curve for OOD detection
"""
function plot_roc_curve(metrics::Dict; title="ROC Curve")
    tpr = metrics["tpr_values"]
    fpr = metrics["fpr_values"]
    auroc = metrics["auroc"]
    
    p = plot(fpr, tpr, linewidth=2, color=:blue, 
            label="ROC (AUC = $(round(auroc, digits=3)))")
    
    # Add diagonal line
    plot!(p, [0, 1], [0, 1], color=:gray, linestyle=:dash, linewidth=1, 
          label="Random")
    
    xlabel!(p, "False Positive Rate")
    ylabel!(p, "True Positive Rate")
    title!(p, title)
    xlims!(p, (0, 1))
    ylims!(p, (0, 1))
    
    return p
end

"""
Plot Precision-Recall curve for OOD detection
"""
function plot_precision_recall_curve(metrics::Dict; title="Precision-Recall Curve")
    precision = metrics["precision_values"]
    recall = metrics["recall_values"]
    aupr = metrics["aupr"]
    
    p = plot(recall, precision, linewidth=2, color=:red, 
            label="PR (AUC = $(round(aupr, digits=3)))")
    
    # Add baseline (random classifier)
    baseline = metrics["n_ood"] / (metrics["n_ood"] + metrics["n_in"])
    hline!(p, [baseline], color=:gray, linestyle=:dash, linewidth=1, 
           label="Random ($(round(baseline, digits=3)))")
    
    xlabel!(p, "Recall")
    ylabel!(p, "Precision")
    title!(p, title)
    xlims!(p, (0, 1))
    ylims!(p, (0, 1))
    
    return p
end

"""
Plot comparison of OOD samples vs in-distribution samples
"""
function plot_ood_samples_comparison(model, x_in, x_ood, ood_name::String="OOD"; 
                                   n_samples::Int=6, data_type::Symbol=:mnist)
    if data_type == :mnist
        return plot_mnist_ood_comparison(model, x_in, x_ood, ood_name, n_samples)
    elseif data_type == :german_credit
        return plot_german_credit_ood_comparison(model, x_in, x_ood, ood_name)
    else
        error("Unsupported data type: $data_type")
    end
end

function plot_mnist_ood_comparison(model, x_in, x_ood, ood_name::String, n_samples::Int)
    if model.la === nothing
        error("Model must have Laplace approximation fitted")
    end
    
    plt_list = []
    
    # Plot in-distribution samples
    for i in 1:min(n_samples÷2, size(x_in, 2))
        sample = x_in[:, i:i]
        pred_probs = predict(model.la, sample)
        confidence = maximum(pred_probs[1])
        entropy = -sum(pred_probs[1] .* log.(pred_probs[1] .+ 1e-8))
        
        title_str = "In-Dist\nConf: $(round(confidence, digits=2))\nEnt: $(round(entropy, digits=2))"
        
        img = reshape(x_in[:, i], 28, 28)
        plt = heatmap(img, color=:grays, aspect_ratio=:equal, 
                     title=title_str, showaxis=false, grid=false, 
                     titlefontsize=8)
        push!(plt_list, plt)
    end
    
    # Plot OOD samples
    for i in 1:min(n_samples÷2, size(x_ood, 2))
        sample = x_ood[:, i:i]
        pred_probs = predict(model.la, sample)
        confidence = maximum(pred_probs[1])
        entropy = -sum(pred_probs[1] .* log.(pred_probs[1] .+ 1e-8))
        
        title_str = "$ood_name\nConf: $(round(confidence, digits=2))\nEnt: $(round(entropy, digits=2))"
        
        img = reshape(x_ood[:, i], 28, 28)
        plt = heatmap(img, color=:grays, aspect_ratio=:equal, 
                     title=title_str, showaxis=false, grid=false, 
                     titlefontsize=8, titlecolor=:red)
        push!(plt_list, plt)
    end
    
    final_plot = plot(plt_list..., layout=(2, n_samples÷2), size=(600, 400))
    return final_plot
end

function plot_german_credit_ood_comparison(model, x_in, x_ood, ood_name::String)
    if model.la === nothing
        error("Model must have Laplace approximation fitted")
    end
    
    # Create scatter plot
    p = scatter(x_in[1, :], x_in[2, :], color=:blue, alpha=0.6, 
               markersize=4, label="In-Distribution")
    scatter!(p, x_ood[1, :], x_ood[2, :], color=:red, alpha=0.6, 
            markersize=4, label=ood_name)
    
    xlabel!(p, "Feature 1")
    ylabel!(p, "Feature 2")
    title!(p, "Data Distribution Comparison")
    
    return p
end

"""
Plot uncertainty scores vs OOD detection results
"""
function plot_uncertainty_vs_ood(scores_in, scores_ood, threshold=nothing)
    # Create combined data
    all_scores = vcat(scores_in, scores_ood)
    labels = vcat(zeros(Int, length(scores_in)), ones(Int, length(scores_ood)))
    
    # Create box plot
    p = boxplot(["In-Distribution", "Out-of-Distribution"], 
               [scores_in, scores_ood],
               color=[:blue, :red], alpha=0.6)
    
    # Add threshold line if provided
    if threshold !== nothing
        hline!(p, [threshold], color=:black, linestyle=:dash, linewidth=2, 
               label="Threshold")
    end
    
    ylabel!(p, "Uncertainty Score")
    title!(p, "Uncertainty Distribution by Data Type")
    
    return p
end

"""
Plot calibration curve for uncertainty quantification
"""
function plot_calibration_curve(confidences, accuracies; n_bins=10)
    # Bin the data
    bin_boundaries = range(0, 1, length=n_bins+1)
    bin_lowers = bin_boundaries[1:end-1]
    bin_uppers = bin_boundaries[2:end]
    
    bin_centers = Float64[]
    bin_accuracies = Float64[]
    bin_counts = Int[]
    
    for i in 1:n_bins
        in_bin = (confidences .>= bin_lowers[i]) .& (confidences .< bin_uppers[i])
        if i == n_bins  # Include upper boundary in last bin
            in_bin = in_bin .| (confidences .== 1.0)
        end
        
        if sum(in_bin) > 0
            push!(bin_centers, (bin_lowers[i] + bin_uppers[i]) / 2)
            push!(bin_accuracies, mean(accuracies[in_bin]))
            push!(bin_counts, sum(in_bin))
        end
    end
    
    # Plot calibration curve
    p = plot(bin_centers, bin_accuracies, 
            marker=:circle, markersize=6, linewidth=2, 
            color=:blue, label="Model")
    
    # Add perfect calibration line
    plot!(p, [0, 1], [0, 1], color=:gray, linestyle=:dash, linewidth=1, 
          label="Perfect Calibration")
    
    xlabel!(p, "Mean Predicted Probability")
    ylabel!(p, "Fraction of Positives")
    title!(p, "Calibration Curve")
    xlims!(p, (0, 1))
    ylims!(p, (0, 1))
    
    return p
end

"""
Create comprehensive OOD detection summary plot
"""
function plot_ood_detection_summary(metrics::Dict, scores_in, scores_ood, 
                                  method_name::String="Entropy")
    # Create subplots
    p1 = plot_ood_scores_distribution(scores_in, scores_ood, method_name, 
                                     metrics["optimal_threshold"])
    p2 = plot_roc_curve(metrics)
    p3 = plot_precision_recall_curve(metrics)
    p4 = plot_uncertainty_vs_ood(scores_in, scores_ood, metrics["optimal_threshold"])
    
    # Combine into summary plot
    summary_plot = plot(p1, p2, p3, p4, layout=(2, 2), size=(800, 600))
    
    return summary_plot
end

"""
Compare Laplace approximation vs MAP (Maximum A Posteriori) predictions
"""
function plot_laplace_vs_map_comparison(model, test_x, test_y=nothing; n_samples=100)
    if model.la === nothing
        error("Model must have Laplace approximation fitted")
    end
    
    # Get subset of test data
    if n_samples > size(test_x, 2)
        n_samples = size(test_x, 2)
    end
    test_subset = test_x[:, 1:n_samples]
    
    # Get predictions with Laplace approximation (probit link)
    laplace_preds = predict(model.la, test_subset; link_approx=:probit)
    
    # Get MAP predictions (plugin estimator)
    map_preds = predict(model.la, test_subset; link_approx=:plugin)
    
    # Calculate confidence (max probability)
    laplace_confidence = [maximum(p) for p in laplace_preds]
    map_confidence = [maximum(p) for p in map_preds]
    
    # Calculate entropy (uncertainty)
    laplace_entropy = [-sum(p .* log.(p .+ 1e-8)) for p in laplace_preds]
    map_entropy = [-sum(p .* log.(p .+ 1e-8)) for p in map_preds]
    
    # Create comparison plots
    p1 = scatter(map_confidence, laplace_confidence, 
                alpha=0.6, markersize=3, color=:blue,
                xlabel="MAP Confidence", ylabel="Laplace Confidence",
                title="Confidence Comparison")
    plot!(p1, [0, 1], [0, 1], color=:red, linestyle=:dash, linewidth=2, label="y=x")
    
    p2 = scatter(map_entropy, laplace_entropy,
                alpha=0.6, markersize=3, color=:red,
                xlabel="MAP Entropy", ylabel="Laplace Entropy", 
                title="Uncertainty Comparison")
    plot!(p2, [0, maximum([map_entropy; laplace_entropy])], 
          [0, maximum([map_entropy; laplace_entropy])], 
          color=:black, linestyle=:dash, linewidth=2, label="y=x")
    
    # Confidence distribution comparison
    p3 = histogram(map_confidence, alpha=0.6, label="MAP", color=:blue, bins=20, normalize=:probability)
    histogram!(p3, laplace_confidence, alpha=0.6, label="Laplace", color=:green, bins=20, normalize=:probability)
    xlabel!(p3, "Confidence")
    ylabel!(p3, "Probability Density")
    title!(p3, "Confidence Distribution")
    
    # Entropy distribution comparison
    p4 = histogram(map_entropy, alpha=0.6, label="MAP", color=:blue, bins=20, normalize=:probability)
    histogram!(p4, laplace_entropy, alpha=0.6, label="Laplace", color=:green, bins=20, normalize=:probability)
    xlabel!(p4, "Entropy")
    ylabel!(p4, "Probability Density")
    title!(p4, "Uncertainty Distribution")
    
    # Combine plots
    final_plot = plot(p1, p2, p3, p4, layout=(2, 2), size=(800, 600))
    
    # Calculate summary statistics
    conf_diff = mean(laplace_confidence .- map_confidence)
    entropy_diff = mean(laplace_entropy .- map_entropy)
    
    println("Laplace vs MAP Comparison:")
    println("  Average confidence difference (Laplace - MAP): $(round(conf_diff, digits=4))")
    println("  Average entropy difference (Laplace - MAP): $(round(entropy_diff, digits=4))")
    println("  Laplace provides $(conf_diff > 0 ? "higher" : "lower") confidence on average")
    println("  Laplace provides $(entropy_diff > 0 ? "higher" : "lower") uncertainty on average")
    
    return final_plot, Dict(
        "laplace_confidence" => laplace_confidence,
        "map_confidence" => map_confidence,
        "laplace_entropy" => laplace_entropy,
        "map_entropy" => map_entropy,
        "conf_diff" => conf_diff,
        "entropy_diff" => entropy_diff
    )
end

"""
Plot confidence comparison between Laplace and MAP methods with accuracy analysis
"""
function plot_confidence_comparison(model, test_x, test_y; confidence_bins=10)
    if model.la === nothing
        error("Model must have Laplace approximation fitted")
    end
    
    if length(test_y) > size(test_x, 2)
        test_y = test_y[1:size(test_x, 2)]
    end
    
    # Get predictions
    laplace_preds = predict(model.la, test_x; link_approx=:probit)
    map_preds = predict(model.la, test_x; link_approx=:plugin)
    
    # Calculate confidence and predictions
    laplace_confidence = [maximum(p) for p in laplace_preds]
    map_confidence = [maximum(p) for p in map_preds]
    
    laplace_pred_classes = [argmax(p) for p in laplace_preds]
    map_pred_classes = [argmax(p) for p in map_preds]
    
    # Adjust for 0-indexed labels if needed (MNIST case)
    if minimum(test_y) == 0
        laplace_pred_classes = laplace_pred_classes .- 1
        map_pred_classes = map_pred_classes .- 1
    end
    
    # Calculate accuracy
    laplace_correct = laplace_pred_classes .== test_y
    map_correct = map_pred_classes .== test_y
    
    # Bin confidence scores and calculate accuracy per bin
    bin_edges = range(0, 1, length=confidence_bins+1)
    
    laplace_bin_acc = Float64[]
    map_bin_acc = Float64[]
    bin_centers = Float64[]
    
    for i in 1:confidence_bins
        bin_mask_laplace = (laplace_confidence .>= bin_edges[i]) .& (laplace_confidence .< bin_edges[i+1])
        bin_mask_map = (map_confidence .>= bin_edges[i]) .& (map_confidence .< bin_edges[i+1])
        
        if i == confidence_bins  # Include upper boundary in last bin
            bin_mask_laplace = bin_mask_laplace .| (laplace_confidence .== 1.0)
            bin_mask_map = bin_mask_map .| (map_confidence .== 1.0)
        end
        
        if sum(bin_mask_laplace) > 0
            push!(laplace_bin_acc, mean(laplace_correct[bin_mask_laplace]))
        else
            push!(laplace_bin_acc, NaN)
        end
        
        if sum(bin_mask_map) > 0
            push!(map_bin_acc, mean(map_correct[bin_mask_map]))
        else
            push!(map_bin_acc, NaN)
        end
        
        push!(bin_centers, (bin_edges[i] + bin_edges[i+1]) / 2)
    end
    
    # Reliability diagram
    p1 = plot(bin_centers, laplace_bin_acc, marker=:circle, linewidth=2, 
             label="Laplace", color=:blue, markersize=6)
    plot!(p1, bin_centers, map_bin_acc, marker=:square, linewidth=2,
          label="MAP", color=:red, markersize=6)
    plot!(p1, [0, 1], [0, 1], color=:gray, linestyle=:dash, linewidth=1, 
          label="Perfect Calibration")
    xlabel!(p1, "Confidence")
    ylabel!(p1, "Accuracy")
    title!(p1, "Reliability Diagram")
    xlims!(p1, (0, 1))
    ylims!(p1, (0, 1))
    
    # Confidence histograms
    p2 = histogram(laplace_confidence, alpha=0.6, label="Laplace", color=:blue, 
                  bins=20, normalize=:probability)
    histogram!(p2, map_confidence, alpha=0.6, label="MAP", color=:red,
              bins=20, normalize=:probability)
    xlabel!(p2, "Confidence")
    ylabel!(p2, "Probability Density")
    title!(p2, "Confidence Distribution")
    
    # Accuracy vs confidence scatter
    p3 = scatter(laplace_confidence, Float64.(laplace_correct), 
                alpha=0.4, label="Laplace", color=:blue, markersize=2,
                xlabel="Confidence", ylabel="Correct (1) / Incorrect (0)",
                title="Confidence vs Correctness")
    scatter!(p3, map_confidence .+ 0.02, Float64.(map_correct), 
            alpha=0.4, label="MAP", color=:red, markersize=2)
    
    # Expected Calibration Error calculation
    laplace_ece = calculate_ece(laplace_confidence, laplace_correct, confidence_bins)
    map_ece = calculate_ece(map_confidence, map_correct, confidence_bins)
    
    # Add ECE to plot
    p4 = bar(["Laplace", "MAP"], [laplace_ece, map_ece], 
            color=[:blue, :red], alpha=0.7,
            title="Expected Calibration Error",
            ylabel="ECE")
    
    final_plot = plot(p1, p2, p3, p4, layout=(2, 2), size=(800, 600))
    
    println("Confidence Analysis:")
    println("  Laplace ECE: $(round(laplace_ece, digits=4))")
    println("  MAP ECE: $(round(map_ece, digits=4))")
    println("  Better calibration: $(laplace_ece < map_ece ? "Laplace" : "MAP")")
    
    return final_plot, Dict(
        "laplace_ece" => laplace_ece,
        "map_ece" => map_ece,
        "laplace_confidence" => laplace_confidence,
        "map_confidence" => map_confidence,
        "laplace_accuracy" => mean(laplace_correct),
        "map_accuracy" => mean(map_correct)
    )
end

"""
Calculate Expected Calibration Error
"""
function calculate_ece(confidences, correct_predictions, n_bins=10)
    bin_edges = range(0, 1, length=n_bins+1)
    ece = 0.0
    total_samples = length(confidences)
    
    for i in 1:n_bins
        bin_mask = (confidences .>= bin_edges[i]) .& (confidences .< bin_edges[i+1])
        if i == n_bins  # Include upper boundary in last bin
            bin_mask = bin_mask .| (confidences .== 1.0)
        end
        
        if sum(bin_mask) > 0
            bin_confidence = mean(confidences[bin_mask])
            bin_accuracy = mean(correct_predictions[bin_mask])
            bin_weight = sum(bin_mask) / total_samples
            
            ece += bin_weight * abs(bin_confidence - bin_accuracy)
        end
    end
    
    return ece
end

"""
Compare AUROC performance between Laplace and MAP methods for OOD detection
"""
function plot_auroc_comparison(model, x_in, x_ood, ood_name="OOD")
    if model.la === nothing
        error("Model must have Laplace approximation fitted")
    end
    
    # Get uncertainty scores for both methods
    println("Calculating Laplace approximation scores...")
    laplace_scores_in = get_laplace_uncertainty_scores(model, x_in)
    laplace_scores_ood = get_laplace_uncertainty_scores(model, x_ood)
    
    println("Calculating MAP scores...")
    map_scores_in = get_map_uncertainty_scores(model, x_in)
    map_scores_ood = get_map_uncertainty_scores(model, x_ood)
    
    # Calculate AUROC for both methods
    laplace_auroc = calculate_auroc_from_scores(laplace_scores_in, laplace_scores_ood)
    map_auroc = calculate_auroc_from_scores(map_scores_in, map_scores_ood)
    
    # Create ROC curves
    laplace_fpr, laplace_tpr = calculate_roc_curve(laplace_scores_in, laplace_scores_ood)
    map_fpr, map_tpr = calculate_roc_curve(map_scores_in, map_scores_ood)
    
    # Plot ROC comparison
    p1 = plot(laplace_fpr, laplace_tpr, linewidth=2, color=:blue,
             label="Laplace (AUC = $(round(laplace_auroc, digits=3)))")
    plot!(p1, map_fpr, map_tpr, linewidth=2, color=:red,
          label="MAP (AUC = $(round(map_auroc, digits=3)))")
    plot!(p1, [0, 1], [0, 1], color=:gray, linestyle=:dash, linewidth=1, label="Random")
    xlabel!(p1, "False Positive Rate")
    ylabel!(p1, "True Positive Rate")
    title!(p1, "ROC Comparison: $ood_name")
    
    # Score distributions
    p2 = histogram(laplace_scores_in, alpha=0.6, label="Laplace In-Dist", 
                  color=:lightblue, bins=30, normalize=:probability)
    histogram!(p2, laplace_scores_ood, alpha=0.6, label="Laplace OOD",
              color=:blue, bins=30, normalize=:probability)
    xlabel!(p2, "Uncertainty Score")
    ylabel!(p2, "Probability Density")
    title!(p2, "Laplace Score Distribution")
    
    p3 = histogram(map_scores_in, alpha=0.6, label="MAP In-Dist",
                  color=:pink, bins=30, normalize=:probability)
    histogram!(p3, map_scores_ood, alpha=0.6, label="MAP OOD",
              color=:red, bins=30, normalize=:probability)
    xlabel!(p3, "Uncertainty Score")
    ylabel!(p3, "Probability Density")
    title!(p3, "MAP Score Distribution")
    
    # AUROC comparison bar chart
    p4 = bar(["Laplace", "MAP"], [laplace_auroc, map_auroc],
            color=[:blue, :red], alpha=0.7,
            title="AUROC Comparison",
            ylabel="AUROC")
    ylims!(p4, (0, 1))
    
    final_plot = plot(p1, p2, p3, p4, layout=(2, 2), size=(800, 600))
    
    improvement = laplace_auroc - map_auroc
    println("OOD Detection Performance:")
    println("  Laplace AUROC: $(round(laplace_auroc, digits=4))")
    println("  MAP AUROC: $(round(map_auroc, digits=4))")
    println("  Improvement: $(round(improvement, digits=4))")
    println("  Better method: $(improvement > 0 ? "Laplace" : "MAP")")
    
    return final_plot, Dict(
        "laplace_auroc" => laplace_auroc,
        "map_auroc" => map_auroc,
        "improvement" => improvement,
        "laplace_scores_in" => laplace_scores_in,
        "laplace_scores_ood" => laplace_scores_ood,
        "map_scores_in" => map_scores_in,
        "map_scores_ood" => map_scores_ood
    )
end

"""
Get uncertainty scores using Laplace approximation
"""
function get_laplace_uncertainty_scores(model, x_data)
    predictions = predict(model.la, x_data; link_approx=:probit)
    return [-sum(p .* log.(p .+ 1e-8)) for p in predictions]  # Entropy
end

"""
Get uncertainty scores using MAP estimation
"""
function get_map_uncertainty_scores(model, x_data)
    predictions = predict(model.la, x_data; link_approx=:plugin)
    return [-sum(p .* log.(p .+ 1e-8)) for p in predictions]  # Entropy
end

"""
Calculate AUROC from in-distribution and OOD scores
"""
function calculate_auroc_from_scores(scores_in, scores_ood)
    y_true = vcat(zeros(Int, length(scores_in)), ones(Int, length(scores_ood)))
    scores_all = vcat(scores_in, scores_ood)
    
    # Sort by scores
    sorted_indices = sortperm(scores_all, rev=true)
    y_sorted = y_true[sorted_indices]
    
    n_ood = sum(y_true)
    n_in = length(y_true) - n_ood
    
    tp = 0
    fp = 0
    auc = 0.0
    
    prev_fpr = 0.0
    for i in 1:length(y_true)
        if y_sorted[i] == 1
            tp += 1
        else
            fp += 1
        end
        
        tpr = tp / n_ood
        fpr = fp / n_in
        
        # Trapezoidal rule
        auc += (fpr - prev_fpr) * tpr
        prev_fpr = fpr
    end
    
    return auc
end

"""
Calculate ROC curve points
"""
function calculate_roc_curve(scores_in, scores_ood)
    y_true = vcat(zeros(Int, length(scores_in)), ones(Int, length(scores_ood)))
    scores_all = vcat(scores_in, scores_ood)
    
    sorted_indices = sortperm(scores_all, rev=true)
    y_sorted = y_true[sorted_indices]
    
    n_ood = sum(y_true)
    n_in = length(y_true) - n_ood
    
    tpr_values = [0.0]
    fpr_values = [0.0]
    
    tp = 0
    fp = 0
    
    for i in 1:length(y_true)
        if y_sorted[i] == 1
            tp += 1
        else
            fp += 1
        end
        
        tpr = tp / n_ood
        fpr = fp / n_in
        
        push!(tpr_values, tpr)
        push!(fpr_values, fpr)
    end
    
    return fpr_values, tpr_values
end

"""
Comprehensive analysis of Laplace approximation improvements
"""
function plot_laplace_improvement_analysis(model, x_in, x_ood_list, ood_names, test_y=nothing)
    if model.la === nothing
        error("Model must have Laplace approximation fitted")
    end
    
    results = Dict()
    improvements = Float64[]
    ood_types = String[]
    
    println("Analyzing Laplace improvements across $(length(ood_names)) OOD types...")
    
    for (i, (x_ood, ood_name)) in enumerate(zip(x_ood_list, ood_names))
        println("  Processing $ood_name...")
        
        # Calculate AUROC for both methods
        laplace_auroc = calculate_auroc_from_scores(
            get_laplace_uncertainty_scores(model, x_in),
            get_laplace_uncertainty_scores(model, x_ood)
        )
        
        map_auroc = calculate_auroc_from_scores(
            get_map_uncertainty_scores(model, x_in),
            get_map_uncertainty_scores(model, x_ood)
        )
        
        improvement = laplace_auroc - map_auroc
        
        results[ood_name] = Dict(
            "laplace_auroc" => laplace_auroc,
            "map_auroc" => map_auroc,
            "improvement" => improvement
        )
        
        push!(improvements, improvement)
        push!(ood_types, ood_name)
    end
    
    # Create improvement comparison plot
    colors = [imp > 0 ? :green : :red for imp in improvements]
    p1 = bar(ood_types, improvements, color=colors, alpha=0.7,
            title="Laplace Approximation Improvement",
            ylabel="AUROC Improvement", xrotation=45)
    hline!(p1, [0], color=:black, linestyle=:dash, linewidth=1)
    
    # AUROC comparison
    laplace_aurocs = [results[name]["laplace_auroc"] for name in ood_names]
    map_aurocs = [results[name]["map_auroc"] for name in ood_names]
    
    x_pos = 1:length(ood_names)
    p2 = bar(x_pos .- 0.2, laplace_aurocs, width=0.4, label="Laplace", 
            color=:blue, alpha=0.7)
    bar!(p2, x_pos .+ 0.2, map_aurocs, width=0.4, label="MAP",
         color=:red, alpha=0.7)
    plot!(p2, xticks=(x_pos, ood_names), xrotation=45)
    title!(p2, "AUROC Comparison")
    ylabel!(p2, "AUROC")
    
    # Summary statistics
    avg_improvement = mean(improvements)
    positive_improvements = sum(improvements .> 0)
    
    p3 = bar(["Average\nImprovement"], [avg_improvement], 
            color=avg_improvement > 0 ? :green : :red, alpha=0.7,
            title="Overall Performance",
            ylabel="Average AUROC Improvement")
    hline!(p3, [0], color=:black, linestyle=:dash, linewidth=1)
    
    # Improvement distribution
    p4 = histogram(improvements, bins=10, alpha=0.7, color=:blue,
                  title="Improvement Distribution",
                  xlabel="AUROC Improvement", ylabel="Frequency")
    vline!(p4, [0], color=:black, linestyle=:dash, linewidth=2)
    vline!(p4, [avg_improvement], color=:red, linestyle=:dash, linewidth=2, 
           label="Average")
    
    final_plot = plot(p1, p2, p3, p4, layout=(2, 2), size=(1000, 700))
    
    # Print summary
    println("\n" * "="^50)
    println("LAPLACE APPROXIMATION IMPROVEMENT ANALYSIS")
    println("="^50)
    println("OOD Type                | Laplace | MAP     | Improvement")
    println("-"^55)
    
    for name in ood_names
        lap_auc = results[name]["laplace_auroc"]
        map_auc = results[name]["map_auroc"]
        imp = results[name]["improvement"]
        
        println("$(rpad(name, 23)) | $(rpad(round(lap_auc, digits=3), 7)) | $(rpad(round(map_auc, digits=3), 7)) | $(round(imp, digits=4))")
    end
    
    println("-"^55)
    println("Average Improvement: $(round(avg_improvement, digits=4))")
    println("Positive Improvements: $positive_improvements/$(length(ood_names))")
    println("Best Improvement: $(maximum(improvements)) ($(ood_names[argmax(improvements)]))")
    println("Worst Case: $(minimum(improvements)) ($(ood_names[argmin(improvements)]))")
    
    if avg_improvement > 0
        println("\n✅ Laplace approximation provides better OOD detection on average!")
    else
        println("\n⚠️  MAP estimation performs better on average for OOD detection.")
    end
    
    return final_plot, results
end

end # module