using Flux
using CUDA
using LaplaceRedux
using Statistics
using Plots
using TaijaPlotting
using LinearAlgebra

function apply_laplace(model, data_train, X_test, y_test, hessian_type, la_name)
    if CUDA.has_cuda()
        model = gpu(model)
        X_test = gpu(X_test)
        y_test = gpu(y_test)
        data_train = [(gpu(x), gpu(y)) for (x, y) in data_train]
    else
        println("CUDA Not Found, using CPU")
    end

    println("Apply $la_name (Hessian: $hessian_type)...")
    println("Model structure: ", model)
    
    # 使用批处理来计算 Hessian
    batch_size = 32  # 进一步减小批量大小
    la = Laplace(model; 
                 likelihood=:classification, 
                 hessian_structure=hessian_type, 
                 subset_of_weights=:last_layer,
                 batch_size=batch_size)
    
    # 分批处理训练数据
    for i in 1:batch_size:length(data_train)
        batch_end = min(i + batch_size - 1, length(data_train))
        batch_data = data_train[i:batch_end]
        try
            println("Processing batch $i to $batch_end")
            fit!(la, batch_data)
        catch e
            println("Warning: Error during fit! - ", e)
            println("Error type: ", typeof(e))
            println("Stacktrace: ", stacktrace())
            continue
        end
        
        # 清理 GPU 内存
        if CUDA.has_cuda()
            CUDA.reclaim()
        end
    end
    
    try
        println("Optimizing prior...")
        optimize_prior!(la; verbosity=1, n_steps=100)
    catch e
        println("Warning: Error during optimize_prior! - ", e)
        println("Error type: ", typeof(e))
        println("Stacktrace: ", stacktrace())
        # 如果优化失败，使用默认参数继续
    end
    
    # 分批处理测试数据
    y_pred_la = []
    for i in 1:batch_size:size(X_test, 2)
        batch_end = min(i + batch_size - 1, size(X_test, 2))
        X_batch = X_test[:, i:batch_end]
        y_batch = y_test[:, i:batch_end]
        
        try
            println("Predicting batch $i to $batch_end")
            batch_pred = predict(la, X_batch, ret_distr=true)
            append!(y_pred_la, batch_pred)
        catch e
            println("Warning: Error during prediction - ", e)
            println("Error type: ", typeof(e))
            println("Stacktrace: ", stacktrace())
            continue
        end
        
        # 清理 GPU 内存
        if CUDA.has_cuda()
            CUDA.reclaim()
        end
    end

    if isempty(y_pred_la)
        error("No predictions were generated successfully")
    end

    y_pred_labels_la = [argmax(p.p) - 1 for p in y_pred_la]
    accuracy_la = mean(y_pred_labels_la .== vec(y_test))
    println("$la_name test set acc: $accuracy_la")

    _labels = 0:9 
    for target in _labels
        try
            bernoulli_distributions = [LaplaceRedux.Bernoulli(p.p[target + 1]) for p in y_pred_la]
            y_onehot_test = Flux.unstack(Array(y_test)', 1)
            plt = TaijaPlotting.Calibration_Plot(la, [y[target + 1] for y in y_onehot_test], bernoulli_distributions; n_bins=10)
            savefig(plt, "calibration_class_$(target)_$(la_name).png")
            println(" $target has a calibration curve: calibration_class_$(target)_$(la_name).png")
        catch e
            println("Warning: Error during calibration plot for class $target - ", e)
            println("Error type: ", typeof(e))
            println("Stacktrace: ", stacktrace())
        end
    end

    confidences = [maximum(p.p) for p in y_pred_la]
    mean_confidence = mean(confidences)
    println("AVG confidence: $mean_confidence")

    y_true = [argmax(Array(y_test)[:, i]) - 1 for i in 1:size(y_test, 2)]
    auroc_list = Float64[]
    for class_idx in 0:9
        try
            y_true_bin = [yt == class_idx ? 1 : 0 for yt in y_true]
            y_score = [p.p[class_idx + 1] for p in y_pred_la]
            roc_obj = roc(y_score, y_true_bin)
            auc = auc(roc_obj)
            push!(auroc_list, auc)
            println("Class $class_idx with AUROC: $auc")
        catch e
            println("Warning: Error calculating AUROC for class $class_idx - ", e)
            println("Error type: ", typeof(e))
            println("Stacktrace: ", stacktrace())
        end
    end
    
    if !isempty(auroc_list)
        mean_auroc = mean(auroc_list)
        println("AUROC: $mean_auroc")
    end
    
    # 最后清理 GPU 内存
    if CUDA.has_cuda()
        CUDA.reclaim()
    end
    
    println("$la_name DONE")
    return la
end 