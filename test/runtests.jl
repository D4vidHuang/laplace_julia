using Test

println("Running comprehensive test suite...")

@testset "Laplace Classification Project Tests" begin
    println("Testing core functionality...")
    include("test_mnist.jl")
    include("test_german_credit.jl") 
    include("test_visualizations.jl")
    
    println("Testing OOD functionality...")
    include("test_ood_datasets.jl")
    include("test_ood_detection.jl")
    include("test_ood_visualizations.jl")
    
    println("Testing Bayesian methods...")
    include("test_bayesian_methods.jl")
end

println("All tests completed!")