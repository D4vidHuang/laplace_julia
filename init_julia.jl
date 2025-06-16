using Pkg

# Activate the current project
Pkg.activate(".")

# Remove existing PyCall if it exists
if haskey(Pkg.project().dependencies, "PyCall")
    Pkg.rm("PyCall")
end

# Add and build PyCall fresh with specific version
Pkg.add(PackageSpec(name="PyCall", version="1.96.4"))

# Configure PyCall to use system Python
ENV["PYTHON"] = Sys.which("python")
Pkg.build("PyCall")

# Add other required packages
packages = ["Plots", "Statistics", "Random", "LinearAlgebra"]
for pkg in packages
    if !haskey(Pkg.project().dependencies, pkg)
        Pkg.add(pkg)
    end
end

# Force precompilation
Pkg.precompile()

println("Julia initialization completed successfully!") 