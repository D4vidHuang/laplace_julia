#!/usr/bin/env julia

"""
Laplace Approximation - Quick Start

Simple launcher for the Laplace approximation framework.
"""

using Pkg

println("🔬 Laplace Approximation Framework")
println("=" * "="^40)

function install_gui_deps()
    println("📦 Installing GUI dependencies...")
    try
        Pkg.activate(".")
        Pkg.add(["Blink", "WebIO", "Interact"])
        println("✅ GUI dependencies installed!")
        println("🔄 Please restart Julia and run this script again to use GUI")
        return true
    catch e
        println("❌ Installation failed: $e")
        return false
    end
end

function launch_gui_mode()
    println("启动GUI... / Launching GUI...")
    try
        # Ensure we're in the right directory
        cd(dirname(@__FILE__))
        
        # Run GUI in a new Julia process to avoid world age issues
        run(`julia -e 'include("gui/laplace_gui.jl"); launch_gui()'`)
        
    catch e
        println("❌ GUI launch failed: $e")
        println("🔄 Trying CLI mode instead...")
        launch_cli_mode()
    end
end

function launch_cli_mode()
    println("启动命令行界面... / Launching CLI...")
    try
        cd(dirname(@__FILE__))
        include("gui/laplace_gui.jl")
        run_cli_interface()
    catch e
        println("❌ CLI launch failed: $e")
    end
end

function run_demo()
    println("运行演示... / Running demo...")
    try
        cd(dirname(@__FILE__))
        include("gui/laplace_gui.jl")
        println("🎯 Demo: Training MNIST classifier...")
        run_classification_training("MNIST", "32,32,10", 5, 0.001)
        println("✅ Demo completed!")
    catch e
        println("❌ Demo failed: $e")
    end
end

function main()
    println("\n选择启动方式 / Choose launch method:")
    println("1. 🖥️  GUI界面 / GUI Interface")
    println("2. 📟 命令行 / Command Line")
    println("3. 📦 安装GUI依赖 / Install GUI Dependencies")
    println("4. 🎯 快速演示 / Quick Demo")
    
    print("\n请输入选择 / Enter choice (1-4): ")
    choice = readline()
    
    if choice == "1"
        launch_gui_mode()
    elseif choice == "2"
        launch_cli_mode()
    elseif choice == "3"
        install_gui_deps()
    elseif choice == "4"
        run_demo()
    else
        println("无效选择 / Invalid choice")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 