#!/usr/bin/env python3
"""
🔬 Laplace Approximation Bayesian Neural Networks - GUI Launcher

Simple launcher script for the Laplace GUI that checks dependencies
and provides helpful error messages.
"""

import sys
import os
import subprocess

def check_python_version():
    """Check if Python version is adequate"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        ("julia", "PyJulia for Julia integration"),
        ("tkinter", "GUI framework"),
        ("PIL", "Image processing (Pillow)"),
        ("numpy", "Numerical computing"),
        ("matplotlib", "Plotting and visualization")
    ]
    
    missing = []
    
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"❌ {package} - {description} (MISSING)")
            missing.append(package)
    
    return missing

def install_missing_packages(missing_packages):
    """Attempt to install missing packages"""
    if not missing_packages:
        return True
        
    print(f"\n🔧 Installing missing packages: {', '.join(missing_packages)}")
    
    # Map package names to pip install names
    pip_names = {
        "PIL": "Pillow",
        "julia": "julia"
    }
    
    for package in missing_packages:
        pip_name = pip_names.get(package, package)
        try:
            print(f"Installing {pip_name}...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", pip_name], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {pip_name} installed successfully")
            else:
                print(f"❌ Failed to install {pip_name}: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error installing {pip_name}: {e}")
            return False
    
    return True

def check_julia():
    """Check if Julia is available"""
    try:
        result = subprocess.run(["julia", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Julia found: {version}")
            return True
        else:
            print("❌ Julia not found in PATH")
            return False
    except FileNotFoundError:
        print("❌ Julia not found. Please install Julia from https://julialang.org/")
        return False

def setup_julia_project():
    """Setup Julia project dependencies"""
    print("🔧 Setting up Julia project...")
    try:
        # Check if Project.toml exists
        if not os.path.exists("Project.toml"):
            print("❌ Project.toml not found. Please run from the project directory.")
            return False
            
        # Instantiate project
        result = subprocess.run([
            "julia", "--project=.", "-e", 
            "using Pkg; Pkg.instantiate(); println(\"Julia project setup complete\")"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Julia project dependencies installed")
            return True
        else:
            print(f"❌ Julia project setup failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Julia project setup error: {e}")
        return False

def launch_gui():
    """Launch the GUI application"""
    print("\n🚀 Launching Laplace GUI...")
    try:
        # Import and run the GUI
        from laplace_gui import main
        main()
    except ImportError as e:
        print(f"❌ Cannot import GUI: {e}")
        print("Make sure laplace_gui.py is in the current directory")
        return False
    except Exception as e:
        print(f"❌ GUI launch failed: {e}")
        return False
    
    return True

def main():
    print("🔬 Laplace Approximation Bayesian Neural Networks")
    print("=" * 50)
    print("GUI Launcher - Checking dependencies...\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check Python dependencies
    missing = check_dependencies()
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("Attempting to install missing packages...")
        
        if not install_missing_packages(missing):
            print("\n❌ Failed to install dependencies automatically.")
            print("Please install manually:")
            for package in missing:
                pip_name = "Pillow" if package == "PIL" else package
                print(f"  pip install {pip_name}")
            sys.exit(1)
    
    print("\n✅ All Python dependencies are available")
    
    # Check Julia
    if not check_julia():
        print("\n💡 Julia installation guide:")
        print("1. Go to https://julialang.org/downloads/")
        print("2. Download and install Julia for your platform")
        print("3. Add Julia to your PATH")
        sys.exit(1)
    
    # Setup Julia project
    if not setup_julia_project():
        print("\n💡 Manual Julia setup:")
        print("1. Open Julia in this directory")
        print("2. Run: using Pkg; Pkg.activate(\".\"); Pkg.instantiate()")
        sys.exit(1)
    
    print("\n✅ All dependencies are ready!")
    
    # Launch GUI
    if not launch_gui():
        print("\n❌ GUI launch failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)