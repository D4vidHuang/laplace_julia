#!/usr/bin/env python3
"""
Test script for GUI-Julia integration
"""

import sys
import os
try:
    import julia
    print("✅ PyJulia is available")
except ImportError:
    print("❌ PyJulia not found. Install with: pip install julia")
    sys.exit(1)

def test_julia_integration():
    """Test basic Julia integration"""
    print("🔧 Testing Julia integration...")
    
    try:
        # Initialize Julia
        print("Initializing Julia...")
        j = julia.Julia(compiled_modules=False)
        
        # Test basic Julia operation
        result = j.eval("2 + 2")
        print(f"✅ Basic Julia operation: 2 + 2 = {result}")
        
        # Activate project
        project_path = os.path.dirname(__file__)
        j.eval(f'using Pkg; Pkg.activate("{project_path}")')
        print("✅ Julia project activated")
        
        # Test module loading
        try:
            j.eval('include("src/MNISTClassifier.jl")')
            j.eval('include("src/GermanCreditClassifier.jl")')
            j.eval('include("src/BayesianMethods.jl")')
            j.eval('include("src/GUIInterface.jl")')
            print("✅ Julia modules loaded successfully")
        except Exception as e:
            print(f"⚠️  Module loading warning: {e}")
            
        try:
            j.eval('using .MNISTClassifier')
            j.eval('using .GermanCreditClassifier')
            j.eval('using .BayesianMethods')
            j.eval('using .GUIInterface')
            print("✅ Julia modules imported successfully")
        except Exception as e:
            print(f"❌ Module import failed: {e}")
            return False
            
        # Test GUI interface functions
        try:
            status = j.eval('get_model_status()')
            print(f"✅ Model status check: {status}")
        except Exception as e:
            print(f"❌ GUI interface test failed: {e}")
            return False
            
        # Test sample data creation
        try:
            result = j.eval('create_sample_data("test_data.csv")')
            print(f"✅ Sample data creation: {result}")
        except Exception as e:
            print(f"⚠️  Sample data creation warning: {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ Julia integration failed: {e}")
        return False

def test_gui_dependencies():
    """Test GUI dependencies"""
    print("🔧 Testing GUI dependencies...")
    
    deps = [
        ("tkinter", "GUI framework"),
        ("PIL", "Image processing"),
        ("numpy", "Numerical computing"),
        ("matplotlib", "Plotting")
    ]
    
    for dep, desc in deps:
        try:
            __import__(dep)
            print(f"✅ {dep} ({desc})")
        except ImportError:
            print(f"❌ {dep} ({desc}) - not found")

def main():
    print("🔬 Laplace GUI Integration Test")
    print("=" * 40)
    
    # Test GUI dependencies
    test_gui_dependencies()
    print()
    
    # Test Julia integration
    success = test_julia_integration()
    print()
    
    if success:
        print("🎉 Integration test PASSED!")
        print("You can now run the GUI with: python laplace_gui.py")
    else:
        print("❌ Integration test FAILED!")
        print("Please check the error messages above and fix any issues.")
        
    return success

if __name__ == "__main__":
    main()