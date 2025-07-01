"""
Simple test to validate the package structure and imports.
"""

def test_imports():
    """Test that all main components can be imported."""
    print("Testing package imports...")
    
    try:
        # Test main package imports
        from los_estimator import LOSEstimator, EstimationParams, EstimationResult, DataLoader, Visualizer
        print("✓ Main package imports successful")
    except ImportError as e:
        print(f"✗ Main package import failed: {e}")
        return False
    
    try:
        # Test core module
        from los_estimator.core.estimator import LOSEstimator
        from los_estimator.core.models import EstimationParams, EstimationResult, SeriesData
        print("✓ Core module imports successful")
    except ImportError as e:
        print(f"✗ Core module import failed: {e}")
        return False
    
    try:
        # Test data module
        from los_estimator.data.loader import DataLoader
        print("✓ Data module imports successful")
    except ImportError as e:
        print(f"✗ Data module import failed: {e}")
        return False
    
    try:
        # Test fitting module
        from los_estimator.fitting.distributions import DistributionFitter
        from los_estimator.fitting.deconvolution import DeconvolutionEngine
        print("✓ Fitting module imports successful")
    except ImportError as e:
        print(f"✗ Fitting module import failed: {e}")
        return False
    
    try:
        # Test visualization module
        from los_estimator.visualization.plots import Visualizer
        print("✓ Visualization module imports successful")
    except ImportError as e:
        print(f"✗ Visualization module import failed: {e}")
        return False
    
    try:
        # Test utils module
        from los_estimator.utils.helpers import generate_run_name, setup_directories
        print("✓ Utils module imports successful")
    except ImportError as e:
        print(f"✗ Utils module import failed: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality without requiring data files."""
    print("\nTesting basic functionality...")
    
    try:
        from los_estimator import LOSEstimator, EstimationParams
        from los_estimator.fitting.distributions import DistributionFitter
        from los_estimator.utils.helpers import generate_run_name
        
        # Test parameter creation
        params = EstimationParams(kernel_width=100, step=14)
        assert params.kernel_width == 100
        assert params.step == 14
        print("✓ EstimationParams creation works")
        
        # Test run name generation
        run_name = generate_run_name(params)
        assert isinstance(run_name, str)
        assert len(run_name) > 0
        print(f"✓ Run name generation works: {run_name}")
        
        # Test distribution fitter
        fitter = DistributionFitter()
        assert len(fitter.DISTRIBUTIONS) > 0
        print(f"✓ DistributionFitter works, has {len(fitter.DISTRIBUTIONS)} distributions")
        
        # Test estimator creation
        estimator = LOSEstimator()
        assert estimator.data_loader is not None
        assert estimator.distribution_fitter is not None
        print("✓ LOSEstimator creation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def test_cli_import():
    """Test CLI import."""
    print("\nTesting CLI import...")
    
    try:
        from los_estimator.cli import main
        print("✓ CLI import successful")
        return True
    except ImportError as e:
        print(f"✗ CLI import failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=== LOS Estimator Package Validation ===")
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_basic_functionality()
    all_passed &= test_cli_import()
    
    print(f"\n=== Test Results ===")
    if all_passed:
        print("✓ All tests passed! Package structure is valid.")
        return 0
    else:
        print("✗ Some tests failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
