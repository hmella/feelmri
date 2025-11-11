"""
Basic import tests for FEelMRI.
Ensures the library and its compiled C++ extensions are properly built and importable.
"""

import pytest

def test_import_top_level():
    """Test that the top-level FEelMRI package can be imported."""
    import feelmri
    assert hasattr(feelmri, "__version__"), "Package missing __version__ attribute"


@pytest.mark.parametrize("module_name", ["Assemble", "MRI", "BlochSimulator", "POD"])
def test_cpp_extensions_importable(module_name):
    """Test that compiled pybind11 modules can be imported."""
    import importlib
    mod = importlib.import_module(f"feelmri.{module_name}")
    assert mod is not None, f"Failed to import module feelmri.{module_name}"