# Test drtsans.auto_wedge and other related wedge methods
import pytest
import numpy as np
from drtsans.auto_wedge import (
    _create_fit_function,
    _set_function_param_value,
    _calculate_function,
)
from drtsans.iq import get_wedges, valid_wedge
from mantid.simpleapi import FlatBackground, Gaussian


def test_create_functions():
    """Test create a set of functions from a string"""
    # set input
    input_function_str = (
        "name=FlatBackground,A0=26452.031557632155;name=Gaussian,Height=383890.0871952363,"
        "PeakCentre=106.28166682679631,Sigma=11.468820017817437;name=Gaussian,"
        "Height=685867.5517201225,PeakCentre=276.6831781469825,Sigma=27.242175478846256"
    )

    # execute
    test_functions = _create_fit_function(input_function_str)

    # verify
    assert (
        len(test_functions) == 3
    ), f"There shall be 3 functions in the list but not {len(test_functions)}"
    # background
    background = test_functions[0]
    assert background.name == "FlatBackground"
    assert background.A0 == pytest.approx(26452.031557632155, 1e-12)
    # gaussian 1
    g1 = test_functions[1]
    assert g1.name == "Gaussian"
    assert g1.Sigma == pytest.approx(11.468820017817437, 1e-12)
    # gaussian 2
    g2 = test_functions[2]
    assert g2.name == "Gaussian"
    assert g2.Height == pytest.approx(685867.5517201225, 1e-9)


def test_set_param_value():
    """Test setting parameter values"""
    # Create input functions
    test_funcs = [FlatBackground(), Gaussian(), Gaussian]

    # Input dictionary
    param_dict = {
        "f0.A0": 3.2,
        "f1.PeakCentre": 1.0,
        "f1.Sigma": 0.42,
        "f1.Height": 10.11,
        "f2.Sigma": 1.24,
        "f2.Height": 5.11,
        "f2.PeakCentre": 7.2,
    }

    # Execute
    for param_name, param_value in param_dict.items():
        _set_function_param_value(test_funcs, param_name, param_value)

    # Verify
    assert test_funcs[0].A0 == pytest.approx(3.2, 0.05)
    assert test_funcs[1].PeakCentre == pytest.approx(1.0, 0.05)
    assert test_funcs[1].Sigma == pytest.approx(0.42, 0.005)
    assert test_funcs[2].Sigma == pytest.approx(1.24, 0.005)
    assert test_funcs[2].Height == pytest.approx(5.11, 0.005)


def test_calculate_functions():
    """Test calculate a combo function"""
    # Create test function
    functions = [None] * 3
    functions[0] = FlatBackground(A0=10.0)
    functions[1] = Gaussian(PeakCentre=5.0, Height=40.0, Sigma=1.0)
    functions[2] = Gaussian(PeakCentre=95.0, Height=80.0, Sigma=2.0)

    # Set up X
    vec_x = np.arange(100)

    # Calculate
    vec_y = _calculate_function(functions, vec_x)

    # Verify
    assert vec_x[5] == pytest.approx(5.0, 1e-6)
    assert vec_y[5] == pytest.approx(50.0, 1e-6)

    assert vec_x[95] == pytest.approx(95.0, 1e-6)
    assert vec_y[95] == pytest.approx(90.0, 1e-6)


def test_calculate_valid_wedge():
    """Test method to transform any wedge to valid wedge angles"""
    # regular: (10, 30)
    validated_wedges = valid_wedge(10.0, 30.0)
    min_angle, max_angle = validated_wedges[0]
    assert min_angle == 10.0 and max_angle == 30.0

    # wedge falls out of region
    with pytest.raises(ValueError):
        valid_wedge(300.0, 318.0)
    with pytest.raises(ValueError):
        valid_wedge(40.0, 40.0)
    with pytest.raises(ValueError):
        valid_wedge(219.0, 40.0)
    with pytest.raises(ValueError):
        valid_wedge(40.0, 221.0)

    # split wedge angle
    validated_wedges = valid_wedge(220.0, -60.0)
    assert isinstance(validated_wedges, list) and len(validated_wedges) == 2
    assert validated_wedges[0] == (220.0, 270.1)
    assert validated_wedges[1] == (-90.1, -60.0)


def test_get_wedge():
    """Test method to get wedge with and without symmetric option"""
    # regular (10, 30)
    wedges = get_wedges(10.0, 30.0, symmetric_wedges=False)
    assert isinstance(wedges, list) and len(wedges) == 1
    assert wedges[0] == (10.0, 30.0)

    # regular (10, 30) with symmetric wedges
    wedges = get_wedges(10.0, 30.0, symmetric_wedges=True)
    assert isinstance(wedges, list) and len(wedges) == 2
    assert wedges[0] == (10.0, 30.0)
    assert wedges[1] == (190.0, 210.0)

    # 90 degree with symmetric wedges
    wedges = get_wedges(85.0, 95.0, symmetric_wedges=True)
    assert isinstance(wedges, list) and len(wedges) == 3
    assert wedges[0] == (85.0, 95.0)
    assert wedges[1] == (265.0, 270.1)
    assert wedges[2] == (-90.1, -85.0)


if __name__ == "__main__":
    pytest.main([__file__])
