#!/usr/bin/env python
import pytest

# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/mono/gpsans/attenuation.py
from drtsans.mono.gpsans import attenuation_factor

# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/samplelogs.py
from drtsans.samplelogs import SampleLogs


def test_attenuation_factor(generic_workspace):
    ws = generic_workspace  # friendly name

    # Test input and expected values provided by Lisa Debeer-Schmitt, 2020-02-26
    wavelength = 4.75
    attenuator = 6  # x2k
    expected_value = 0.001037270673420313
    expected_error = 7.005200329552345e-05

    # Add sample logs
    SampleLogs(ws).insert("wavelength", wavelength, "Angstrom")
    SampleLogs(ws).insert("attenuator", attenuator)

    value, error = attenuation_factor(ws)
    assert value == pytest.approx(expected_value)
    assert error == pytest.approx(expected_error)


def test_attenuation_factor_open_close(generic_workspace):
    ws = generic_workspace  # friendly name

    # add wavelength
    SampleLogs(ws).insert("wavelength", 1.54, "Angstrom")

    # add Undefined attenuator
    attenuator = 0  # Undefined
    SampleLogs(ws).insert("attenuator", attenuator)
    assert attenuation_factor(ws) == (1, 0)

    # add Close attenuator
    attenuator = 1  # Close
    SampleLogs(ws).insert("attenuator", attenuator)
    assert attenuation_factor(ws) == (1, 0)

    # add Open attenuator
    attenuator = 2  # Open
    SampleLogs(ws).insert("attenuator", attenuator)
    assert attenuation_factor(ws) == (1, 0)


def test_attenuation_factor_missing_logs(generic_workspace):
    """This test that correct error messages are return if the required
    attenuator or wavelenght logs is missing
    """
    ws = generic_workspace  # friendly name

    # Missing attenuator and wavelength log
    with pytest.raises(RuntimeError) as excinfo:
        attenuation_factor(ws)
    assert "attenuator" in str(
        excinfo.value
    )  # Should complain about missing attenuator

    # Add in attenuator so only missing wavelength log
    SampleLogs(ws).insert("attenuator", 4)
    with pytest.raises(RuntimeError) as excinfo:
        attenuation_factor(ws)
    assert "wavelength" in str(
        excinfo.value
    )  # Should complain about missing wavelength


if __name__ == "__main__":
    pytest.main([__file__])
