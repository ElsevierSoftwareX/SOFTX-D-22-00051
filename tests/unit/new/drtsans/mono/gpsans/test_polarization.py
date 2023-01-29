import pytest
from drtsans.settings import unique_workspace_dundername as uwd
from drtsans import half_polarization
from drtsans.api import _calc_flipping_ratio  # private function

# https://docs.mantidproject.org/nightly/algorithms/CreateSingleValuedWorkspace-v1.html
from mantid.simpleapi import CreateSingleValuedWorkspace


def test_flipping_ratio():
    """Test for the calclation of the flipping ratio in section 9.1. This was used to determine
    that the uncertainty needs to be calculated separately because of accumulation of numeric
    error.

    dev - Pete Peterson <petersonpf@ornl.gov>
    SME - Lisa DeBeer-Schmitt <debeerschmlm@ornl.gov>
          Mike Fitzsimmons <fitzsimmonsm@ornl.gov>
    """
    # this is called "P" in the document
    polarization = CreateSingleValuedWorkspace(
        DataValue=0.95, ErrorValue=0.01, OutputWorkspace=uwd()
    )
    flipping_ratio_expected = (1.0 + 0.95) / (1.0 - 0.95)
    flipping_ratio_err_expected = (
        2.0 * polarization.readE(0)[0]
    ) / 0.0025  # denominator is (1-p)^2

    flipping_ratio = _calc_flipping_ratio(polarization)

    assert flipping_ratio.extractY() == flipping_ratio_expected
    assert flipping_ratio.extractE() == pytest.approx(flipping_ratio_err_expected)


def test_half_polarization():
    """Test the calculation and application of the half polarization from section 9.1
    and requires reading section 9.0 for definition variables.

    dev - Pete Peterson <petersonpf@ornl.gov>
    SME - Lisa DeBeer-Schmitt <debeerschmlm@ornl.gov>
          Mike Fitzsimmons <fitzsimmonsm@ornl.gov>
    """
    # this is called "P" in the document
    polarization = CreateSingleValuedWorkspace(
        DataValue=0.95, ErrorValue=0.01, OutputWorkspace=uwd()
    )
    # this is called "e" in the document
    efficiency = CreateSingleValuedWorkspace(
        DataValue=0.998, ErrorValue=0.001, OutputWorkspace=uwd()
    )

    # values for the measured flipper off (M0) and flipper on (M1)
    M0 = CreateSingleValuedWorkspace(
        DataValue=10000, ErrorValue=100, OutputWorkspace=uwd()
    )
    M1 = CreateSingleValuedWorkspace(
        DataValue=8100, ErrorValue=90, OutputWorkspace=uwd()
    )

    # expected results
    SpinUpExp = CreateSingleValuedWorkspace(
        DataValue=10050.100, ErrorValue=103.2046, OutputWorkspace=uwd()  # was 103.205
    )
    SpinDownExp = CreateSingleValuedWorkspace(
        DataValue=8046.0925, ErrorValue=93.2163, OutputWorkspace=uwd()
    )

    # do the calculation
    SpinUp, SpinDown = half_polarization(M0, M1, polarization, efficiency)

    # compare with what was expected
    assert SpinUp.extractY()[0][0] == pytest.approx(SpinUpExp.extractY()[0][0])
    assert SpinUp.extractE()[0][0] == pytest.approx(SpinUpExp.extractE()[0][0])
    assert SpinDown.extractY()[0][0] == pytest.approx(SpinDownExp.extractY()[0][0])
    assert SpinDown.extractE()[0][0] == pytest.approx(SpinDownExp.extractE()[0][0])


if __name__ == "__main__":
    pytest.main([__file__])
