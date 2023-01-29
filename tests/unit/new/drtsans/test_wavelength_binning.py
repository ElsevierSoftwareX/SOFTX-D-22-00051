# https://docs.mantidproject.org/nightly/algorithms/Rebin-v1.html
import pytest
import numpy as np
from mantid.simpleapi import Rebin


@pytest.fixture
def fake_events():
    def _(wavelengths):
        from mantid.simpleapi import CreateSampleWorkspace
        import mantid.kernel

        ws = CreateSampleWorkspace(
            WorkspaceType="Event",
            Function="Flat background",
            NumBanks=1,
            BankPixelWidth=1,
            NumEvents=0,
            XUnit="Wavelength",
            XMax=10,
            BinWidth=1,
        )
        sp0 = ws.getSpectrum(0)
        date = mantid.kernel.DateAndTime("2019-09-09T00:00")
        for lam in wavelengths:
            sp0.addEventQuickly(lam, date)
        return ws

    return _


def test_linear(fake_events):
    r"""Test linear binning by creating a workspace with fake events at specific wavelengths,
    binning into a histogram with specific binning parameters, and test against expected output.
    dev - Jiao Lin <linjiao@ornl.gov>
    SME - William Heller <hellerwt@ornl.gov>

    **Mantid algorithms used:**
    :ref:`Rebin <algm-Rebin-v1>`

    For details see https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/208
    """
    # create a workspace with fake events with the given wavelengths
    ws = fake_events([2.57, 3.05, 2.76, 3.13, 2.84])
    # binning the events to linear bins with boundaries at 2.5, 2.6, 2.7, ..., 3.2
    # start, step, end of bin boundaries
    start, step, end = 2.5, 0.1, 3.2
    # rebin using Mantid algorithm Rebin
    ws = Rebin(InputWorkspace=ws, Params="{}, {}, {}".format(start, step, end))
    # verify
    assert np.allclose(ws.readX(0), np.arange(2.5, 3.21, 0.1))
    assert np.allclose(ws.readY(0), [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    return


def test_log(fake_events):
    r"""Test constant dlambda/lambda binning by creating a workspace with fake events at specific wavelengths,
    binning into a histogram with specific binning parameters, and test against expected output.
    dev - Jiao Lin <linjiao@ornl.gov>
    SME - William Heller <hellerwt@ornl.gov>

    **Mantid algorithms used:**
    :ref:`Rebin <algm-Rebin-v1>`

    For details see https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/207
    """
    # create a workspace with fake events
    ws = fake_events([2.57, 3.05, 2.76, 3.13, 2.84])
    # binning the events to loglinear bins with boundaries at 2.5, 2.625, 2.75625, 2.894063, 3.038, 3.1907
    # this is the constant dlambda/lambda
    dlambda_over_lambda = 0.125 / 2.5
    # start, step, end of bin boundaries. negative means loglinear!
    start, step, end = 2.5, -dlambda_over_lambda, 3.36
    # rebin using Mantid algorithm Rebin
    ws = Rebin(InputWorkspace=ws, Params="{}, {}, {}".format(start, step, end))
    # verify
    assert np.allclose(
        ws.readX(0), [2.5, 2.625, 2.75625, 2.894063, 3.038766, 3.190704, 3.36]
    )
    assert np.allclose(ws.readY(0), [1.0, 0.0, 2.0, 0.0, 2.0, 0.0])
    return


if __name__ == "__main__":
    pytest.main([__file__])
