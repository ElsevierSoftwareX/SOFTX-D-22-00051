# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/api.py
from drtsans import subtract_background
from drtsans.dataobjects import DataType, IQazimuthal, IQmod
from drtsans.settings import unique_workspace_dundername as uwd
from tests.conftest import assert_wksp_equal

# https://docs.mantidproject.org/nightly/algorithms/CompareWorkspaces-v1.html
# https://docs.mantidproject.org/nightly/algorithms/CreateWorkspace-v1.html
# https://docs.mantidproject.org/nightly/algorithms/DeleteWorkspace-v1.html
from mantid.simpleapi import CompareWorkspaces, CreateWorkspace, DeleteWorkspace
import numpy as np
import pytest


class _Data1D(object):
    """This is a factory class for generating the 1d example data

    The equations are taken from the spreadsheet supplied for 1d data
    """

    scale_factor = 0.92
    Sig_scale = 0.005 * scale_factor

    Q_Scale = np.linspace(0.01, 0.99, 99)  # 99 numbers from 0.01 to 0.99
    I_Background_1d = (
        np.power(Q_Scale * 10, -4) + 0.57
    )  # power-law maximum (Porod-scattering)
    Sig_background_1d = I_Background_1d * 0.02

    I_Data_1d = scale_factor * I_Background_1d + 0.2 * (
        0.5 - np.absolute(0.5 - Q_Scale)
    )
    Sig_data_1d = 0.01 * I_Data_1d

    I_output_1d = I_Data_1d - scale_factor * I_Background_1d
    Sig_output_1d = np.sqrt(
        np.power(Sig_data_1d, 2)
        + np.power(scale_factor * Sig_background_1d, 2)
        + np.power(Sig_scale * I_Background_1d, 2)
    )

    def __init__(self, mode):
        self.mode = mode

    def create(self, datatype):
        """This function creates data based on supplied information. There are pre-defined ``datatype``
        of ``data`` and ``background``. All others are custom.
        """
        if datatype == "data":
            y = self.I_Data_1d
            e = self.Sig_data_1d
        elif datatype == "background":
            y = self.I_Background_1d
            e = self.Sig_background_1d
        elif datatype == "output":
            y = self.I_output_1d
            e = self.Sig_output_1d
        else:
            raise RuntimeError('Unknown data type="{}"'.format(datatype))

        # create a workspace with the correct signal and uncertainties and random name
        if self.mode == DataType.WORKSPACE2D:
            return CreateWorkspace(
                DataX=self.Q_Scale,
                DataY=y,
                DataE=e,
                UnitX="momentumtransfer",
                OutputWorkspace=uwd(),
            )
        elif self.mode == DataType.IQ_MOD:
            return IQmod(intensity=y, error=e, mod_q=self.Q_Scale)
        else:
            raise NotImplementedError(
                'Cannot create test data of type "{}"'.format(self.mode)
            )


# -------------------- 1d tests
def test_data_not_background_1d():
    """This tests that the ``data`` is not equal to the ``background``

    dev - Pete Peterson <petersonpf@ornl.gov>
    SME - Ken Littrell <littrellkc@ornl.gov>
    """
    factory = _Data1D(DataType.WORKSPACE2D)

    data = factory.create("data")
    background = factory.create("background")

    assert not CompareWorkspaces(data, background).Result

    DeleteWorkspace(data)
    DeleteWorkspace(background)


@pytest.mark.parametrize("mode", [DataType.WORKSPACE2D, DataType.IQ_MOD])
def test_subtract_background_1d(mode):
    """This tests that ``data - scale * background`` and its uncertainties gives the expected result.

    dev - Pete Peterson <petersonpf@ornl.gov>
    SME - Ken Littrell <littrellkc@ornl.gov>
    """
    factory = _Data1D(mode)

    # create workspaces with the input data
    data = factory.create("data")
    background = factory.create("background")
    expected = factory.create("output")

    # do the calculation using the framework in-place
    observed = subtract_background(
        data, background, scale=factory.scale_factor, scale_error=factory.Sig_scale
    )

    # check the results
    assert_wksp_equal(observed, expected, rtol=1e-7)

    # cleanup
    if mode == DataType.WORKSPACE2D:
        # cleanup workspaces that were created
        for wksp in [data, background, expected]:
            DeleteWorkspace(wksp)


# -------------------- 2d tests
class _Data2D(object):
    """This is a factory class for generating the 1d example data

    The equations are taken from the spreadsheet supplied for 1d data
    """

    scale_factor = 0.92
    Sig_scale = 0.005 * scale_factor

    def __init__(self, mode):
        self.mode = mode

    def _get_Q(self):
        qx = []
        for i in range(10):
            qx.append(np.zeros(10) + ((i + 1) * 0.01))

        qy = []
        for i in range(10):
            qy.append(np.linspace(0.01, 0.1, 10))  # 99 numbers from 0.01 to 0.99

        if self.mode == "linearized":
            return np.array(qx).ravel(), np.array(qy).ravel()
        elif self.mode == "full_2d" or self.mode == "2d_edge":
            return np.array(qx), np.array(qy)
        else:
            raise NotImplementedError(
                'Do not know how to create data in mode "{}"'.format(self.mode)
            )

    def create(self, datatype):
        """This function creates data based on supplied information. There are pre-defined ``datatype``
        of ``data`` and ``background``. All others are custom.
        """
        qx, qy = self._get_Q()

        if datatype == "data":
            background = self.create("background")

            y = self.scale_factor * background.intensity + 0.2 * (
                0.5 - np.absolute(0.5 - qx)
            )
            e = 0.01 * y
        elif datatype == "background":
            qscalar = np.sqrt(np.square(qx) + np.square(qy))
            y = (
                np.power((qscalar * 10.0), -4) + 0.57
            )  # power-law maximum (Porod-scattering)
            e = y * 0.01
        elif datatype == "output":
            signal = self.create("data")
            background = self.create("background")

            y = signal.intensity - self.scale_factor * background.intensity
            e = np.sqrt(
                np.square(signal.error)
                + np.square(self.scale_factor * background.error)
                + np.square(self.Sig_scale * background.intensity)
            )
        else:
            raise RuntimeError('Unknown data type="{}"'.format(datatype))

        if self.mode == "2d_edge":
            qx = np.array(qx)[::, 0]
            qy = np.array(qy)[0]

        # create a workspace with the correct signal and uncertainties
        return IQazimuthal(intensity=y, error=e, qx=qx, qy=qy)


@pytest.mark.parametrize("mode", ["linearized", "full_2d", "2d_edge"])
def test_subtract_background_2d(mode):
    """This tests that ``data - scale * background`` and its uncertainties gives the expected result.

    The ``mode`` argument changes how the data is stored from ``linearized`` (1d) array, ``full_2d``
    array of (2d) values, and ``2d_edge`` wich only specifies qx and qy of the edges.

    dev - Pete Peterson <petersonpf@ornl.gov>
    SME - Ken Littrell <littrellkc@ornl.gov>
    """
    factory = _Data2D(mode)

    # create workspaces with the input data
    data = factory.create("data")
    background = factory.create("background")
    expected = factory.create("output")

    # do the calculation using the framework in-place
    observed = subtract_background(
        data, background, scale=factory.scale_factor, scale_error=factory.Sig_scale
    )

    # check the results
    assert_wksp_equal(observed, expected, rtol=1e-7)


if __name__ == "__main__":
    pytest.main([__file__])
