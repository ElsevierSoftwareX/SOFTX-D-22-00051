import pytest
import numpy as np

# CreateWorkspace <https://docs.mantidproject.org/nightly/algorithms/CreateWorkspace-v1.html>
from mantid.simpleapi import mtd, CreateWorkspace

# unique_workspace_dundername within <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/settings.py> # noqa: 501
# subtract_dark_current <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/tof/eqsans/dark_current.py>  # noqa: E501
# SampleLogs within <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/samplelogs.py>
from drtsans.settings import unique_workspace_dundername
from drtsans.samplelogs import SampleLogs
from drtsans.tof.eqsans.dark_current import subtract_dark_current


@pytest.fixture(scope="module")
def data_test_16a():
    r"""
    Input and expected output taken from the intro to issue #174
    <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/174#dark-current-normalized-at-eq-sans>
    """
    l_min, l_step, l_max = 2.5, 0.1, 6.0
    f = 0.0006746  # sample_run_duration*(t_frame-t_low-t_high)*lstep/(dark_current_duration*t_frame*(l_max-l_min))
    return dict(
        t_frame=1.0 / 60.0 * 1000000.0,  # time duration of a frame, in microseconds
        t_low=500,  # sample measurement tof range, in microseconds
        t_high=2000,
        sample_run_duration=100.0,  # sample measurement duration, in seconds
        dark_current_duration=3600.0,  # dark current duration, in seconds
        l_min=l_min,  # minimum wavelength
        l_max=l_max,
        l_step=l_step,  # wavelength bin width
        n_pixels=30,  # 30 pixels in the detector
        # dark current intensities and errors
        I_dc=[
            [
                40.0,
                45.0,
                50.0,
                55.0,
                60.0,
            ],  # intensity per pixel, integrated over all wavelength bins
            [65.0, 70.0, 75.0, 80.0, 85.0],
            [90.0, 95.0, 100.0, 105.0, 110.0],
            [115.0, 120.0, 125.0, 130.0, 135.0],
            [140.0, 145.0, 150.0, 155.0, 160.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        I_dc_err=[
            [6.3246, 6.7082, 7.0711, 7.4162, 7.746],
            [8.0623, 8.3666, 8.6603, 8.9443, 9.2195],
            [9.4868, 9.7468, 10.0, 10.247, 10.4881],
            [10.7238, 10.9545, 11.1803, 11.4018, 11.619],
            [11.8322, 12.0416, 12.2474, 12.4499, 12.6491],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        # sample data intensities and errors. Numers are selected in such a way that subtracting the dark
        # current will yield no intensities.
        I_data=[
            [
                0.9444,
                1.0625,
                1.1806,
                1.2986,
                1.4167,
            ],  # intensity per pixel, integrated over all
            [1.5347, 1.6528, 1.7708, 1.8889, 2.0069],  # wavelength bins
            [2.125, 2.2431, 2.3611, 2.4792, 2.5972],
            [2.7153, 2.8333, 2.9514, 3.0694, 3.1875],
            [3.3056, 3.4236, 3.5417, 3.6597, 3.7778],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        I_data_err=[
            [0.0252, 0.0268, 0.0282, 0.0296, 0.0309],
            [0.0322, 0.0334, 0.0346, 0.0357, 0.0368],
            [0.0379, 0.0389, 0.0399, 0.0409, 0.0419],
            [0.0428, 0.0437, 0.0446, 0.0455, 0.0464],
            [0.0472, 0.0481, 0.0489, 0.0497, 0.0505],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        # Sample data normalized by the dark current. As stated before, the sample data was selected such
        # that the normalization will yield no intensities.
        I_data_norm=[
            [0.0, 0.0, 0.0, 0.0, 0.0],  # data minus normalized dark current
            [0.0, 0.0, 0.0, 0.0, 0.0],  # is all zeroes by construction
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        I_data_norm_err=[
            [0.001, 0.0011, 0.0011, 0.0012, 0.0012],
            [0.0013, 0.0013, 0.0014, 0.0014, 0.0015],
            [0.0015, 0.0016, 0.0016, 0.0017, 0.0017],
            [0.0017, 0.0018, 0.0018, 0.0018, 0.0019],
            [0.0019, 0.0019, 0.002, 0.002, 0.002],
            [f, f, f, f, f],
        ],
        precision=1.0e-4,  # precision to compare reduction framework to test results
    )


def test_subtract_dark_current(data_test_16a):
    """Test of dark current subtraction from data. Dark current must be normalized

    For details see https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/156
    and also https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/174

    dev - Jose Borreguero <borreguerojm@ornl.gov>, Steven Hahn <hahnse@ornl.gov>, Jiao Lin <linjiao@ornl.gov>
    SME - Changwoo Do <doc1@ornl.gov>

    **Mantid algorithms used:**
    :ref:`CreateWorkspace <algm-CreateWorkspace-v1>`,
        <https://docs.mantidproject.org/nightly/algorithms/CreateWorkspace-v1.html>

    **drtsans functions used:**
    ~drtsans.settings.unique_workspace_dundername
        <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/settings.py>
    ~drtsans.samplelogs.SampleLogs
        <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/samplelogs.py>
    ~drtsans.tof.eqsans.dark_current.normalize_dark_current
        <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/tof/eqsans/dark_current.py>
    """
    wavelength_bin_boundaries = np.arange(
        data_test_16a["l_min"],
        data_test_16a["l_max"] + data_test_16a["l_step"] / 2.0,
        data_test_16a["l_step"],
    )

    # The intensity in a detector pixel is the same for all wavelength bins by construction in the test. Thus,
    # we repeat the one given value per detector pixel to be the same for all wavelength bins.
    dark_intensities_list = np.array(data_test_16a["I_dc"]).flatten()
    dark_intensities_list = np.repeat(
        dark_intensities_list[:, np.newaxis], len(wavelength_bin_boundaries) - 1, axis=1
    )
    dark_errors_list = np.array(data_test_16a["I_dc_err"]).flatten()
    dark_errors_list = np.repeat(
        dark_errors_list[:, np.newaxis], len(wavelength_bin_boundaries) - 1, axis=1
    )

    # The test does not sum the dark current intensities over all wavelength channels, but the reduction framework
    # does as it follows the master document. In order to match the test results we have to first divide the
    # dark current intensities by the number of wavelength channels
    dark_intensities_list /= len(wavelength_bin_boundaries) - 1
    dark_errors_list /= len(wavelength_bin_boundaries) - 1

    # The dark current workspace now becomes:
    dark_workspace = (
        unique_workspace_dundername()
    )  # arbitrary name for the dark current workspace
    CreateWorkspace(
        DataX=wavelength_bin_boundaries,
        UnitX="Wavelength",
        DataY=dark_intensities_list,
        DataE=dark_errors_list,
        NSpec=data_test_16a["n_pixels"],
        OutputWorkspace=dark_workspace,
    )

    # Initialize the dark current logs. Only the duration of the run is necessary, which is recorded by the data
    # acquisition software.
    dark_sample_log = SampleLogs(dark_workspace)
    dark_sample_log.insert("duration", data_test_16a["dark_current_duration"])

    # Same procedure now in order to create the workspace for the sample run
    data_intensities_list = np.array(data_test_16a["I_data"]).flatten()
    data_intensities_list = np.repeat(
        data_intensities_list[:, np.newaxis], len(wavelength_bin_boundaries) - 1, axis=1
    )
    data_intensities_list /= len(wavelength_bin_boundaries) - 1
    data_errors_list = np.array(data_test_16a["I_data_err"]).flatten()
    data_errors_list = np.repeat(
        data_errors_list[:, np.newaxis], len(wavelength_bin_boundaries) - 1, axis=1
    )
    data_errors_list /= len(wavelength_bin_boundaries) - 1

    data_workspace = (
        unique_workspace_dundername()
    )  # arbitrary name for the sample workspace
    CreateWorkspace(
        DataX=wavelength_bin_boundaries,
        UnitX="Wavelength",
        DataY=data_intensities_list,
        DataE=data_errors_list,
        NSpec=data_test_16a["n_pixels"],
        OutputWorkspace=data_workspace,
    )

    # Initialize the sample logs. In the reduction framework this would have happened after loading the events file
    # and converting to wavelength
    data_sample_logs = SampleLogs(data_workspace)
    SampleLogs(data_workspace).insert(
        "duration", data_test_16a["sample_run_duration"], "second"
    )
    data_sample_logs.insert("tof_frame_width", data_test_16a["t_frame"])
    data_sample_logs.insert(
        "tof_frame_width_clipped",
        data_test_16a["t_frame"] - data_test_16a["t_low"] - data_test_16a["t_high"],
    )  # noqa: E501
    data_sample_logs.insert("wavelength_min", data_test_16a["l_min"], unit="Angstrom")
    data_sample_logs.insert("wavelength_max", data_test_16a["l_max"], unit="Angstrom")
    data_sample_logs.insert(
        "wavelength_lead_min", data_test_16a["l_min"], unit="Angstrom"
    )
    data_sample_logs.insert(
        "wavelength_lead_max", data_test_16a["l_max"], unit="Angstrom"
    )
    data_sample_logs.insert("is_frame_skipping", False)

    # Call the reduction workflow function
    subtract_dark_current(data_workspace, dark_workspace)

    # Compare the normalized intensities.
    computed_intensities = np.transpose(mtd[data_workspace].extractY())[0].ravel()
    test_intensities = np.array(data_test_16a["I_data_norm"]).ravel()
    assert computed_intensities == pytest.approx(
        test_intensities, abs=data_test_16a["precision"]
    )

    # Compare the errors of the normalized intensities
    computed_errors = np.transpose(mtd[data_workspace].extractE())[0].ravel()
    test_errors = np.array(data_test_16a["I_data_norm_err"]).ravel()
    assert computed_errors == pytest.approx(test_errors, abs=data_test_16a["precision"])
    assert computed_errors[-1] == pytest.approx(
        test_errors[-1], abs=data_test_16a["precision"] * 0.01
    )


if __name__ == "__main__":
    pytest.main([__file__])
