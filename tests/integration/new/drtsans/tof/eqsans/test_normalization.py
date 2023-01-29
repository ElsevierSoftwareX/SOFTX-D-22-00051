import pytest
import numpy as np

# CreateWorkspace <https://docs.mantidproject.org/nightly/algorithms/CreateWorkspace-v1.html>
from mantid.simpleapi import CreateWorkspace
from mantid.api import mtd

# unique_workspace_dundername within <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/settings.py> # noqa: 501
# SampleLogs within <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/samplelogs.py>
# time, monitor within <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/mono/normalization.py>
from drtsans.settings import unique_workspace_dundername
from drtsans.samplelogs import SampleLogs
from drtsans.tof.eqsans import (
    load_events,
    transform_to_wavelength,
    normalize_by_time,
    normalize_by_monitor,
    normalize_by_proton_charge_and_flux,
    set_init_uncertainties,
)


def test_normalize_by_time(reference_dir):
    r"""
    Test normalization by time duration.
    """
    output_workspace = unique_workspace_dundername()
    load_events(
        "EQSANS_68168",
        data_dir=reference_dir.new.eqsans,
        output_workspace=output_workspace,
    )
    workspace, bands = transform_to_wavelength(output_workspace)
    workspace = set_init_uncertainties(workspace)
    time_duration = SampleLogs(output_workspace).duration.value

    # Let's pick one spectrum (spectrum 42) and verify we are dividing its intensity by the time duration
    intensity, uncertainty = np.copy(workspace.readY(42)), np.copy(workspace.readE(42))
    workspace_normalized = normalize_by_time(output_workspace)

    # Verify we selected 'duration' as the log entry to find out the duration of the run
    assert SampleLogs(output_workspace).normalizing_duration.value == "duration"

    assert workspace_normalized.readY(42) == pytest.approx(
        intensity / time_duration, abs=1.0e-6
    )
    assert workspace_normalized.readE(42) == pytest.approx(uncertainty / time_duration)

    workspace_normalized.delete()  # some cleanup


@pytest.fixture(scope="module")
def data_test_16a_by_time():
    r"""
    Input and expected output for the normalization by time test, taken from the intro to issue #174
    <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/174#normalization-by-monitor-spectrum>
    """
    return dict(
        t_sam=5,  # sample duration
        n_pixels=25,
        wavelength_bin_boundaries=[2.5, 3.5],
        precision=1.0e-4,  # precision when comparing output with data from this test
        I_sam=[
            [40.0, 45.0, 50.0, 55.0, 60.0],
            [65.0, 70.0, 75.0, 80.0, 85.0],
            [90.0, 95.0, 100.0, 105.0, 110.0],
            [115.0, 120.0, 125.0, 130.0, 135.0],
            [140.0, 145.0, 150.0, 155.0, 160.0],
        ],
        I_sam_err=[
            [6.3246, 6.7082, 7.0711, 7.4162, 7.746],
            [8.0623, 8.3666, 8.6603, 8.9443, 9.2195],
            [9.4868, 9.7468, 10.0, 10.247, 10.4881],
            [10.7238, 10.9545, 11.1803, 11.4018, 11.619],
            [11.8322, 12.0416, 12.2474, 12.4499, 12.6491],
        ],
        I_samnorm=[
            [8.0, 9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0, 17.0],
            [18.0, 19.0, 20.0, 21.0, 22.0],
            [23.0, 24.0, 25.0, 26.0, 27.0],
            [28.0, 29.0, 30.0, 31.0, 32.0],
        ],
        I_samnorm_error=[
            [1.26492, 1.34164, 1.41422, 1.48324, 1.5492],
            [1.61246, 1.67332, 1.73206, 1.78886, 1.8439],
            [1.89736, 1.94936, 2.0, 2.0494, 2.09762],
            [2.14476, 2.1909, 2.23606, 2.28036, 2.3238],
            [2.36644, 2.40832, 2.44948, 2.48998, 2.52982],
        ],
    )


def test_normalization_by_time(data_test_16a_by_time):
    r"""
    Normalize sample intensities by run duration.
    Addresses section of the 6.1 the master document

    devs - Steven Hahn <hahnse@ornl.gov>,
           Jose Borreguero <borreguerojm@ornl.gov>
    SME  - Changwoo Do <doc1@ornl.gov>

    **Mantid algorithms used:**
    :ref:`CreateWorkspace <algm-CreateWorkspace-v1>`,
    <https://docs.mantidproject.org/nightly/algorithms/CreateWorkspace-v1.html>

    **drtsans functions used:**
    ~drtsans.settings.unique_workspace_dundername
    <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/settings.py>
    ~drtsans.samplelogs.SampleLogs
    <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/samplelogs.py>
    ~drtsans.tof.normalization.normalize_by_time
    <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/tof/eqsans/normalization.py>
    """
    # Create a sample workspace with the input data
    data_workspace = unique_workspace_dundername()
    CreateWorkspace(
        DataX=data_test_16a_by_time["wavelength_bin_boundaries"],
        DataY=np.array(data_test_16a_by_time["I_sam"]).ravel(),
        DataE=np.array(data_test_16a_by_time["I_sam_err"]).ravel(),
        NSpec=data_test_16a_by_time["n_pixels"],
        OutputWorkspace=data_workspace,
    )
    SampleLogs(data_workspace).insert(
        "duration", data_test_16a_by_time["t_sam"], "Second"
    )

    # Carry out the normalization by time
    normalize_by_time(data_workspace)

    # Compare calculated normalized intensities with test output
    test_intensities = np.array(data_test_16a_by_time["I_samnorm"]).ravel()
    assert mtd[data_workspace].extractY().ravel() == pytest.approx(
        test_intensities, abs=data_test_16a_by_time["precision"]
    )

    # Compare calculated errors of normalized intensities with test output
    test_errors = np.array(data_test_16a_by_time["I_samnorm_error"]).ravel()
    assert mtd[data_workspace].extractE().ravel() == pytest.approx(
        test_errors, abs=data_test_16a_by_time["precision"]
    )


@pytest.fixture(scope="module")
def data_test_16a_by_monitor():
    r"""
    Input and expected output for the normalization by monitor test, taken from the intro to issue #174
    <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/174#normalization-by-monitor-spectrum>
    """
    return dict(
        precision=1e-04,  # desired precision for comparisons,
        n_pixels=25,
        wavelength_bin_boundaries=[
            2.0,
            2.1,
            2.2,
            2.3,
            2.4,
            2.5,
            2.6,
            2.7,
            2.8,
            2.9,
            3.0,
        ],
        # The intensity in a detector pixel is the same for all wavelength bins. I_sam below shows the
        # intensity of one wavelength bin on all detectors.
        I_sam=[
            [40.0, 45.0, 50.0, 55.0, 60.0],
            [65.0, 70.0, 75.0, 80.0, 85.0],
            [90.0, 95.0, 100.0, 105.0, 110.0],
            [115.0, 120.0, 125.0, 130.0, 135.0],
            [140.0, 145.0, 150.0, 155.0, 160.0],
        ],
        # Same as I_sam but now showing the uncertainties.
        I_sam_err=[
            [6.3246, 6.7082, 7.0711, 7.4162, 7.746],
            [8.0623, 8.3666, 8.6603, 8.9443, 9.2195],
            [9.4868, 9.7468, 10.0, 10.247, 10.4881],
            [10.7238, 10.9545, 11.1803, 11.4018, 11.619],
            [11.8322, 12.0416, 12.2474, 12.4499, 12.6491],
        ],
        flux_to_monitor_ratios=[5, 5, 4, 4, 3, 3, 3, 3, 2, 2],  # flux to monitor ratio
        monitor_counts=[20, 40, 30, 25, 20, 10, 5, 5, 5, 5],  # monitor spectrum
        I_samnorm=[
            [
                [0.4, 0.45, 0.5, 0.55, 0.6],
                [0.65, 0.7, 0.75, 0.8, 0.85],
                [0.9, 0.95, 1.0, 1.05, 1.1],
                [1.15, 1.2, 1.25, 1.3, 1.35],
                [1.4, 1.45, 1.5, 1.55, 1.6],
            ],
            [
                [0.2, 0.225, 0.25, 0.275, 0.3],
                [0.325, 0.35, 0.375, 0.4, 0.425],
                [0.45, 0.475, 0.5, 0.525, 0.55],
                [0.575, 0.6, 0.625, 0.65, 0.675],
                [0.7, 0.725, 0.75, 0.775, 0.8],
            ],
            [
                [0.3333, 0.375, 0.4167, 0.4583, 0.5],
                [0.5417, 0.5833, 0.625, 0.6667, 0.7083],
                [0.75, 0.7917, 0.8333, 0.875, 0.9167],
                [0.9583, 1.0, 1.0417, 1.0833, 1.125],
                [1.1667, 1.2083, 1.25, 1.2917, 1.3333],
            ],
            [
                [0.4, 0.45, 0.5, 0.55, 0.6],
                [0.65, 0.7, 0.75, 0.8, 0.85],
                [0.9, 0.95, 1.0, 1.05, 1.1],
                [1.15, 1.2, 1.25, 1.3, 1.35],
                [1.4, 1.45, 1.5, 1.55, 1.6],
            ],
            [
                [0.6667, 0.75, 0.8333, 0.9167, 1.0],
                [1.0833, 1.1667, 1.25, 1.3333, 1.4167],
                [1.5, 1.5833, 1.6667, 1.75, 1.8333],
                [1.9167, 2.0, 2.0833, 2.1667, 2.25],
                [2.3333, 2.4167, 2.5, 2.5833, 2.6667],
            ],
            [
                [1.3333, 1.5, 1.6667, 1.8333, 2.0],
                [2.1667, 2.3333, 2.5, 2.6667, 2.8333],
                [3.0, 3.1667, 3.3333, 3.5, 3.6667],
                [3.8333, 4.0, 4.1667, 4.3333, 4.5],
                [4.6667, 4.8333, 5.0, 5.1667, 5.3333],
            ],
            [
                [2.6667, 3.0, 3.3333, 3.6667, 4.0],
                [4.3333, 4.6667, 5.0, 5.3333, 5.6667],
                [6.0, 6.3333, 6.6667, 7.0, 7.3333],
                [7.6667, 8.0, 8.3333, 8.6667, 9.0],
                [9.3333, 9.6667, 10.0, 10.3333, 10.6667],
            ],
            [
                [2.6667, 3.0, 3.3333, 3.6667, 4.0],
                [4.3333, 4.6667, 5.0, 5.3333, 5.6667],
                [6.0, 6.3333, 6.6667, 7.0, 7.3333],
                [7.6667, 8.0, 8.3333, 8.6667, 9.0],
                [9.3333, 9.6667, 10.0, 10.3333, 10.6667],
            ],
            [
                [4.0, 4.5, 5.0, 5.5, 6.0],
                [6.5, 7.0, 7.5, 8.0, 8.5],
                [9.0, 9.5, 10.0, 10.5, 11.0],
                [11.5, 12.0, 12.5, 13.0, 13.5],
                [14.0, 14.5, 15.0, 15.5, 16.0],
            ],
            [
                [4.0, 4.5, 5.0, 5.5, 6.0],
                [6.5, 7.0, 7.5, 8.0, 8.5],
                [9.0, 9.5, 10.0, 10.5, 11.0],
                [11.5, 12.0, 12.5, 13.0, 13.5],
                [14.0, 14.5, 15.0, 15.5, 16.0],
            ],
        ],
    )


def test_normalization_by_monitor(data_test_16a_by_monitor):
    r"""
    Normalize sample intensities by flux at monitor using also flux-to-monitor ratios.
    Addresses section of the 6.2 the master document

    devs - Steven Hahn <hahnse@ornl.gov>,
           Jose Borreguero <borreguerojm@ornl.gov>
    SME  - Changwoo Do <doc1@ornl.gov>

    **Mantid algorithms used:**
    :ref:`CreateWorkspace <algm-CreateWorkspace-v1>`,
    <https://docs.mantidproject.org/nightly/algorithms/CreateWorkspace-v1.html>

    **drtsans functions used:**
    ~drtsans.settings.unique_workspace_dundername
    <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/settings.py>
    ~drtsans.samplelogs.SampleLogs
    <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/samplelogs.py>
    ~drtsans.tof.normalization.normalize_by_monitor
    <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/tof/eqsans/normalization.py>
    """
    # Input intensities from the test, only one value per detector pixel
    intensities_list = np.array(data_test_16a_by_monitor["I_sam"]).flatten()
    errors_list = np.array(data_test_16a_by_monitor["I_sam_err"]).flatten()

    # The intensity in a detector pixel is the same for all wavelength bins. Thus, we replicate the one value per
    # detector pixel to be the same for all wavelength bins
    number_wavelength_bins = (
        len(data_test_16a_by_monitor["wavelength_bin_boundaries"]) - 1
    )
    intensities_list = np.repeat(
        intensities_list[:, np.newaxis], number_wavelength_bins, axis=1
    )
    errors_list = np.repeat(errors_list[:, np.newaxis], number_wavelength_bins, axis=1)

    # Create the workspace with the intensities and errors. It has 25 spectra and each spectra has 10 wavelength bins
    data_workspace = CreateWorkspace(
        DataX=data_test_16a_by_monitor["wavelength_bin_boundaries"],
        DataY=intensities_list,
        DataE=errors_list,
        NSpec=data_test_16a_by_monitor["n_pixels"],
        OutputWorkspace=unique_workspace_dundername(),
    )
    SampleLogs(data_workspace).insert(
        "is_frame_skipping", False
    )  # Monitor normalization does not work in skip-frame

    # In the reduction framework, counts at the monitor will be stored in a Mantid workspace
    monitor_workspace = CreateWorkspace(
        DataX=data_test_16a_by_monitor["wavelength_bin_boundaries"],
        DataY=data_test_16a_by_monitor["monitor_counts"],
        DataE=np.zeros(number_wavelength_bins),
        NSpec=1,
        OutputWorkspace=unique_workspace_dundername(),
    )

    # In the reduction framework, the flux-to-monitor file will be loaded to a Mantid workspace
    flux_to_monitor_workspace = CreateWorkspace(
        DataX=data_test_16a_by_monitor["wavelength_bin_boundaries"],
        DataY=data_test_16a_by_monitor["flux_to_monitor_ratios"],
        DataE=np.zeros(number_wavelength_bins),
        NSpec=1,
        OutputWorkspace=unique_workspace_dundername(),
    )
    # Carry out the normalization with the reduction framework
    data_workspace = normalize_by_monitor(
        data_workspace, flux_to_monitor_workspace, monitor_workspace
    )

    # Compare to test data. Notice that data_test_16a_by_monitor['I_samnorm'] has shape (10, 5, 5) but
    # data_workspace.extractY() has shape (25, 10). A transpose operation is necessary
    test_intensities = np.transpose(
        np.array(data_test_16a_by_monitor["I_samnorm"]).reshape((10, 25))
    )
    assert data_workspace.extractY() == pytest.approx(
        test_intensities, abs=data_test_16a_by_monitor["precision"]
    )


@pytest.fixture(scope="module")
def data_test_16a_by_proton_charge_and_flux():
    r"""
    Input and expected output for the normalization by proton charge test, taken from the intro to issue #174
    <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/174#normalize-by-proton-charge-and-flux-spectrum>
    """
    return dict(
        precision=1e-04,  # desired precision for comparisons,
        n_pixels=25,
        wavelength_bin_boundaries=[
            2.0,
            2.1,
            2.2,
            2.3,
            2.4,
            2.5,
            2.6,
            2.7,
            2.8,
            2.9,
            3.0,
        ],
        # The intensity in a detector pixel is the same for all wavelength bins. I_sam below shows the
        # intensity of one wavelength bin on all detectors.
        I_sam=[
            [40.0, 45.0, 50.0, 55.0, 60.0],
            [65.0, 70.0, 75.0, 80.0, 85.0],
            [90.0, 95.0, 100.0, 105.0, 110.0],
            [115.0, 120.0, 125.0, 130.0, 135.0],
            [140.0, 145.0, 150.0, 155.0, 160.0],
        ],
        # Same as I_sam but now showing the uncertainties.
        I_sam_err=[
            [6.3246, 6.7082, 7.0711, 7.4162, 7.746],
            [8.0623, 8.3666, 8.6603, 8.9443, 9.2195],
            [9.4868, 9.7468, 10.0, 10.247, 10.4881],
            [10.7238, 10.9545, 11.1803, 11.4018, 11.619],
            [11.8322, 12.0416, 12.2474, 12.4499, 12.6491],
        ],
        phi=[100, 90, 80, 70, 60, 50, 40, 30, 20, 10],  # flux spectrum
        proton_sam=10.0,  # proton charge
        I_samnorm=[
            [
                [0.04, 0.045, 0.05, 0.055, 0.06],
                [0.065, 0.07, 0.075, 0.08, 0.085],
                [0.09, 0.095, 0.1, 0.105, 0.11],
                [0.115, 0.12, 0.125, 0.13, 0.135],
                [0.14, 0.145, 0.15, 0.155, 0.16],
            ],
            [
                [0.0444, 0.05, 0.0556, 0.0611, 0.0667],
                [0.0722, 0.0778, 0.0833, 0.0889, 0.0944],
                [0.1, 0.1056, 0.1111, 0.1167, 0.1222],
                [0.1278, 0.1333, 0.1389, 0.1444, 0.15],
                [0.1556, 0.1611, 0.1667, 0.1722, 0.1778],
            ],
            [
                [0.05, 0.0563, 0.0625, 0.0688, 0.075],
                [0.0813, 0.0875, 0.0938, 0.1, 0.1062],
                [0.1125, 0.1187, 0.125, 0.1313, 0.1375],
                [0.1437, 0.15, 0.1562, 0.1625, 0.1688],
                [0.175, 0.1812, 0.1875, 0.1938, 0.2],
            ],
            [
                [0.0571, 0.0643, 0.0714, 0.0786, 0.0857],
                [0.0929, 0.1, 0.1071, 0.1143, 0.1214],
                [0.1286, 0.1357, 0.1429, 0.15, 0.1571],
                [0.1643, 0.1714, 0.1786, 0.1857, 0.1929],
                [0.2, 0.2071, 0.2143, 0.2214, 0.2286],
            ],
            [
                [0.0667, 0.075, 0.0833, 0.0917, 0.1],
                [0.1083, 0.1167, 0.125, 0.1333, 0.1417],
                [0.15, 0.1583, 0.1667, 0.175, 0.1833],
                [0.1917, 0.2, 0.2083, 0.2167, 0.225],
                [0.2333, 0.2417, 0.25, 0.2583, 0.2667],
            ],
            [
                [0.08, 0.09, 0.1, 0.11, 0.12],
                [0.13, 0.14, 0.15, 0.16, 0.17],
                [0.18, 0.19, 0.2, 0.21, 0.22],
                [0.23, 0.24, 0.25, 0.26, 0.27],
                [0.28, 0.29, 0.3, 0.31, 0.32],
            ],
            [
                [0.1, 0.1125, 0.125, 0.1375, 0.15],
                [0.1625, 0.175, 0.1875, 0.2, 0.2125],
                [0.225, 0.2375, 0.25, 0.2625, 0.275],
                [0.2875, 0.3, 0.3125, 0.325, 0.3375],
                [0.35, 0.3625, 0.375, 0.3875, 0.4],
            ],
            [
                [0.1333, 0.15, 0.1667, 0.1833, 0.2],
                [0.2167, 0.2333, 0.25, 0.2667, 0.2833],
                [0.3, 0.3167, 0.3333, 0.35, 0.3667],
                [0.3833, 0.4, 0.4167, 0.4333, 0.45],
                [0.4667, 0.4833, 0.5, 0.5167, 0.5333],
            ],
            [
                [0.2, 0.225, 0.25, 0.275, 0.3],
                [0.325, 0.35, 0.375, 0.4, 0.425],
                [0.45, 0.475, 0.5, 0.525, 0.55],
                [0.575, 0.6, 0.625, 0.65, 0.675],
                [0.7, 0.725, 0.75, 0.775, 0.8],
            ],
            [
                [0.4, 0.45, 0.5, 0.55, 0.6],
                [0.65, 0.7, 0.75, 0.8, 0.85],
                [0.9, 0.95, 1.0, 1.05, 1.1],
                [1.15, 1.2, 1.25, 1.3, 1.35],
                [1.4, 1.45, 1.5, 1.55, 1.6],
            ],
        ],
    )


def test_normalize_by_proton_charge_and_flux(data_test_16a_by_proton_charge_and_flux):
    r"""
    Normalize sample intensities by flux and proton charge.
    Addresses section of the 6.3 the master document

    devs - Steven Hahn <hahnse@ornl.gov>,
           Jose Borreguero <borreguerojm@ornl.gov>
    SME  - Changwoo Do <doc1@ornl.gov>

    **Mantid algorithms used:**
    :ref:`CreateWorkspace <algm-CreateWorkspace-v1>`,
    <https://docs.mantidproject.org/nightly/algorithms/CreateWorkspace-v1.html>

    **drtsans functions used:**
    ~drtsans.settings.unique_workspace_dundername
    <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/settings.py>
    ~drtsans.samplelogs.SampleLogs
    <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/samplelogs.py>
    ~drtsans.tof.normalization.normalize_by_proton_charge_and_flux
    <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/tof/eqsans/normalization.py>
    """
    test_data = data_test_16a_by_proton_charge_and_flux  # handy shortcut
    # Input intensities from the test, only one value per detector pixel
    intensities_list = np.array(test_data["I_sam"]).flatten()
    errors_list = np.array(test_data["I_sam_err"]).flatten()

    # The intensity in a detector pixel is the same for all wavelength bins. Thus, we replicate the one value per
    # detector pixel to be the same for all wavelength bins
    number_wavelength_bins = len(test_data["wavelength_bin_boundaries"]) - 1
    intensities_list = np.repeat(
        intensities_list[:, np.newaxis], number_wavelength_bins, axis=1
    )
    errors_list = np.repeat(errors_list[:, np.newaxis], number_wavelength_bins, axis=1)

    # Create the workspace with the intensities and errors. It has 25 spectra and each spectra has 10 wavelength bins
    data_workspace = CreateWorkspace(
        DataX=test_data["wavelength_bin_boundaries"],
        DataY=intensities_list,
        DataE=errors_list,
        NSpec=test_data["n_pixels"],
        OutputWorkspace=unique_workspace_dundername(),
    )
    # Insert the proton charge in the logs of the workspace
    SampleLogs(data_workspace).insert("gd_prtn_chrg", test_data["proton_sam"])

    # In the reduction framework, the flux file will be loaded to a Mantid workspace
    flux_workspace = CreateWorkspace(
        DataX=test_data["wavelength_bin_boundaries"],
        DataY=test_data["phi"],
        OutputWorkspace=unique_workspace_dundername(),
    )

    # Carry out the normalization with the reduction framework
    data_workspace = normalize_by_proton_charge_and_flux(data_workspace, flux_workspace)

    # Compare to test data. Notice that test_data['I_samnorm'] has shape (10, 5, 5) but
    # data_workspace.extractY() has shape (25, 10). A transpose operation is necessary
    test_intensities = np.transpose(np.array(test_data["I_samnorm"]).reshape((10, 25)))
    assert data_workspace.extractY() == pytest.approx(
        test_intensities, abs=test_data["precision"]
    )


if __name__ == "__main__":
    pytest.main([__file__])
