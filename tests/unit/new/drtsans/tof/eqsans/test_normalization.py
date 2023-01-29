import pytest
from pytest import approx
from os.path import join as pj

r"""
Hyperlinks to Mantid algorithms
SumSpectra <https://docs.mantidproject.org/nightly/algorithms/SumSpectra-v1.html>
"""
from mantid.simpleapi import SumSpectra

r"""
Hyperlinks to drtsans functions
amend_config, unique_workspace_dundername <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/settings.py>
SampleLogs <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/samplelogs.py>
load_events <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/tof/eqsans/load.py>
prepare_monitors <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/tof/eqsans/api.py>
normalize_by_time,...load_flux_to_monitor_ratio_file <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/tof/eqsans/nomalization.py>
"""  # noqa: E501
from drtsans.settings import amend_config, unique_workspace_dundername
from drtsans.samplelogs import SampleLogs
from drtsans.tof.eqsans import (
    load_events,
    transform_to_wavelength,
    set_init_uncertainties,
    prepare_monitors,
    normalize_by_flux,
    normalize_by_proton_charge_and_flux,
    normalize_by_time,
    normalize_by_monitor,
)
from drtsans.tof.eqsans.normalization import (
    load_beam_flux_file,
    load_flux_to_monitor_ratio_file,
)


@pytest.fixture(scope="module")
def beam_flux(reference_dir):
    r"""Filepath to the flux file"""
    return pj(reference_dir.new.eqsans, "test_normalization", "beam_profile_flux.txt")


@pytest.fixture(scope="module")
def flux_to_monitor(reference_dir):
    r"""Filepath to the flux-to-monitor-ratio file"""
    return pj(
        reference_dir.new.eqsans, "test_normalization", "flux_to_monitor_ratio.nxs"
    )


@pytest.fixture(scope="module")
def data_ws(reference_dir):
    r"""Two Mantid workspaces containing intensities versus wavelength for each of the EQSANS pixel-detectors.
    The two workspaces correspond to runs 92353 and 88565."""
    ws = dict()
    with amend_config(data_dir=reference_dir.new.eqsans):
        for run in ("92353", "88565"):
            w = load_events(
                "EQSANS_{}.nxs.h5".format(run),
                output_workspace=unique_workspace_dundername(),
            )
            ws[run], bands = transform_to_wavelength(w, output_workspace=w.name())
            ws[run] = set_init_uncertainties(ws[run])
    return ws


@pytest.fixture(scope="module")
def monitor_ws(reference_dir):
    r"""Single-spectrum Workspace containing wavelength-dependent monitor counts for run 88565."""
    ws = dict()
    with amend_config(data_dir=reference_dir.new.eqsans):
        for run in ("88565",):
            ws[run] = prepare_monitors(run)
    return ws


def test_load_beam_flux_file(beam_flux, data_ws):
    r"""
    (This test was introduced prior to the testset with the instrument team)
    Loads flux file
    /SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/sns/eqsans/test_normalization/beam_profile_flux.txt
    into a Mantid workspace.
    """
    flux_workspace = load_beam_flux_file(beam_flux, data_workspace=data_ws["92353"])
    assert flux_workspace.readY(0)[0] == approx(959270.0, abs=1.0)
    assert max(flux_workspace.readY(0)) == approx(966276.0, abs=1.0)
    assert flux_workspace.dataX(0) == approx(data_ws["92353"].dataX(0))


def test_normalize_by_proton_charge_and_flux(beam_flux, data_ws):
    r"""
    (This test was introduced prior to the testset with the instrument team)

    """
    data_workspace = data_ws["92353"]  # intensities versus wavelength for run 92353
    # Load flux file
    # /SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/sns/eqsans/test_normalization/beam_profile_flux.txt
    # into a workspace
    flux_workspace = load_beam_flux_file(beam_flux, data_workspace=data_workspace)

    # Use drtsans normalizing function
    normalized_data_workspace = normalize_by_proton_charge_and_flux(
        data_workspace, flux_workspace, output_workspace=unique_workspace_dundername()
    )

    # We run a simplified comparison. We merge all spectra of the individual pixel-detectors onto a single spectrum
    normalized_total_intensities = SumSpectra(
        normalized_data_workspace, OutputWorkspace=unique_workspace_dundername()
    ).dataY(0)
    unnormalized_total_intensities = SumSpectra(
        data_workspace, OutputWorkspace=unique_workspace_dundername()
    ).dataY(0)

    # Manually normalize the unnormalized_total_intensities and compare to the result from using drtsans
    # normalizing function
    good_proton_charge = SampleLogs(data_workspace).getProtonCharge()
    manual_normalized_intensities = unnormalized_total_intensities / (
        flux_workspace.readY(0) * good_proton_charge
    )

    # compare the two spectra don't deviate more than 1%.
    assert normalized_total_intensities == approx(
        manual_normalized_intensities, rel=0.01
    )


def test_load_flux_to_monitor_ratio_file(flux_to_monitor, data_ws):
    r"""
    (This test was introduced prior to the testset with the instrument team)
    Loads flux-to-monitor-ratio file
    /SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/sns/eqsans/test_normalization/flux_to_monitor_ratio.nxs
    onto a mantid workspace
    """
    # Passing just the file to function load_flux_to_monitor_ratio_file will result in a workspace with the
    # wavelength binning as in the file.
    flux_to_monitor_workspace = load_flux_to_monitor_ratio_file(flux_to_monitor)
    # check that the workspace is a histogram (the number of wavelength boundaries is the number of ratios plus one)
    assert len(flux_to_monitor_workspace.dataX(0)) == 1 + len(
        flux_to_monitor_workspace.dataY(0)
    )
    # check the number of wavelength bin boundaries is that of the input file.
    assert len(flux_to_monitor_workspace.dataX(0)) == 48664

    # Passing the file and a reference workspace to function load_flux_to_monitor_ratio_file will result
    # in a workspace with the wavelength binning as in the reference workspace.
    data_workspace = data_ws["88565"]  # our reference workspace
    flux_to_monitor_workspace = load_flux_to_monitor_ratio_file(
        flux_to_monitor, data_workspace=data_workspace
    )
    # Check the wavelength bin boundaries are those of the reference workspace.
    assert flux_to_monitor_workspace.dataX(0) == approx(
        data_workspace.dataX(0), abs=1e-3
    )
    assert max(flux_to_monitor_workspace.dataY(0)) == approx(
        0.569, abs=1e-3
    )  # a simple check


def test_normalize_by_monitor(flux_to_monitor, data_ws, monitor_ws):
    r"""
    (This test was introduced prior to the testset with the instrument team)
    """
    # First we try normalization in frame-skipping mode, which should raise an exception
    data_workspace, monitor_workspace = data_ws["92353"], monitor_ws["88565"]
    with pytest.raises(ValueError, match="not possible in frame-skipping"):
        # the below statement will raise an exception of class ValueError with an error message that
        # should contain the above "match" string
        data_workspace_normalized = normalize_by_monitor(
            data_workspace,
            flux_to_monitor,
            monitor_workspace,
            output_workspace=unique_workspace_dundername(),
        )
    # Second we try normalization if non-skipping mode
    data_workspace, monitor_workspace = (
        data_ws["88565"],
        monitor_ws["88565"],
    )  # data and monitor for run 88565
    # Use drtsans function to normalize by monitor counts
    data_workspace_normalized = normalize_by_monitor(
        data_workspace,
        flux_to_monitor,
        monitor_workspace,
        output_workspace=unique_workspace_dundername(),
    )

    # Simplified test by checking the intensity integrated over all pixel detectors and over all wavelength bins
    # after normalization
    # First we add the spectrum of all pixels into a single pixel
    data_workspace_normalized = SumSpectra(
        data_workspace_normalized, OutputWorkspace=data_workspace_normalized.name()
    )
    # Second we integrate over all wavelength bins and check the value  will not change as the code in the
    # repository evolves
    assert sum(data_workspace_normalized.dataY(0)) == approx(0.621, abs=1e-03)

    data_workspace_normalized.delete()  # clean-up from memory of temporary workspaces


def test_normalize_by_time(data_ws):
    r"""
    (This test was introduced prior to the testset with the instrument team)
    """
    data_workspace = data_ws["92353"]  # intensities versus wavelength for run 92353

    # use drtsans normalizing function
    data_workspace_normalized = normalize_by_time(
        data_workspace, output_workspace=unique_workspace_dundername()
    )

    # check we looked log entry 'duration' in order to find out the time duration of the run
    assert (
        SampleLogs(data_workspace_normalized).normalizing_duration.value == "duration"
    )
    run_duration = SampleLogs(
        data_workspace_normalized
    ).duration.value  # run duration, in seconds
    assert run_duration == pytest.approx(101.8, abs=0.1)

    # Simplified test by checking the intensity integrated over all pixel detectors and over all wavelength bins
    # after normalization
    # First we add the spectrum of all pixels into a single pixel
    data_workspace_normalized = SumSpectra(
        data_workspace_normalized, OutputWorkspace=data_workspace_normalized.name()
    )
    # Second we integrate over all wavelength bins and check the value will not change as the code in the repository
    # evolves
    assert sum(data_workspace_normalized.dataY(0)) == approx(2560.5, abs=1.0)

    data_workspace_normalized.delete()  # clean-up from memory of temporary workspaces


def test_normalize_by_flux(beam_flux, flux_to_monitor, data_ws, monitor_ws):
    r"""
    (This test was introduced prior to the testset with the instrument team)
    Function normalize_by_flux is a front to the three time of normalization we can carry out.
    """
    #
    # First we normalize by flux and proton charge with method='proton charge'
    #
    data_workspace = data_ws["92353"]
    data_workspace_normalized = normalize_by_flux(
        data_workspace,
        beam_flux,
        method="proton charge",
        output_workspace=unique_workspace_dundername(),
    )
    # we carry a simplified test whereby we will sum all pixel-detector spectra into a single spectrum
    summed_normalized = SumSpectra(
        data_workspace_normalized, OutputWorkspace=unique_workspace_dundername()
    )
    summed_normalized_intensities = summed_normalized.readY(
        0
    )  # there's only one spectrum, that with index 0

    # Compare the output of calling function "normalize_by_flux" to a "manual" normalization by carrying out the
    # individual normalizing steps one after the other.
    flux_workspace = load_beam_flux_file(
        beam_flux, data_workspace=data_workspace
    )  # first load the flux file
    proton_charge = SampleLogs(
        data_workspace
    ).getProtonCharge()  # find the proton charge
    summed = SumSpectra(data_workspace, OutputWorkspace=unique_workspace_dundername())
    manual_summed_normalized_intensities = summed.readY(0) / (
        flux_workspace.readY(0) * proton_charge
    )

    # compare now output of calling function "normalize_by_flux" to the "manual" normalization
    assert summed_normalized_intensities == pytest.approx(
        manual_summed_normalized_intensities, rel=0.001
    )

    [
        ws.delete()
        for ws in [data_workspace_normalized, flux_workspace, summed, summed_normalized]
    ]  # clean-up

    #
    # Second we normalize by monitor and flux-to-monitor ratio with method='monitor'
    #
    data_workspace, monitor_workspace = data_ws["88565"], monitor_ws["88565"]
    data_workspace_normalized = normalize_by_flux(
        data_workspace,
        flux_to_monitor,
        method="monitor",
        monitor_workspace=monitor_workspace,
        output_workspace=unique_workspace_dundername(),
    )
    # we carry a simplified test whereby we will sum all pixel-detector spectra into a single spectrum
    summed_normalized = SumSpectra(
        data_workspace_normalized, OutputWorkspace=unique_workspace_dundername()
    )
    # then we integrate this single spectrum over all wavelengths
    total_normalized_intensity = sum(summed_normalized.readY(0))
    # here we just check that the result will not change as the code in the repository evolves
    assert total_normalized_intensity == approx(0.621, abs=1e-3)

    [ws.delete() for ws in [data_workspace_normalized, summed_normalized]]  # clean-up

    #
    # Third we normalize by run duration with method='time'
    #
    data_workspace = data_ws["92353"]  # intensities versus wavelength for run 92353
    # call normalization by time using the log entry 'duration' when searching for the duration of the run
    data_workspace_normalized = normalize_by_flux(
        data_workspace,
        "duration",
        method="time",
        output_workspace=unique_workspace_dundername(),
    )

    # check we looked log entry 'duration' in order to find out the time duration of the run
    assert (
        SampleLogs(data_workspace_normalized).normalizing_duration.value == "duration"
    )
    run_duration = SampleLogs(
        data_workspace_normalized
    ).duration.value  # run duration, in seconds
    assert run_duration == pytest.approx(101.8, abs=0.1)

    # Simplified test by checking the intensity integrated over all pixel detectors and over all wavelength bins
    # after normalization
    # First we add the spectrum of all pixels into a single pixel
    data_workspace_normalized = SumSpectra(
        data_workspace_normalized, OutputWorkspace=data_workspace_normalized.name()
    )
    # Second we integrate over all wavelength bins and check the value will not change as the code in the repository
    # evolves
    assert sum(data_workspace_normalized.dataY(0)) == approx(2560, abs=1.0)

    data_workspace_normalized.delete()  # clean-up from memory of temporary workspaces


if __name__ == "__main__":
    pytest.main([__file__])
