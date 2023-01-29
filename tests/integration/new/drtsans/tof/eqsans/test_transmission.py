from math import nan
import numpy as np
import pytest
from os.path import join as pjn

r"""
Hyperlinks to Mantid algorithms
CompareWorkspaces <https://docs.mantidproject.org/nightly/algorithms/CompareWorkspaces-v1.html>
CreateWorkspace <https://docs.mantidproject.org/nightly/algorithms/CreateWorkspace-v1.html>
LoadNexus <https://docs.mantidproject.org/nightly/algorithms/LoadNexus-v1.html>
SumSpectra <https://docs.mantidproject.org/nightly/algorithms/SumSpectra-v1.html>
"""
from mantid.simpleapi import CompareWorkspaces, CreateWorkspace, LoadNexus, SumSpectra
from mantid.api import mtd

r"""
Hyperlinks to drtsans functions
amend_config, namedtuplefy, unique_workspace_dundername available at:
    <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/settings.py>
SampleLogs <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/samplelogs.py>
insert_aperture_logs <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/tof/eqsans/geometry.py>
prepare_data <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/tof/eqsans/api.py>
calculate_transmission, fit_raw_transmission available at:
    <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/tof/eqsans/transmission.py>
apply_transmission_correction <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/transmission.py>
find_beam_center, center_detector <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/beam_finder.py>
"""  # noqa: E501
from drtsans.settings import amend_config, namedtuplefy, unique_workspace_dundername
from drtsans.samplelogs import SampleLogs
from drtsans.tof.eqsans.geometry import insert_aperture_logs
from drtsans.tof.eqsans.api import prepare_data
from drtsans.tof.eqsans import (
    calculate_transmission,
    apply_transmission_correction,
    fit_raw_transmission,
    find_beam_center,
    center_detector,
)


@pytest.fixture(scope="module")
@namedtuplefy  # converts a dictionary into a namedtuple for more pythonic dereferencing of dictionary keys
def test_data_9a_part_1():
    r"""
    Data for the test that calculates the raw transmission at zero scattering angle starting with sample and
    empty-beam datasets, integrated on wavelength, over a predefined region of interest.

    Dropbox link to the test data:
        <https://www.dropbox.com/s/8ttwjfa1u0q4cyq/calculate_transmission_test.xlsx>

    Returns
    -------
    dict
    """
    return dict(
        wavelength_range=[2.0, 5.0],  # data integrated over this wavelength range
        emtpy_reference=[
            [1, 2, 3, 1, 1, 2, 2, 3, 0, 0],  # uncertainties by taking the square root
            [3, 2, 2, 4, 5, 4, 6, 3, 2, 1],
            [1, 4, 6, 9, 13, 15, 5, 8, 3, 0],
            [7, 3, 8, 19, 25, 18, 65, 12, 4, 1],
            [2, 5, 9, 28, 79, 201, 41, 16, 2, 5],
            [0, 7, 11, 23, 128, 97, 50, 17, 3, 2],
            [3, 3, 9, 20, 27, 23, 18, 7, 4, 3],
            [1, 2, 5, 9, 9, 15, 4, nan, nan, 1],
            [2, 4, 2, 4, 3, 4, 1, nan, nan, 1],
            [
                4,
                0,
                1,
                3,
                1,
                2,
                0,
                0,
                2,
                0,
            ],
        ],
        sample=[
            [4, 5, 4, 3, 1, 5, 5, 7, 3, 3],
            [4, 4, 3, 5, 7, 5, 7, 3, 2, 4],
            [4, 5, 5, 9, 12, 12, 7, 8, 5, 3],
            [9, 3, 10, 15, 20, 18, 49, 13, 4, 2],
            [3, 5, 9, 23, 60, 155, 35, 13, 3, 8],
            [2, 9, 13, 20, 96, 76, 41, 15, 6, 3],
            [3, 5, 10, 15, 22, 20, 17, 10, 5, 6],
            [1, 6, 7, 10, 8, 13, 5, nan, nan, 4],
            [2, 6, 3, 7, 6, 4, 3, nan, nan, 1],
            [3, 4, 1, 4, 5, 6, 3, 3, 3, 1],
        ],
        radius=2.0,
        radius_unit="mm",
        transmission=0.7888,
        transmission_uncertainty=0.0419,
        precision=1.0e-04,  # precision when comparing test data with drtsans calculations
    )


@pytest.mark.parametrize(
    "workspace_with_instrument",
    [dict(name="EQSANS", Nx=10, Ny=10, dx=1.0e-3, dy=1.0e-3, zc=1.0)],
    indirect=True,
)
def test_calculate_transmission_single_bin(
    test_data_9a_part_1, reference_dir, workspace_with_instrument
):
    r"""
    This test calculates the raw transmission at zero scattering angle starting with sample and empty-beam datasets,
    integrated on wavelength, over a predefined region of interest.

    This test implements Issue #175, addressing section 7.3 of the master document.
    Dropbox links to the test:
        <https://www.dropbox.com/s/1lejukntcx3g8p1/calculate_transmission_test.pdf>
        <https://www.dropbox.com/s/8ttwjfa1u0q4cyq/calculate_transmission_test.xlsx>

    Test introduces a detector array with the following properties:
        - 10 tubes.
        - 10 pixels per tube.
        - square pixels of size 1 mili-meter.
        - distance from sample of 1 meter (this is irrelevant for the test, though).

    dev - Jose Borreguero <borreguerojm@ornl.gov>
    SME - Changwoo Do <doc1@ornl.gov>

    - List of Mantid algorithms employed:
        FindCenterOfMassPosition <https://docs.mantidproject.org/nightly/algorithms/FindCenterOfMassPosition-v2.html>
    """
    data = test_data_9a_part_1  # let's cut-down on the verbosity

    # Load the empty reference into a workspace
    reference_counts = np.array(data.emtpy_reference, dtype=float).reshape((10, 10, 1))
    reference_workspace = (
        unique_workspace_dundername()
    )  # some temporary random name for the workspace
    workspace_with_instrument(
        axis_values=data.wavelength_range,
        intensities=reference_counts,
        uncertainties=np.sqrt(reference_counts),
        view="array",
        output_workspace=reference_workspace,
    )
    assert mtd[reference_workspace].readY(44)[0] == 128

    # Load the sample into a workspace
    sample_counts = np.array(data.sample, dtype=float).reshape((10, 10, 1))
    sample_workspace = (
        unique_workspace_dundername()
    )  # some temporary random name for the workspace
    workspace_with_instrument(
        axis_values=data.wavelength_range,
        intensities=sample_counts,
        uncertainties=np.sqrt(sample_counts),
        view="array",
        output_workspace=sample_workspace,
    )
    assert mtd[sample_workspace].readY(44)[0] == 96

    # Find the beam center using the empty reference, and then center both reference and sample runs
    # We pass centering options to the underlying Mantid algorithm finding the center of the beam
    # <FindCenterOfMassPosition://docs.mantidproject.org/nightly/algorithms/FindCenterOfMassPosition-v1.html>
    beam_center = find_beam_center(
        reference_workspace,
        centering_options={"BeamRadius": data.radius, "Tolerance": 0.1 * data.radius},
    )
    center_detector(reference_workspace, *beam_center[:-1])
    center_detector(sample_workspace, *beam_center[:-1])

    # Calculate raw (no fitting) transmission at zero angle using drtsans
    transmission = calculate_transmission(
        sample_workspace,
        reference_workspace,
        radius=data.radius,
        radius_unit=data.radius_unit,
        fit_function=None,
    )

    # Verify transmission and associated uncertainty
    assert transmission.readY(0)[0] == pytest.approx(
        data.transmission, abs=data.precision
    )
    assert transmission.readE(0)[0] == pytest.approx(
        data.transmission_uncertainty, abs=data.precision
    )


@pytest.fixture(scope="module")
@namedtuplefy  # converts a dictionary into a namedtuple for more pythonic dereferencing of dictionary keys
def test_data_9a_part_2():
    r"""
    Data for the test that fits a raw transmission at zero scattering angle, wavelength dependent, with a linear model.

    Dropbox link to the test data:
        <https://www.dropbox.com/s/up96plrq60jjdcg/fit_transmission_and_calc_test.xlsx>

    Returns
    -------
    dict
    """
    return dict(
        wavelength_bin_boundaries=[
            2.25,
            2.75,
            3.25,
            3.75,
            4.25,
            4.75,
            5.25,
            5.75,
            6.25,
        ],
        raw_transmissions=[0.91, 0.90, 0.86, 0.83, 0.80, 0.78, 0.75, 0.72],
        raw_uncertainties=[0.04, 0.03, 0.03, 0.04, 0.03, 0.04, 0.03, 0.03],
        fitted_transmissions=[
            0.918456,
            0.890107,
            0.861748,
            0.833394,
            0.805040,
            0.776686,
            0.748320,
            0.719978,
        ],
        fitted_uncertainties=[
            0.051776,
            0.054412,
            0.057371,
            0.060607,
            0.064076,
            0.067744,
            0.07158,
            0.075558,
        ],
        slope=(
            -0.056708,
            0.0100089,
        ),  # slope value and error for the linear fit of the raw transmission values
        intercept=(
            1.060225,
            0.045217,
        ),  # intercept value and error for the linear fit of the raw transmission values
        precision=1.0e06,  # precision when comparing test data with drtsans calculations
    )


def test_fit_transmission_and_calc(test_data_9a_part_2):
    r"""
    This test fits a raw transmission at zero scattering angle, wavelength dependent, with a linear model.

    This test implements Issue #175, addressing section 7.3 of the master document.
    Dropbox links to the test:
        <https://www.dropbox.com/s/dxg613usfg34vbx/fit_transmission_and_calc_test.pdf>
        <https://www.dropbox.com/s/up96plrq60jjdcg/fit_transmission_and_calc_test.xlsx>

    dev - Jose Borreguero <borreguerojm@ornl.gov>
    SME - Changwoo Do <doc1@ornl.gov>

    - List of Mantid algorithms employed:
        Fit <https://docs.mantidproject.org/nightly/algorithms/Fit-v1.html>
    """
    data = test_data_9a_part_2  # let's cut-down on the verbosity

    # Load the raw transmissions at zero scattering angle into a workspace.
    raw_transmission_workspace = (
        unique_workspace_dundername()
    )  # some temporary random name for the workspace
    CreateWorkspace(
        DataX=data.wavelength_bin_boundaries,
        UnitX="Wavelength",
        DataY=data.raw_transmissions,
        DataE=data.raw_uncertainties,
        OutputWorkspace=raw_transmission_workspace,
    )

    # We need to "manually" insert information about whether this run is skipped mode. This information would
    # already be present in a prepared workspace.
    sample_logs = SampleLogs(raw_transmission_workspace)
    sample_logs.insert("is_frame_skipping", 0)
    sample_logs.insert(
        "wavelength_lead_min", data.wavelength_bin_boundaries[0], unit="Angstrom"
    )
    sample_logs.insert(
        "wavelength_lead_max", data.wavelength_bin_boundaries[-1], unit="Angstrom"
    )

    # use drtsans to fit the raw transmission values
    fitted_transmission_workspace = (
        unique_workspace_dundername()
    )  # some temporary random name for the workspace
    fit_results = fit_raw_transmission(
        raw_transmission_workspace, output_workspace=fitted_transmission_workspace
    )

    # Verify fitted transmission values
    assert mtd[fitted_transmission_workspace].readY(0) == pytest.approx(
        data.fitted_transmissions, abs=data.precision
    )
    assert mtd[fitted_transmission_workspace].readE(0) == pytest.approx(
        data.fitted_uncertainties, abs=data.precision
    )

    # Verify values for the fit parameters
    parameter_table_workspace = fit_results.lead_mantid_fit.OutputParameters
    _, slope_value, slope_error = list(parameter_table_workspace.row(0).values())
    assert (slope_value, slope_error) == pytest.approx(data.slope, abs=data.precision)
    _, intercept_value, intercept_error = list(
        parameter_table_workspace.row(1).values()
    )
    assert (intercept_value, intercept_error) == pytest.approx(
        data.intercept, abs=data.precision
    )


@pytest.fixture(scope="module")
@namedtuplefy
def transmission_fixture(reference_dir):
    data_dir = pjn(reference_dir.new.eqsans, "test_transmission")
    cmp_dir = pjn(data_dir, "compare")

    def quick_compare(tentative, asset):
        r"""asset: str, name of golden standard nexus file"""
        ws = LoadNexus(
            pjn(cmp_dir, asset), OutputWorkspace=unique_workspace_dundername()
        )
        return CompareWorkspaces(tentative, ws, Tolerance=1.0e-4).Result

    a = LoadNexus(pjn(data_dir, "sample.nxs"), OutputWorkspace=unique_workspace_dundername())
    insert_aperture_logs(a)  # source and sample aperture diameters
    b = LoadNexus(pjn(data_dir, "direct_beam.nxs"), OutputWorkspace=unique_workspace_dundername())
    insert_aperture_logs(b)
    c = LoadNexus(pjn(data_dir, "sample_skip.nxs"), OutputWorkspace=unique_workspace_dundername())
    d = LoadNexus(pjn(data_dir, "direct_beam_skip.nxs"), OutputWorkspace=unique_workspace_dundername())
    for workspace in (a, c):
        sample_logs = SampleLogs(workspace)
        sample_logs.insert("low_tof_clip", 0.0, unit="ms")
        sample_logs.insert("low_tof_clip", 0.0, unit="ms")
    return dict(
        data_dir=data_dir,
        sample=a,
        reference=b,
        sample_skip=c,
        reference_skip=d,
        compare=quick_compare,
    )


def test_masked_beam_center(reference_dir, transmission_fixture):
    r"""
    (this test was written previously to the testset with the instrument team)
    Test for an exception raised when the beam centers are masked
    """
    mask = pjn(transmission_fixture.data_dir, "beam_center_masked.xml")
    with amend_config(data_dir=reference_dir.new.eqsans):
        sample_workspace = prepare_data(
            "EQSANS_88975", mask=mask, output_workspace=unique_workspace_dundername()
        )
        reference_workspace = prepare_data(
            "EQSANS_88973", mask=mask, output_workspace=unique_workspace_dundername()
        )
    with pytest.raises(RuntimeError, match=r"Transmission at zero-angle is NaN"):
        calculate_transmission(sample_workspace, reference_workspace)
    [workspace.delete() for workspace in (sample_workspace, reference_workspace)]


def test_calculate_raw_transmission(transmission_fixture):
    r"""
    (this test was written previously to the testset with the instrument team)
    """
    raw = calculate_transmission(
        transmission_fixture.sample, transmission_fixture.reference, fit_function=None
    )
    assert transmission_fixture.compare(raw, "raw_transmission.nxs")
    # big radius because detector is not centered
    raw = calculate_transmission(
        transmission_fixture.sample_skip,
        transmission_fixture.reference_skip,
        radius=50,
        fit_function=None,
    )
    assert transmission_fixture.compare(raw, "raw_transmission_skip.nxs")


def test_calculate_fitted_transmission(transmission_fixture):
    r"""
    (this test was written previously to the testset with the instrument team)
    Gold data is changed due to a bugfix on Mantid.Fit's error bar calculation
    """
    fitted_transmission_workspace = calculate_transmission(
        transmission_fixture.sample, transmission_fixture.reference
    )
    assert transmission_fixture.compare(
        fitted_transmission_workspace, "fitted_transmission_mtd6.nxs"
    )

    # big radius because detector is not centered
    fitted_transmission_workspace = calculate_transmission(
        transmission_fixture.sample_skip, transmission_fixture.reference_skip, radius=50
    )
    assert transmission_fixture.compare(
        fitted_transmission_workspace, "fitted_transmission_skip_mtd6.nxs"
    )


def test_apply_transmission(transmission_fixture):
    r"""
    (this test was written previously to the testset with the instrument team)
    """
    trans = calculate_transmission(
        transmission_fixture.sample, transmission_fixture.reference
    )
    corr = apply_transmission_correction(
        transmission_fixture.sample,
        trans,
        output_workspace=unique_workspace_dundername(),
    )
    corr = SumSpectra(corr, OutputWorkspace=corr.name())
    transmission_fixture.compare(corr, "sample_corrected.nxs")
    # big radius because detector is not centered
    trans = calculate_transmission(
        transmission_fixture.sample_skip, transmission_fixture.reference_skip, radius=50
    )
    corr = apply_transmission_correction(
        transmission_fixture.sample_skip,
        trans,
        output_workspace=unique_workspace_dundername(),
    )
    corr = SumSpectra(corr, OutputWorkspace=corr.name())
    transmission_fixture.compare(corr, "sample_corrected_skip.nxs")


if __name__ == "__main__":
    pytest.main([__file__])
