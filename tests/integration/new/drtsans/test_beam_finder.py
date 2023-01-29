import pytest
from pytest import approx
import re
from drtsans.beam_finder import center_detector, find_beam_center, fbc_options_json
from drtsans.sensitivity import apply_sensitivity_correction

# https://docs.mantidproject.org/nightly/algorithms/LoadEmptyInstrument-v1.html
# https://docs.mantidproject.org/nightly/algorithms/MaskDetectors-v1.html
# <https://docs.mantidproject.org/nightly/algorithms/CreateWorkspace-v1.html>
# https://docs.mantidproject.org/nightly/algorithms/LoadInstrument-v1.html
from mantid.simpleapi import (
    LoadEmptyInstrument,
    MaskDetectors,
    CreateWorkspace,
    DeleteWorkspace,
    LoadInstrument,
)
import numpy as np

# Note for testing beam center: The FindCenterOfMassPosition algorithm
# needs some pixels outside the given examples, to be able to perform
# the iterative approach. So, for a 2x2 example, we need a 4x4 instrument
# to pad with zeros on each side


@pytest.mark.parametrize(
    "generic_workspace",
    [
        {
            "name": "GENERICSANS",
            "Nx": 4,
            "Ny": 4,
            "dx": 1.0,
            "dy": 1.0,
            "xc": 1.5,
            "yc": 1.5,
            "axis_values": [1.0, 2.0] * 16,
            "intensities": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                5.0,
                3.0,
                0.0,
                0.0,
                11.0,
                7.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "uncertainties": [1.0] * 16,
        }
    ],
    indirect=True,
)
def test_beam_finder_trivial(generic_workspace):
    r"""
    Testing section 3.1 in the master document
    Find beam center of a simple 2 x 2 detector
    Functions to test: drtsans.tof.eqsans.beam_finder.find_beam_center
    Underlying Mantid algorithms:
        FindCenterOfMassPosition https://docs.mantidproject.org/nightly/algorithms/FindCenterOfMassPosition-v2.html
    dev - Andrei Savici <saviciat@ornl.gov>
    SME - Venky Pingali <pingalis@ornl.gov>
    """
    ws = generic_workspace  # simple name to make the rest easier

    inst = ws.getInstrument()
    assert inst.getDetector(5).getPos() == approx([2, 1, 5], abs=1e-5)
    assert inst.getDetector(6).getPos() == approx([2, 2, 5], abs=1e-5)
    assert inst.getDetector(9).getPos() == approx([1, 1, 5], abs=1e-5)
    assert inst.getDetector(10).getPos() == approx([1, 2, 5], abs=1e-5)
    x_cen, y_cen, _ = find_beam_center(ws)
    assert x_cen == approx(1.328293, abs=1e-5)
    assert y_cen == approx(1.421136, abs=1e-5)

    DeleteWorkspace(ws)


@pytest.mark.parametrize(
    "generic_workspace",
    [
        {
            "intensities": [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 4, 2, 1, 3, 0, 2, 7, 1, 3, 1, 0],
                [0, 0, 4, 2, 3, 7, 5, 3, 4, 2, 2, 0],
                [0, 1, 2, 5, 9, 11, 9, 8, 6, 2, 3, 0],
                [0, 3, 4, 9, 20, 23, 28, 19, 9, 4, 1, 0],
                [0, 1, 3, 9, 27, 128, 79, 25, 13, 5, 1, 0],
                [0, 2, 4, 15, 23, 97, 201, 18, 15, 4, 2, 0],
                [0, 0, 1, 4, 18, 50, 41, 65, 5, 6, 2, 0],
                [0, 0, np.nan, np.nan, 7, 17, 16, 12, 8, 3, 3, 0],
                [0, 2, np.nan, np.nan, 4, 3, 2, 4, 3, 2, 0, 0],
                [0, 0, 1, 1, 3, 2, 5, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            "axis_values": [6.0, 7.0],
            "name": "BIOSANS",
            "dx": 1,
            "dy": 1,
            "xc": 5.5,
            "yc": 5.5,
        }
    ],
    indirect=True,
)
def test_beam_finder_larger_workspace(generic_workspace):
    r"""
    Testing section 3.1 in the master document
    Find beam center of a larger 10(+2) x 10(+2) detector
    Functions to test: drtsans.tof.eqsans.beam_finder.find_beam_center

    Underlying Mantid algorithms:
        FindCenterOfMassPosition https://docs.mantidproject.org/nightly/algorithms/FindCenterOfMassPosition-v2.html
    dev - Andrei Savici <saviciat@ornl.gov>
    SME - William Heller <hellerwt@ornl.gov>
    """
    ws = generic_workspace
    # masking
    intensities = ws.extractY()
    mask = np.where(np.isnan(intensities))[0].tolist()
    MaskDetectors(Workspace=ws, DetectorList=mask)
    # test masking
    inst = ws.getInstrument()
    assert inst.getDetector(mask[0]).getPos() == approx([3, 2, 5], abs=1e-5)
    assert inst.getDetector(mask[1]).getPos() == approx([3, 3, 5], abs=1e-5)
    assert inst.getDetector(mask[2]).getPos() == approx([2, 2, 5], abs=1e-5)
    assert inst.getDetector(mask[3]).getPos() == approx([2, 3, 5], abs=1e-5)
    spec = ws.spectrumInfo()
    for i in mask:
        assert spec.isMasked(i)
    # test functions
    x_cen, y_cen, _ = find_beam_center(ws)
    assert x_cen == approx(5.594458, abs=1e-5)
    assert y_cen == approx(6.098827, abs=1e-5)
    #
    DeleteWorkspace(ws)


def test_center_detector():
    r"""Testing moving detector using EQSANS instrument

    Functions to test: drtsans.tof.eqsans.beam_finder.center_detector
    Underlying Mantid algorithms:
        MoveInstrumentComponent https://docs.mantidproject.org/nightly/algorithms/MoveInstrumentComponent-v1.html

    dev - Andrei Savici <saviciat@ornl.gov>
    SME - William Heller <hellerwt@ornl.gov>
    """
    # look at the original instrument
    w_eqsans = LoadEmptyInstrument(InstrumentName="EQ-SANS")
    inst = w_eqsans.getInstrument()
    assert inst.getDetector(0).getPos() == approx(
        [0.524185, -0.520957, -0.0316256], abs=1e-5
    )
    assert inst.getDetector(49151).getPos() == approx(
        [-0.525015, 0.520957, -0.0234656], abs=1e-5
    )

    # move detector
    xcenter = 0.5
    ycenter = 0.7
    center_detector(w_eqsans, center_x=xcenter, center_y=ycenter)
    inst = w_eqsans.getInstrument()
    assert inst.getDetector(0).getPos() == approx(
        [0.024185, -1.220957, -0.0316256], abs=1e-5
    )
    assert inst.getDetector(49151).getPos() == approx(
        [-1.025015, -0.179043, -0.0234656], abs=1e-5
    )

    # cleanup
    DeleteWorkspace(w_eqsans)


# Detector pixel geometry (x_center, y_center, z-center, height and width) are defined outside of the function
# (test_find_beam_center_arbitrary_assembly(arbitrary_assembly_IDF)) as they are needed to constract the
# assembly of the detector which we enter the test  function as argument to the function.
# Thus the data must be defined outside the test function.
# If we need to write the code that constructs the detector geometry inside the test function,
# then we would have to rewrite the code every time we needed an arbitrary assembly of detectors.
# We are following one of the basic principles of software engineering, which is "reuse reuse reuse".
# By writing encapsulating the code that constructs the detector geometry in a fixture,
# we write the code once and then can call this fixture as many times as we want.

# x_center of the detector given in
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/264/beam_center_testR1.xlsx
# with additional padding in the boundary


x = np.array(
    [
        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.10, 1.20, 1.30, 1.40],
        [1.5, 5.0, 9.9, 15.1, 20.1, 24.9, 30.0, 35.0, 39.9, 40],
        [1.6, 5.1, 9.9, 15.2, 20.2, 25.0, 30.0, 35.0, 39.9, 40.1],
        [1.7, 5.0, 10.0, 15.0, 20.1, 24.9, 30.1, 35.0, 39.9, 40.2],
        [1.8, 4.9, 10.1, 14.9, 20.0, 24.8, 30.2, 35.1, 40.0, 40.3],
        [1.9, 5.0, 10.0, 14.9, 20.1, 24.9, 30.1, 34.9, 39.8, 40.4],
        [2.0, 5.0, 10.1, 14.8, 20.1, 24.9, 30.2, 34.9, 39.8, 40.5],
        [2.1, 4.8, 9.8, 15.0, 19.9, 24.7, 29.9, 35.0, 39.9, 40.6],
        [2.2, 4.9, 10.2, 15.0, 20.0, 24.8, 30.3, 35.3, 40.2, 40.7],
        [2.3, 2.4, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 40.8],
    ]
)

# y_center of the detector
y = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 40.2, 40.7, 40.2, 40.0, 39.8, 39.8, 39.9, 40.3, 0],
        [0, 35.0, 35.1, 34.9, 35.1, 34.8, 35.0, 35.0, 35.2, 0],
        [0, 30.1, 30.2, 30.0, 30.2, 29.9, 30.1, 30.1, 30.3, 0],
        [0, 24.9, 24.6, 24.7, 25.0, 24.9, 25.0, 25.1, 24.9, 0],
        [0, 20.1, 19.8, 19.9, 20.2, 20.1, 20.2, 20.3, 20.1, 0],
        [0, 15.0, 15.1, 14.9, 15.1, 14.8, 15.0, 15.0, 15.2, 0],
        [0, 10.1, 10.2, 10.0, 9.9, 9.9, 9.8, 10.0, 10.0, 0],
        [0, 5.0, 4.7, 4.8, 5.1, 5.0, 5.1, 5.2, 5.0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)

# z_center of the detector (though the z_center was not specified by the instrument scientist,
# however, to define an instrument in Mantid, an arbitrary z center of 10 mm is defined)
z = np.full((10, 10), 10)

# pixel center of the detector with x,y,z coordinates
pixel_centers = np.vstack((x * 0.001, y * 0.001, z * 0.001)).reshape(3, -1).T.tolist()

# height of the detector pixels
height = [
    5.20,
    5.60,
    5.30,
    4.90,
    5.00,
    4.80,
    4.90,
    5.10,
    5.10,
    5.10,
    5.10,
    5.20,
    5.60,
    5.30,
    4.90,
    5.00,
    4.80,
    4.90,
    5.10,
    5.10,
    5.10,
    4.90,
    4.90,
    4.90,
    4.90,
    4.90,
    4.90,
    4.90,
    4.90,
    5.10,
    5.10,
    5.20,
    5.60,
    5.30,
    5.20,
    5.00,
    5.10,
    5.00,
    5.40,
    5.10,
    5.10,
    4.80,
    4.80,
    4.80,
    4.80,
    4.80,
    4.80,
    4.80,
    4.80,
    5.10,
    5.10,
    5.10,
    4.70,
    5.00,
    5.10,
    5.30,
    5.20,
    5.30,
    4.90,
    5.10,
    5.10,
    4.90,
    4.90,
    4.90,
    5.20,
    4.90,
    5.20,
    5.00,
    5.20,
    0,
    0,
    5.10,
    5.50,
    5.20,
    4.80,
    4.90,
    4.70,
    4.80,
    5.00,
    5.10,
    5.10,
    5.00,
    4.70,
    4.80,
    5.10,
    5.00,
    5.10,
    5.20,
    5.00,
    5.10,
    5.10,
    5.10,
    5.10,
    5.10,
    5.10,
    5.10,
    5.10,
    5.10,
    5.10,
    5.10,
]

# width of the detector pixels
width = [
    5.00,
    5.00,
    5.00,
    5.00,
    5.00,
    5.00,
    5.00,
    5.00,
    5.00,
    5.00,
    5.00,
    5.00,
    4.90,
    5.20,
    5.00,
    4.80,
    5.10,
    5.00,
    4.90,
    4.9,
    4.9,
    5.10,
    4.80,
    5.30,
    5.00,
    4.80,
    5.00,
    5.00,
    4.90,
    4.9,
    4.9,
    5.00,
    5.00,
    5.00,
    5.10,
    4.80,
    5.20,
    4.90,
    4.90,
    4.9,
    4.9,
    4.90,
    5.20,
    4.80,
    5.10,
    4.80,
    5.40,
    4.90,
    4.90,
    4.9,
    4.9,
    5.00,
    5.00,
    4.90,
    5.20,
    4.80,
    5.20,
    4.80,
    4.90,
    5.00,
    5.00,
    5.00,
    5.10,
    4.70,
    5.30,
    4.80,
    5.30,
    4.70,
    4.90,
    4.9,
    4.9,
    4.80,
    5.00,
    5.20,
    4.90,
    4.80,
    5.20,
    5.10,
    4.90,
    4.9,
    4.9,
    4.90,
    5.30,
    4.80,
    5.00,
    4.80,
    5.50,
    5.00,
    4.90,
    5.00,
    5.00,
    5.00,
    5.00,
    5.00,
    5.00,
    5.00,
    5.00,
    5.00,
    5.00,
    5.00,
]


@pytest.mark.parametrize(
    "arbitrary_assembly_IDF",
    [{"width": width, "height": height, "pixel_centers": pixel_centers}],
    indirect=True,
)
def test_find_beam_center_arbitrary_assembly(arbitrary_assembly_IDF):
    r"""Testing finding the beam center using EQSANS instrument for arbitrary assembly of pixels

    Underlying Mantid algorithms:
        MoveInstrumentComponent https://docs.mantidproject.org/nightly/algorithms/MoveInstrumentComponent-v1.html

    dev - Fahima Islam <islamff@ornl.gov>
    SME - William Heller <hellerwt@ornl.gov>
    """
    axis_values = np.array([6.9, 7.1])
    intensities = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 50, 34, 34, 45, 51, 23, 31, 34, 0],
            [0, 34, 45, 23, 43, 47, 41, 34, 36, 0],
            [0, 64, 56, 69, 98, 101, 70, 45, 45, 0],
            [0, 43, 70, 110, 210, 225, 95, 51, 43, 0],
            [0, 41, 75, 97, 215, 235, 105, 38, 41, 0],
            [0, 57, 61, 71, 103, 108, 75, 39, 38, 0],
            [0, 54, 34, 44, 56, 55, 43, 42, 54, 0],
            [0, 49, 46, 34, 53, 37, 45, 45, 43, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    uncertainities = np.sqrt(intensities)

    given_mask = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, np.nan, np.nan, 1, 0],
            [0, 1, 1, 1, 1, 1, np.nan, np.nan, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    sensitivity = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1.03, 0.98, 1.02, 0.98, 1.03, 1.02, 1.02, 1.03, 0],
            [0, 1.01, 1.00, 1.00, 0.99, 1.00, 1.03, 0.99, 1.01, 0],
            [0, 0.98, 0.99, 0.97, 1.03, 1.01, 0.99, 0.97, 1.02, 0],
            [0, 1.03, 1.02, 1.02, 1.02, 1.02, 1.02, 1.01, 1.03, 0],
            [0, 1.00, 1.00, 1.00, 1.01, 1.00, 1.03, 0.99, 1.02, 0],
            [0, 1.02, 0.97, 0.98, 1.01, 0.99, 0.98, 0.98, 1.00, 0],
            [0, 1.00, 0.98, 1.03, 0.97, 1.03, 1.02, 0.98, 1.00, 0],
            [0, 1.01, 1.02, 1.00, 1.03, 1.03, 1.00, 0.99, 1.01, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    uncertainities_sensitivity = 0.01 * sensitivity

    # creating the workspace with counts
    workspace = CreateWorkspace(
        DataX=axis_values,
        DataY=intensities,
        DataE=uncertainities,
        Nspec=100,
        UnitX="wavelength",
        OutputWorkspace="count",
    )
    instrument_name = re.search(
        r'instrument name="([A-Za-z0-9_-]+)"', arbitrary_assembly_IDF
    ).groups()[0]
    LoadInstrument(
        Workspace=workspace,
        InstrumentXML=arbitrary_assembly_IDF,
        RewriteSpectraMap=True,
        InstrumentName=instrument_name,
    )

    # Applying masking
    mask = np.where(np.isnan(given_mask))[0].tolist()
    MaskDetectors(Workspace=workspace, DetectorList=mask)

    # creating the workspace with sensitivity
    workspace_sensitivity = CreateWorkspace(
        DataX=axis_values,
        DataY=sensitivity,
        DataE=uncertainities_sensitivity,
        Nspec=100,
        UnitX="wavelength",
        OutputWorkspace="sensitivity",
    )

    LoadInstrument(
        Workspace=workspace_sensitivity,
        InstrumentXML=arbitrary_assembly_IDF,
        RewriteSpectraMap=True,
        InstrumentName=instrument_name,
    )

    # Applying sensitivity correction
    sensitivity_corrected_counts = apply_sensitivity_correction(
        input_workspace=workspace,
        sensitivity_workspace=workspace_sensitivity,
        output_workspace="sensitivity_corrected",
    )

    # Finding the beam center
    x_cen, y_cen, _ = find_beam_center(
        sensitivity_corrected_counts, solid_angle_method="GenericShape"
    )
    assert x_cen * 1000 == approx(27.41, abs=0.9)
    assert y_cen * 1000 == approx(22.5, abs=0.9)

    # cleanup
    DeleteWorkspace(workspace)
    DeleteWorkspace(workspace_sensitivity)
    DeleteWorkspace(sensitivity_corrected_counts)


def test_fbc_options_json():
    reduction_input = {"beamCenter": {"method": "center_of_mass"}}
    fbc_options = fbc_options_json(reduction_input)
    assert len(fbc_options) == 1
    assert fbc_options["method"] == "center_of_mass"
    reduction_input = {
        "beamCenter": {
            "method": "center_of_mass",
            "com_centering_options": {"CenterX": 0.0},
        }
    }
    fbc_options = fbc_options_json(reduction_input)
    assert len(fbc_options) == 2
    assert fbc_options["method"] == "center_of_mass"
    assert fbc_options["centering_options"] == {"CenterX": 0.0}
    reduction_input = {"beamCenter": {"method": "gaussian"}}
    fbc_options = fbc_options_json(reduction_input)
    assert len(fbc_options) == 1
    assert fbc_options["method"] == "gaussian"
    reduction_input = {
        "beamCenter": {
            "method": "gaussian",
            "gaussian_centering_options": {"CenterX": {"val": 0.0}},
        }
    }
    fbc_options = fbc_options_json(reduction_input)
    assert len(fbc_options) == 2
    assert fbc_options["method"] == "gaussian"
    assert fbc_options["centering_options"] == {"CenterX": {"val": 0.0}}


if __name__ == "__main__":
    pytest.main([__file__])
