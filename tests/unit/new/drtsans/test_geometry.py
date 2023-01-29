import numpy as np
from os.path import join as path_join
import pytest

from mantid.simpleapi import (
    AddSampleLogMultiple,
    LoadEmptyInstrument,
    LoadEventNexus,
    MoveInstrumentComponent,
)
from drtsans.settings import unique_workspace_dundername
from drtsans.samplelogs import SampleLogs
from drtsans import geometry as geo


@pytest.fixture(scope="module")
def wss():
    r"""Just one workspace for each instrument"""

    # Load an EQSANS instrument and mess with the instrument components
    _eq_ws = LoadEmptyInstrument(InstrumentName="EQSANS")
    geo.sample_detector_distance(_eq_ws)
    for component, shift in (
        ("detector1", 1.3),
        ("sample-position", 0.02),
        ("moderator", 0.3),
    ):
        MoveInstrumentComponent(_eq_ws, ComponentName=component, Z=shift)

    # ssd: source-sample-distance, sdd: sample-detector-distance
    return dict(biosans=None, eqsans=dict(ws=_eq_ws, ssd=13842, sdd=1280), gpsans=None)


@pytest.mark.offline
def test_source_sample_distance(wss):
    for v in wss.values():
        if v is not None:
            assert geo.source_sample_distance(v["ws"]) == pytest.approx(
                v["ssd"], rel=0.01
            )


@pytest.mark.offline
def test_sample_detector_distance(wss):
    for v in wss.values():
        if v is not None:
            assert geo.sample_detector_distance(v["ws"]) == pytest.approx(
                v["sdd"], rel=0.01
            )


@pytest.mark.offline
def test_source_detector_distance(wss):
    for v in wss.values():
        if v is not None:
            assert geo.source_detector_distance(v["ws"]) == pytest.approx(
                v["ssd"] + v["sdd"], rel=0.01
            )


def test_detector_translation():
    r"""Ascertain sub-components are moved when main detector is moved"""
    translation = np.array([0.01, 0.1, 1.0])
    detector_name = "detector1"
    for instrument_name in ("EQ-SANS", "CG2"):
        workspace = LoadEmptyInstrument(
            InstrumentName=instrument_name,
            OutputWorkspace=unique_workspace_dundername(),
        )
        instrument = workspace.getInstrument()
        component_detector = instrument.getComponentByName(detector_name)
        component_bank = instrument.getComponentByName("bank42")
        component_detector = instrument.getDetector(42)
        initial_positions = [
            c.getPos() for c in (component_detector, component_bank, component_detector)
        ]
        MoveInstrumentComponent(
            workspace,
            ComponentName=detector_name,
            RelativePosition=True,
            **dict(zip(("X", "Y", "Z"), translation))
        )
        final_positions = [
            c.getPos() for c in (component_detector, component_bank, component_detector)
        ]
        for i, final_position in enumerate(final_positions):
            assert final_position == pytest.approx(
                np.array(initial_positions[i]) + translation, abs=1e-4
            )
        workspace.delete()


@pytest.mark.parametrize(
    "instrument, component, detmin, detmax",
    [
        ("EQ-SANS", "", 0, 49151),
        ("BIOSANS", "", 0, 44 * 8 * 256 - 1),
        ("BIOSANS", "detector1", 0, 24 * 8 * 256 - 1),
        ("BIOSANS", "wing_detector", 24 * 8 * 256, 44 * 8 * 256 - 1),
    ],
)
def test_bank_detector_ids(instrument, component, detmin, detmax):
    wksp = LoadEmptyInstrument(
        InstrumentName=instrument, OutputWorkspace=unique_workspace_dundername()
    )
    num_detectors = detmax - detmin + 1

    # None test
    detIDs = geo.bank_detector_ids(wksp, component=component, masked=None)
    assert detIDs.size == num_detectors
    assert detIDs.min() == detmin
    assert detIDs.max() == detmax

    detIDs = geo.bank_detector_ids(wksp, component=component, masked=False)
    assert detIDs.size == num_detectors
    assert detIDs.min() == detmin
    assert detIDs.max() == detmax

    detIDs = geo.bank_detector_ids(wksp, component=component, masked=True)
    assert len(detIDs) == 0


@pytest.mark.parametrize(
    "instrument, component, wksp_index_min, wksp_index_max",
    [
        ("EQ-SANS", "", 1, 49151 + 2),
        ("BIOSANS", "", 2, 44 * 8 * 256 + 2),
        ("BIOSANS", "detector1", 2, 24 * 8 * 256 + 2),
        ("BIOSANS", "wing_detector", 24 * 8 * 256 + 2, 44 * 8 * 256 + 2),
    ],
)
def test_bank_workspace_indices(instrument, component, wksp_index_min, wksp_index_max):
    wksp = LoadEmptyInstrument(
        InstrumentName=instrument, OutputWorkspace=unique_workspace_dundername()
    )

    wksp_indices = geo.bank_workspace_index_range(wksp, component)
    assert wksp_indices[0] >= 0
    assert wksp_indices[1] <= wksp.getNumberHistograms()
    assert wksp_indices[0] == wksp_index_min
    assert wksp_indices[1] == wksp_index_max


@pytest.mark.parametrize(
    "workspace_with_instrument",
    [
        {
            "instrument_geometry": "n-pack",
            "n_tubes": 2,
            "n_pixels": 2,
            "spacing": 0.0,
            "x_center": 0.0,
            "y_center": 0.0,
            "z_center": 0.0,  # detector center
            "diameter": 0.02,
            "height": 0.02,  # pixel dimensions
            "x_center": 0.0,
            "y_center": 0.0,
            "z_center": 0.0,
        }
    ],
    indirect=True,
)
def test_pixel_centers(workspace_with_instrument):
    r"""
    Pixel centers for a detector array made up of two tubes, each with two pixels.
    There's no spacing between tubes.
    Detector is 1 meter away from the sample
    The shape of a detector pixel is a cylinder of 20mm diameter and 20mm in height.
    """
    # The generated IDF file still puts pixel position at the bottome edge center
    input_workspace = unique_workspace_dundername()  # temporary workspace
    workspace_with_instrument(
        axis_values=[2.0, 2.1],
        intensities=[[1.0, 2.0], [3.0, 2.0]],
        output_workspace=input_workspace,
    )
    pixel_positions = geo.pixel_centers(input_workspace, [0, 1, 2, 3])
    expected = 1.0e-03 * np.array(
        [[10, -20, 0.0], [10, 0, 0.0], [-10, -20, 0.0], [-10, 0, 0.0]]
    )  # in meters
    assert pixel_positions == pytest.approx(expected)


def test_logged_smearing_pixel_size(workspace_with_instrument):
    workspace = workspace_with_instrument()

    # Assert default value of `None` when no smearing pixels are provided
    logged_values = geo.logged_smearing_pixel_size(workspace)
    assert list(logged_values) == [None, None]

    # Assert values are correctly retrieved when smearing pixels are provided
    values, names, units = (
        [0.0042, 0.0024],
        ["smearingPixelSizeX", "smearingPixelSizeY"],
        ["m", "m"],
    )
    AddSampleLogMultiple(
        Workspace=workspace, LogNames=names, LogValues=values, LogUnits=units
    )
    # the order of `logged_values` should be the same as that of `values`
    logged_values = geo.logged_smearing_pixel_size(workspace)
    assert logged_values == pytest.approx(logged_values)


def test_sample_aperture_diameter(serve_events_workspace, reference_dir):
    input_workspace = serve_events_workspace("EQSANS_92353.nxs.h5")
    # diameter is retrieved from log 'beamslit4', and we convert the 10mm into 0.01 meters
    assert geo.sample_aperture_diameter(input_workspace, unit="m") == pytest.approx(
        0.01, abs=0.1
    )
    # verify entry 'sample_aperture_diameter' has been added to the logs
    assert SampleLogs(input_workspace).single_value(
        "sample_aperture_diameter"
    ) == pytest.approx(10.0, abs=0.1)
    # test a run containing "sample_aperture_radius" instead of "sample_aperture_diameter"
    workspace = LoadEventNexus(
        Filename=path_join(reference_dir.new.gpsans, "geometry", "CG2_1338.nxs.h5"),
        OutputWorkspace=unique_workspace_dundername(),
        MetaDataOnly=True,
        LoadLogs=True,
    )
    assert geo.sample_aperture_diameter(workspace, unit="mm") == pytest.approx(
        14.0, abs=0.1
    )
    workspace.delete()


def test_source_aperture_diameter(reference_dir):
    # test a run containing "sample_aperture_radius" instead of "sample_aperture_diameter"
    workspace = LoadEventNexus(
        Filename=path_join(reference_dir.new.gpsans, "geometry", "CG2_1338.nxs.h5"),
        OutputWorkspace=unique_workspace_dundername(),
        MetaDataOnly=True,
        LoadLogs=True,
    )
    assert geo.source_aperture_diameter(workspace, unit="mm") == pytest.approx(
        40.0, abs=0.1
    )
    workspace.delete()


def test_translate_source_by_z(reference_dir):
    filename = path_join(reference_dir.new.gpsans, "geometry", "CG2_1338.nxs.h5")
    workspace = LoadEventNexus(
        Filename=filename,
        OutputWorkspace=unique_workspace_dundername(),
        MetaDataOnly=False,
        LoadLogs=True,
    )
    geo.translate_source_by_z(workspace)
    assert workspace.getInstrument().getComponentByName(
        "moderator"
    ).getPos().Z() == pytest.approx(-7.283, abs=0.1)


@pytest.mark.parametrize(
    "instrument, pixel_size",
    [
        ("BIOSANS", (0.00804, 0.00409)),
        ("EQSANS", (0.00804, 0.00409)),
        ("GPSANS", (0.00804, 0.00409)),
    ],
)
def test_nominal_pixel_size(instrument, pixel_size):
    workspace = LoadEmptyInstrument(
        InstrumentName=instrument, OutputWorkspace=unique_workspace_dundername()
    )
    assert geo.nominal_pixel_size(workspace) == pytest.approx(pixel_size, abs=1.0e-04)
    workspace.delete()


if __name__ == "__main__":
    pytest.main([__file__])
