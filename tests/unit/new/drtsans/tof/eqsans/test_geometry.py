from collections import namedtuple
import os
import pytest

r"""
Hyperlinks to Mantid algorithms
LoadInstrument <https://docs.mantidproject.org/nightly/algorithms/LoadInstrument-v1.html>
"""
from mantid.simpleapi import LoadInstrument

from drtsans.geometry import main_detector_panel
from drtsans.tof.eqsans.geometry import (
    detector_id,
    pixel_coordinates,
    sample_aperture_diameter,
    source_aperture,
    source_aperture_diameter,
    source_aperture_sample_distance,
    source_monitor_distance,
    translate_detector_by_z,
)
from drtsans.samplelogs import SampleLogs


def test_translate_detector_by_z(serve_events_workspace, reference_dir):
    # Load instrument with main panel at Z=0, then translate according to the logs
    workspace = serve_events_workspace("EQSANS_92353.nxs.h5")
    assert main_detector_panel(workspace).getPos()[-1] == pytest.approx(
        0.0, abs=1e-3
    )  # detector1 at z=0
    translate_detector_by_z(workspace)
    assert main_detector_panel(workspace).getPos()[-1] == pytest.approx(
        4.0, abs=1e-3
    )  # now at z=4.0

    # Load instrument with main panel at Z=0, then apply latest IDF which will move the main panel. Subsequent
    # application of translate_detector_by_z will have no effect
    workspace = serve_events_workspace("EQSANS_92353.nxs.h5")
    assert main_detector_panel(workspace).getPos()[-1] == pytest.approx(
        0.0, abs=1e-3
    )  # detector1 at z=0
    idf = os.path.join(reference_dir.new.eqsans, "instrument", "EQ-SANS_Definition.xml")
    LoadInstrument(workspace, FileName=idf, RewriteSpectraMap=True)
    assert main_detector_panel(workspace).getPos()[-1] == pytest.approx(
        4.0, abs=1e-3
    )  # now at z=4.0
    translate_detector_by_z(workspace)
    assert main_detector_panel(workspace).getPos()[-1] == pytest.approx(
        4.0, abs=1e-3
    )  # no effect


def test_sample_aperture_diameter(serve_events_workspace):
    ws = serve_events_workspace("EQSANS_92353.nxs.h5")
    sad = sample_aperture_diameter(ws)
    # ISSUE1887 TODO Enabled assert sad == approx(10)
    sad = SampleLogs(ws).single_value("sample_aperture_diameter")
    # ISSUE1887 TODO assert sad == approx(10)
    assert sad > 0


# Test function source_aperture() with three different aperture settings for the three
# slits [(1, 7, 4,), (4, 1, 6), (6, 3, 3)] and for two different run numbers [9999, 10000].
# Compare the diameter and Aperture-Sample-Distance (asd) from running source_aperture() against the values
# stored here
DataSourceAperture = namedtuple(
    "DataSourceAperture", "run_number vBeamSlit vBeamSlit2 vBeamSlit3 diameter asd"
)
data_source_aperture = [
    DataSourceAperture(9999, 1, 7, 4, 0.005, 4.042),
    DataSourceAperture(10000, 1, 7, 4, 0.005, 4.042),
    DataSourceAperture(9999, 4, 1, 6, 0.0, 2.966),
    DataSourceAperture(10000, 4, 1, 6, 0.0, 2.966),
    DataSourceAperture(9999, 6, 3, 3, 0.01, 2.966),
    DataSourceAperture(10000, 6, 3, 3, 0.01, 2.966),
]


@pytest.mark.parametrize("data", data_source_aperture)
@pytest.mark.parametrize(
    "generic_workspace", [{"name": "EQ-SANS", "l1": -14.122}], indirect=True
)
def test_source_aperture(generic_workspace, data):
    r"""
    Test function source_aperture for different aperture settings and run numbers.
    Use a mock EQ-SANS instrument with the moderator 14.122 meters away from the sample.
    """
    workspace = generic_workspace
    sample_logs = SampleLogs(workspace)
    sample_logs.insert("run_number", data.run_number)
    for log_key in ["vBeamSlit", "vBeamSlit2", "vBeamSlit3"]:
        data_index = data._fields.index(
            log_key
        )  # which item in object `data` stores info for this particular slit?
        diameter_index = data[data_index]  # diameter index for the particular slit
        times = [
            0.0,
            3600,
        ]  # the run started at time 0.0 and ended after one hour, here in seconds
        sample_logs.insert_time_series(log_key, times, [diameter_index, diameter_index])
    result = source_aperture(workspace)
    assert result.diameter == pytest.approx(data.diameter, abs=1.0e-05)
    assert result.distance_to_sample == pytest.approx(data.asd, abs=1.0e-05)


def test_source_aperture_diameter(serve_events_workspace):
    ws = serve_events_workspace("EQSANS_92353.nxs.h5")
    sad = source_aperture_diameter(ws)
    # ISSUE187 TODO Enable assert sad == approx(20)
    sad = SampleLogs(ws).single_value("source_aperture_diameter")
    # ISSUE187 TODO Enable assert sad == approx(20)
    assert sad > 0


@pytest.mark.parametrize("data", data_source_aperture)
@pytest.mark.parametrize(
    "generic_workspace", [{"name": "EQ-SANS", "l1": -14.122}], indirect=True
)
def test_source_aperture_sample_distance(generic_workspace, data):
    workspace = generic_workspace
    sample_logs = SampleLogs(workspace)
    sample_logs.insert("run_number", data.run_number)
    for log_key in ["vBeamSlit", "vBeamSlit2", "vBeamSlit3"]:
        data_index = data._fields.index(
            log_key
        )  # which item in object `data` stores info for this particular slit?
        diameter_index = data[data_index]  # diameter index for the particular slit
        times = [
            0.0,
            3600,
        ]  # the run started at time 0.0 and ended after one hour, here in seconds
        sample_logs.insert_time_series(log_key, times, [diameter_index, diameter_index])
    assert source_aperture_sample_distance(workspace, unit="m") == pytest.approx(
        data.asd, abs=1.0e-05
    )
    # Check the distance was inserted in the metadata, in units of mili-meters
    assert SampleLogs(workspace).source_aperture_sample_distance.value == pytest.approx(
        1000 * data.asd, abs=1.0e-05
    )


def test_source_monitor_distance(serve_events_workspace):
    ws = serve_events_workspace("EQSANS_92353.nxs.h5")
    smd = source_monitor_distance(ws, unit="m")
    assert smd == pytest.approx(10.122, abs=0.001)
    smd = SampleLogs(ws).single_value("source-monitor-distance")
    assert smd == pytest.approx(10122, abs=1)


def test_detector_id():
    pixel_coords = [
        (1, 0),  # eightpack 0, tube id 4, pixel 0
        (42, 42),  # eigtpack 5, tube id 1, pixel 42
        (126, 255),
    ]  # eigthpack 15, tube id 3, pixel 255]
    assert detector_id(pixel_coords) == [1024, 10538, 31743]
    assert [detector_id(p) for p in pixel_coords] == [1024, 10538, 31743]


def test_pixel_coordinates():
    detector_ids = [1024, 10538, 31743]
    assert pixel_coordinates(detector_ids) == [(1, 0), (42, 42), (126, 255)]
    assert [tuple(pixel_coordinates(det)) for det in detector_ids] == [
        (1, 0),
        (42, 42),
        (126, 255),
    ]


if __name__ == "__main__":
    pytest.main([__file__])
