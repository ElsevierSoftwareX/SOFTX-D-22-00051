import numpy as np
from os.path import join
import pytest
from mantid.simpleapi import (
    CompareWorkspaces,
    Load,
    LoadEmptyInstrument,
    MoveInstrumentComponent,
)
from drtsans import calculate_solid_angle, solid_angle_correction


def test_solid_angle(reference_dir):
    # Load empty instrument
    wsInput = LoadEmptyInstrument(InstrumentName="eqsans")

    # Move the detector 5m away from the sample
    MoveInstrumentComponent(
        Workspace=wsInput, ComponentName="detector1", RelativePosition="0", Z="5"
    )

    # Apply solid angle correction
    wsOutput = solid_angle_correction(wsInput, detector_type="VerticalTube")

    # Let's do some validation
    assert wsOutput.getNumberHistograms(), 49153
    reference_workspace = Load(
        Filename=join(reference_dir.new.eqsans, "test_solid_angle.nxs")
    )
    assert CompareWorkspaces(wsOutput, reference_workspace)


def test_solid_angle_optional_output(reference_dir):
    # Load empty instrument
    wsInput = LoadEmptyInstrument(InstrumentName="eqsans")

    # Move the detector 5m away from the sample
    MoveInstrumentComponent(
        Workspace=wsInput, ComponentName="detector1", RelativePosition="0", Z="5"
    )

    # Apply solid angle correction
    wsOutput = solid_angle_correction(
        wsInput, detector_type="VerticalTube", output_workspace="wsOutput"
    )

    # Let's do some validation
    assert wsOutput.getNumberHistograms(), 49153
    reference_workspace = Load(
        Filename=join(reference_dir.new.eqsans, "test_solid_angle.nxs")
    )
    assert CompareWorkspaces(wsOutput, reference_workspace)


def test_solid_angle_input_output(reference_dir):
    # Load empty instrument
    wsInput = LoadEmptyInstrument(InstrumentName="eqsans")

    # Move the detector 5m away from the sample
    MoveInstrumentComponent(
        Workspace=wsInput, ComponentName="detector1", RelativePosition="0", Z="5"
    )

    # Apply solid angle correction
    wsInput = solid_angle_correction(
        wsInput, detector_type="VerticalTube", output_workspace="wsInput"
    )

    # Let's do some validation
    assert wsInput.getNumberHistograms(), 49153
    reference_workspace = Load(
        Filename=join(reference_dir.new.eqsans, "test_solid_angle.nxs")
    )
    assert CompareWorkspaces(wsInput, reference_workspace)


@pytest.mark.parametrize(
    "instrument, num_monitor",
    [("BIOSANS", 2), ("GPSANS", 2), ("EQSANS", 1)],
    ids=["BIOSANS", "GPSANS", "EQSANS"],
)
def test_solid_angle_calculation(instrument, num_monitor):
    ws = LoadEmptyInstrument(InstrumentName=instrument)
    ws = calculate_solid_angle(ws)

    values = ws.extractY()
    # monitors have zero solid angle
    np.testing.assert_equal(values[:1], 0.0)
    # everything other than monitors should have a positive solid angle
    assert np.all(values[num_monitor:] > 0.0)

    # cleanup
    ws.delete()


if __name__ == "__main__":
    pytest.main([__file__])
