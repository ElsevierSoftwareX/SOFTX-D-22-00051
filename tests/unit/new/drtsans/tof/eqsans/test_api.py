import pytest
import os
from collections import namedtuple
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from mantid.simpleapi import CreateWorkspace, mtd

from drtsans.tof.eqsans import reduction_parameters
from drtsans.tof.eqsans.api import (
    load_all_files,
    prepare_data_workspaces,
    pre_process_single_configuration,
)
from drtsans.samplelogs import SampleLogs
from drtsans.settings import unique_workspace_dundername as uwd


ws_mon_pair = namedtuple("ws_mon_pair", ["data", "monitor"])


@pytest.mark.skipif(
    not os.path.exists("/SNS/EQSANS/IPTS-22747/nexus/EQSANS_105428.nxs.h5"),
    reason="Required data is not available",
)
def test_load_all_files_simple():
    specs = {
        "iptsNumber": 22747,
        "sample": {"runNumber": 105428, "thickness": 1.0},
        "beamCenter": {"runNumber": 105428},
        "outputFileName": "test",
        "configuration": {
            "outputDir": "/tmp",
            "useDefaultMask": False,
            "sampleOffset": "0",
            "LogQBinsPerDecade": 10,
            "normalization": "Total charge",
            "absoluteScaleMethod": "standard",
            "StandardAbsoluteScale": "5.18325",
        },
    }
    reduction_input = reduction_parameters(specs, instrument_name="EQSANS")
    loaded = load_all_files(reduction_input)

    assert loaded.sample is not None
    assert len(loaded.sample) == 1
    history = loaded.sample[0].data.getHistory()

    assert history.size() == 11
    assert history.getAlgorithm(0).name() == "LoadEventNexus"
    assert (
        history.getAlgorithm(0).getProperty("Filename").value
        == "/SNS/EQSANS/IPTS-22747/nexus/EQSANS_105428.nxs.h5"
    )
    assert history.getAlgorithm(2).name() == "MoveInstrumentComponent"
    assert history.getAlgorithm(3).name() == "ChangeBinOffset"
    assert history.getAlgorithm(4).name() == "SetInstrumentParameter"
    # assert history.getAlgorithm(4).name() == "ModeratorTZero"
    assert history.getAlgorithm(6).name() == "MoveInstrumentComponent"
    assert history.getAlgorithm(7).name() == "ConvertUnits"
    assert history.getAlgorithm(8).name() == "Rebin"
    assert history.getAlgorithm(9).name() == "SetUncertainties"
    assert history.getAlgorithm(10).name() == "AddSampleLogMultiple"

    assert loaded.background.data is None
    assert loaded.background_transmission.data is None
    assert loaded.empty.data is None
    assert loaded.sample_transmission.data is None
    assert loaded.dark_current.data is not None
    assert loaded.sensitivity is not None
    assert loaded.mask is not None

    # Verify that if something is changed that it gets applied correctly on reload, use default mask as test
    # First check the current value
    assert not loaded.sample[0].data.detectorInfo().isMasked(1)

    # Change reduction input and rerun load_all_files
    reduction_input["configuration"]["useDefaultMask"] = True
    reduction_input["configuration"]["defaultMask"] = "{'Pixel':'1'}"
    loaded = load_all_files(reduction_input)

    # Check that the value has changed
    assert loaded.sample[0].data.detectorInfo().isMasked(1)


@pytest.mark.parametrize("generic_workspace", [{"name": "ws_raw_histo"}], indirect=True)
def test_prepare_data_workspaces_simple(generic_workspace):
    ws = generic_workspace  # friendly name

    output = prepare_data_workspaces(ws_mon_pair(data=ws, monitor=None))
    # this should make a clone of the workspace
    assert ws is not output
    # and change the workspace name automatically
    assert ws.name() == "ws_raw_histo"
    assert output.name() == "ws_processed_histo"

    output2 = prepare_data_workspaces(
        ws_mon_pair(data=ws, monitor=None), output_workspace="foobar"
    )
    # the ws name should change to what is set
    assert ws.name() == "ws_raw_histo"
    assert output2.name() == "foobar"


def test_prepare_data_workspaces_dark_current():
    # Create dark current workspace, insert the duration of the dark
    # current run as one of the log entries in the dark current
    # workspace.
    dark_current_workspace = uwd()  # arbitrary name for the dark current workspace
    CreateWorkspace(
        DataX=[2.5, 3.5],
        DataY=np.full(2, 100.0),
        DataE=np.full(2, 10.0),
        NSpec=2,
        OutputWorkspace=dark_current_workspace,
    )
    SampleLogs(dark_current_workspace).insert("duration", 3600.0, "second")

    # Create a sample run workspace.
    data_workspace = uwd()  # arbitrary name for the sample workspace
    CreateWorkspace(
        DataX=[2.5, 3.5],
        DataY=np.array([1.0, 2.0]),
        DataE=np.array([1.0, np.sqrt(2)]),
        NSpec=2,
        OutputWorkspace=data_workspace,
    )
    # Insert the duration of the sample run. The log key must be the
    # same as that used for the dark current, which turns out to be
    # 'duration'
    SampleLogs(data_workspace).insert("duration", 36.0, "second")

    # Chech that this fail for the correct reason, I didn't add the all the required logs
    with pytest.raises(AttributeError) as excinfo:
        prepare_data_workspaces(
            ws_mon_pair(data=mtd[data_workspace], monitor=None),
            dark_current=ws_mon_pair(data=mtd[dark_current_workspace], monitor=None),
            solid_angle=False,
        )
    assert str(excinfo.value) == '"tof_frame_width_clipped" not found in sample logs'


@pytest.mark.parametrize(
    "generic_workspace", [{"intensities": [[1, 2], [3, 4]]}], indirect=True
)
def test_prepare_data_workspaces_flux_method(generic_workspace):
    ws = generic_workspace  # friendly name
    SampleLogs(ws).insert("duration", 2.0)
    SampleLogs(ws).insert("monitor", 2e9)

    # No normalization
    output = prepare_data_workspaces(
        ws_mon_pair(data=ws, monitor=None), flux_method=None, solid_angle=False
    )
    assert output.getHistory().size() == 3
    assert_almost_equal(output.extractY(), [[1], [2], [3], [4]])

    # Normalize by time
    output = prepare_data_workspaces(
        ws_mon_pair(data=ws, monitor=None), flux_method="time", solid_angle=False
    )
    assert output.getHistory().size() == 4
    assert_almost_equal(output.extractY(), [[0.5], [1], [1.5], [2]])

    # need to add test for proton charge and monitor


def test_prepare_data_workspaces_apply_mask(generic_workspace):
    ws = generic_workspace

    # mask_ws
    output = prepare_data_workspaces(
        ws_mon_pair(data=ws, monitor=None), mask_ws=[0, 2], solid_angle=False
    )
    history = output.getHistory()
    assert history.size() == 4
    alg3 = history.getAlgorithm(3)
    assert alg3.name() == "MaskDetectors"
    assert alg3.getPropertyValue("DetectorList") == "0,2"


@pytest.mark.parametrize(
    "generic_workspace", [{"intensities": [[1, 1], [1, 1]]}], indirect=True
)
def test_prepare_data_workspaces_solid_angle(generic_workspace):
    ws = generic_workspace  # friendly name

    # No normalization
    output = prepare_data_workspaces(
        ws_mon_pair(data=ws, monitor=None), solid_angle=True
    )
    # CreateWorkspace, LoadInstrument, CloneWorkspace, CloneWorkspace,
    # ClearMaskFlag, SolidAngle, Divide, ReplaceSpecialValues
    assert output.getHistory().size() == 8
    assert_almost_equal(
        output.extractY(), [[25.6259267], [25.6259267], [25.6259267], [25.6259267]]
    )


def test_prepare_data_workspaces_sensitivity():
    # Create dark current workspace, insert the duration of the dark
    # current run as one of the log entries in the dark current
    # workspace.
    sensitivity_workspace = uwd()  # arbitrary name for the dark current workspace
    CreateWorkspace(
        DataX=[2.5, 3.5],
        DataY=np.full(2, 2.0),
        DataE=np.full(2, np.sqrt(2)),
        NSpec=2,
        OutputWorkspace=sensitivity_workspace,
    )

    # Create a sample run workspace.
    data_workspace = uwd()  # arbitrary name for the sample workspace
    CreateWorkspace(
        DataX=[2.5, 3.5],
        DataY=np.array([1.0, 2.0]),
        DataE=np.array([1.0, np.sqrt(2)]),
        NSpec=2,
        OutputWorkspace=data_workspace,
    )

    output = prepare_data_workspaces(
        ws_mon_pair(data=mtd[data_workspace], monitor=None),
        sensitivity_workspace=sensitivity_workspace,
        solid_angle=False,
    )

    assert output.getHistory().size() == 6

    assert_almost_equal(output.extractY(), [[0.5], [1.0]])
    assert_almost_equal(output.extractE(), [[0.6123724], [1.0]])


@pytest.mark.parametrize(
    "generic_workspace", [{"intensities": [[1, 2], [3, 4]]}], indirect=True
)
def test_process_single_configuration_thickness_absolute_scale(generic_workspace):
    ws = generic_workspace

    # This should only run prepare_data_workspaces,
    # normalize_by_thickness and scale by absolute_scale
    # The output result should be scaled by y_out = y_in * absolute_scale / thickness

    output = pre_process_single_configuration(
        ws_mon_pair(data=ws, monitor=None),
        bkg_ws_raw=ws_mon_pair(data=None, monitor=None),
        solid_angle=False,
    )

    # CreateWorkspace, LoadInstrument, CloneWorkspace,
    # CreateSingleValuedWorkspace, Divide,
    # CreateSingleValuedWorkspace, Multiply
    assert output.getHistory().size() == 7

    assert_equal(output.extractY(), [[1], [2], [3], [4]])

    output = pre_process_single_configuration(
        ws_mon_pair(data=ws, monitor=None),
        bkg_ws_raw=ws_mon_pair(data=None, monitor=None),
        solid_angle=False,
        absolute_scale=1.5,
    )
    assert_equal(output.extractY(), [[1.5], [3], [4.5], [6]])

    output = pre_process_single_configuration(
        ws_mon_pair(data=ws, monitor=None),
        bkg_ws_raw=ws_mon_pair(data=None, monitor=None),
        solid_angle=False,
        thickness=0.1,
    )
    assert_equal(output.extractY(), [[10], [20], [30], [40]])

    output = pre_process_single_configuration(
        ws_mon_pair(data=ws, monitor=None),
        bkg_ws_raw=ws_mon_pair(data=None, monitor=None),
        solid_angle=False,
        absolute_scale=1.5,
        thickness=0.1,
    )
    assert_equal(output.extractY(), [[15], [30], [45], [60]])


if __name__ == "__main__":
    pytest.main([__file__])
