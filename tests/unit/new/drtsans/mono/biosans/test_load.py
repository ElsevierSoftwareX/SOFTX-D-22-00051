import pytest

from mantid import mtd
from mantid.simpleapi import GroupWorkspaces
from drtsans.geometry import sample_detector_distance
from drtsans.mono.biosans import (
    load_histogram,
    load_events,
    load_and_split,
    set_init_uncertainties,
    transform_to_wavelength,
    sum_data,
    load_events_and_histogram,
)
from drtsans.samplelogs import SampleLogs


def test_load_events(reference_dir):
    events_workspace = load_events(
        "CG3_961.nxs.h5", data_dir=reference_dir.new.biosans, overwrite_instrument=True
    )
    assert events_workspace.name() == "BIOSANS_961"

    sample_logs = SampleLogs(events_workspace)
    assert sample_logs.monitor.value == 19173627

    x, y, z = events_workspace.getInstrument().getComponentByName("detector1").getPos()
    assert -1000 * x == pytest.approx(
        sample_logs.single_value("detector_trans_Readback"), abs=0.001
    )
    assert z == pytest.approx(
        sample_logs.single_value("sample_detector_distance"), abs=0.001
    )


def test_transform_to_wavelength(reference_dir):
    workspace = load_events("CG3_961.nxs.h5", data_dir=reference_dir.new.biosans)
    workspace = transform_to_wavelength(workspace)
    assert workspace.getAxis(0).getUnit().caption() == "Wavelength"


def test_api_load(biosans_f):
    ws = load_histogram(filename=biosans_f["beamcenter"])
    assert ws.name() == "BioSANS_exp402_scan0006_0001"

    # check logs
    sl = SampleLogs(ws)
    wavelength_log = sl.find_log_with_units("wavelength", "Angstrom")
    assert round(wavelength_log) == 6.0
    wavelength_spread_log = sl.find_log_with_units("wavelength-spread", "Angstrom")
    assert round(wavelength_spread_log, 1) == 0.8
    wavelength_spread_ratio_log = sl.single_value("wavelength-spread-ratio")
    assert wavelength_spread_ratio_log == pytest.approx(0.1323, abs=1e-3)

    ws_name = "xpto"
    ws = load_histogram(filename=biosans_f["beamcenter"], output_workspace=ws_name)
    assert ws.name() == ws_name
    assert ws_name in mtd.getObjectNames()

    ws_name = "xptoxpto"
    ws = load_histogram(
        filename=biosans_f["beamcenter"],
        output_workspace=ws_name,
        wavelength=12,
        wavelength_spread=1,
        sample_det_cent=9,
    )
    assert ws_name in mtd.getObjectNames()

    # check logs when some parameters don't come directly from the metadata
    sl = SampleLogs(ws)
    wavelength_log = sl.find_log_with_units("wavelength", "Angstrom")
    assert round(wavelength_log) == 12.0
    wavelength_spread_log = sl.find_log_with_units("wavelength-spread", "Angstrom")
    assert wavelength_spread_log == 1.0
    wavelength_spread_ratio_log = sl.single_value("wavelength-spread-ratio")
    assert wavelength_spread_ratio_log == pytest.approx(
        wavelength_spread_log / wavelength_log, abs=1e-3
    )


def test_sum_data(reference_dir):
    # Merge the same file twice
    workspace1 = load_events(
        "CG3_961.nxs.h5",
        data_dir=reference_dir.new.biosans,
        output_workspace="workspace1",
    )

    with pytest.raises(ValueError) as excinfo:
        sum_data("workspace1", "merged")
    assert "is not a Workspace2D" in str(
        excinfo.value
    )  # Should complain about wrong workspace type

    workspace1 = transform_to_wavelength(workspace1)
    workspace1 = set_init_uncertainties(workspace1)
    workspace2 = load_events(
        "CG3_960.nxs.h5",
        data_dir=reference_dir.new.biosans,
        output_workspace="workspace2",
    )
    workspace2 = transform_to_wavelength(workspace2)
    workspace2 = set_init_uncertainties(workspace2)

    sample_logs1 = SampleLogs(workspace1)
    sample_logs2 = SampleLogs(workspace2)

    merged_workspaces = sum_data([workspace1, workspace2], output_workspace="merged")

    merged_sample_logs = SampleLogs(merged_workspaces)

    # Check monitor and duration increase as the sum
    assert sample_logs1.monitor.value == 19173627
    assert sample_logs2.monitor.value == 1039
    assert merged_sample_logs.monitor.value == 19173627 + 1039
    assert sample_logs1.duration.value == pytest.approx(1809.4842529296875, abs=1e-11)
    assert sample_logs2.duration.value == pytest.approx(0.0833325386047363, abs=1e-11)
    assert merged_sample_logs.duration.value == pytest.approx(
        1809.4842529296875 + 0.08333253860473633, abs=1e-11
    )

    # Check Time Series properties increase length
    assert sample_logs1.wavelength.size() == 692
    assert sample_logs2.wavelength.size() == 2
    assert merged_sample_logs.wavelength.size() == 692 + 2

    # Check integrated intensity increases as the total sum
    assert mtd[str(workspace1)].extractY().sum() == 11067715
    assert mtd[str(workspace2)].extractY().sum() == 1
    assert mtd[str(merged_workspaces)].extractY().sum() == 11067715 + 1

    # Test different input formats
    # List of workspace names
    merged_workspaces_2 = sum_data(
        ["workspace1", "workspace2"], output_workspace="merged2"
    )
    assert SampleLogs(merged_workspaces_2).duration.value == pytest.approx(
        1809.4842529296875 + 0.08333253860473633, abs=1e-11
    )

    # Comma separated list of workspace space
    merged_workspaces_3 = sum_data("workspace1, workspace2", output_workspace="merged3")
    assert SampleLogs(merged_workspaces_3).duration.value == pytest.approx(
        1809.4842529296875 + 0.08333253860473633, abs=1e-11
    )

    # Workspace group
    ws_group = GroupWorkspaces("workspace1, workspace2")
    merged_workspaces_4 = sum_data(ws_group, output_workspace="merged4")
    assert SampleLogs(merged_workspaces_4).duration.value == pytest.approx(
        1809.4842529296875 + 0.08333253860473633, abs=1e-11
    )


def test_load_events_and_histogram(reference_dir):
    workspace = load_events_and_histogram(
        "CG3_961.nxs.h5",
        sample_to_si_name="CG3:CS:SampleToSi",
        si_nominal_distance=0.071,
        data_dir=reference_dir.new.biosans,
    )
    assert workspace.getAxis(0).getUnit().caption() == "Wavelength"
    assert workspace.name() == "BIOSANS_961"

    sample_logs = SampleLogs(workspace)
    assert sample_logs.monitor.value == 19173627
    assert sample_logs.duration.value == pytest.approx(1809.4842529296875, abs=1e-11)
    assert sample_logs.wavelength.size() == 692
    assert mtd[str(workspace)].extractY().sum() == 11067715

    workspace2 = load_events_and_histogram(
        "CG3_961.nxs.h5, CG3_960.nxs.h5",
        data_dir=reference_dir.new.biosans,
        sample_to_si_name="CG3:CS:SampleToSi",
        si_nominal_distance=0.071,
    )
    assert workspace2.getAxis(0).getUnit().caption() == "Wavelength"
    assert workspace2.name() == "BIOSANS_961_960"

    sample_logs2 = SampleLogs(workspace2)

    assert sample_logs2.monitor.value == 19173627 + 1039
    assert sample_logs2.duration.value == pytest.approx(
        1809.4842529296875 + 0.08333253860473633, abs=1e-11
    )
    assert sample_logs2.wavelength.size() == 692 + 2
    assert mtd[str(workspace2)].extractY().sum() == 11067715 + 1


def test_load_and_split(reference_dir):
    # Check that is fails with missing required paramters
    with pytest.raises(ValueError) as excinfo:
        load_and_split(
            "CG3_961.nxs.h5",
            data_dir=reference_dir.new.biosans,
            sample_to_si_name="CG3:CS:SampleToSi",
            si_nominal_distance=0.071,
        )
    assert "Must provide with time_interval or log_name and log_value_interval" == str(
        excinfo.value
    )

    filtered_ws = load_and_split(
        "CG3_961.nxs.h5",
        data_dir=reference_dir.new.biosans,
        time_interval=1000,
        sample_to_si_name="CG3:CS:SampleToSi",
        si_nominal_distance=0.071,
    )

    assert filtered_ws.size() == 2

    assert SampleLogs(filtered_ws.getItem(0)).duration.value == pytest.approx(
        1000, abs=1e-11
    )
    assert SampleLogs(filtered_ws.getItem(1)).duration.value == pytest.approx(
        809.48427990000005, abs=1e-11
    )

    assert SampleLogs(filtered_ws.getItem(0)).monitor.value == 10553922
    assert SampleLogs(filtered_ws.getItem(1)).monitor.value == 8619525

    assert filtered_ws.getItem(0).getNumberEvents() == 6149184
    assert filtered_ws.getItem(1).getNumberEvents() == 4918534

    # check metadata is set correctly
    assert SampleLogs(filtered_ws.getItem(0)).slice.value == 1
    assert SampleLogs(filtered_ws.getItem(1)).slice.value == 2
    assert SampleLogs(filtered_ws.getItem(0)).number_of_slices.value == 2
    assert SampleLogs(filtered_ws.getItem(1)).number_of_slices.value == 2
    assert (
        SampleLogs(filtered_ws.getItem(0)).slice_parameter.value
        == "relative time from start"
    )
    assert (
        SampleLogs(filtered_ws.getItem(1)).slice_parameter.value
        == "relative time from start"
    )
    assert SampleLogs(filtered_ws.getItem(0)).slice_interval.value == 1000
    assert SampleLogs(filtered_ws.getItem(1)).slice_interval.value == 1000
    assert SampleLogs(filtered_ws.getItem(0)).slice_start.value == 0
    assert SampleLogs(filtered_ws.getItem(1)).slice_start.value == 1000
    assert SampleLogs(filtered_ws.getItem(0)).slice_end.value == 1000
    assert SampleLogs(filtered_ws.getItem(1)).slice_end.value == pytest.approx(
        1809.4842799, abs=1e-5
    )
    assert SampleLogs(filtered_ws.getItem(0)).slice_start.units == "seconds"
    assert SampleLogs(filtered_ws.getItem(1)).slice_start.units == "seconds"
    assert SampleLogs(filtered_ws.getItem(0)).slice_end.units == "seconds"
    assert SampleLogs(filtered_ws.getItem(1)).slice_end.units == "seconds"

    # Verify the others
    # SampleToSi = 60. mm
    # SDD = 3.060004084948254
    # Sample position is expected to be at -(60 - 71) = 11 mm
    # Verify geometry related values
    # workspace 0
    ws0 = filtered_ws.getItem(0)
    # workspace 1
    ws1 = filtered_ws.getItem(1)
    # Verify
    very_geometry_meta(
        [ws0, ws1],
        expected_sample_detector_distance=3.06,
        expected_sample_si_distance=60,
        expected_sample_position_z=11.0 * 1e-3,
    )


def test_load_and_split_overwrite_geometry(reference_dir):
    # Check that is fails with missing required paramters
    with pytest.raises(ValueError) as excinfo:
        load_and_split(
            "CG3_961.nxs.h5",
            data_dir=reference_dir.new.biosans,
            sample_to_si_name="CG3:CS:SampleToSi",
            si_nominal_distance=0.071,
        )
    assert "Must provide with time_interval or log_name and log_value_interval" == str(
        excinfo.value
    )

    filtered_ws = load_and_split(
        "CG3_961",
        data_dir=reference_dir.new.biosans,
        time_interval=1000,
        sample_to_si_name="CG3:CS:SampleToSi",
        si_nominal_distance=0.071,
        sample_detector_distance_value=10.0,
        sample_to_si_value=0.041,
    )

    assert filtered_ws.size() == 2

    assert SampleLogs(filtered_ws.getItem(0)).duration.value == pytest.approx(
        1000, abs=1e-11
    )
    assert SampleLogs(filtered_ws.getItem(1)).duration.value == pytest.approx(
        809.48427990000005, abs=1e-11
    )

    assert SampleLogs(filtered_ws.getItem(0)).monitor.value == 10553922
    assert SampleLogs(filtered_ws.getItem(1)).monitor.value == 8619525

    assert filtered_ws.getItem(0).getNumberEvents() == 6149184
    assert filtered_ws.getItem(1).getNumberEvents() == 4918534

    # check metadata is set correctly
    assert SampleLogs(filtered_ws.getItem(0)).slice.value == 1
    assert SampleLogs(filtered_ws.getItem(1)).slice.value == 2
    assert SampleLogs(filtered_ws.getItem(0)).number_of_slices.value == 2
    assert SampleLogs(filtered_ws.getItem(1)).number_of_slices.value == 2
    assert (
        SampleLogs(filtered_ws.getItem(0)).slice_parameter.value
        == "relative time from start"
    )
    assert (
        SampleLogs(filtered_ws.getItem(1)).slice_parameter.value
        == "relative time from start"
    )
    assert SampleLogs(filtered_ws.getItem(0)).slice_interval.value == 1000
    assert SampleLogs(filtered_ws.getItem(1)).slice_interval.value == 1000
    assert SampleLogs(filtered_ws.getItem(0)).slice_start.value == 0
    assert SampleLogs(filtered_ws.getItem(1)).slice_start.value == 1000
    assert SampleLogs(filtered_ws.getItem(0)).slice_end.value == 1000
    assert SampleLogs(filtered_ws.getItem(1)).slice_end.value == pytest.approx(
        1809.4842799, abs=1e-5
    )
    assert SampleLogs(filtered_ws.getItem(0)).slice_start.units == "seconds"
    assert SampleLogs(filtered_ws.getItem(1)).slice_start.units == "seconds"
    assert SampleLogs(filtered_ws.getItem(0)).slice_end.units == "seconds"
    assert SampleLogs(filtered_ws.getItem(1)).slice_end.units == "seconds"

    # Verify the others
    # SampleToSi = 0.041 mm
    # SDD = 10.
    # Sample position is expected to be at -(41 - 71) = 30 mm
    # Verify geometry related values
    # workspace 0
    ws0 = filtered_ws.getItem(0)
    # workspace 1
    ws1 = filtered_ws.getItem(1)
    # Verify
    very_geometry_meta(
        [ws0, ws1],
        expected_sample_detector_distance=10.0,
        expected_sample_si_distance=41.0,
        expected_sample_position_z=30.0 * 1e-3,
    )

    return


def very_geometry_meta(
    workspace_list,
    expected_sample_detector_distance,
    expected_sample_si_distance,
    expected_sample_position_z,
):
    """Assuming there are 2 workspaces in the group

    Parameters
    ----------
    workspace_list: ~list
        list of workspaces
    expected_sample_detector_distance: float
        .. ...  unit = meter
    expected_sample_si_distance: float
        ... ... unit = millimeter
    expected_sample_position_z: float
        ... ...

    Returns
    -------

    """
    for index, workspace in enumerate(workspace_list):
        print("[TEST] workspace {}: {}".format(index, str(workspace)))

        # check SDD: unit meter
        sdd = sample_detector_distance(workspace, unit="m", search_logs=False)
        assert sdd == pytest.approx(expected_sample_detector_distance, 1e-4)

        # check sample silicon window distance: unit millimeter
        swd = SampleLogs(workspace)["CG3:CS:SampleToSi"].value
        assert swd == pytest.approx(expected_sample_si_distance, 1e-4)

        # sample position: unit meter
        sample_pos_z = workspace.getInstrument().getSample().getPos()[2]
        assert sample_pos_z == pytest.approx(expected_sample_position_z, 1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
