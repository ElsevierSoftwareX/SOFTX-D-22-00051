import pytest
from pytest import approx
import numpy as np

from mantid.simpleapi import Rebin, SumSpectra, mtd
from drtsans.settings import amend_config, unique_workspace_name as uwn
from drtsans.tof.eqsans.load import (
    load_events,
    load_events_monitor,
    sum_data,
    load_and_split,
    load_events_and_histogram,
)
from drtsans.load import load_and_split as generic_load_and_split
from drtsans.tof.eqsans.correct_frame import (
    transform_to_wavelength,
    set_init_uncertainties,
)
from drtsans.samplelogs import SampleLogs


def test_load_events(reference_dir):
    # default workspace name is file hint
    with amend_config(data_dir=reference_dir.new.eqsans):
        ws_test_load_events = load_events("EQSANS_92353.nxs.h5")
    assert ws_test_load_events.name() == "EQSANS_92353"

    ws_name = uwn()
    with amend_config(data_dir=reference_dir.new.eqsans):
        ws = load_events("EQSANS_92353.nxs.h5", output_workspace=ws_name)
    assert ws.name() == ws_name

    assert ws.getTofMin() == pytest.approx(11288, abs=1)
    assert ws.getTofMax() == pytest.approx(61309, abs=1)

    ws = Rebin(ws, Params=[10000, 1000, 62000], PreserveEvents=False)
    ws = SumSpectra(ws)
    assert len(np.nonzero(ws.dataY(0))[0]) == 35


def test_load_events_monitor(reference_dir):
    # Raises for a run in skip frame mode
    with pytest.raises(RuntimeError, match="cannot correct monitor"):
        load_events_monitor("EQSANS_92353.nxs.h5", data_dir=reference_dir.new.eqsans)

    w = load_events_monitor("EQSANS_88901.nxs.h5", data_dir=reference_dir.new.eqsans)
    assert w.name() == "EQSANS_88901_monitors"
    assert w.getSpectrum(0).getTofMin() == approx(30680, abs=1)
    assert w.getSpectrum(0).getTofMax() == approx(47347, abs=1)


def test_merge_Data(reference_dir):
    ws0 = load_events("EQSANS_101595.nxs.h5", data_dir=reference_dir.new.eqsans)
    ws0, bands0 = transform_to_wavelength(ws0)
    ws0 = set_init_uncertainties(ws0)
    ws1 = load_events("EQSANS_104088.nxs.h5", data_dir=reference_dir.new.eqsans)
    ws1, bands1 = transform_to_wavelength(ws1)
    ws1 = set_init_uncertainties(ws1)
    ws2 = load_events("EQSANS_105428.nxs.h5", data_dir=reference_dir.new.eqsans)
    ws2, bands2 = transform_to_wavelength(ws2)
    ws2 = set_init_uncertainties(ws2)

    sample_logs0 = SampleLogs(ws0)
    sample_logs1 = SampleLogs(ws1)
    sample_logs2 = SampleLogs(ws2)

    merged_workspaces = sum_data([ws0, ws1, ws2], output_workspace="merged")

    merged_sample_logs = SampleLogs(merged_workspaces)

    # Check duration increase as the sum
    assert sample_logs0.duration.value == pytest.approx(215.531066895, abs=1e-9)
    assert sample_logs1.duration.value == pytest.approx(289.029266357, abs=1e-9)
    assert sample_logs2.duration.value == pytest.approx(72.3323135376, abs=1e-9)
    assert merged_sample_logs.duration.value == pytest.approx(
        215.531066895 + 289.029266357 + 72.3323135376, abs=1e-9
    )

    # Check proton charge is correct
    assert sample_logs0.getProtonCharge() == pytest.approx(83.37074628055555, abs=1e-9)
    assert sample_logs1.getProtonCharge() == pytest.approx(111.1237739861111, abs=1e-9)
    assert sample_logs2.getProtonCharge() == pytest.approx(27.799524525, abs=1e-9)
    assert merged_sample_logs.getProtonCharge() == pytest.approx(
        83.37074628055555, +111.1237739861111 + 27.799524525, abs=1e-9
    )

    # Check Time Series properties increase length
    assert sample_logs0.proton_charge.size() == 12933
    assert sample_logs1.proton_charge.size() == 17343
    assert sample_logs2.proton_charge.size() == 4341
    assert merged_sample_logs.proton_charge.size() == 12933 + 17343 + 4341

    # Check integrated intensity increases as the total sum
    assert mtd[str(ws0)].extractY().sum() == 284923
    assert mtd[str(ws1)].extractY().sum() == 1368485
    assert mtd[str(ws2)].extractY().sum() == 65694
    assert mtd[str(merged_workspaces)].extractY().sum() == 284923 + 1368485 + 65694

    mtd.remove(str(ws0))
    mtd.remove(str(ws1))
    mtd.remove(str(ws2))
    mtd.remove(str(merged_workspaces))


def test_load_events_and_histogram(reference_dir):
    ws0 = load_events_and_histogram(
        "EQSANS_101595.nxs.h5", data_dir=reference_dir.new.eqsans
    )

    assert ws0.data.getAxis(0).getUnit().caption() == "Wavelength"
    assert ws0.data.name() == "EQSANS_101595"
    assert ws0.monitor is None

    sample_logs0 = SampleLogs(ws0.data)

    assert sample_logs0.duration.value == pytest.approx(215.531066895, abs=1e-9)
    assert sample_logs0.getProtonCharge() == pytest.approx(83.37074628055555, abs=1e-9)
    assert sample_logs0.proton_charge.size() == 12933

    ws1 = load_events_and_histogram(
        "EQSANS_101595.nxs.h5,EQSANS_104088.nxs.h5,EQSANS_105428.nxs.h5",
        data_dir=reference_dir.new.eqsans,
        keep_events=False,
    )

    assert ws1.data.getAxis(0).getUnit().caption() == "Wavelength"
    assert ws1.data.name() == "EQSANS_101595_104088_105428"
    assert ws1.monitor is None

    sample_logs1 = SampleLogs(ws1.data)
    assert sample_logs1.duration.value == pytest.approx(
        215.531066895 + 289.029266357 + 72.3323135376, abs=1e-9
    )
    assert sample_logs1.getProtonCharge() == pytest.approx(
        83.37074628055555, +111.1237739861111 + 27.799524525, abs=1e-9
    )
    assert sample_logs1.proton_charge.size() == 12933 + 17343 + 4341

    mtd.remove(str(ws0))
    mtd.remove(str(ws1))


def test_generic_load_and_split(reference_dir):
    # split by the SampleTemp log
    filtered_ws, filtered_ws_monitors = generic_load_and_split(
        "EQSANS_104088.nxs.h5",
        data_dir=reference_dir.new.eqsans,
        log_name="SampleTemp",
        log_value_interval=0.1,
        monitors=True,
    )

    assert filtered_ws.size() == 3
    assert filtered_ws_monitors.size() == 3

    assert filtered_ws.name() == "EQSANS_104088"
    assert filtered_ws_monitors.name() == "EQSANS_104088_monitors"

    assert SampleLogs(filtered_ws.getItem(0)).duration.value == pytest.approx(
        8.007968453, abs=1e-7
    )
    assert SampleLogs(filtered_ws.getItem(1)).duration.value == pytest.approx(
        277.040577412, abs=1e-7
    )
    assert SampleLogs(filtered_ws.getItem(2)).duration.value == pytest.approx(
        3.997389135, abs=1e-7
    )

    assert filtered_ws.getItem(0).getNumberEvents() == 38038
    assert filtered_ws.getItem(1).getNumberEvents() == 1311447
    assert filtered_ws.getItem(2).getNumberEvents() == 19000

    assert filtered_ws_monitors.getItem(0).getNumberEvents() == 1440
    assert filtered_ws_monitors.getItem(1).getNumberEvents() == 49869
    assert filtered_ws_monitors.getItem(2).getNumberEvents() == 720

    # check metadata is set correctly
    assert SampleLogs(filtered_ws.getItem(0)).slice.value == 1
    assert SampleLogs(filtered_ws.getItem(1)).slice.value == 2
    assert SampleLogs(filtered_ws.getItem(2)).slice.value == 3
    assert SampleLogs(filtered_ws.getItem(0)).number_of_slices.value == 3
    assert SampleLogs(filtered_ws.getItem(1)).number_of_slices.value == 3
    assert SampleLogs(filtered_ws.getItem(2)).number_of_slices.value == 3
    assert SampleLogs(filtered_ws.getItem(0)).slice_parameter.value == "SampleTemp"
    assert SampleLogs(filtered_ws.getItem(1)).slice_parameter.value == "SampleTemp"
    assert SampleLogs(filtered_ws.getItem(2)).slice_parameter.value == "SampleTemp"
    assert SampleLogs(filtered_ws.getItem(0)).slice_interval.value == 0.1
    assert SampleLogs(filtered_ws.getItem(1)).slice_interval.value == 0.1
    assert SampleLogs(filtered_ws.getItem(2)).slice_interval.value == 0.1
    assert SampleLogs(filtered_ws.getItem(0)).slice_start.value == 19.85
    assert SampleLogs(filtered_ws.getItem(1)).slice_start.value == 19.95
    assert SampleLogs(filtered_ws.getItem(2)).slice_start.value == 20.05
    assert SampleLogs(filtered_ws.getItem(0)).slice_end.value == 19.95
    assert SampleLogs(filtered_ws.getItem(1)).slice_end.value == 20.05
    assert SampleLogs(filtered_ws.getItem(2)).slice_end.value == 20.15
    assert SampleLogs(filtered_ws.getItem(0)).slice_start.units == "C"
    assert SampleLogs(filtered_ws.getItem(1)).slice_start.units == "C"
    assert SampleLogs(filtered_ws.getItem(2)).slice_start.units == "C"
    assert SampleLogs(filtered_ws.getItem(0)).slice_end.units == "C"
    assert SampleLogs(filtered_ws.getItem(1)).slice_end.units == "C"
    assert SampleLogs(filtered_ws.getItem(2)).slice_start.units == "C"

    mtd.remove(str(filtered_ws))
    mtd.remove(str(filtered_ws_monitors))


def test_load_and_split(reference_dir):
    # split by the SampleTemp log
    filtered_ws, bands = load_and_split(
        "EQSANS_104088.nxs.h5",
        data_dir=reference_dir.new.eqsans,
        log_name="SampleTemp",
        log_value_interval=0.1,
    )

    assert filtered_ws.size() == 3

    assert filtered_ws.name() == "EQSANS_104088"

    assert SampleLogs(filtered_ws.getItem(0)).duration.value == pytest.approx(
        8.007968453, abs=1e-7
    )
    assert SampleLogs(filtered_ws.getItem(1)).duration.value == pytest.approx(
        277.040577412, abs=1e-7
    )
    assert SampleLogs(filtered_ws.getItem(2)).duration.value == pytest.approx(
        3.997389135, abs=1e-7
    )

    assert filtered_ws.getItem(0).getAxis(0).getUnit().caption() == "Wavelength"
    assert filtered_ws.getItem(1).getAxis(0).getUnit().caption() == "Wavelength"
    assert filtered_ws.getItem(2).getAxis(0).getUnit().caption() == "Wavelength"

    # check values for Y and E don't change unexpectedly
    assert filtered_ws.getItem(0).extractY().max() == 4
    assert filtered_ws.getItem(1).extractY().max() == 25
    assert filtered_ws.getItem(2).extractY().max() == 3
    assert filtered_ws.getItem(0).extractE().max() == pytest.approx(2, abs=1e-7)
    assert filtered_ws.getItem(1).extractE().max() == pytest.approx(5, abs=1e-7)
    assert filtered_ws.getItem(2).extractE().max() == pytest.approx(1.7320508, abs=1e-7)

    # check metadata is set correctly
    assert SampleLogs(filtered_ws.getItem(0)).slice.value == 1
    assert SampleLogs(filtered_ws.getItem(1)).slice.value == 2
    assert SampleLogs(filtered_ws.getItem(2)).slice.value == 3
    assert SampleLogs(filtered_ws.getItem(0)).number_of_slices.value == 3
    assert SampleLogs(filtered_ws.getItem(1)).number_of_slices.value == 3
    assert SampleLogs(filtered_ws.getItem(2)).number_of_slices.value == 3
    assert SampleLogs(filtered_ws.getItem(0)).slice_parameter.value == "SampleTemp"
    assert SampleLogs(filtered_ws.getItem(1)).slice_parameter.value == "SampleTemp"
    assert SampleLogs(filtered_ws.getItem(2)).slice_parameter.value == "SampleTemp"
    assert SampleLogs(filtered_ws.getItem(0)).slice_interval.value == 0.1
    assert SampleLogs(filtered_ws.getItem(1)).slice_interval.value == 0.1
    assert SampleLogs(filtered_ws.getItem(2)).slice_interval.value == 0.1
    assert SampleLogs(filtered_ws.getItem(0)).slice_start.value == 19.85
    assert SampleLogs(filtered_ws.getItem(1)).slice_start.value == 19.95
    assert SampleLogs(filtered_ws.getItem(2)).slice_start.value == 20.05
    assert SampleLogs(filtered_ws.getItem(0)).slice_end.value == 19.95
    assert SampleLogs(filtered_ws.getItem(1)).slice_end.value == 20.05
    assert SampleLogs(filtered_ws.getItem(2)).slice_end.value == 20.15
    assert SampleLogs(filtered_ws.getItem(0)).slice_start.units == "C"
    assert SampleLogs(filtered_ws.getItem(1)).slice_start.units == "C"
    assert SampleLogs(filtered_ws.getItem(2)).slice_start.units == "C"
    assert SampleLogs(filtered_ws.getItem(0)).slice_end.units == "C"
    assert SampleLogs(filtered_ws.getItem(1)).slice_end.units == "C"
    assert SampleLogs(filtered_ws.getItem(2)).slice_start.units == "C"

    mtd.remove(str(filtered_ws))


if __name__ == "__main__":
    pytest.main([__file__])
