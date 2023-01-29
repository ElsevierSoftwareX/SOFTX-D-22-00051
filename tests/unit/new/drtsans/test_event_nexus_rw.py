import pytest
import numpy as np
import os
import h5py
from drtsans.files.event_nexus_rw import EventNeXusWriter
from drtsans.files.event_nexus_rw import (
    generate_events_from_histogram,
    generate_monitor_events_from_count,
)
from drtsans.files.event_nexus_rw import (
    convert_events_to_histogram,
    TofHistogram,
    DasLog,
)


def test_write_event_nexus():
    assert EventNeXusWriter


def test_convert_to_histogram(reference_dir):
    """Test method to convert a single bank's TOF events to histogram

    Parameters
    ----------
    reference_dir

    Returns
    -------

    """
    # Parse NeXus file manually for the values nodes
    nexus_name = os.path.join(reference_dir.new.gpsans, "CG2_9166.nxs.h5")
    nexus_h5 = h5py.File(nexus_name, "r")

    # test with bank 9
    bank9_entry = nexus_h5["/entry/bank9_events"]
    bank9_histogram = convert_events_to_histogram(bank9_entry)
    #  pixel_id_array, pixel_counts_array, pulse_duration, tof_array.min(), tof_array.max()

    # close  file
    nexus_h5.close()

    # verify
    assert bank9_histogram.counts.shape == (992,)
    # pixel ID 17000 (ws index 17000 too) at index 597 has 224 counts
    assert bank9_histogram.pixel_ids[597] == 17000
    assert bank9_histogram.counts[597] == 224
    assert (
        bank9_histogram.pixel_ids.min() >= 16384
        and bank9_histogram.pixel_ids.max() < 17408
    )
    assert bank9_histogram.pulse_duration == pytest.approx(0.01666667, 1.0e-4)
    assert bank9_histogram.tof_min >= 0.0
    assert bank9_histogram.tof_max == pytest.approx(16666.2, 0.1)


def test_convert_histogram_to_events(reference_dir):
    """

    Returns
    -------

    """
    # Create a TofHistogram from bank9
    nexus_name = os.path.join(reference_dir.new.gpsans, "CG2_9166.nxs.h5")
    nexus_h5 = h5py.File(nexus_name, "r")
    # test with bank 9
    bank9_entry = nexus_h5["/entry/bank9_events"]
    bank9_histogram = convert_events_to_histogram(bank9_entry)
    total_counts = bank9_entry["total_counts"][0]
    # close  file
    nexus_h5.close()

    # generate events
    nexus_events = generate_events_from_histogram(bank9_histogram, 0.1)

    # Verification
    # event index only contain the starting event index of each pulse. Its aggregated value is useless
    assert nexus_events.event_id.shape[0] >= nexus_events.event_index.sum()
    assert nexus_events.event_id.shape == nexus_events.event_time_offset.shape
    assert nexus_events.event_index.shape == nexus_events.event_time_zero.shape
    assert nexus_events.event_time_offset.min() == bank9_histogram.tof_min
    assert nexus_events.event_time_offset.max() <= bank9_histogram.tof_max
    # check number of events:
    assert nexus_events.event_id.shape[0] == total_counts


def test_convert_monitor_counts_to_events():
    """Test convert monitor counts to events

    Returns
    -------

    """
    # Generate pulse time: 0.16 second for 250 pulses
    event_time_zero_array = np.arange(250).astype("float") * 0.16

    tof_min = 0.0
    tof_max = 10000

    # Case 1: Counts can be spread evenly
    num_counts = 25000

    # Execute
    monitor_events_even = generate_monitor_events_from_count(
        num_counts, event_time_zero_array, tof_min, tof_max
    )

    # Verify
    assert monitor_events_even.event_time_offset.shape == (num_counts,)
    assert monitor_events_even.event_index.shape == (250,)
    assert monitor_events_even.event_index[-1] == num_counts - 100

    # Case 2: Counts cannot be spread unevenly
    num_counts = 25013

    # Execute
    monitor_events_uneven = generate_monitor_events_from_count(
        num_counts, event_time_zero_array, tof_min, tof_max
    )

    # Verify
    assert monitor_events_uneven.event_time_offset.shape == (num_counts,)
    assert monitor_events_uneven.event_index.shape == (250,)
    assert monitor_events_uneven.event_index[-1] == num_counts - 101
    assert monitor_events_uneven.event_index[-12] == num_counts - 101 * 12


def test_generate_event_nexus(cleanfile):
    """Test for generating an event nexus file

    Returns
    -------

    """
    # Initialize writer
    event_nexus_writer = EventNeXusWriter(
        beam_line="TEST1", instrument_name="TestMonoSANS"
    )

    # Set variables
    # set instrument
    event_nexus_writer.set_instrument_info(2, "<IDF in XML>")
    # set counts
    for bank_id in range(1, 2 + 1):
        histogram_i = TofHistogram(
            np.arange(4) + bank_id * 4,
            np.arange(4) ** 3 + bank_id * 4,
            0.20,
            1000,
            2000,
        )
        event_nexus_writer.set_bank_histogram(bank_id, histogram_i)
    # END-FOR
    # set meta
    for meta_name, meta_value, unit in [
        ("SampleToSi", 81, "mm"),
        ("SampleToDetector", 2.323, "m"),
    ]:
        meta_data = DasLog(
            meta_name, np.array([0.0, 1000.0]), np.array([meta_value]), unit, None
        )
        event_nexus_writer.set_meta_data(meta_data)

    # Generate
    # TODO / FIXME - use a tempfile
    out_nexus_name = "/tmp/fake.nxs.h5"
    # clean
    if os.path.exists(out_nexus_name):
        os.remove(out_nexus_name)

    cleanfile(out_nexus_name)

    start_time = "2020-02-19T01:02:03.123456-05:00"
    end_time = "2020-02-19T01:05:03.654321-05:00"
    event_nexus_writer.generate_event_nexus(out_nexus_name, start_time, end_time, 12345)

    # Verify file existence
    assert os.path.exists(
        out_nexus_name
    ), f"Output event nexus file {out_nexus_name} does not exist"

    # Import file
    nexus_h5 = h5py.File(out_nexus_name, "r")

    # check directory
    assert "entry" in nexus_h5
    assert "DASlogs" in nexus_h5["entry"]
    assert "SampleToSi" in nexus_h5["entry"]["DASlogs"]
    assert "SampleToDetector" in nexus_h5["entry"]["DASlogs"]
    assert "instrument" in nexus_h5["entry"]
    assert "instrument_xml" in nexus_h5["entry"]["instrument"]
    assert "bank1_events" in nexus_h5["entry"]
    assert "bank2_events" in nexus_h5["entry"]
    assert "start_time" in nexus_h5["entry"]
    assert "end_time" in nexus_h5["entry"]


if __name__ == "__main__":
    pytest.main(__file__)
