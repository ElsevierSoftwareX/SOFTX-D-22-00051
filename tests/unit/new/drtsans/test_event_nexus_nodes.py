import pytest
import numpy as np
import os
import h5py
from drtsans.files.hdf5_rw import parse_h5_entry
from drtsans.files.event_nexus_nodes import InstrumentNode
from drtsans.files.event_nexus_nodes import BankNode
from drtsans.files.event_nexus_nodes import DasLogNode
from drtsans.files.event_nexus_nodes import MonitorNode


def test_create_monitor_node(reference_dir):
    """Test to create a Monitor node

    Parameters
    ----------
    reference_dir

    Returns
    -------

    """
    # Parse NeXus file manually for the values nodes
    source_nexus = os.path.join(reference_dir.new.gpsans, "CG2_9166.nxs.h5")

    # Parse the source HDF5
    nexus_h5 = h5py.File(source_nexus, "r")
    source_root = parse_h5_entry(nexus_h5)

    # Get a bank node
    bank9_entry = nexus_h5["/entry/monitor1"]
    event_indexes = bank9_entry["event_index"][()]
    event_time_offsets = bank9_entry["event_time_offset"][()]
    event_time_zeros = bank9_entry["event_time_zero"][(())]
    run_start_time = bank9_entry["event_time_zero"].attrs["offset"].decode()

    # check type
    assert isinstance(event_indexes, np.ndarray)

    # Create bank node for bank 9
    monitor_node = MonitorNode(name="/entry/monitor1", monitor_name="monitor")
    monitor_node.set_monitor_events(
        event_indexes, event_time_offsets, run_start_time, event_time_zeros
    )

    # Verify
    expected_monitor_node = source_root.get_child("/entry").get_child(
        "monitor1", is_short_name=True
    )
    expected_monitor_node.match(monitor_node)

    # Close file
    nexus_h5.close()


def test_create_events_node(reference_dir):
    """Test to create a Bank event node

    Parameters
    ----------
    reference_dir

    Returns
    -------

    """
    # Parse NeXus file manually for the values nodes
    source_nexus = os.path.join(reference_dir.new.gpsans, "CG2_9166.nxs.h5")

    # Parse the source HDF5
    nexus_h5 = h5py.File(source_nexus, "r")
    source_root = parse_h5_entry(nexus_h5)

    # Get a bank node
    bank9_entry = nexus_h5["/entry/bank9_events"]
    event_ids = bank9_entry["event_id"][()]
    event_indexes = bank9_entry["event_index"][()]
    event_time_offsets = bank9_entry["event_time_offset"][()]
    event_time_zeros = bank9_entry["event_time_zero"][(())]
    run_start_time = bank9_entry["event_time_zero"].attrs["offset"].decode()

    # check type
    assert isinstance(event_ids, np.ndarray)

    # Create bank node for bank 9
    bank9_node = BankNode(name="/entry/bank9_events", bank_name="bank9")
    bank9_node.set_events(
        event_ids, event_indexes, event_time_offsets, run_start_time, event_time_zeros
    )

    # Verify
    expected_bank9_node = source_root.get_child("/entry").get_child(
        "bank9_events", is_short_name=True
    )
    expected_bank9_node.match(bank9_node)

    # Close file
    nexus_h5.close()


def test_create_das_log_node(reference_dir):
    """Test to create a DAS log node

    Example: /entry/DASlogs/sample_detector_distance

    Parameters
    ----------
    reference_dir

    Returns
    -------

    """
    # Parse NeXus file manually for the values nodes
    source_nexus = os.path.join(reference_dir.new.gpsans, "CG2_9166.nxs.h5")

    # Parse the source HDF5
    nexus_h5 = h5py.File(source_nexus, "r")
    source_root = parse_h5_entry(nexus_h5)

    # Get times and value for /entry/DASlogs/sample_detector_distance
    ssd_entry = nexus_h5["entry"]["DASlogs"]["sample_detector_distance"]
    ssd_times = ssd_entry["time"][()]
    ssd_start_time = ssd_entry["time"].attrs["start"]
    ssd_value = ssd_entry["value"][()]
    ssd_value_unit = ssd_entry["value"].attrs["units"]

    # Set up a DAS log node
    ssd_test_node = DasLogNode(
        log_name="/entry/DASlogs/sample_detector_distance",
        log_times=ssd_times,
        log_values=ssd_value,
        start_time=ssd_start_time,
        log_unit=ssd_value_unit,
    )

    ssd_test_node.set_device_info(
        device_id=13,
        device_name=b"Mot-Galil3",
        target=b"/entry/DASlogs/CG2:CS:SampleToDetRBV",
    )

    # match: to entry/DASlogs/sample_detector_distance
    expected_node = source_root.get_child("entry", is_short_name=True).get_child(
        "DASlogs", is_short_name=True
    )
    expected_node = expected_node.get_child(
        "sample_detector_distance", is_short_name=True
    )
    expected_node.match(ssd_test_node)

    # Close HDF5
    nexus_h5.close()


def test_create_instrument_node(reference_dir):
    """Test to create an instrument node

    Returns
    -------

    """
    # Parse NeXus file manually
    source_nexus = os.path.join(reference_dir.new.gpsans, "CG2_9166.nxs.h5")

    # Parse the source HDF5
    nexus_h5 = h5py.File(source_nexus, "r")
    source_root = parse_h5_entry(nexus_h5)

    # IDF in XML
    xml_idf = nexus_h5["entry"]["instrument"]["instrument_xml"]["data"][0]

    # Create new instrument node
    test_node = InstrumentNode()
    test_node.set_idf(
        xml_idf, idf_type=b"text/xml", description=b"XML contents of the instrument IDF"
    )
    test_node.set_instrument_info(
        target_station_number=1, beam_line=b"CG2", name=b"CG2", short_name=b"CG2"
    )

    # Verify
    # attributes
    source_instrument = source_root.get_child("/entry").get_child("/entry/instrument")

    # attributes
    # cannot get b'NXinstrument'
    assert (
        source_instrument.attributes == test_node.attributes
    ), "{} shall be same as {}" "".format(
        source_instrument.attributes, test_node.attributes
    )

    # beam line
    for child_name in ["beamline", "instrument_xml", "name", "target_station_number"]:
        child_name = f"/entry/instrument/{child_name}"
        src_child_node = source_instrument.get_child(child_name)
        test_child_node = test_node.get_child(child_name)
        src_child_node.match(test_child_node)

    # Close HDF5
    nexus_h5.close()


if __name__ == "__main__":
    pytest.main(__file__)
