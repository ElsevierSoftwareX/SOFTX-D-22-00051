import numpy as np
from collections import namedtuple
from drtsans.files.hdf5_rw import FileNode, GroupNode
from drtsans.files.hdf5_rw import DataSetNode
from drtsans.files.event_nexus_nodes import (
    InstrumentNode,
    BankNode,
    MonitorNode,
    DasLogNode,
    DasLogsCollectionNode,
)
import h5py
import dateutil
from mantid.simpleapi import logger


__all__ = [
    "TofHistogram",
    "NexusEvents",
    "EventNeXusWriter",
    "generate_events_from_histogram",
    "convert_events_to_histogram",
    "DasLog",
    "DasDevice",
    "RunTime",
    "init_event_nexus",
    "parse_event_nexus",
]

# Specify parameter
# Histogram converted from TOF events
TofHistogram = namedtuple(
    "TofHistogram", ["pixel_ids", "counts", "pulse_duration", "tof_min", "tof_max"]
)

# TOF events generated from histogram
NexusEvents = namedtuple(
    "NexusEvents", ["event_id", "event_index", "event_time_offset", "event_time_zero"]
)

# Run time in ISO time string
RunTime = namedtuple("RunTime", ["start", "stop"])

# DAS log: time is relative time to run start in seconds
DasLog = namedtuple("DasLog", ["name", "times", "values", "unit", "device"])

# Sample log device parameters
DasDevice = namedtuple("DasDevice", ["id", "name", "target"])


class EventNeXusWriter(object):
    """
    Write an Event NeXus file
    """

    def __init__(self, beam_line, instrument_name):
        """Initialization"""
        self._beam_line = beam_line
        self._instrument_name = instrument_name

        # entry node
        self._entry_node = None
        self._root_node = None

        # Initialize
        self._init_event_nexus()

        # Bank of events
        self._banks_dict = dict()

        # Instrument
        self._num_banks = None
        self._idf_xml = None

        # Meta data
        self._meta_data_dict = dict()

        # Run start and stop time
        self._run_time = None

    def _init_event_nexus(self):
        """Initialize event Nexus node

        Returns
        -------
        None

        """
        # create a new file node
        self._root_node = init_event_nexus()

        # get entry node
        self._entry_node = self._root_node.get_child("/entry")

        # Set DAS log node
        self._log_collection_node = DasLogsCollectionNode()
        self._entry_node.set_child(self._log_collection_node)

        # Create new instrument node
        self._instrument_node = InstrumentNode()
        self._entry_node.set_child(self._instrument_node)

    def set_instrument_info(self, num_banks, instrument_xml):
        """Set information related to SANS instrument including IDF in plain XML string and
        number of banks

        Parameters
        ----------
        num_banks: int
            number of banks
        instrument_xml: str
            instrument definition in plain XML string

        Returns
        -------

        """
        self._num_banks = num_banks
        self._idf_xml = instrument_xml

    def set_bank_histogram(self, bank_id, bank_histogram):
        """Set a single bank's counts

        Parameters
        ----------
        bank_id
        bank_histogram: TofHistogram
            Histogram for simulating TOF

        Returns
        -------

        """
        self._banks_dict[bank_id] = bank_histogram

    def set_meta_data(self, meta_data):
        """

        Parameters
        ----------
        meta_data: DasLog
            Das log object

        Returns
        -------

        """
        self._meta_data_dict[meta_data.name] = meta_data

    def _set_instrument_node(self, xml_idf):
        """Set instrument node

        Parameters
        ----------
        xml_idf: str
            IDF XML

        Returns
        -------

        """
        # Set values
        self._instrument_node.set_idf(
            xml_idf,
            idf_type=b"text/xml",
            description=b"XML contents of the instrument IDF",
        )
        self._instrument_node.set_instrument_info(
            target_station_number=1,
            beam_line=np.string_(self._instrument_name),
            name=np.string_(self._instrument_name),
            short_name=np.string_(self._instrument_name),
        )

    def _set_log_node(self, log_name, relative_log_times, log_values, log_unit, device):
        """Set a DAS log node

        Parameters
        ----------
        log_name
        relative_log_times
        log_values
        log_unit
        device: DasDevice
            DAS device parameters

        Returns
        -------

        """

        # Set up a DAS log node
        das_log_node = DasLogNode(
            log_name=f"/entry/DASlogs/{log_name}",
            log_times=relative_log_times,
            log_values=log_values,
            start_time=self._run_time.start,
            log_unit=log_unit,
        )

        # Set device information
        if device is not None:
            # about device target
            if device.target is None:
                device_target = np.string_(device.name).decode()
            else:
                device_target = np.string_(device.target).decode()
            # add full path if device.target is only a short name
            if not device_target.startswith("/entry"):
                device_target = f"/entry/DASlogs/{device_target}"

            das_log_node.set_device_info(
                device_id=device.id,
                device_name=np.string_(device.name),
                target=device_target.encode(),
            )

        # append to parent node
        self._log_collection_node.set_child(das_log_node)

    def _set_das_logs_node(self):
        """Set DAS log node in a mixed way

        Returns
        -------

        """
        # Set logs
        for das_log in self._meta_data_dict.values():
            self._set_log_node(
                das_log.name,
                das_log.times,
                das_log.values,
                das_log.unit,
                das_log.device,
            )

    def _set_run_time(self, start_time, stop_time):
        """Set run start and stop time

        Parameters
        ----------
        start_time: str
            run start time in ISO standard
        stop_time: str
            run stop time in ISO standard

        Returns
        -------

        """
        # Calculate duration
        t0 = dateutil.parser.parse(start_time)
        tf = dateutil.parser.parse(stop_time)
        duration_s = (tf - t0).total_seconds()

        # Set up the list
        entry_value_tuples = [
            ("/entry/start_time", np.array([np.string_(start_time)])),
            ("/entry/end_time", np.array([np.string_(stop_time)])),
            ("/entry/duration", np.array([duration_s]).astype("float32")),
        ]

        for child_node_name, value in entry_value_tuples:
            # Init regular DataSetNode and set value
            child_node = DataSetNode(child_node_name)
            child_node.set_value(value)
            # Link as the child of entry
            self._entry_node.set_child(child_node)

    def _set_single_bank_node(self, bank_id, bank_histogram):
        """Test writing bank 9 from histogram

        Parameters
        ----------
        bank_id: int
            bank ID (from 1 to 48)
        bank_histogram: TofHistogram
            Histogram converted from TOF bank information

        Returns
        -------
        BankNode
            newly generated bank node

        """
        # Retrieve information from specified bank
        # bank_entry = source_h5[f'/entry/bank{bank_id}_events']
        # bank_histogram = convert_events_to_histogram(bank_entry)
        # run_start_time = bank_entry['event_time_zero'].attrs['offset'].decode()

        # generate events
        nexus_events = generate_events_from_histogram(bank_histogram, 10.0)

        # Create bank node for bank
        bank_node = BankNode(
            name=f"/entry/bank{bank_id}_events", bank_name=f"bank{bank_id}"
        )
        bank_node.set_events(
            nexus_events.event_id,
            nexus_events.event_index,
            nexus_events.event_time_offset,
            self._run_time.start,
            nexus_events.event_time_zero,
        )

        # Link with parent
        self._entry_node.set_child(bank_node)

        return bank_node

    def _set_monitor_node(self, monitor_counts, event_time_zeros):
        """

        Parameters
        ----------
        monitor_counts: int, float
            monitor counts
        event_time_zeros

        Returns
        -------

        """
        # Generate a monitor node
        target_monitor_node = MonitorNode("/entry/monitor1", "monitor1")

        tof_min = 0.0
        tof_max = 10000.0
        monitor_events = generate_monitor_events_from_count(
            monitor_counts, event_time_zeros, tof_min, tof_max
        )

        target_monitor_node.set_monitor_events(
            event_index_array=monitor_events.event_index,
            event_time_offset_array=monitor_events.event_time_offset,
            run_start_time=self._run_time.start,
            event_time_zero_array=event_time_zeros,
        )

        self._entry_node.set_child(target_monitor_node)

    def set_run_number(self, run_number):
        """Set run number if required

        Parameters
        ----------
        run_number: int
            run number

        Returns
        -------

        """
        # Set up the list
        entry_value_tuple = ("/entry/run_number", run_number)

        # Init regular DataSetNode and set value
        child_node = DataSetNode(entry_value_tuple[0])
        # need to convert to
        child_node.set_value(np.array([np.string_(f"{entry_value_tuple[1]}")]))
        # Link as the child of entry
        self._entry_node.set_child(child_node)

    def generate_event_nexus(self, nexus_name, start_time, stop_time, monitor_counts):
        """Generate an event Nexus file from scratch
        Parameters
        ----------
        nexus_name: str
            Output NeXus file name
        start_time
        stop_time
        monitor_counts

        Returns
        -------

        """
        # Set time
        self._run_time = RunTime(start_time, stop_time)

        # set instrument node
        self._set_instrument_node(self._idf_xml)

        # set DAS logs
        self._set_das_logs_node()

        # set run start and stop information
        self._set_run_time(start_time, stop_time)

        # set Bank 1 - 48
        max_pulse_time_array = None
        for bank_id in range(1, self._num_banks + 1):
            bank_node_i = self._set_single_bank_node(
                bank_id=bank_id, bank_histogram=self._banks_dict[bank_id]
            )
            event_time_zeros = bank_node_i.get_child(
                "event_time_zero", is_short_name=True
            ).value
            if (
                max_pulse_time_array is None
                or event_time_zeros.shape[0] > max_pulse_time_array.shape[0]
            ):
                max_pulse_time_array = event_time_zeros

        # Set monitor node
        self._set_monitor_node(monitor_counts, max_pulse_time_array)

        # write
        self._root_node.write(nexus_name)


def init_event_nexus():
    """Initialize an event Nexus file buffer including
    - entry

    Returns
    -------
    FileNode
        root event NeXus node

    """
    # create a new file node
    nexus_root_node = FileNode()

    # create an '/entry' node
    entry_node = GroupNode("/entry")
    nexus_root_node.set_child(entry_node)

    # add attribution as NX_class
    entry_node.add_attributes({"NX_class": "NXentry"})

    return nexus_root_node


def generate_monitor_events_from_count(
    monitor_counts, event_time_zero_array, min_tof, max_tof
):
    """Generate monitor events from a single monitor count

    Parameters
    ----------
    monitor_counts
    event_time_zero_array: numpy.ndarray
        event time zero array (for pulse time)
    min_tof: float
        minimum TOF value for faked neutron events
    max_tof: float
        maximum TOF value for faked neutron events

    Returns
    -------
    NexusEvents
        Generated TOF events for monitor counts

    """
    # Create event_id list
    event_id_array = np.zeros((monitor_counts,), dtype="uint32")

    # number of events per pulse
    num_pulses = event_time_zero_array.shape[0]
    num_events_per_pulse = monitor_counts // num_pulses
    if num_events_per_pulse < 1:
        logger.warning("num_events_per_pulse = monitor_counts // num_pulses")
        logger.warning(f"\t={monitor_counts}//{num_pulses}=0")
        logger.warning("forcing num_events_per_pulse to 1")
        num_events_per_pulse = 1

    # Time of flight array
    # number of pulses with regular value or more value
    num_plus_one = monitor_counts % num_pulses
    num_regular = num_pulses - num_plus_one

    # base TOF array
    # resolution
    resolution = (max_tof - min_tof) / num_events_per_pulse
    base_tof_array = np.arange(num_events_per_pulse) * resolution + min_tof
    event_time_offset_array = np.tile(base_tof_array, num_regular)
    # create event index array
    event_index_array = np.arange(num_regular).astype("uint64") * num_events_per_pulse

    # plus 1 ones
    if num_plus_one > 0:
        # resolution
        resolution = (max_tof - min_tof) / (num_events_per_pulse + 1)
        base_tof_array_p1 = np.arange(num_events_per_pulse + 1) * resolution + min_tof
        event_time_offset_plus1 = np.tile(base_tof_array_p1, num_plus_one)
        # create event index array: incrementing by (num_event_per_pulse + 1)
        event_index_array_plus1 = np.arange(num_plus_one).astype("uint64") * (
            num_events_per_pulse + 1
        )
        event_shift = event_index_array[-1] + num_events_per_pulse
        # shift
        event_index_array_plus1 += int(event_shift)

        # concatenate
        event_time_offset_array = np.concatenate(
            (event_time_offset_array, event_time_offset_plus1)
        )
        event_index_array = np.concatenate((event_index_array, event_index_array_plus1))

    # construct output
    faked_nexus_events = NexusEvents(
        event_id_array,
        event_index_array,
        event_time_offset_array,
        event_time_zero_array,
    )

    return faked_nexus_events


def parse_event_nexus(source_nexus_name, num_banks, logs_white_list=None):
    """Parse an event Nexus file for minimal required information

    Parameters
    ----------
    source_nexus_name
    num_banks: int
        number of banks. assuming bank numbers are consecutive

    Returns
    -------

    """
    # Import source
    source_nexus_h5 = h5py.File(source_nexus_name, "r")

    # Run start and stop
    run_start = source_nexus_h5["entry"]["start_time"][0]
    run_stop = source_nexus_h5["entry"]["end_time"][0]

    # IDF in XML
    xml_idf = source_nexus_h5["entry"]["instrument"]["instrument_xml"]["data"][0]
    # Retrieve information from specified bank
    bank_histograms = dict()
    for bank_id in range(1, num_banks + 1):
        bank_entry = source_nexus_h5[f"/entry/bank{bank_id}_events"]
        bank_histogram = convert_events_to_histogram(bank_entry)
        bank_histograms[bank_id] = bank_histogram

    # Retrieve information from specified bank
    monitor_entry = source_nexus_h5["/entry/monitor1"]
    monitor_counts = monitor_entry["event_time_offset"][()].shape[0]

    # add sample logs
    source_logs_node = source_nexus_h5["entry"]["DASlogs"]

    # Specify white list
    if logs_white_list is None:
        logs_white_list = [
            "CG2:CS:SampleToSi",
            "sample_detector_distance",
            "wavelength",
            "wavelength_spread",
            "source_aperture_diameter",
            "sample_aperture_diameter",
            "detector_trans_Readback",
            "attenuator",
        ]
    das_log_dict = dict()

    for log_name in logs_white_list:
        # print(f'Reading sample log {log_name}')
        log_times = source_logs_node[log_name]["time"][()]
        log_value = source_logs_node[log_name]["value"][()]
        try:
            log_value_unit = source_logs_node[log_name]["value"].attrs["units"]
        except KeyError:
            log_value_unit = None
        device_name = source_logs_node[log_name]["device_name"][0]
        device_id = source_logs_node[log_name]["device_id"][0]
        try:
            device_target = source_logs_node[log_name]["target"][0]
        except KeyError:
            device_target = None
        # Set to proper data structure
        device = DasDevice(device_id, device_name, device_target)
        das_log = DasLog(log_name, log_times, log_value, log_value_unit, device)
        # Add to dictionary
        das_log_dict[log_name] = das_log

    # close original file
    source_nexus_h5.close()

    return xml_idf, bank_histograms, monitor_counts, run_start, run_stop, das_log_dict


def generate_events_from_histogram(bank_histogram, tof_resolution=0.1, verbose=False):
    """Convert histogram (counts on detector pixels) to 'fake' events

    Parameters
    ----------
    bank_histogram: TofHistogram
        Histogram for a single bank
    tof_resolution: float
        resolution of TOF
    verbose: bool
        flag to print out events information

    Returns
    -------
    NexusEvents
        Generated TOF events in NeXus format

    """
    # get total counts
    total_counts = bank_histogram.counts.sum()
    if verbose:
        print(
            f"Pixel ID range: {bank_histogram.pixel_ids.min()} - {bank_histogram.pixel_ids.max()}; "
            f"Total counts = {total_counts} type = {type(total_counts)}"
        )

    # Create event_id
    event_id_array = np.ndarray(shape=(total_counts,), dtype="uint32")
    # repeat each pixel for its 'counts' times to simulate the number of events
    start_index = 0
    for pid_index, pixel_id in enumerate(bank_histogram.pixel_ids):
        # get counts
        stop_index = start_index + bank_histogram.counts[pid_index]
        # set repetition
        event_id_array[start_index:stop_index] = pixel_id
        # promote to next round
        start_index = stop_index

    # Get pulse related parameters
    # num_events_per_pulse = int((bank_histogram.tof_max - bank_histogram.tof_min) / tof_resolution)
    tof_max = 20000
    tof_min = 10000
    num_events_per_pulse = int((tof_max - tof_min) / tof_resolution)
    num_pulses = (
        total_counts // num_events_per_pulse
    )  # This value is just a whole number. It could +1
    if verbose:
        print(
            f"Original TOF range = {bank_histogram.tof_min}, {bank_histogram.tof_max}"
        )
        print(f"event/pulse = {num_events_per_pulse}; number pulses = {num_pulses}")

    # event_time_offset, event_index
    single_pulse_tof = (
        np.arange(num_events_per_pulse, dtype="float32") * tof_resolution
        + bank_histogram.tof_min
    )
    # if verbose:
    #     print(f'single pulse TOF: {single_pulse_tof}')
    event_time_offset_array = np.tile(single_pulse_tof, num_pulses)
    # event indexes: number of events of each pulse: same value for the first N pulses completely filled
    event_index_array = np.arange(num_pulses).astype("uint64") * num_events_per_pulse
    # event_time_zero: range as [0, 1, ....] * pulse duration. such as [0, 0.16, 0.33, ...]
    event_time_zero_array = np.arange(num_pulses) * bank_histogram.pulse_duration

    # for the rest of events in the REAL last pulse: partially filled
    last_pulse_event_number = total_counts - num_pulses * num_events_per_pulse
    if last_pulse_event_number > 0:
        # num_pulses += 1
        # add the TOF of the last pulse to add
        event_time_offset_array = np.concatenate(
            (event_time_offset_array, single_pulse_tof[0:last_pulse_event_number])
        )
        # add event indexes for the last added pulse
        if len(event_time_zero_array) > 0:
            # last pulse time
            event_index_array = np.concatenate(
                (
                    event_index_array,
                    np.array(
                        [event_index_array[-1] + num_events_per_pulse], dtype="uint64"
                    ),
                )
            )
        else:
            # number of total count is less than number of events per pulse
            # THIS IS TO AVOID A BUG IN MANTID
            # Init even time zero array with 1 pulse
            event_time_zero_array = np.arange(1) * bank_histogram.pulse_duration
            event_index_array = np.array([0, 0], dtype="uint64")

        # add last pulse time (event time zeor)
        prev_last_pulse_time = event_time_zero_array[-1]
        last_pulse_time = prev_last_pulse_time + bank_histogram.pulse_duration
        event_time_zero_array = np.concatenate(
            (event_time_zero_array, np.array([last_pulse_time]))
        )

    # construct output
    faked_nexus_events = NexusEvents(
        event_id_array,
        event_index_array,
        event_time_offset_array,
        event_time_zero_array,
    )

    return faked_nexus_events


def convert_events_to_histogram(bank_entry):
    """Convert events information in bank to histogram

    Parameters
    ----------
    bank_entry:

    Returns
    -------
    TofHistogram
        Histogram converted from TOF bank information

    """

    # Calculate counts on each pixel:w
    bank_event_ids = bank_entry["event_id"]

    # Count each value's appearance in bank event ids array
    # result is from 0 to largest pixel ID in the event_id array
    per_pixel_counts = np.bincount(bank_event_ids)
    # ignore those pixels without any counts
    pixel_id_array = np.where(per_pixel_counts > 0)[0]
    pixel_counts_array = per_pixel_counts[per_pixel_counts > 0]

    # Get pulse time duration information
    pulse_start_times = bank_entry["event_time_zero"][()]
    pulse_duration = (pulse_start_times[1:] - pulse_start_times[:-1]).mean()

    # Get reasonable TOF information
    tof_array = bank_entry["event_time_offset"][()]

    # Define namedtuple to record histogram from TOF events
    histogram = TofHistogram(
        pixel_id_array,
        pixel_counts_array,
        pulse_duration,
        tof_array.min(),
        tof_array.max(),
    )

    return histogram
