# Create Event NeXus file
import numpy as np
import dateutil
import datetime
import math
import drtsans
from drtsans.files.hdf5_rw import DataSetNode, GroupNode

__all__ = [
    "MonitorNode",
    "BankNode",
    "DataSetNode",
    "DasLogsCollectionNode",
    "InstrumentNode",
]


class MonitorNode(drtsans.files.hdf5_rw.GroupNode):
    """
    Node to record monitor counts
    """

    def __init__(self, name, monitor_name):
        """

        Parameters
        ----------
        name
        monitor_name
        """
        self._monitor_name = monitor_name

        super(MonitorNode, self).__init__(name)

        # add NX_class
        self.add_attributes({"NX_class": b"NXmonitor"})

    def set_monitor_events(
        self,
        event_index_array,
        event_time_offset_array,
        run_start_time,
        event_time_zero_array,
    ):
        """Monitor counts are recorded as events

        Parameters
        ----------
        event_time_offset_array: numpy.ndarray
            TOF of each neutron event.  Size is equal to number of events
        event_time_zero_array: numpy.ndarray
            Staring time, as seconds offset to run start time, of each pulse
        run_start_time: str
            Run start time in ISO standard
        event_index_array: numpy.ndarray
            Index of staring event in each pulse.  Size is equal to number of pulses

        Returns
        -------

        """
        # Check inputs
        assert event_time_zero_array.shape == event_index_array.shape

        # Total counts
        total_counts = len(event_time_offset_array)

        # For all children except event time
        for child_name, child_value, child_units in [
            ("event_index", event_index_array, None),
            ("event_time_offset", event_time_offset_array, b"microsecond"),
            ("total_counts", [total_counts], None),
        ]:
            child_node = DataSetNode(name=self._create_child_name(child_name))
            child_node.set_value(np.array(child_value))

            if child_units is not None:
                child_node.add_attributes({"units": child_units})
            self.set_child(child_node)

        # Set pulse time node (event time zero)
        node_name = self._create_child_name("event_time_zero")
        pulse_time_node = generate_event_time_zero_node(
            node_name, event_time_zero_array, run_start_time
        )
        # link to self/its parent
        self.set_child(pulse_time_node)


class BankNode(drtsans.files.hdf5_rw.GroupNode):
    """Node for bank entry such as /entry/bank12"""

    def __init__(self, name, bank_name):
        """Initialization

        Parameters
        ----------
        name: str
            Bank node name
        bank_name: str
            name of bank, such as 'bank10' or 'bank39'
        """
        self._bank_name = bank_name

        super(BankNode, self).__init__(name)

        # add NX_class
        self.add_attributes({"NX_class": b"NXevent_data"})

    def set_events(
        self,
        event_id_array,
        event_index_array,
        event_time_offset_array,
        run_start_time,
        event_time_zero_array,
    ):
        """

        Parameters
        ----------
        event_id_array: numpy.ndarray
            pixel IDs for each event.  Size is equal to number of events
        event_time_offset_array: numpy.ndarray
            TOF of each neutron event.  Size is equal to number of events
        event_time_zero_array: numpy.ndarray
            Staring time, as seconds offset to run start time, of each pulse
        run_start_time: str
            Run start time in ISO standard
        event_index_array: numpy.ndarray
            Index of staring event in each pulse.  Size is equal to number of pulses

        Returns
        -------

        """
        # Check inputs
        assert event_id_array.shape == event_time_offset_array.shape
        assert event_time_zero_array.shape == event_index_array.shape

        # Total counts
        total_counts = len(event_id_array)

        # For all children except event time
        for child_name, child_value, child_units in [
            ("event_id", event_id_array, None),
            ("event_index", event_index_array, None),
            ("event_time_offset", event_time_offset_array, b"microsecond"),
            ("total_counts", [total_counts], None),
        ]:
            child_node = DataSetNode(name=self._create_child_name(child_name))
            child_node.set_value(np.array(child_value))
            # add target
            target_value = f"/entry/instrument/{self._bank_name}/{child_name}".encode()
            child_node.add_attributes({"target": target_value})

            if child_units is not None:
                child_node.add_attributes({"units": child_units})
            self.set_child(child_node)

        # Set pulse time node
        self._set_pulse_time_node(event_time_zero_array, run_start_time)

    def _set_pulse_time_node(self, event_time_zero_array, run_start_time):
        """Set pulse time zero node

        # add attriutes including
        # offset : run start time
        # offset_nanoseconds
        # offset_seconds

        Parameters
        ----------
        event_time_zero_array: numpy.ndarray
            Staring time, as seconds offset to run start time, of each pulse
        run_start_time: str
            run start time in ISO format

        Returns
        -------

        """
        # create child node name with full path
        node_name = self._create_child_name("event_time_zero")
        # create child node for event time zero (pulse time)
        pulse_time_node = generate_event_time_zero_node(
            node_name, event_time_zero_array, run_start_time
        )
        # link to self/its parent
        self.set_child(pulse_time_node)


def generate_event_time_zero_node(node_name, event_time_zero_array, run_start_time):
    # calculate run start time offset
    offset_second, offset_ns = calculate_time_offsets(run_start_time)

    # Special for event_time_zero node
    pulse_time_node = DataSetNode(name=node_name)

    # set value
    pulse_time_node.set_value(event_time_zero_array)
    # set up attribution dictionary
    pulse_attr_dict = {
        "units": b"second",
        "target": b"/entry/DASlogs/frequency/time",
        "offset": np.string_(run_start_time),
        "offset_nanoseconds": offset_ns,
        "offset_seconds": offset_second,
    }
    pulse_time_node.add_attributes(pulse_attr_dict)

    return pulse_time_node


class InstrumentNode(drtsans.files.hdf5_rw.GroupNode):
    """
    Node for instrument entry (i.e., /entry/instrument)
    """

    def __init__(self):
        """ """
        super(InstrumentNode, self).__init__(name="/entry/instrument")

        # add the NeXus class attributes
        self.add_attributes({"NX_class": b"NXinstrument"})

    def set_instrument_info(self, target_station_number, beam_line, name, short_name):
        """Set instrument information

        Parameters
        ----------
        target_station_number: int
            target station number.  1 is used for HFIR
        beam_line: Bytes
            CG2, CG3
        name: Bytes
            CG2, CG3
        short_name: Bytes
            CG2, CG3

        Returns
        -------
        None

        """
        # target station node
        target_station_node = DataSetNode(name=f"{self.name}/target_station_number")
        target_station_node.set_value(np.array(target_station_number))
        self.set_child(target_station_node)

        # beam line
        beam_line_node = DataSetNode(name=f"{self.name}/beamline")
        beam_line_node.set_string_value(beam_line)
        self.set_child(beam_line_node)

        # beam line name
        name_node = DataSetNode(name=f"{self.name}/name")
        name_node.set_string_value(name)
        self.set_child(name_node)
        name_node.add_attributes({"short_name": short_name})

    def set_idf(self, idf_str, idf_type, description):
        """Set instrument xml

        Parameters
        ----------
        idf_str: Bytes
            IDF XML string
        idf_type: Bytes
            IDF type
        description: Bytes
            Description

        Returns
        -------

        """
        # Create the instrument_xml node
        xml_node_name = f"{self.name}/instrument_xml"
        xml_node = GroupNode(name=xml_node_name)
        xml_node.add_attributes({"NX_class": b"NXnote"})
        self.set_child(xml_node)

        # add data node
        data_node = DataSetNode(name=f"{xml_node_name}/data")
        data_node.set_string_value(idf_str)
        xml_node.set_child(data_node)

        # add description
        des_node = DataSetNode(name=f"{xml_node_name}/description")
        des_node.set_string_value(description)
        xml_node.set_child(des_node)

        # add type
        type_node = DataSetNode(name=f"{xml_node_name}/type")
        type_node.set_string_value(idf_type)
        xml_node.set_child(type_node)


class DasLogNode(drtsans.files.hdf5_rw.GroupNode):
    """
    Node for one specific DAS log such as /entry/DASlogs/sample_detector_distance
    """

    def __init__(self, log_name, log_times, start_time, log_values, log_unit):
        """DAS log node for specific

        Parameters
        ----------
        log_name: str
            full path log name as /entry/DASlogs/{log_name}
        log_times: numpy.ndarray
            relative sample log time
        start_time: Bytes
            ISO standard time for run start
        log_values: numpy.ndarray
            sample log values
        log_unit: Byes
            log unit
        """
        super(DasLogNode, self).__init__(name=log_name)
        self._log_times = log_times
        self._run_start = start_time
        self._log_values = log_values
        self._log_unit = log_unit

        # Standard NX_class type for DASlogs' node
        self.add_attributes({"NX_class": b"NXlog"})

        # Set log value and related terms
        self._set_log_values()
        # Set log times
        self._set_log_times()

    def _set_log_times(self):
        """Set log times' node
        - time

        Returns
        -------

        """
        time_offset_second, time_offset_ns = calculate_time_offsets(self._run_start)

        # Now I set up time related attributes
        time_node = DataSetNode(name=self._create_child_name("time"))
        time_node.set_value(self._log_times)
        time_node.add_attributes(
            {
                "offset_nanoseconds": time_offset_ns,
                "offset_seconds": time_offset_second,
                "start": self._run_start,
                "units": b"second",
            }
        )
        self.set_child(time_node)

    def _set_log_values(self):
        """Set time and value including
        - average_value
        - average_value_error
        - maximum_value
        - minimum_value
        - value

        Returns
        -------

        """
        if any([isinstance(me, np.bytes_) for me in self._log_values]):
            # string type log requires no calculation
            child_node = DataSetNode(name=self._create_child_name("value"))
            child_node.set_value(self._log_values)
            child_node.add_attributes({"units": self._log_unit})
            self.set_child(child_node)
        else:
            # value type log requires some calculation
            average_value = self._log_values.mean()
            average_value_error = self._log_values.std()
            min_value = np.min(self._log_values)
            max_value = np.max(self._log_values)

            for child_name, child_value in [
                ("average_value", [average_value]),
                ("average_value_error", [average_value_error]),
                ("maximum_value", [max_value]),
                ("minimum_value", [min_value]),
                ("value", self._log_values),
            ]:
                child_node = DataSetNode(name=self._create_child_name(child_name))
                child_node.set_value(np.array(child_value))
                child_node.add_attributes({"units": self._log_unit})
                self.set_child(child_node)

    def set_device_info(self, device_id, device_name, target):
        """Set node for device related information

        Parameters
        ----------
        device_id
        device_name
        target

        Returns
        -------

        """
        # Create Device ID node
        # Need to make sure all strings are Bytes
        for node_name, info_value in [
            ("device_id", [device_id]),
            ("device_name", [np.string_(device_name)]),
            ("target", [np.string_(target)]),
        ]:
            child_node = DataSetNode(name=self._create_child_name(node_name))
            child_node.set_value(np.array(info_value))
            self.set_child(child_node)

        return


class DasLogsCollectionNode(drtsans.files.hdf5_rw.GroupNode):
    """
    Node for '/entry/DASlogs'
    """

    def __init__(self):
        """
        Initialization
        """
        super(DasLogsCollectionNode, self).__init__(name="/entry/DASlogs")
        self.add_attributes({"NX_class": b"NXcollection"})


def calculate_time_offsets(iso_time):
    """Calculate time offset from 1990.01.01

    Parameters
    ----------
    iso_time: str, bytes
        time in ISO format
    Returns
    -------
    tuple
        offset time in second (whole number), offset time in nanosecond (whole number)

    """
    # convert
    if isinstance(iso_time, bytes):
        iso_time = iso_time.decode()

    # convert date time in IOS string to datetime instance
    run_start_time = dateutil.parser.parse(iso_time)
    epoch_time = datetime.datetime(
        1990, 1, 1, tzinfo=datetime.timezone(datetime.timedelta(0))
    )
    # offsets
    time_offset = run_start_time.timestamp() - epoch_time.timestamp()
    time_offset_second = int(time_offset)
    # nanosecond shift
    if iso_time.count(".") == 0:
        # zero sub-second offset
        time_offset_ns = 0
    elif iso_time.count(".") == 1:
        # non-zero sub-second offset
        # has HH:MM:SS.nnnsssnnnss-05 format
        sub_second_str = iso_time.split(".")[1].split("-")[0]
        sub_seconds = float(sub_second_str)
        # convert from sub seconds to nano seconds
        # example: 676486982
        digits = int(math.log(sub_seconds) / math.log(10)) + 1
        time_offset_ns = int(sub_seconds * 10 ** (9 - digits))
    else:
        # more than 1 '.': not knowing the case.  Use robust solution
        time_offset_ns = int((time_offset - time_offset_second) * 1e9)

    return time_offset_second, time_offset_ns
