# This module contains the class and static method to convert legacy
# CG2's data file in XML format to event Nexus
import os
import numpy as np
from drtsans.mono.spice_xml_parser import SpiceXMLParser
from drtsans.files.event_nexus_rw import DasLog, EventNeXusWriter, TofHistogram
import h5py
from typing import List
from mantid.simpleapi import LoadHFIRSANS
from mantid.simpleapi import mtd, logger
import abc
from abc import ABC


# SPICE NeXus meta data unit name conversion.  Note that the units are same but with difference names.
SPICE_NEXUS_UNIT_NAME_MAP = {
    "wavelength": "A",
    "wavelength_spread": None,
    "ww_rot_Readback": "deg",
    "attenuator": None,
}


class EventNexusConverter(ABC):
    """
    Class to provide service to convert to event NeXus from various input
    """

    def __init__(self, beam_line, instrument_name, num_banks):
        """

        Parameters
        ----------
        beam_line: str
            beam line such as CG2, CG3
        instrument_name: str
            instrument name such as CG2, CG3
        """
        # beam line name
        self._beam_line = beam_line
        self._instrument_name = instrument_name
        self._num_banks = num_banks

        # instrument XML IDF content
        self._idf_content = None

        # counts
        self._spice_detector_counts = (
            None  # 1D array ranging from PID 0 (aka workspace index 0)
        )
        self._bank_pid_dict = dict()
        self._bank_counts_dict = dict()
        self._monitor_counts = None

        # sample logs
        self._das_logs = dict()

        # run time
        self._run_start = None
        self._run_stop = None

        # run number
        self._run_number = None

    @property
    def detector_counts(self):
        """Get detector counts

        Returns
        -------
        ~numpy.array
            detector counts

        """
        return self._spice_detector_counts[:]

    def generate_event_nexus(self, target_nexus):
        """Generate event NeXus properly

        num_banks: int
            CG2 = 48, CG3 = 88

        Parameters
        ----------
        target_nexus: str
            name of the output Nexus file

        Returns
        -------

        """
        #  Map detector counts from SPICE to detector counts/bank/detector ID of NeXus
        # detector counts
        self._map_detector_and_counts()

        num_banks = self._num_banks

        # Set constants
        pulse_duration = 0.1  # second
        tof_min = 1000.0
        tof_max = 20000.0

        # Generate event nexus writer
        event_nexus_writer = EventNeXusWriter(
            beam_line=self._beam_line, instrument_name=self._instrument_name
        )

        # set instrument
        event_nexus_writer.set_instrument_info(num_banks, self._idf_content)

        # set counts
        for bank_id in range(1, num_banks + 1):
            # create TofHistogram instance
            # start_pid, end_pid = self.get_pid_range(bank_id)
            # pix_ids = np.arange(start_pid, end_pid + 1)
            # counts = self._detector_counts[start_pid:end_pid + 1]
            # counts = counts.astype("int64")
            pix_ids = self._bank_pid_dict[bank_id]
            counts = self._bank_counts_dict[bank_id]
            assert pix_ids is not None
            assert counts is not None
            histogram = TofHistogram(pix_ids, counts, pulse_duration, tof_min, tof_max)

            # set to writer
            event_nexus_writer.set_bank_histogram(bank_id, histogram)

        # set meta
        for das_log in self._das_logs.values():
            event_nexus_writer.set_meta_data(das_log)

        # set fake run number
        event_nexus_writer.set_run_number(self._run_number)

        # Write file
        event_nexus_writer.generate_event_nexus(
            target_nexus, self._run_start, self._run_stop, self._monitor_counts
        )

    def load_sans_xml(self, xml_file_name, das_log_map, prefix=""):
        """Load data and meta data from legacy SANS XML data file

        Parameters
        ----------
        xml_file_name: str
            name of SANS XML file
        prefix: str
            prefix for output workspace name
        das_log_map: ~dict
            meta data map between event NeXus and SPICE

        Returns
        -------

        """
        spice_log_dict, pt_number = self._retrieve_meta_data(xml_file_name, das_log_map)
        self._das_logs = self.convert_log_units(spice_log_dict)

        # output workspace name
        sans_ws_name = os.path.basename(xml_file_name).split(".xml")[0]
        sans_ws_name = f"{prefix}{sans_ws_name}"

        # load
        logger.notice(f"Load {xml_file_name}")
        LoadHFIRSANS(Filename=xml_file_name, OutputWorkspace=sans_ws_name)

        # get counts and reshape to (N, )
        sans_ws = mtd[sans_ws_name]
        counts = (
            sans_ws.extractY().transpose().reshape((sans_ws.getNumberHistograms(),))
        )

        # monitor counts
        monitor_counts = int(counts[0])
        self._monitor_counts = monitor_counts

        # get detector counts and convert to int 64
        self._spice_detector_counts = counts[2:].astype("int64")

        # NOTE:
        # monitor counts cannot be zero since we need it as the denominator during
        # normalization.
        if abs(self._monitor_counts) < 1e-6:
            logger.warning("current XML contains: monitor_count=0")

        # get run start time: force to a new time
        self._run_start = sans_ws.run().getProperty("run_start").value
        self._run_stop = sans_ws.run().getProperty("end_time").value

        self._run_number = pt_number

    def _map_detector_and_counts(self):
        # self._detector_counts = counts[2:]
        raise RuntimeError("This is virtual")

    def load_idf(self, template_nexus_file):
        """Load IDF content from a template NeXus file

        Parameters
        ----------
        template_nexus_file

        Returns
        -------

        """
        # Import source
        try:
            source_nexus_h5 = h5py.File(template_nexus_file, "r")
        except OSError as os_err:
            raise RuntimeError(f"Unable to load {template_nexus_file} due to {os_err}")
        # IDF in XML
        self._idf_content = source_nexus_h5["entry"]["instrument"]["instrument_xml"][
            "data"
        ][0]
        # Close
        source_nexus_h5.close()

        return

    def mask_spice_detector_pixels(self, pixel_index_list: List[int]):
        """Mask detector pixels with SPICE counts by set the counts to zero

        Parameters
        ----------
        pixel_index_list: ~list
            list of integers as workspace index detector pixels

        """
        # Sanity check
        if self._spice_detector_counts is None:
            raise RuntimeError(
                "Detector counts array has not been set up yet.  Load data first"
            )

        # Set masked pixels
        for pid in pixel_index_list:
            try:
                self._spice_detector_counts[pid] = 0
            except IndexError as index_error:
                raise RuntimeError(
                    f"Pixel ID {pid} is out of range {self._spice_detector_counts.shape}. "
                    f"FYI: {index_error}"
                )

    @staticmethod
    def _retrieve_meta_data(spice_file_name, das_spice_log_map):
        """Retrieve meta from workspace

        Parameters
        ----------
        spice_file_name: str
            full path of SPICE data file in XML format
        das_spice_log_map: ~dict
            DAS log conversion map between event NeXus and spice

        Returns
        -------
        ~dict
            key: Nexus das log name, value: (log value, log unit). if das log is not found, value will be None

        """
        # Load SPICE file
        spice_reader = SpiceXMLParser(spice_file_name)

        # Mandatory key list
        # NOTE: Missing node listed in this list will leads to error in processing.
        #       Early failure is enfored for missing entry in this list
        mandatory_das_logs = [
            "CG3:CS:SampleToSi",
            "sample_detector_distance",
            "wavelength",
            "wavelength_spread",
            "source_aperture_diameter",
            "sample_aperture_diameter",
            "detector_trans_Readback",
            "ww_rot_Readback",
            "source_aperture_sample_aperture_distance",
        ]

        das_log_values = dict()
        for nexus_log_name, spice_tuple in das_spice_log_map.items():
            # read value from XML node
            # FIXME - this is a temporary solution in order to work with both new and old
            # FIXME - meta data map
            spice_log_name = spice_tuple[0]
            default_unit = spice_tuple[1]
            if len(spice_tuple) >= 3:
                data_type = spice_tuple[2]
            else:
                data_type = float  # default
            # try to query the value from XML tree
            try:
                value, unit = spice_reader.get_node_value(spice_log_name, data_type)
            except KeyError as key_err:
                if nexus_log_name in mandatory_das_logs:
                    raise ValueError(
                        f"!Aborting: Cannot find mandaory node {nexus_log_name}"
                    )
                else:
                    logger.warning(str(key_err))
                    logger.warning(f"skipping {nexus_log_name}")
                    value = None
                    unit = None

            if value is not None:
                # set default
                if unit is None:
                    unit = default_unit
                else:
                    # check unit
                    if unit != default_unit:
                        raise RuntimeError(
                            f"SPICE log {spice_log_name} has unit {unit} different from "
                            f"expected {default_unit}"
                        )

                das_log_values[nexus_log_name] = value, unit

        # Get pt number
        # NOTE: the image path contains some information we can use for sanity check
        _expn, _scnn, _scpn = 0, 0, 0
        try:
            imgpath, _ = spice_reader.get_node_value("ImagePath", str)
            tmp = imgpath.split(".")[0].split("_")[1:]
        except KeyError as key_err:
            logger.warning(str(key_err))
            imgpath = ""
            tmp = []

        if len(tmp) == 3:
            _expn = int(tmp[0].replace("exp", ""))
            _scnn = int(tmp[1].replace("scan", ""))
            _scpn = int(tmp[2].replace("scan", ""))
        _pt_parts = [_expn, _scnn, _scpn]

        if "bio" in imgpath.lower():
            logger.notice("Possible BioSANS data found")
            logger.notice("run_number = exp#scan#scanPt#")
            lbs = ["Experiment_number", "Scan_Number", "Scan_Point_Number"]
            for i, lb in enumerate(lbs):
                try:
                    _pt_parts[i], _ = spice_reader.get_node_value(lb, int)
                except KeyError as key_err:
                    logger.warning(str(key_err))
                    logger.warning(f"Defaulting {lb} to {_pt_parts[i]}")
            # stich to form a pt_number
            pt_number = "".join([str(me).zfill(4) for me in _pt_parts])
        else:
            # Default method to handle non BioSANS type SPICE data
            try:
                pt_number, unit = spice_reader.get_node_value("Scan_Point_Number", int)
            except KeyError as key_err:
                pt_number = key_err

        # Close file
        spice_reader.close()

        return das_log_values, pt_number

    @staticmethod
    def convert_log_units(spice_log_dict):
        """Convert log unit from SPICE log to Nexus log

        Parameters
        ----------
        spice_log_dict:  ~dict
            key: Nexus das log name, value: (log value, log unit)

        Returns
        -------
         ~dict
            key: DAS log name, value: DasLog

        """
        nexus_log_dict = dict()

        for nexus_das_log_name in spice_log_dict:
            # get value
            log_value, log_unit = spice_log_dict[nexus_das_log_name]

            # use the name of the NeXus das log value unit
            if nexus_das_log_name in SPICE_NEXUS_UNIT_NAME_MAP:
                log_unit = SPICE_NEXUS_UNIT_NAME_MAP[nexus_das_log_name]

            # form das log
            nexus_das_log = DasLog(
                nexus_das_log_name,
                np.array([0.0]),
                np.array([log_value]),
                log_unit,
                None,
            )
            # add
            nexus_log_dict[nexus_das_log_name] = nexus_das_log

        return nexus_log_dict

    @abc.abstractmethod
    def get_pid_range(self, bank_id):
        """Set GPSANS bank and pixel ID relation

        Parameters
        ----------
        bank_id: int
            bank ID from 1 to 48

        Returns
        -------
        tuple
            start PID, end PID (assuming PID are consecutive in a bank and end PID is inclusive)

        """
        raise RuntimeError("Class method EventNexusConverter.get_pid_range is abstract")
