#!/usr/bin/env python

import os
import yaml
import numpy as np
from typing import List, Union
from drtsans.mono.convert_xml_to_nexus import EventNeXusWriter
from drtsans.files.event_nexus_rw import parse_event_nexus
from drtsans.mono.convert_xml_to_nexus import EventNexusConverter


class CG3EventNexusConvert(EventNexusConverter):
    """
    Converting legacy SPICE file for BIOSANS to Event NeXus
    """

    def __init__(self):
        """
        initialization

        work for 88 banks
        """
        super(CG3EventNexusConvert, self).__init__("CG3", "CG3", self.num_banks)

    @property
    def num_banks(self):
        """For BioSANS (CG3), the total number of banks is a fixed value: 88"""
        return 88

    def _map_detector_and_counts(self):
        """Map detector counts and pixel IDs from SPICE-era IDF to NeXus-era IDF

        SPICE: pixel ID is consecutive from lower left corner of detector, up to the top and then from the next tube
        as

        255  511   ...
        .    ...   ...
        .    ...   ...
        1    257   ...
        0    256   512 ....
        ----------------------------------------
        tube0   tube1   tube2   ...

        NeXus: in each bank, the pixel IDs start with 4 front tubes and then 4 back tubes as

        255  1279   ...
        .    ...    ...
        .    ...    ...
        1    1025   257
        0    1024   256  ....
        ---------------------------
        bank1  bank 25   bank 2  bank 26 ......

        and the front 4 tubes are in one bank and the back 4 tubes are in another bank

        Therefore, the algorithm shall re-assign the counts to tubes
        """
        # map SPICE tube to NeXus bank/tube
        # TODO - make these into constants and do a sanity check for any data
        num_pixel_per_tube = 256
        num_main_8packs = 192 // 8
        num_wing_8packs = 160 // 8
        # sanity_check()

        # initialize the output
        for nexus_bank_id in range(1, 1 + self._num_banks):
            # NeXus PID range for each bank
            start_pid, end_pid = self.get_pid_range(nexus_bank_id)
            # assign to tubes: pixel ID shall be ordered according to SPICE workspace indexes
            pix_ids = np.arange(start_pid, end_pid + 1)
            # initialize dictionary items for PID and counts
            self._bank_pid_dict[nexus_bank_id] = pix_ids
            self._bank_counts_dict[nexus_bank_id] = np.zeros_like(pix_ids)

        # map from SPICE tubes to Nexus bank/tube
        for tube_group in range(num_main_8packs + num_wing_8packs):
            # each 8 pack/tube group has 2 banks: bank shift is for the front bank in the 8 pack's shift from
            # first bank
            if tube_group < num_main_8packs:
                group_bank_shift = tube_group
            else:
                group_bank_shift = tube_group + num_main_8packs

            for tube_index in range(8):
                # event tube: front panel
                # odd tube: back panel shift another half detector (i.e., 1/2 banks in detector or number of 8 packs)
                tube_bank_shift = tube_index % 2

                # consider main and wing
                if tube_group < num_main_8packs:
                    # main
                    bank_id = group_bank_shift + tube_bank_shift * num_main_8packs
                else:
                    # wing
                    bank_id = group_bank_shift + tube_bank_shift * num_wing_8packs
                bank_id += 1  # Nexus bank ID starts from 1

                # spice tube index
                spice_tube_index = tube_group * 8 + tube_index
                bank_tube_index = tube_index // 2

                # map counts to
                spice_count_start_index = spice_tube_index * num_pixel_per_tube
                bank_count_start_index = bank_tube_index * num_pixel_per_tube
                self._bank_counts_dict[bank_id][
                    bank_count_start_index : bank_count_start_index + num_pixel_per_tube
                ] = self._spice_detector_counts[
                    spice_count_start_index : spice_count_start_index
                    + num_pixel_per_tube
                ]

    def get_pid_range(self, bank_id):
        """Set GPSANS bank and pixel ID relation

        Parameters
        ----------
        bank_id: int
            bank ID from 1 to 88

        Returns
        -------
        tuple
            start PID, end PID (assuming PID are consecutive in a bank and end PID is inclusive)

        """
        # NOTE:
        # For legacy data, the hardware configuration is fixed, therefore it is hardcoded
        # in this method.  DO NOT TOUCH!!!

        # Check input valid
        if bank_id < 1 or bank_id > self.num_banks:
            raise RuntimeError(
                f"CG3 (BioSANS) has 88 banks indexed from 1 to 88. "
                f"Bank {bank_id} is out of range."
            )

        # calculate starting PID
        if bank_id <= 24:
            # from 1 to 24: front panel
            start_pid = (bank_id - 1) * 2 * 1024
        elif bank_id <= 48:
            # from 25 to 48: back panel
            start_pid = ((bank_id - 25) * 2 + 1) * 1024
        elif bank_id <= 68:
            # from 49 to 68: even bank from 49152 (main detector pixel number)
            start_pid = (bank_id - 49) * 2 * 1024 + 48 * 1024
        else:
            # from 69 to 88
            start_pid = ((bank_id - 69) * 2 + 1) * 1024 + 48 * 1024

        # calculate end PID
        end_pid = start_pid + 1023

        return start_pid, end_pid


def convert_spice_to_nexus(
    ipts_number: int,
    exp_number: int,
    scan_number: int,
    pt_number: int,
    template_nexus: str,
    masked_detector_pixels: List[int] = list(),
    output_dir: str = None,
    spice_dir: str = None,
    spice_data: str = Union[None, str],
):
    """
    Convert legacy SPICE file for bioSANS/cg3 to Event NeXus

    Parameters
    ----------
    ipts_number: int
        IPTS
    exp_number: int
        experiment number
    scan_number: int
        scan
    pt_number: int
        pt
    masked_detector_pixels: ~list
        List of pixels (mantid workspace indexes) to mask
    template_nexus: str
        path to a GPSANS nED event Nexus file especially for IDF
    output_dir: None or str
        output directory of the converted data
    spice_dir: None or str
        data file directory for SPICE file.  None using default
    spice_data: None or str
        full data file.  It is specified, there is no need to construct SPICE file name anymore

    Returns
    -------
    str
        generated event Nexus file

    """
    if spice_dir is not None:
        # construct SPICE file from IPTS, experiment and etc.
        # path processing
        spice_dir = (
            f"/HFIR/CG3/IPTS-{ipts_number}/exp{exp_number}/Datafiles"
            if spice_dir is None
            else spice_dir
        )

        # construct  SPICE
        spice_data = f"BioSANS_exp{exp_number}_scan{scan_number:04}_{pt_number:04}.xml"
        spice_data = os.path.join(spice_dir, spice_data)
        output_dir = (
            f"/HFIR/CG3/IPTS-{ipts_number}/shared/spice_nexus"
            if output_dir is None
            else output_dir
        )

        # Input (Path&File) validation
        assert os.path.exists(
            spice_dir
        ), f"SPICE data directory {spice_dir} cannot be found"

    # check SPICE file
    assert os.path.exists(spice_data), f"SPICE file {spice_data} cannot be located"
    assert os.path.exists(
        template_nexus
    ), f"Template NeXus file {template_nexus} cannot be located"

    # Check output directory exist.  If not, create it
    if not os.path.exists(output_dir):
        try:
            os.mkdir(output_dir)
        except (OSError, IOError) as dir_err:
            raise RuntimeError(
                f"Output directory {output_dir} doesn't exist."
                f"Unable to create {output_dir} due to {dir_err}"
            )

    # output file name
    out_nexus_file = f"CG3_{exp_number:04}{scan_number:04}{pt_number:04}.nxs.h5"
    out_nexus_file = os.path.join(output_dir, out_nexus_file)

    # load mapping reference from yaml
    _file_parent_dir = os.path.dirname(os.path.realpath(__file__))
    with open(
        os.path.join(_file_parent_dir, "cg3_to_nexus_mapping.yml"), "r"
    ) as stream:
        das_log_map = yaml.safe_load(stream)

    # init converter
    converter = CG3EventNexusConvert()
    # load instrument definition (IDF)
    converter.load_idf(template_nexus)
    # load SPICE (xml file)
    converter.load_sans_xml(spice_data, das_log_map)
    # mask detector
    converter.mask_spice_detector_pixels(masked_detector_pixels)
    # generate event nexus
    converter.generate_event_nexus(out_nexus_file)

    return out_nexus_file


# note: the following function is a legacy function
def generate_event_nexus(source_nexus, target_nexus, das_log_list=None):
    """Generate event NeXus properly from a source Nexus file

    This method will be migrated to drtsans.mono.biaosans

    Parameters
    ----------
    source_nexus: str
        source nexus file
    target_nexus: str
        target nexus file
    das_log_list: ~list
        list of DAS logs

    Returns
    -------

    """
    cg3_num_banks = 88

    DAS_LOGs = [
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

    das_log_list = DAS_LOGs if das_log_list is None else das_log_list

    # Import essential experimental data from source event nexus file
    nexus_contents = parse_event_nexus(source_nexus, 88, das_log_list)
    # Generate event nexus writer
    event_nexus_writer = EventNeXusWriter(beam_line="CG3", instrument_name="CG3")

    # set instrument: 88 banks (2 detectors)
    event_nexus_writer.set_instrument_info(cg3_num_banks, nexus_contents[0])

    # set counts: 88 banks (2 detectors)
    for bank_id in range(1, cg3_num_banks + 1):
        event_nexus_writer.set_bank_histogram(bank_id, nexus_contents[1][bank_id])

    # set meta
    for das_log in nexus_contents[5].values():
        event_nexus_writer.set_meta_data(das_log)

    # time
    start_time = nexus_contents[3]
    end_time = nexus_contents[4]

    # Write file
    event_nexus_writer.generate_event_nexus(
        target_nexus, start_time, end_time, nexus_contents[2]
    )


if __name__ == "__main__":
    print("converting Legacy cg3 SPICE file to Event Nexus")
    print("e.g.")
    print("cg3_spice_to_nexus.py SPICE_FILE_TO_CONVERT")
