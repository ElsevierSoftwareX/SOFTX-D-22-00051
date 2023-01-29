from drtsans.mono.convert_xml_to_nexus import EventNexusConverter
import os
import numpy as np


class CG2EventNexusConvert(EventNexusConverter):
    """
    SPICE to NeXus converter
    """

    def __init__(self):
        """
        initialization

        work for 48 banks
        """
        super(CG2EventNexusConvert, self).__init__("CG2", "CG2", 48)

    def _map_detector_and_counts(self):
        # map to bank
        for bank_id in range(1, self._num_banks + 1):
            # create TofHistogram instance
            start_pid, end_pid = self.get_pid_range(bank_id)
            pix_ids = np.arange(start_pid, end_pid + 1)
            counts = self._spice_detector_counts[start_pid : end_pid + 1]

            self._bank_pid_dict[bank_id] = pix_ids
            self._bank_counts_dict[bank_id] = counts

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
        # Check input valid
        if bank_id < 1 or bank_id > 48:
            raise RuntimeError(
                f"CG2 (GP-SANS) has 88 banks indexed from 1 to 48. "
                f"Bank {bank_id} is out of range."
            )

        # calculate starting PID
        if bank_id <= 24:
            # from 1 to 24: front panel
            start_pid = (bank_id - 1) * 2 * 1024
        else:
            # from 25 to 48: back panel
            start_pid = ((bank_id - 25) * 2 + 1) * 1024

        # calculate end PID
        end_pid = start_pid + 1023

        return start_pid, end_pid


def convert_spice_to_nexus(
    ipts_number,
    exp_number,
    scan_number,
    pt_number,
    template_nexus,
    output_dir=None,
    spice_dir=None,
):
    """Convert one SPICE to NeXus

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
    template_nexus: str
        path to a GPSANS nED event Nexus file especially for IDF
    output_dir: None or str
        output directory of the converted data
    spice_dir: None or str
        data file directory for SPICE file.  None using default

    Returns
    -------
    str
        generated event Nexus file

    """
    # Set the SPICE dir
    if spice_dir is None:
        # Build the default path to the SPICE files
        spice_dir = f"/HFIR/CG2/IPTS-{ipts_number}/exp{exp_number}/Datafiles"
    print(f"[INFO] SPICE file will be read from to directory {spice_dir}")
    # verify path
    assert os.path.exists(
        spice_dir
    ), f"SPICE data directory {spice_dir} cannot be found"
    spice_data_file = os.path.join(
        spice_dir, f"CG2_exp{exp_number}_scan{scan_number:04}_{pt_number:04}.xml"
    )
    assert os.path.exists(
        spice_data_file
    ), f"SPICE file {spice_data_file} cannot be located"

    # Template Nexus file
    template_nexus_file = template_nexus
    assert os.path.exists(
        template_nexus_file
    ), f"Template NeXus file {template_nexus_file} cannot be located"

    # Specify the default output directory
    if output_dir is None:
        output_dir = f"/HFIR/CG2/IPTS-{ipts_number}/shared/spice_nexus/Exp{exp_number}"
    if not os.path.exists(output_dir):
        raise RuntimeError(
            f"Output NeXus directory {output_dir} does not exist."
            f"Create directory {output_dir} and grand access to all IPTS users"
        )

    # output file name
    out_nexus_file = f"CG2_{exp_number:04}{scan_number:04}{pt_number:04}.nxs.h5"
    out_nexus_file = os.path.join(output_dir, out_nexus_file)
    print(f"[INFO] NeXus file will be written to {out_nexus_file}")

    # Load meta data and convert to NeXus format
    das_log_map = {
        "CG2:CS:SampleToSi": ("sample_to_flange", "mm"),  # same
        "sample_detector_distance": ("sdd", "m"),  # same
        "wavelength": ("lambda", "angstroms"),  # angstroms -> A
        "wavelength_spread": ("dlambda", "fraction"),  # fraction -> None
        "source_aperture_diameter": ("source_aperture_size", "mm"),  # same
        "sample_aperture_diameter": ("sample_aperture_size", "mm"),  # same
        "detector_trans_Readback": ("detector_trans", "mm"),  # same
        "source_distance": (
            "source_distance",
            "m",
        ),  # same. source-aperture-sample-aperture
        "beamtrap_diameter": ("beamtrap_diameter", "mm"),  # not there
        "dcal_Readback": ("dcal", "mm"),  # required by pixel calibration
        "attenuator": ("attenuator_pos", "mm"),  # special
    }

    # init converter
    converter = CG2EventNexusConvert()
    # load instrument definition (IDF)
    converter.load_idf(template_nexus_file)
    # load SPICE (xml file)
    converter.load_sans_xml(spice_data_file, das_log_map)
    # generate event nexus
    converter.generate_event_nexus(out_nexus_file)

    return out_nexus_file
