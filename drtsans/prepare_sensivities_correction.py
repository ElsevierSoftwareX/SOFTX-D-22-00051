import numpy as np
import h5py
from drtsans.load import load_events
import drtsans.mono.gpsans
import drtsans.mono.biosans
import drtsans.tof.eqsans

r"""
Links to mantid algorithms
https://docs.mantidproject.org/nightly/algorithms/SaveNexusProcessed-v1.html
https://docs.mantidproject.org/nightly/algorithms/MaskAngle-v1.html
https://docs.mantidproject.org/nightly/algorithms/Integration-v1.html
https://docs.mantidproject.org/nightly/algorithms/LoadEventNexus-v1.html
https://docs.mantidproject.org/nightly/algorithms/MaskDetectors-v1.html
https://docs.mantidproject.org/nightly/algorithms/CreateWorkspace-v1.html
"""
import os
from mantid.simpleapi import (
    SaveNexusProcessed,
    MaskAngle,
    Integration,
    MaskDetectors,
    CreateWorkspace,
)
from mantid.api import mtd
from mantid.kernel import logger

# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans%2Fmask_utils.py
from drtsans.mask_utils import circular_mask_from_beam_center, apply_mask

# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans%2Fprocess_uncertainties.py
from drtsans.process_uncertainties import set_init_uncertainties

from drtsans.sensitivity_correction_moving_detectors import (
    calculate_sensitivity_correction as calculate_sensitivity_correction_moving,
)
from drtsans.sensitivity_correction_patch import (
    calculate_sensitivity_correction as calculate_sensitivity_correction_patch,
)

# Constants
CG2 = "CG2"
CG3 = "CG3"
EQSANS = "EQSANS"
PIXEL = "Pixel"
MOVING_DETECTORS = "Moving Detectors"
PATCHING_DETECTORS = "Patching Detectors"

# As this script is a wrapper to handle script prepare_sensitivity
# (https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/scripts%2Fprepare_sensitivities.py),
# by which user just needs to set up instrument name but not is required to import the right modules
# (for example drtsans.mono.gpsans or drtsans.tof.eqsans),
# therefore it has to import the correct ones according instrument name in string.
# Using dictionary with instrument name as key is solution for it.

# prepare data  in .../api.py
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans%2Fmono%2Fgpsans%2Fapi.py
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans%2Fmono%2Fbiosans%2Fapi.py
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans%2Ftof%2Feqsans%2Fapi.py
PREPARE_DATA = {
    CG2: drtsans.mono.gpsans.api.prepare_data,
    CG3: drtsans.mono.biosans.api.prepare_data,
    EQSANS: drtsans.tof.eqsans.api.prepare_data,
}

# Find beam center in .../find_beam_center.py
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans%2Fmono%2Fbiosans%2Fbeam_finder.py
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans%2Fbeam_finder.py
FIND_BEAM_CENTER = {
    CG2: drtsans.mono.gpsans.find_beam_center,
    CG3: drtsans.mono.biosans.find_beam_center,
    EQSANS: drtsans.tof.eqsans.find_beam_center,
}

# Center detector in
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans%2Fmono%2Fbiosans%2Fbeam_finder.py
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans%2Fbeam_finder.py
CENTER_DETECTOR = {
    CG2: drtsans.mono.gpsans.center_detector,
    CG3: drtsans.mono.biosans.center_detector,
    EQSANS: drtsans.tof.eqsans.center_detector,
}

# Calculate transmission
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans%2Ftof%2Feqsans%2Ftransmission.py
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans%2Ftransmission.py
CALCULATE_TRANSMISSION = {
    CG2: drtsans.mono.gpsans.calculate_transmission,
    CG3: drtsans.mono.biosans.calculate_transmission,
    EQSANS: drtsans.tof.eqsans.calculate_transmission,
}

# Apply transmission correction
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans%2Ftof%2Feqsans%2Ftransmission.py
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans%2Ftransmission.py
APPLY_TRANSMISSION = {
    CG2: drtsans.mono.gpsans.apply_transmission_correction,
    CG3: drtsans.mono.biosans.apply_transmission_correction,
    EQSANS: drtsans.tof.eqsans.apply_transmission_correction,
}

# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans%2Fsolid_angle.py
SOLID_ANGLE_CORRECTION = {
    CG2: drtsans.mono.gpsans.solid_angle_correction,
    CG3: drtsans.mono.biosans.solid_angle_correction,
    EQSANS: drtsans.solid_angle_correction,
}

# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans%2Fsensitivity_correction_moving_detectors.py
# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans%2Fsensitivity_correction_patch.py
CALCULATE_SENSITIVITY_CORRECTION = {
    MOVING_DETECTORS: calculate_sensitivity_correction_moving,
    PATCHING_DETECTORS: calculate_sensitivity_correction_patch,
}


class PrepareSensitivityCorrection(object):
    """Workflow (container) class to prepare sensitivities correction file

    It tries to manage the various configuration and approaches for instrument scientists to prepare
    sensitivities file for EQSANS, GPSANS and BIOSANS.

    """

    def __init__(self, instrument, is_wing_detector=False):
        """Initialization

        Parameters
        ----------
        instrument : str
            instrument name, CG2, CG2, EQSANS
        is_wing_detector : bool
            Flag to calculate sensivitities for 'wing' detector special to BIOSANS/CG3
        """
        if instrument not in [CG2, CG3, EQSANS]:
            raise RuntimeError("Instrument {} is not supported".format(instrument))
        self._instrument = instrument

        # flood runs
        self._flood_runs = None
        # direct beam (center) runs
        self._direct_beam_center_runs = None
        # mask
        self._default_mask = None
        self._extra_mask_dict = dict()
        self._beam_center_radius = None  # mm

        # Transmission correction (BIOSANS)
        self._transmission_reference_runs = None
        self._transmission_flood_runs = None
        self._theta_dep_correction = False
        self._biosans_beam_trap_factor = 2

        # Dark current
        self._dark_current_runs = None

        # Pixel calibration default
        self._apply_calibration = False

        # Apply solid angle correction or not?
        self._solid_angle_correction = False

        # BIOSANS special
        # If the instrument is Bio-SANS, then check to see if we are working with the wing detector
        if self._instrument == CG3:
            self._is_wing_detector = is_wing_detector
        else:
            self._is_wing_detector = False

        # BIO-SANS special application to
        # mask the area around the direct beam to remove it and the associated parasitic scattering
        # Mask angles of wing detector pixels to be masked from beam center run.
        self._wing_det_mask_angle = None
        # Mask angles on main detector pixels to mask on beam center.
        self._main_det_mask_angle = None

    def set_pixel_calibration_flag(self, apply_calibration):
        """Set the flag to apply pixel calibrations.

        Parameters
        ----------
        apply_calibration : bool, str
            Flag for applying the pixel calibration. No calibration (False), Default (True), Calibration file (str)
        """
        self._apply_calibration = apply_calibration

    def set_solid_angle_correction_flag(self, apply_correction):
        """Set the flag to apply solid angle correction

        Parameters
        ----------
        apply_correction : bool
            Flag for applying

        Returns
        -------

        """
        self._solid_angle_correction = apply_correction

    def set_flood_runs(self, flood_runs):
        """Set flood run numbers

        Parameters
        ----------
        flood_runs : ~list or int or tuple
            list of run number as integers

        Returns
        -------
        None

        """
        # Process flood runs
        if isinstance(flood_runs, (int, str)):
            self._flood_runs = [flood_runs]
        else:
            self._flood_runs = list(flood_runs)

    def set_dark_current_runs(self, dark_current_runs):
        """Set dark current runs

        Parameters
        ----------
        dark_current_runs : ~list or ~tuple or int
            Dark current run(s)'s run number(s)

        Returns
        -------
        None

        """
        if dark_current_runs is None:
            self._dark_current_runs = None
        elif isinstance(dark_current_runs, (int, str)):
            self._dark_current_runs = [dark_current_runs]
        else:
            self._dark_current_runs = list(dark_current_runs)

    def set_direct_beam_runs(self, direct_beam_runs):
        """Set direct beam runs

        Parameters
        ----------
        direct_beam_runs : ~list or int or tuple

        Returns
        -------

        """
        if isinstance(direct_beam_runs, (int, str)):
            # defined as a single value
            self._direct_beam_center_runs = [direct_beam_runs]
        else:
            # shall be a list or tuple
            self._direct_beam_center_runs = list(direct_beam_runs)

    def set_masks(
        self, default_mask, pixels, wing_det_mask_angle=None, main_det_mask_angle=None
    ):
        """Set masks

        Parameters
        ----------
        default_mask : str or ~mantid.api.MaskWorkspace or :py:obj:`list` or None
            Mask to be applied. If :py:obj:`list`, it is a list of
            detector ID's. If `None`, it is expected that `maskbtp` is not empty.
            mask file name
        pixels : str or None
            pixels to mask.  Example: '1-8,249-256'
        wing_det_mask_angle : float or None
            angle to mask (MaskAngle) to (BIOSANS) wing detector
        main_det_mask_angle : float or None
            angle to mask (MaskAngle) to main detector

        Returns
        -------
        None

        """
        # default mask file or list of detector IDS or MaskWorkspace
        if default_mask is not None:
            self._default_mask = default_mask

        # pixels to mask
        if pixels is not None:
            self._extra_mask_dict[PIXEL] = pixels

        # angles to mask (BIOSANS)
        if self._instrument == CG3:
            if wing_det_mask_angle is not None:
                self._wing_det_mask_angle = wing_det_mask_angle
            if main_det_mask_angle is not None:
                self._main_det_mask_angle = main_det_mask_angle

    def set_beam_center_radius(self, radius):
        """Set beam center radius

        Parameters
        ----------
        radius : float
            radius in mm
        Returns
        -------

        """
        self._beam_center_radius = radius

    def set_transmission_correction(
        self, transmission_flood_runs, transmission_reference_runs, beam_trap_factor=2
    ):
        """Set transmission beam run and transmission flood runs

        Parameters
        ----------
        transmission_flood_runs : int or tuple or list
            transmission flood runs
        transmission_reference_runs : int or tuple or list
            transmission reference runs
        beam_trap_factor : float, int
            factor to beam trap size for masking angle

        Returns
        -------

        """

        def format_run_or_runs(run_s):
            """Format input run or runs to list of run or file names

            Parameters
            ----------
            run_s: int or str or ~list
                a run, a NeXus file name, or a list of runs

            Returns
            -------
            ~list

            """
            if isinstance(run_s, (int, str)):
                # an integer or string as run number
                run_list = [run_s]
            else:
                # a sequence, tuple or list
                run_list = list(run_s)

            return run_list

        # Only BIO SANS use transmission correction
        if self._instrument != CG3:
            return

        # transmission reference
        self._transmission_reference_runs = format_run_or_runs(
            transmission_reference_runs
        )

        # transmission flood
        self._transmission_flood_runs = format_run_or_runs(transmission_flood_runs)

        # if isinstance(transmission_reference_run, int):
        #     # a run number
        #     self._transmission_reference_runs = [transmission_reference_run]
        # else:
        #     self._transmission_reference_runs = list(transmission_reference_run)
        #
        # if isinstance(transmission_flood_runs, int):
        #     self._transmission_flood_runs = [transmission_flood_runs]
        # else:
        #     self._transmission_flood_runs = list(transmission_flood_runs)

        # Set the beam trap factor for transmission reference and flood run to mask angle
        self._biosans_beam_trap_factor = beam_trap_factor

    def set_theta_dependent_correction_flag(self, flag):
        """Set the flag to do theta dep with transmission correction

        Parameters
        ----------
        flag : bool
            True to do the correction

        Returns
        -------

        """
        self._theta_dep_correction = flag

    def _prepare_flood_data(
        self, flood_run, beam_center, dark_current_run, enforce_use_nexus_idf
    ):
        """Prepare flood data including
        (1) load
        (2) mask: default, pixels
        (3) center detector
        (4) optionally solid angle correction
        (5) optionally dark current correction
        (6) normalization

        Parameters
        ----------
        flood_run: int, str
            flood run number of flood file path
        beam_center
        dark_current_run
        enforce_use_nexus_idf: bool
            flag to enforce to use IDF XML in NeXus file; otherwise, it may use IDF from Mantid library
        Returns
        -------

        """

        # Prepare data
        # get right prepare_data method specified to instrument type
        prepare_data = PREPARE_DATA[self._instrument]

        instrument_specific_param_dict = dict()
        if self._instrument == CG3:
            instrument_specific_param_dict["center_y_wing"] = beam_center[2]
        if self._instrument in [CG2, CG3]:
            instrument_specific_param_dict["overwrite_instrument"] = False

        # Determine normalization method
        if self._instrument == EQSANS:
            # EQSANS requirs additional file with flux_method.  So set flux_method to None
            flux_method = None
        else:
            # BIOSANS and GPSANS does not require extra flux file for normalization by monitor
            flux_method = "monitor"

        # Determine dark current: None or INSTRUMENT_RUN
        if dark_current_run is not None:
            if isinstance(dark_current_run, str) and os.path.exists(dark_current_run):
                # dark current run (given) is a data file: do nothing
                pass
            else:
                # dark current is a run number either as an integer or a string cast from integer
                dark_current_run = "{}_{}".format(self._instrument, dark_current_run)

        # Load data with masking: returning to a list of workspace references
        # processing includes: load, mask, normalize by monitor
        if isinstance(flood_run, int):
            flood_run = "{}_{}".format(self._instrument, flood_run)
        else:
            # check file existence
            assert os.path.exists(flood_run)

        # prepare data
        if self._instrument in [CG2, CG3]:
            instrument_specific_param_dict[
                "enforce_use_nexus_idf"
            ] = enforce_use_nexus_idf
        flood_ws = prepare_data(
            data=flood_run,  # self._flood_runs[index]),
            pixel_calibration=self._apply_calibration,
            mask=self._default_mask,
            btp=self._extra_mask_dict,
            center_x=beam_center[0],
            center_y=beam_center[1],
            dark_current=dark_current_run,
            flux_method=flux_method,
            solid_angle=self._solid_angle_correction,
            **instrument_specific_param_dict,
        )

        # Integration all the wavelength for EQSANS
        if flood_ws.blocksize() != 1:
            # More than 1 bins in spectra: do integration to single bin
            # This is for EQSANS specially
            # output workspace name shall be unique and thus won't overwrite any existing one
            flood_ws = Integration(
                InputWorkspace=flood_ws, OutputWorkspace=str(flood_ws)
            )

        return flood_ws

    @staticmethod
    def _get_masked_detectors(workspace):
        """Get the detector masking information

        Parameters
        ----------
        workspace : ~mantid.api.MatrixWorkspace
            Workspace to get masked detectors' masking status

        Returns
        -------
        numpy.ndarray
            (N, 1) bool array, True for being masked

        """
        # The masked pixels after `set_uncertainties()` will have zero uncertainty.
        # Thus, it is an efficient way to identify them by check uncertainties (E) close to zero
        masked_array = workspace.extractE() < 1e-5

        return masked_array

    @staticmethod
    def _set_mask_value(
        flood_workspace, det_mask_array, use_moving_detector_method=True
    ):
        """Set masked pixels' values to NaN or -infinity according to mask type and sensitivity correction
        algorithm

        Parameters
        ----------
        flood_workspace :
            Flood data workspace space to have masked pixels' value set
        det_mask_array : numpy.ndarray
            Array to indicate pixel to be masked or not
        use_moving_detector_method : bool
            True for calculating sensitivities by moving detector algorithm;
            otherwise for detector patching algorithm

        Returns
        -------

        """
        # Complete mask array.  Flood workspace has been processed by set_uncertainties.  Therefore all the masked
        # pixels' uncertainties are zero, which is different from other pixels
        total_mask_array = flood_workspace.extractE() < 1e-6

        # Loop through each detector pixel to check its masking state to determine whether its value shall be
        # set to NaN, -infinity or not changed (i.e., for pixels without mask)
        num_spec = flood_workspace.getNumberHistograms()
        problematic_pixels = list()
        for i in range(num_spec):
            if total_mask_array[i][0] and use_moving_detector_method:
                # Moving detector algorithm.  Any masked detector pixel is set to NaN
                flood_workspace.dataY(i)[0] = np.nan
                flood_workspace.dataE(i)[0] = np.nan
            elif (
                total_mask_array[i][0]
                and not use_moving_detector_method
                and det_mask_array[i][0]
            ):
                # Patch detector method: Masked as the bad pixels and thus set to NaN
                flood_workspace.dataY(i)[0] = np.nan
                flood_workspace.dataE(i)[0] = np.nan
            elif total_mask_array[i][0]:
                # Patch detector method: Pixels that have not been masked as bad pixels, but have been
                # identified as needing to have values set by the patch applied. To identify them, the
                # value is set to -INF.
                flood_workspace.dataY(i)[0] = np.NINF
                flood_workspace.dataE(i)[0] = np.NINF
            elif (
                not total_mask_array[i][0]
                and not use_moving_detector_method
                and det_mask_array[i][0]
            ):
                # Logic error: impossible case
                problematic_pixels.append(i)
        # END-FOR

        # Array
        if len(problematic_pixels) > 0:
            raise RuntimeError(
                f"Impossible case: pixels {problematic_pixels} has local detector mask is on, "
                f"but total mask is off"
            )

        logger.debug(
            "Patch detector method: Pixels that have not been masked as bad pixels, but have been"
            "identified as needing to have values set by the patch applied. To identify them, the"
            "value is set to -INF."
        )
        logger.notice(
            "Number of infinities = {}".format(
                len(np.where(np.isinf(flood_workspace.extractY()))[0])
            )
        )

        logger.debug(
            "Moving/Patch detector algorithm.  Any masked detector pixel is set to NaN"
        )
        logger.notice(
            "Number of NaNs       = {}".format(
                len(np.where(np.isnan(flood_workspace.extractY()))[0])
            )
        )

        return flood_workspace

    def execute(
        self,
        use_moving_detector_method,
        min_threshold,
        max_threshold,
        output_nexus_name,
        enforce_use_nexus_idf=False,
        debug_mode=False,
    ):
        """Main workflow method to calculate sensitivities correction

        Parameters
        ----------
        use_moving_detector_method : bool
            Flag to use 'moving detectors' method; Otherwise, use 'detector patch' method
        min_threshold : float
            minimum threshold of normalized count for GOOD pixels
        max_threshold : float
            maximum threshold of normalized count for GOOD pixels
        output_nexus_name : str
            path to the output processed NeXus file
        enforce_use_nexus_idf: bool
            flag to enforce to use IDF XML in NeXus file; otherwise, it may use IDF from Mantid library
        debug_mode: bool
            flag for debugging mode

        Returns
        -------
        None

        """
        # Number of pair of workspaces to process
        num_workspaces_set = len(self._flood_runs)

        # Load beam center runs and calculate beam centers
        beam_centers = list()
        for i in range(num_workspaces_set):
            beam_center_i = self._calculate_beam_center(i, enforce_use_nexus_idf)
            beam_centers.append(beam_center_i)
            logger.notice(
                "Calculated beam center ({}-th) = {}".format(i, beam_center_i)
            )

        # Set default value to dark current runs
        if self._dark_current_runs is None:
            self._dark_current_runs = [None] * num_workspaces_set

        # Load and process flood data with (1) mask (2) center detector and (3) solid angle correction
        flood_workspaces = list()
        for i in range(num_workspaces_set):
            flood_ws_i = self._prepare_flood_data(
                self._flood_runs[i],
                beam_centers[i],
                self._dark_current_runs[i],
                enforce_use_nexus_idf,
            )
            logger.notice(
                f"Load {i}-th flood run {self._flood_runs[i]} to " f"{flood_ws_i}"
            )
            flood_workspaces.append(flood_ws_i)

        # Retrieve masked detectors
        if not use_moving_detector_method:
            bad_pixels_list = list()
            for i in range(num_workspaces_set):
                bad_pixels_list.append(self._get_masked_detectors(flood_workspaces[i]))
        else:
            bad_pixels_list = [None] * num_workspaces_set

        # Mask beam centers
        for i in range(num_workspaces_set):
            flood_workspaces[i] = self._mask_beam_center(
                flood_workspaces[i], beam_centers[i]
            )

        # Transmission correction as an option
        if self._instrument == CG3 and self._transmission_reference_runs is not None:
            # Must have transmission run specified and cannot be wing detector (of CG3)
            for i in range(num_workspaces_set):
                flood_workspaces[i] = self._apply_transmission_correction(
                    flood_workspaces[i],
                    self._transmission_reference_runs[i],
                    self._transmission_flood_runs[i],
                    beam_centers[i],
                    enforce_use_nexus_idf,
                )

        # Set the masked pixels' counts to nan and -infinity
        for i in range(num_workspaces_set):
            flood_workspaces[i] = self._set_mask_value(
                flood_workspaces[i], bad_pixels_list[i], use_moving_detector_method
            )

        info = "Preparation of data is over.\n"
        for fws in flood_workspaces:
            info += (
                f"{str(fws)}: Number of infinities = {len(np.where(np.isinf(fws.extractY()))[0])},"
                f"Number of NaNs = {len(np.where(np.isnan(fws.extractY()))[0])}\n"
            )
        logger.notice(info)

        # Debug output
        if debug_mode:
            for flood_ws in flood_workspaces:
                SaveNexusProcessed(
                    InputWorkspace=flood_ws, Filename=f"{str(flood_ws)}_flood.nxs"
                )

        # Decide algorithm to prepare sensitivities
        if self._instrument in [CG2, CG3] and use_moving_detector_method is True:
            if debug_mode:
                # nan.sum all the input flood runs to check the coverage of summed counts
                self.sum_input_runs(flood_workspaces)

            # Prepare by using moving detector algorithm
            calculate_sensitivity_correction = CALCULATE_SENSITIVITY_CORRECTION[
                MOVING_DETECTORS
            ]

            # Calculate sensitivities for each file
            sens_ws = calculate_sensitivity_correction(
                flood_workspaces,
                threshold_min=min_threshold,
                threshold_max=max_threshold,
            )

        else:
            # Prepare by using the sensitivity patch method for a single detector (image)
            # Such as GPSANS, BIOSANS Main detector, BIOSANS wing detector, EQSANS
            calculate_sensitivity_correction = CALCULATE_SENSITIVITY_CORRECTION[
                PATCHING_DETECTORS
            ]

            # Default polynomial order: CG3 uses order 3.  Others use order 2.
            if self._instrument == CG3:
                polynomial_order = 3
            else:
                polynomial_order = 2

            # component name
            if self._is_wing_detector:
                detector = "wing_detector"
            else:
                detector = "detector1"

            # This only processes a single image, even for the Bio-SANS.
            # Each detector on the Bio-SANS must be treated independently
            sens_ws = calculate_sensitivity_correction(
                flood_workspaces[0],
                min_threshold=min_threshold,
                max_threshold=max_threshold,
                poly_order=polynomial_order,
                min_detectors_per_tube=50,
                component_name=detector,
            )

        # Export
        self._export_sensitivity(sens_ws, output_nexus_name, self._flood_runs[0])

    def _export_sensitivity(self, sensitivity_ws, output_nexus_name, parent_flood_run):
        """Process and export sensitivities to a processed NeXus file

        Parameters
        ----------
        sensitivity_ws :  ~mantid.api.MatrixWorkspace
            MatrixWorkspace containing sensitivity and error
        output_nexus_name : str
            Output NeXus file  name
        parent_flood_run : int
            Flood run number to create parent workspace for sensitivity workspace

        Returns
        -------
        None

        """
        # Create a new workspace for output
        instrument_name = {CG2: "GPSANS", CG3: "BIOSANS", EQSANS: "EQSANS_"}[
            self._instrument
        ]
        if isinstance(parent_flood_run, int):
            event_nexus = "{}{}".format(instrument_name, parent_flood_run)
        else:
            # must be a nexus file already
            event_nexus = parent_flood_run
            assert os.path.exists(event_nexus)

        parent_ws = load_events(run=event_nexus, MetaDataOnly=True)

        # Create new sensitivity workspace
        new_sens_name = "{}_new".format(str(sensitivity_ws))
        new_sensitivity_ws = CreateWorkspace(
            DataX=sensitivity_ws.extractX().flatten(),
            DataY=sensitivity_ws.extractY().flatten(),
            DataE=sensitivity_ws.extractE().flatten(),
            NSpec=parent_ws.getNumberHistograms(),
            ParentWorkspace=parent_ws,
            OutputWorkspace=new_sens_name,
        )

        # Mask detectors
        mask_ws_indexes = list()
        for iws in range(new_sensitivity_ws.getNumberHistograms()):
            # get the workspace with -infinity or NaN for masking
            if np.isnan(new_sensitivity_ws.readY(iws)[0]) or np.isinf(
                new_sensitivity_ws.readY(iws)[0]
            ):
                mask_ws_indexes.append(iws)
        MaskDetectors(Workspace=new_sensitivity_ws, WorkspaceIndexList=mask_ws_indexes)

        # Set all the mask values to NaN
        new_sensitivity_ws = mtd[new_sens_name]
        for iws in mask_ws_indexes:
            new_sensitivity_ws.dataY(iws)[0] = np.nan

        # Save
        SaveNexusProcessed(
            InputWorkspace=new_sensitivity_ws, Filename=output_nexus_name
        )

    def _calculate_beam_center(self, index, enforce_use_nexus_idf):
        """Find beam centers for all flood runs

        Beam center run shall be
        (1) masked properly (default mask + top/bottom)
        (2) NOT corrected by solid angle

        Parameters
        ----------
        index : int
            beam center run index mapped to flood run
        enforce_use_nexus_idf: bool
            flag to enforce to use IDF XML in NeXus file; otherwise, it may use IDF from Mantid library
        Returns
        -------
        ~tuple
            beam center as xc, yc and possible wc for BIOSANS

        """
        if (
            self._direct_beam_center_runs is None
            and self._instrument == CG3
            and self._wing_det_mask_angle is not None
        ):
            # CG3, flood run as direct beam center and mask angle is defined
            # In this case, run shall be loaded to another workspace for masking
            beam_center_run = self._flood_runs[index]
        elif self._direct_beam_center_runs is None:
            raise RuntimeError(
                "Beam center runs must be given for {}".format(self._instrument)
            )
        else:
            # Direct beam run is specified
            beam_center_run = self._direct_beam_center_runs[index]
        if isinstance(beam_center_run, str):
            # beam center run shall be a file path
            assert os.path.exists(
                beam_center_run
            ), f"Bean center run {beam_center_run} cannot be found"
        else:
            # run number (integer)
            beam_center_run = "{}_{}".format(self._instrument, beam_center_run)

        # Prepare data
        # Only applied for BIOSANS with mask_angle case!!! and GPSANS moving detector
        # It is not necessary for EQSANS because data won't be modified at all!
        prepare_data = PREPARE_DATA[self._instrument]

        # Add instrument_specific_parameters
        instrument_specific_param_dict = dict()
        if self._instrument in [CG2, CG3]:
            # HFIR spedific
            instrument_specific_param_dict["overwrite_instrument"] = False

        # Determine normalization method
        if self._instrument == EQSANS:
            # EQSANS requirs additional file with flux_method.  So set flux_method to None
            flux_method = None
        else:
            # BIOSANS and GPSANS does not require extra flux file for normalization by monitor
            flux_method = "monitor"

        # FIXME - data shall be more flexible here for beam center run path
        if self._instrument in [CG2, CG3]:
            instrument_specific_param_dict[
                "enforce_use_nexus_idf"
            ] = enforce_use_nexus_idf
        beam_center_workspace = prepare_data(
            data=beam_center_run,
            pixel_calibration=self._apply_calibration,
            center_x=0.0,  # force to not to center
            center_y=0.0,
            mask=self._default_mask,
            btp=self._extra_mask_dict,
            flux_method=flux_method,
            solid_angle=False,
            output_workspace="BC_{}_{}".format(self._instrument, beam_center_run),
            **instrument_specific_param_dict,
        )
        # Mask angle for CG3: apply mask on angle
        if self._instrument == CG3 and self._wing_det_mask_angle is not None:
            # mask wing detector
            apply_mask(beam_center_workspace, Components="wing_detector")
            # mask 2-theta angle on main detector
            MaskAngle(
                Workspace=beam_center_workspace,
                MinAngle=self._wing_det_mask_angle,
                Angle="TwoTheta",
            )

        # Find detector center
        find_beam_center = FIND_BEAM_CENTER[self._instrument]
        beam_center = find_beam_center(beam_center_workspace)

        return beam_center[:-1]

    def _mask_beam_center(self, flood_ws, beam_center):
        """Mask beam center

        Mask beam center with 3 algorithms
        1. if beam center mask is present, mask by file
        2. Otherwise if beam center workspace is specified, find beam center from this workspace and mask
        3. Otherwise find beam center for flood workspace and mask itself

        Parameters
        ----------
        flood_ws : ~mantid.api.MatrixWorkspace
            Mantid workspace for flood data
        beam_center : tuple or str
            if tuple, beam centers (xc, yc, wc) / (xc, yc); str: beam center masks file
        Returns
        -------

        """
        # Calculate masking (masked file or detectors)
        if isinstance(beam_center, str):
            # beam center mask XML file: apply mask
            apply_mask(
                flood_ws, mask=beam_center
            )  # data_ws reference shall not be invalidated here!
        elif self._main_det_mask_angle is not None and self._instrument == CG3:
            # CG3 special: Mask 2-theta angle
            # Mask wing detector right top/bottom corners
            if self._is_wing_detector is False:
                # main detector: mask wing
                component = "wing_detector"
            else:
                # wing detector: mask main
                component = "detector1"
            apply_mask(flood_ws, Components=component)
            # mask 2theta
            MaskAngle(
                Workspace=flood_ws, MaxAngle=self._main_det_mask_angle, Angle="TwoTheta"
            )
        else:
            # calculate beam center mask from beam center workspace
            # Mask the new beam center by 65 mm (Lisa's magic number)
            masking = list(
                circular_mask_from_beam_center(flood_ws, self._beam_center_radius)
            )
            # Mask
            apply_mask(
                flood_ws, mask=masking
            )  # data_ws reference shall not be invalidated here!

        # Set uncertainties
        # output: masked are zero intensity and zero error
        masked_flood_ws = set_init_uncertainties(flood_ws)

        return masked_flood_ws

    def _apply_transmission_correction(
        self,
        flood_ws,
        transmission_beam_run,
        transmission_flood_run,
        beam_center,
        enforce_use_nexus_idf,
    ):
        """Calculate and pply transmission correction

        Parameters
        ----------
        flood_ws : MarixWorkspace
            Flood run workspace to transmission correct workspace
        transmission_beam_run : int or str
            run number for transmission beam run
        transmission_flood_run : int or str
            run number for transmission flood run
        beam_center : ~tuple
            detector center
        enforce_use_nexus_idf: bool
            flag to enforce to use IDF XML in NeXus file; otherwise, it may use IDF from Mantid library
        Returns
        -------
        MatrixWorkspace
            Flood workspace with transmission corrected

        """
        prepare_data = PREPARE_DATA[self._instrument]

        instrument_specific_param_dict = dict()
        if self._instrument == CG3:
            instrument_specific_param_dict["center_y_wing"] = beam_center[2]
        if self._instrument in [CG2, CG3]:
            # HFIR specific
            instrument_specific_param_dict["overwrite_instrument"] = False

        # Load, mask default and pixels, and normalize
        if self._instrument in [CG2, CG3]:
            instrument_specific_param_dict[
                "enforce_use_nexus_idf"
            ] = enforce_use_nexus_idf

        if isinstance(transmission_beam_run, str) and os.path.exists(
            transmission_beam_run
        ):
            sans_data = transmission_beam_run
        elif (
            isinstance(transmission_beam_run, str)
            and transmission_beam_run.isdigit()
            or isinstance(transmission_beam_run, int)
        ):
            sans_data = "{}_{}".format(self._instrument, transmission_beam_run)
        else:
            raise TypeError(
                f"Transmission run {transmission_beam_run} of type {type(transmission_beam_run)} "
                f"is not supported to load a NeXus run from it"
            )

        transmission_workspace = prepare_data(
            data=sans_data,
            pixel_calibration=self._apply_calibration,
            mask=self._default_mask,
            btp=self._extra_mask_dict,
            flux_method="monitor",
            solid_angle=False,
            center_x=beam_center[0],
            center_y=beam_center[1],
            output_workspace="TRANS_{}_{}".format(
                self._instrument, transmission_beam_run
            ),
            **instrument_specific_param_dict,
        )
        # Apply mask
        if self._instrument == CG3:
            apply_mask(transmission_workspace, Components="wing_detector")
            MaskAngle(
                Workspace=transmission_workspace,
                MinAngle=self._biosans_beam_trap_factor * self._main_det_mask_angle,
                Angle="TwoTheta",
            )

        # Load, mask default and pixels, normalize transmission flood run
        if self._instrument in [CG2, CG3]:
            instrument_specific_param_dict[
                "enforce_use_nexus_idf"
            ] = enforce_use_nexus_idf

        if not os.path.exists(transmission_flood_run):
            # given run number: form to CG3_XXX
            mtd_trans_run = "{}_{}".format(self._instrument, transmission_flood_run)
        else:
            # already a file path
            mtd_trans_run = transmission_flood_run
        transmission_flood_ws = prepare_data(
            data=mtd_trans_run,
            pixel_calibration=self._apply_calibration,
            mask=self._default_mask,
            btp=self._extra_mask_dict,
            flux_method="monitor",
            solid_angle=False,
            center_x=beam_center[0],
            center_y=beam_center[1],
            output_workspace="TRANS_{}_{}".format(
                self._instrument, transmission_flood_run
            ),
            **instrument_specific_param_dict,
        )
        # Apply mask
        if self._instrument == CG3:
            apply_mask(transmission_flood_ws, Components="wing_detector")
            MaskAngle(
                Workspace=transmission_flood_ws,
                MinAngle=self._biosans_beam_trap_factor * self._main_det_mask_angle,
                Angle="TwoTheta",
            )
        elif self._instrument == EQSANS:
            raise RuntimeError("Never tested EQSANS with Transmission correction")

        # Zero-Angle Transmission Co-efficients
        calculate_transmission = CALCULATE_TRANSMISSION[self._instrument]
        transmission_corr_ws = calculate_transmission(
            transmission_flood_ws, transmission_workspace
        )
        average_zero_angle = np.mean(transmission_corr_ws.readY(0))
        average_zero_angle_error = np.linalg.norm(transmission_corr_ws.readE(0))
        logger.notice(
            f"Transmission Coefficient is {average_zero_angle:.3f} +/- "
            f"{average_zero_angle_error:.3f}."
            f"Transmission flood {str(transmission_flood_ws)} and "
            f"transmission {str(transmission_workspace)}"
        )

        # Apply calculated transmission
        apply_transmission_correction = APPLY_TRANSMISSION[self._instrument]
        flood_ws = apply_transmission_correction(
            flood_ws,
            trans_workspace=transmission_corr_ws,
            theta_dependent=self._theta_dep_correction,
        )

        return flood_ws

    @staticmethod
    def sum_input_runs(flood_workspaces):
        """Do NaN sum to all input flood workspaces

        Parameters
        ----------
        flood_workspaces

        Returns
        -------

        """
        from mantid.simpleapi import CloneWorkspace

        # Do NaN sum
        y_list = [ws.extractY() for ws in flood_workspaces]
        y_matrix = np.array(y_list)

        nan_sum_matrix = np.nansum(y_matrix, axis=0)

        # clone a workspace
        cloned = CloneWorkspace(
            InputWorkspace=flood_workspaces[0], OutputWorkspace="FloodSum"
        )

        for iws in range(cloned.getNumberHistograms()):
            cloned.dataY(iws)[0] = nan_sum_matrix[iws][0]

        # output
        SaveNexusProcessed(InputWorkspace=cloned, Filename="SummedFlood.nxs")


def debug_output(workspace, output_file):
    """Exporting a workspace to NeXus file and HDF5 for debugging purpose

    Parameters
    ----------
    workspace : numpy.ndarray
        data to plot
    output_file : str
        output file name as reference

    Returns
    -------
    None

    """
    # Save Nexus
    SaveNexusProcessed(InputWorkspace=workspace, Filename=output_file)

    data = workspace.extractY()
    data_error = workspace.extractE()

    # Export sensitivities calculated in file for quick review
    # Export to hdf5 for a quick review
    sens_h5 = h5py.File("{}.h5".format(output_file.split(".")[0]), "w")
    sens_group = sens_h5.create_group("Data")
    sens_group.create_dataset("Data", data=data)
    if data_error is not None:
        sens_group.create_dataset("Data error", data=data_error)
    sens_h5.close()
