from typing import Union
from drtsans.mono.spice_data import SpiceRun
from drtsans.prepare_sensivities_correction import PrepareSensitivityCorrection


def prepare_spice_sensitivities_correction(
    is_wing_detector: bool,
    flood_run: SpiceRun,
    direct_beam_run: SpiceRun,
    dark_current_run: SpiceRun,
    apply_solid_angle_correction: bool,
    transmission_flood_run: SpiceRun,
    transmission_reference_run: SpiceRun,
    beam_trap_size_factor: float,
    apply_theta_dependent_correction: bool,
    universal_mask_file: Union[str, None],
    pixels_to_mask: str,
    beam_center_mask_radius: float,
    main_detector_mask_angle: float,
    wing_detector_mask_angle: float,
    min_count_threshold: float,
    max_count_threshold: float,
    sensitivity_file_name: str,
    nexus_dir: str = None,
):
    """Prepare sensitivities from SPICE files

    Parameters
    ----------
    is_wing_detector: bool
        Flag to indicate the operation is on wing detector
    flood_run: SpiceRun
        flood run
    direct_beam_run: SpiceRun or None
        direct beam run
    dark_current_run: SpiceRun
        dark current run
    apply_solid_angle_correction: bool
        Flag to apply solid angle correction to flood run
    transmission_flood_run: SpiceRun
        transmission flood run
    transmission_reference_run: SpiceRun
        transmission reference run
    beam_trap_size_factor: float
        size factor of beam trap given by user
    apply_theta_dependent_correction: bool
        Flag to apply theta dependent correction to transmission run
    universal_mask_file: str
        path to mask file applied to all the runs
    pixels_to_mask: str
        lists of pixels (IDs) to mask
    beam_center_mask_radius: float
        radius of mask for beam center in mm
    main_detector_mask_angle: float
        angle for main detector mask
    wing_detector_mask_angle: float
        angle for wing detector mask
    min_count_threshold: float
        minimum normalized count threshold as a good pixel
    max_count_threshold: float
        maximum normalized count threshold as a good pixel
    sensitivity_file_name: str
        output file name with full path
    nexus_dir: str or None
        directory for nexus file.  None for default.

    """

    CG3 = "CG3"

    # Set up sensitivities preparation configurations
    preparer = PrepareSensitivityCorrection(CG3, is_wing_detector)
    # Load flood runs
    preparer.set_flood_runs([flood_run.unique_nexus_name(nexus_dir, True)])

    # Process beam center runs
    if direct_beam_run is not None:
        preparer.set_direct_beam_runs(
            [direct_beam_run.unique_nexus_name(nexus_dir, True)]
        )

    # Set extra masks
    preparer.set_masks(
        universal_mask_file,
        pixels_to_mask,
        wing_det_mask_angle=wing_detector_mask_angle,
        main_det_mask_angle=main_detector_mask_angle,
    )

    # Set beam center radius
    if beam_center_mask_radius is not None:
        preparer.set_beam_center_radius(beam_center_mask_radius)

    # Transmission
    if transmission_reference_run is not None:
        trans_flood_file = transmission_flood_run.unique_nexus_name(nexus_dir, True)
        trans_ref_file = transmission_reference_run.unique_nexus_name(nexus_dir, True)
        preparer.set_transmission_correction(
            transmission_flood_runs=[trans_flood_file],
            transmission_reference_runs=[trans_ref_file],
            beam_trap_factor=beam_trap_size_factor,
        )
        preparer.set_theta_dependent_correction_flag(apply_theta_dependent_correction)

    # Dark runs
    if dark_current_run is not None:
        dark_curr_file = dark_current_run.unique_nexus_name(nexus_dir, True)
        preparer.set_dark_current_runs([dark_curr_file])

    # solid angle correction
    preparer.set_solid_angle_correction_flag(apply_solid_angle_correction)

    # Run
    moving_detector = False

    preparer.execute(
        moving_detector,
        min_count_threshold,
        max_count_threshold,
        sensitivity_file_name,
        enforce_use_nexus_idf=True,
        debug_mode=True,
    )

    return
