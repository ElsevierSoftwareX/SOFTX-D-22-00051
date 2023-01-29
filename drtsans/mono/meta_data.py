# Method in this module is to set meta data to SANS Mantid Workspaces
from mantid.simpleapi import AddSampleLogMultiple, AddSampleLog
from mantid.kernel import logger
from drtsans.samplelogs import SampleLogs
from drtsans.geometry import sample_detector_distance

__all__ = ["set_meta_data", "get_sample_detector_offset", "parse_json_meta_data"]


# Constants for JSON
SAMPLE = "Sample"
BEAM_CENTER = "BeamCenter"
BACKGROUND = "Background"
EMPTY_TRANSMISSION = "EmptyTrans"
TRANSMISSION = "Transmission"
TRANSMISSION_BACKGROUND = "Background_transmission"
BLOCK_BEAM = "BlockBeam"
DARK_CURRENT = "DarkCurrent"


def parse_json_meta_data(
    reduction_input,
    meta_name,
    unit_conversion_factor,
    beam_center_run=False,
    background_run=False,
    empty_transmission_run=False,
    transmission_run=False,
    background_transmission=False,
    block_beam_run=False,
    dark_current_run=False,
):
    """Parse user specified meta data from JSON from configuration part

    Parameters
    ----------
    reduction_input: dict
        dictionary parsed from JSON
    meta_name: str
        meta data name
    unit_conversion_factor: float
        unit conversion conversion factor from user preferred unit to drt-sans preferred unit
    beam_center_run: bool
        flag whether beam center run will have the same meta data
    background_run: bool
        flag whether beam center run will have the same meta data
    empty_transmission_run: bool
        flag whether beam center run will have the same meta data
    transmission_run
        flag whether beam center run will have the same meta data
    background_transmission
        flag whether beam center run will have the same meta data
    block_beam_run
        flag whether beam center run will have the same meta data
    dark_current_run
        flag whether beam center run will have the same meta data

    Returns
    -------
    dict
        keys: strings stands for sample (run), background (run) and etc used in JSON; value: JSON value
        if value is None, it indicates that the meta data won't be overwritten

    """
    # Init return dictionary
    overwrite_dict = dict()
    for run_type in [
        BEAM_CENTER,
        BACKGROUND,
        EMPTY_TRANSMISSION,
        TRANSMISSION,
        TRANSMISSION_BACKGROUND,
        BLOCK_BEAM,
        DARK_CURRENT,
    ]:
        overwrite_dict[run_type] = None

    if isinstance(reduction_input["configuration"][meta_name], dict):
        # new JSON code: parse each key
        _parse_new_meta_data_json(
            reduction_input, meta_name, unit_conversion_factor, overwrite_dict
        )

    else:
        # current JSON format
        # Parse for sample run
        try:
            # Get sample run's overwrite value
            overwrite_value = (
                float(reduction_input["configuration"][meta_name])
                * unit_conversion_factor
            )
            overwrite_dict[SAMPLE] = overwrite_value

            # Apply to runs other than sample run
            for run_type, overwrite_flag in [
                (BEAM_CENTER, beam_center_run),
                (BACKGROUND, background_run),
                (EMPTY_TRANSMISSION, empty_transmission_run),
                (TRANSMISSION, transmission_run),
                (TRANSMISSION_BACKGROUND, background_transmission),
                (BLOCK_BEAM, block_beam_run),
                (DARK_CURRENT, dark_current_run),
            ]:
                if overwrite_flag:
                    overwrite_dict[run_type] = overwrite_value

        except ValueError:
            # Overwritten value error
            overwrite_value = None
            overwrite_dict[SAMPLE] = overwrite_value
        except TypeError:  # as type_err:
            overwrite_value = None
            # msg = 'Meta data {} does not exist'.format(meta_name)
            overwrite_dict[SAMPLE] = overwrite_value
            # print(msg + '\n' + str(type_err))
            # raise TypeError(msg + '\n' + str(type_err))
        except KeyError as key_error:
            # Required value cannot be found
            raise KeyError(
                "JSON file shall have key as [configuration][{}]. Error message: {}"
                "".format(meta_name, key_error)
            )

    return overwrite_dict


def _parse_new_meta_data_json(
    reduction_input, meta_name, unit_conversion_factor, meta_value_dict
):
    """Parse JSON with new format such that each run will have its own value

    Parameters
    ----------
    reduction_input: dict
        reduction inputs from JSON
    meta_name: str
        meta data name
    unit_conversion_factor: float
        unit conversion
    meta_value_dict: dict
        meta data value dictionary for all runs

    Returns
    -------
    None

    """
    # Parse for sample run
    run_type = SAMPLE
    try:
        # Get sample run's overwrite value
        overwrite_value = (
            float(reduction_input["configuration"][meta_name][run_type])
            * unit_conversion_factor
        )
    except ValueError:
        # Overwritten value error
        overwrite_value = None
    except KeyError as key_error:
        # Required value cannot be found
        raise KeyError(
            "JSON file shall have key as configuration:{}:{}. Error message: {}"
            "".format(meta_name, run_type, key_error)
        )
    meta_value_dict[SAMPLE] = overwrite_value

    # Parse for other runs
    try:
        for run_type in [
            BEAM_CENTER,
            BACKGROUND,
            EMPTY_TRANSMISSION,
            TRANSMISSION,
            TRANSMISSION_BACKGROUND,
            BLOCK_BEAM,
            DARK_CURRENT,
        ]:
            over_write_value_temp = reduction_input["configuration"][meta_name][
                run_type
            ]
            if over_write_value_temp is True:
                # input is True/true: follow SAMPLE run
                meta_value_dict[run_type] = overwrite_value
            elif over_write_value_temp is False:
                # input is False/false
                pass
            else:
                # otherwise do the conversion
                meta_value_dict[run_type] = (
                    float(over_write_value_temp) * unit_conversion_factor
                )
        # END-FOR
    except ValueError as value_error:
        # Overwritten value error
        raise RuntimeError(
            "JSON value of key configuration:{}:{} has a value error.  Error message: {}"
            "".format(meta_name, run_type, value_error)
        )
    except KeyError as key_error:
        # Required value cannot be found
        raise KeyError(
            "JSON file shall have key as configuration:{}:{}. Error message: {}"
            "".format(meta_name, run_type, key_error)
        )

    return


def parse_json_wave_length_and_spread(reduction_input):
    """Parse wave length and wave length spread from JSON dict

    * drt-sans should be supporting overwriting wavelength only and
    * overwriting both (NOT overwriting wavelength spread only).

    Parameters
    ----------
    reduction_input: ~dict
        reduction configuration parsed from JSON
    Returns
    -------
    ~tuple
        wave length and wave length spread in format of dictionary

    """
    # Parse wave length with instrument scientists' preferred defaults
    wave_length_dict = parse_json_meta_data(
        reduction_input,
        "wavelength",
        1.0,
        beam_center_run=True,
        background_run=True,
        empty_transmission_run=True,
        transmission_run=True,
        background_transmission=True,
        block_beam_run=True,
        dark_current_run=False,
    )

    # Parse wave length with instrument scientists' preferred defaults
    wave_length_spread_dict = parse_json_meta_data(
        reduction_input,
        "wavelengthSpread",
        1.0,
        beam_center_run=True,
        background_run=True,
        empty_transmission_run=True,
        transmission_run=True,
        background_transmission=True,
        block_beam_run=True,
        dark_current_run=False,
    )

    # Check valid or not
    if wave_length_dict[SAMPLE] is None and wave_length_spread_dict[SAMPLE] is not None:
        # the case that is not allowed such that only wave length spread is overwritten
        raise RuntimeError(
            "It is not allowed to overwrite wavelengthSpread only but not wavelength"
        )

    return wave_length_dict, wave_length_spread_dict


def set_meta_data(
    workspace,
    wave_length=None,
    wavelength_spread=None,
    sample_offset=0.0,
    sample_aperture_diameter=None,
    sample_thickness=None,
    source_aperture_diameter=None,
    smearing_pixel_size_x=None,
    smearing_pixel_size_y=None,
):
    """Set meta data to SANS Mantid Workspace as run properties

    Parameters
    ----------
    workspace: str, ~mantid.api.MatrixWorkspace
        Mantid workspace instance or workspace name
    wave_length: float, None
        wave length in Angstrom
    wavelength_spread: float, None
        wave length spread in Angstrom
    sample_offset: float
        offset of sample from origin in unit mm
    sample_aperture_diameter: float, None
        sample aperture diameter in mm
    sample_thickness: None, float
        sample thickness in unit cm
    source_aperture_diameter: float, None
        source aperture size radius in unit mm
    smearing_pixel_size_x: float, None
        pixel size in x direction in unit as meter, only for Q-resolution calculation
    smearing_pixel_size_y: float, None
        pixel size in Y direction in unit as meter, only for Q-resolution calculation

    Returns
    -------

    """
    # Init list for sample log name, value and unit
    meta_data_list = list()

    # Wave length and wave length spread shall be set Number Series
    # Wave length
    if wave_length is not None:
        # meta_data_list.append(('wavelength', np.array([wave_length, wave_length]), 'A'))
        AddSampleLog(
            workspace,
            LogName="wavelength",
            LogText="{}".format(wave_length),
            LogType="Number Series",
            LogUnit="A",
        )

    # Wave length spread
    if wavelength_spread is not None:
        # meta_data_list.append(('wavelength_spread', np.array([wavelength_spread, wavelength_spread]), 'A'))
        AddSampleLog(
            workspace,
            LogName="wavelength_spread",
            LogText="{}".format(wavelength_spread),
            LogType="Number Series",
        )

    # Add the sample log dictionary to add
    if sample_aperture_diameter is not None:
        meta_data_list.append(
            ("sample_aperture_diameter", sample_aperture_diameter, "mm")
        )

    # Source aperture radius
    if source_aperture_diameter is not None:
        meta_data_list.append(
            ("source_aperture_diameter", source_aperture_diameter, "mm")
        )

    # Sample offset
    meta_data_list.append(("sample_offset", sample_offset, "mm"))

    # Sample thickness
    if sample_thickness is not None:
        meta_data_list.append(("sample_thickness", sample_thickness, "cm"))

    # Pixel size
    if smearing_pixel_size_x is not None and smearing_pixel_size_y is not None:
        meta_data_list.append(("smearingPixelSizeX", smearing_pixel_size_x, "m"))
        meta_data_list.append(("smearingPixelSizeY", smearing_pixel_size_y, "m"))
    elif smearing_pixel_size_x is None and smearing_pixel_size_y is None:
        pass
    else:
        raise RuntimeError(
            "Pixel size X ({}) and Y ({}) must be set together"
            "".format(smearing_pixel_size_x, smearing_pixel_size_y)
        )

    # Add log value
    if len(meta_data_list) > 0:
        # only work on non-empty meta data list
        log_names, log_values, log_units = zip(*meta_data_list)

        # add meta data (as sample logs) to workspace
        AddSampleLogMultiple(
            Workspace=workspace,
            LogNames=log_names,
            LogValues=log_values,
            LogUnits=log_units,
        )


def get_sample_detector_offset(
    workspace,
    sample_si_meta_name,
    zero_sample_offset_sample_si_distance,
    overwrite_sample_si_distance=None,
    overwrite_sample_detector_distance=None,
):
    """Get sample offset and detector offset from meta data

    This method is based on the assumption and fact that
    "sample position is set to nominal position (0, 0, 0) regardless of sample log SampleToSi"
    "detector1 is set to [0, 0, sample_detector_distance]
    It will be re-implemented

    Parameters
    ----------
    workspace: str, ~mantid.api.MatrixWorkspace
        Mantid workspace instance or workspace name
    sample_si_meta_name : str
        Sample to Si (window) meta data name
    zero_sample_offset_sample_si_distance: float
        default sample to Si window distance, i.e., distance without sample offset. unit = meter
    overwrite_sample_si_distance: float or None
        sample to Si window distance to overwrite.  Unit = mm (consistent with the unit of original meta data)
    overwrite_sample_detector_distance : float or None
        sample detector distance to overwrite. Unit = m (consistent with the unit of original meta data)

    Returns
    -------
    ~tuple
        sample offset (float) in unit meter and detector offset (float) in unit meter

    """
    # Calculate the sample offset and detector offset without overwriting value
    # This is caused by incorrect IDF which does not use SampleToSi.
    sample_logs = SampleLogs(workspace)
    # read sample log for SampleToSi and convert to meter from mm
    sample_to_si = sample_logs.find_log_with_units(sample_si_meta_name, "mm") * 1e-3
    logger.notice(
        "[META INIT] User SSD = {}, SWD = {},"
        "".format(overwrite_sample_detector_distance, overwrite_sample_si_distance)
    )
    logger.notice("[META] EPICS Sample to Si = {} meter".format(sample_to_si))
    logger.notice(
        "[META] Hardcoded Sample to nominal distance = {} meter"
        "".format(zero_sample_offset_sample_si_distance)
    )

    # Offsets: shift both sample and detector to conserve sample-detector distance
    # Move instrument_component sample (relative) to [0, 0, 0.071 - SampleToSi/1000]
    sample_offset = zero_sample_offset_sample_si_distance - sample_to_si
    # Move instrument_component detector1 relative [0, 0, 0.071 - SampleToSi/1000]
    detector_offset = sample_offset

    # Get sample detector distance by calculation from instrument geometry directly
    real_sample_det_distance = sample_detector_distance(
        workspace, unit="m", search_logs=False
    )
    logger.notice(
        "[META] EPICS Sample detector distance = {} (calculated)".format(
            real_sample_det_distance
        )
    )

    # With overwriting distance(s)
    if (
        overwrite_sample_si_distance is not None
        or overwrite_sample_detector_distance is not None
    ):
        # 2 cases to handle.  The order must be conserved
        if overwrite_sample_si_distance is not None:
            # Sample-Si distance is overwritten. NeXus-recorded sample-detector-distance is thus inaccurate.
            # # convert unit of (overwrite)-sample-Si-distance to meter
            # overwrite_sample_si_distance *= 1E-3

            # Shift the sample position only without moving detector
            overwrite_offset = sample_to_si - overwrite_sample_si_distance
            logger.notice(
                "[META-Overwrite SSD] SampleToSi = {}, SampleToSiOverwrite = {}, "
                "Original SampleOffset = {}"
                "".format(sample_to_si, overwrite_sample_si_distance, sample_offset)
            )
            sample_offset += overwrite_offset
            real_sample_det_distance -= overwrite_offset
            sample_to_si = overwrite_offset

        if overwrite_sample_detector_distance is not None:
            # Sample-detector distance is overwritten, i.e., fix the sample position and move detector to
            # make the sample-detector-distance to the overwriting value
            # Move instrument_component detector1 relatively
            # [0, 0, sample_detector_distance_overwrite - sample_detector_distance_nexus]
            overwrite_offset = (
                overwrite_sample_detector_distance - real_sample_det_distance
            )
            detector_offset += overwrite_offset
            real_sample_det_distance += overwrite_offset
    # END-IF

    logger.notice(
        "[META FINAL] Sample offset = {}, Detector offset = {}, Sample-detector-distance = {}, "
        "Sample-Si-window-distance = {}"
        "".format(
            sample_offset, detector_offset, real_sample_det_distance, sample_to_si
        )
    )

    return sample_offset, detector_offset
