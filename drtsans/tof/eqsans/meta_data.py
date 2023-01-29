# Method in this module is to set meta data to EQSANS Mantid Workspaces
from mantid.simpleapi import AddSampleLogMultiple

__all__ = ["set_meta_data"]


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
        sample aperture diameter in unit mm
    sample_thickness: None, float
        sample thickness in unit cm
    source_aperture_diameter: float, None
        source aperture diameter in unit meter
    smearing_pixel_size_x: float, None
        pixel size in x direction in unit as meter
    smearing_pixel_size_y: float, None
        pixel size in Y direction in unit as meter

    Returns
    -------

    """
    # Exception
    if wave_length is not None or wavelength_spread is not None:
        raise RuntimeError(
            "Wave length and wave length spread are not allowed to set to EQ-SANS"
        )

    # Log value dictionary: 3-tuple (log name, log value, unit)
    meta_data_list = list()

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
