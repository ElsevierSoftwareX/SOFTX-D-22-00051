from drtsans.settings import namedtuplefy
from drtsans.samplelogs import SampleLogs
from drtsans.geometry import (
    get_instrument,
    source_sample_distance,
    sample_detector_distance,
    translate_sample_by_z,
    translate_detector_by_z,
)

__all__ = [
    "beam_radius",
    "sample_aperture_diameter",
    "source_aperture_diameter",
    "source_aperture_sample_distance",
    "translate_sample_by_z",
    "translate_detector_by_z",
]


def source_monitor_distance(source, unit="mm", log_key=None, search_logs=True):
    r"""
    Report the distance (always positive!) between source and monitor.

    If logs are not used or distance fails to be found in the logs, then
    calculate the distance using the instrument configuration file.

    Parameters
    ----------
    source: PyObject
        Instrument object, MatrixWorkspace, workspace name, file name,
        run number
    unit: str
        'mm' (millimeters), 'm' (meters)
    log_key: str
        Only search for the given string in the logs. Do not use default
        log keys
    search_logs: bool
        Report the value found in the logs.

    Returns
    -------
    float
        distance between source and sample, in selected units
    """
    m2units = dict(mm=1e3, m=1.0)
    mm2units = dict(mm=1.0, m=1e-3)
    sl = SampleLogs(source)

    # Search the logs for the distance
    if search_logs is True:
        lk = "source-monitor-distance" if log_key is not None else log_key
        try:
            return sl.single_value(lk) * mm2units[unit]
        except Exception:
            pass

    # Calculate the distance using the instrument definition file
    instrument = get_instrument(source)
    monitor = instrument.getComponentByName("monitor1")
    smd = monitor.getDistance(instrument.getSource())

    # Insert in the logs if not present
    if "source-monitor-distance" not in sl.keys():
        sl.insert("source-monitor-distance", smd * 1.0e3, unit="mm")

    return smd * m2units[unit]


def sample_aperture_diameter(run, unit="mm"):
    r"""
    Find the sample aperture diameter from the logs.

    Log keys searched are 'sample_aperture_diameter' (override beamslit4) and 'beamslit4'.

    Parameters
    ----------
    run: Mantid Run instance, :py:obj:`~mantid.api.MatrixWorkspace`, file name, run number
        Input from which to find the aperture
    unit: str
        return aperture in requested length unit, either 'm' or 'mm'

    Returns
    -------
    float
        Sample aperture diameter, in requested units
    """
    sl = SampleLogs(run)
    sad = None
    for log_key in ("sample_aperture_diameter", "beamslit4"):
        if log_key in sl.keys():
            sad = sl.single_value(log_key)
            break
    if sad is None:
        pnames = [p.name for p in run.run().getProperties()]
        raise RuntimeError(
            'Unable to retrieve sample aperture diameter as neither log "sample_aperture_diameter" '
            'nor "beamslit4" is in the sample logs.  Available logs are {}'
            "".format(pnames)
        )

    if "sample_aperture_diameter" not in sl.keys():
        sl.insert("sample_aperture_diameter", sad, unit="mm")
    if unit == "m":
        sad /= 1000.0
    return sad


@namedtuplefy
def source_aperture(other, unit="m"):
    r"""
    Find the source aperture diameter and position

    After the moderator (source) there are three consecutive discs (termed slits), each with eight holes of
    different apertures. Appropriate log entries (VbeamSlit, VbeamSlit2, VbeamSlit3) indicate the aperture index for
    each of the three disc slits. Thus, the position of the source and the source aperture are not the same. The most
    restrictive slit will define the source aperture. Restriction is determined by the smallest angle subtended
    from the slit to the sample, here assumed to be a pinhole.

    Log entries beamslit, beamslit2, and beamslit3 store the required rotation angle for each wheel in order to align
    the appropriate slit with the neutron beam. These angles are not used in the reduction.

    Parameters
    ----------
    other: Run, MatrixWorkspace, file name, run number
    unit: str
        Length unit, either 'm' or 'mm'

    Returns
    -------
    namedtuple
        Fields of the name tuple
        - float: diameter, in requested units
        - float: distance to sample, in requested units
    """
    sample_logs = SampleLogs(other)
    if (
        "sample_aperture_diameter" in sample_logs
        and "source_aperture_sample_distance" in sample_logs
    ):
        diameter = float(
            sample_logs.sample_aperture_diameter.value
        )  # assumed in mili meters
        asd = float(
            sample_logs.source_aperture_sample_distance.value
        )  # assumed in mili meters
    else:  # determine the aperture using the three diameter-variable slits
        source_aperture_distance = [
            10080,
            11156,
            12150,
        ]  # source to aperture distance, in mili-meters
        number_slits = len(source_aperture_distance)

        # Slit diameters for the different slits. Diameters have changed during time so we use the run number to
        # identify which sets of diameters to pick
        # Entries are of the form: (start_run_number, end_run_number): [slits set]
        index_to_diameters = {
            (0, 9999): [
                [5.0, 10.0, 10.0, 15.0, 20.0, 20.0, 25.0, 40.0],
                [0.0, 10.0, 10.0, 15.0, 15.0, 20.0, 20.0, 40.0],
                [0.0, 10.0, 10.0, 15.0, 15.0, 20.0, 20.0, 40.0],
            ],
            (10000, float("inf")): [
                [5.0, 10.0, 15.0, 20.0, 25.0, 25.0, 25.0, 25.0],
                [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 25.0, 25.0],
                [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 25.0, 25.0],
            ],
        }

        # Find the appropriate set of slit diameters
        run_number = int(sample_logs.run_number.value)
        for (start, end) in index_to_diameters:
            if start <= run_number <= end:
                index_to_diameter = index_to_diameters[(start, end)]
                break

        # entries vBeamSlit, vBeamSlit2, and vBeamSlit3 contain the slit number, identifying the slit diameter
        # for each of the three slits
        diameter_indexes = [
            int(sample_logs[log_key].value.mean()) - 1
            for log_key in ["vBeamSlit", "vBeamSlit2", "vBeamSlit3"]
        ]

        # Determine which of the three slits subtend the smallest angle with the sample, assumed to be a pinhole.
        # The angle is estimated as the ratio aperture_diameter / aperture_sample_distance.
        # `slit_index` runs from zero to two, specifying the slits we're selecting
        # `diameter_index` runs from zero to seven, specifying the diameter we're selecting
        ssd = source_sample_distance(other, unit="mm")
        diameter, asd = float("inf"), 1.0  # start with an infinite diameter / asd ratio
        for slit_index in range(number_slits):  # iterate over the three slits
            diameter_index = diameter_indexes[
                slit_index
            ]  # index specifies which of the 8 apertures to choose from
            tentative_asd = (
                ssd - source_aperture_distance[slit_index]
            )  # distance from the aperture to the sample
            tentative_diameter = index_to_diameter[slit_index][
                diameter_index
            ]  # slit diameter
            if (
                tentative_diameter / tentative_asd < diameter / asd
            ):  # we found a smaller subtending angle
                diameter = tentative_diameter
                asd = tentative_asd
    if unit == "m":
        diameter /= 1000.0
        asd /= 1000.0
    return dict(diameter=diameter, distance_to_sample=asd, unit=unit)


def source_aperture_diameter(run, unit="mm"):
    r"""
    Find the source aperture diameter

    Either report log vale or compute this quantity.

    After the moderator (source) there are three consecutive discs
    (termed wheels), each with eight holes in them (eight slits).
    Appropriate log entries (VbeamSlit, VbeamSlit2, VbeamSlit3) indicate
    the slit index for each of the three wheels. Thus, the position
    of the source and the source aperture are not the same. The most
    restrictive slit will define the source aperture

    Log entries beamslit, beamslit2, and beamslit3 store the required
    rotation angle for each wheel in order to align the appropriate slit
    with the neutron beam. These angles are not used in reduction.

    If the aperture is computed, then the value is stored
    in log key "source_aperture_diameter", with mili meter units

    Parameters
    ----------
    run: Mantid Run instance, MatrixWorkspace, file name, run number
    unit: str
        Length unit, either 'm' or 'mm'

    Returns
    -------
    float
        Source aperture diameter, in requested units
    """
    log_key = "source_aperture_diameter"
    sample_logs = SampleLogs(run)
    if log_key in sample_logs.keys():
        source_aperture_diameter_entry = sample_logs.single_value(
            log_key
        )  # units are 'mm'
    else:
        source_aperture_diameter_entry = source_aperture(run, unit="mm").diameter
        sample_logs.insert(log_key, source_aperture_diameter_entry, unit="mm")
    if unit == "m":
        source_aperture_diameter_entry /= 1000.0

    return source_aperture_diameter_entry


def source_aperture_sample_distance(run, unit="mm"):
    r"""
    Find the distance from the source aperture to the sample.

    Either report log vale or compute this quantity. If the distance has to be computed, then stores the value
    in log key "source_aperture_sample_distance", with mili-meter units.

    Parameters
    ----------
    run: Mantid Run instance, MatrixWorkspace, file name, run number
    unit: str
        Length unit, either 'm' or 'mm'

    Returns
    -------
    float
    """
    log_key = "source_aperture_sample_distance"
    sample_logs = SampleLogs(run)
    if log_key in sample_logs.keys():
        sasd = sample_logs.single_value(log_key)  # units are 'mm'
    else:
        sasd = source_aperture(run, unit="mm").distance_to_sample
        sample_logs.insert(log_key, sasd, unit="mm")
    if unit == "m":
        sasd /= 1000.0

    return sasd


def insert_aperture_logs(ws):
    r"""
    Insert source and sample aperture diameters in the logs, as well as
    the distance between the source aperture and the sample. Units are in mm

    Parameters
    ----------
    ws: MatrixWorkspace
        Insert metadata in this workspace's logs
    """
    sample_logs = SampleLogs(ws)
    if "sample_aperture_diameter" not in sample_logs.keys():
        sample_aperture_diameter(ws, unit="mm")  # function will insert the log
    if "source_aperture_diameter" not in sample_logs.keys():
        sad = source_aperture_diameter(ws, unit="mm")
        sample_logs.insert("source_aperture_diameter", sad, unit="mm")
    if "source_aperture_sample_distance" not in sample_logs.keys():
        sds = source_aperture(ws, unit="mm").distance_to_sample
        sample_logs.insert("source_aperture_sample_distance", sds, unit="mm")


def detector_id(pixel_coordinates, tube_size=256):
    r"""
    Find the detector ID given 2D pixel coordinates in the main detector

    Coordinate (0, 0) refers to the left-low corner of the detector when viewed
    from the sample. Similarly, coordinate (191, 255) refers to the right-up corner.
    Takes into account that the front and back panel are interleaved.

    Parameters
    ----------
    pixel_coordinates: tuple, list
        (x, y) coordinates in pixels, or list of (x, y) coordinates.
    tube_size: int
        Number of pixels in a tube
    Returns
    -------
    int, list
        detector ID or list of detector ID's depending on the input pixel_coordinates
    """
    pixel_xy_list = (
        [pixel_coordinates]
        if isinstance(pixel_coordinates[0], int)
        else pixel_coordinates
    )
    detector_ids = list()
    for pixel_xy in pixel_xy_list:
        x, y = pixel_xy
        eightpack_index = x // 8
        consecutive_tube_index = (
            x % 8
        )  # tube index within the eightpack containing the tube
        tube_index_permutation = [0, 4, 1, 5, 2, 6, 3, 7]
        tube_index = tube_index_permutation[consecutive_tube_index]
        detector_ids.append((eightpack_index * 8 + tube_index) * tube_size + y)
    return detector_ids if len(detector_ids) > 1 else detector_ids[0]


def pixel_coordinates(detector_id, tube_size=256):
    r"""
    Find 2D pixel coordinates in the main detector, given a detector ID.

    Coordinate (0, 0) refers to the left-low corner of the detector when viewed from the sample. Similarly,
    coordinate (191, 255) refers to the right-up corner.
    Takes into account that the front and back panel are interleaved.


    Parameters
    ----------
    detector_id: int, list
        A single detector ID or a list of detector ID's
    tube_size: int
        Number of pixels in a tube

    Returns
    -------
    tuple
        (x, y) pixel coordinates if only one detector, else a list of (x, y) pixel coordinates
    """
    tube_index_permutation = [0, 2, 4, 6, 1, 3, 5, 7]
    detector_ids = (
        [detector_id] if isinstance(detector_id, int) else detector_id
    )  # assume iterable
    pixel_xy = list()
    for det_id in detector_ids:
        y = det_id % tube_size
        eithpack_index = det_id // (8 * tube_size)
        tube_id = (
            det_id // tube_size - 8 * eithpack_index
        )  # tube index within the eightpack containing the tube
        x = eithpack_index * 8 + tube_index_permutation[tube_id]
        pixel_xy.append((x, y))
    return pixel_xy if len(pixel_xy) > 1 else pixel_xy[0]


def beam_radius(input_workspace, unit="mm"):
    r"""
    Calculate the beam radius impinging on the detector bank.

    .. math::

           R_{beam} = R_{sampleAp} + SDD * (R_{sampleAp} + R_{sourceAp}) / SSD

    Parameters
    ----------
    input_workspace: ~mantid.api.MatrixWorkspace, str
        Input workspace, contains all necessary info in the logs
    unit: str
        Either 'mm' or 'm'

    Returns
    -------
    float
        Estimated beam radius
    """
    # retrieve source aperture radius
    source_aperture_diam = source_aperture_diameter(input_workspace, unit=unit)
    r_src_ap = 0.5 * source_aperture_diam

    # retrieve source aperture to sample distance
    source_aperture_sample_dist = source_aperture_sample_distance(
        input_workspace, unit=unit
    )

    # retrieve sample aperture radius
    sample_aperture_diam = sample_aperture_diameter(input_workspace, unit=unit)
    r_sam_ap = 0.5 * sample_aperture_diam

    # retrieve sample detector distance
    sample_detector_dist = sample_detector_distance(input_workspace)

    # calculate beam radius
    r_beam = (
        r_sam_ap
        + (r_sam_ap + r_src_ap) * sample_detector_dist / source_aperture_sample_dist
    )

    return r_beam
