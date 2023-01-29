from mantid.simpleapi import mtd, LoadEventNexus

from drtsans.settings import amend_config, unique_workspace_dundername as uwd
from drtsans import geometry
from drtsans.tof.eqsans import geometry as e_geometry, correct_frame


def load_w(
    run, low_tof_clip=0, high_tof_clip=0, dw=0.1, data_dir=None, output_workspace=None
):
    r"""
    Load a run, correct the TOF frame, and convert to wavelength

    Parameters
    ----------
    run: str
        Run number or filename. Passed onto Mantid's `Load` algorithm
    low_tof_clip: float
        Lower TOF clipping
    high_tof_clip: float
        Upper TOF clipping
    data_dir: str, list
        Additional one or more data search directories
    dw: float
        Wavelength bin width

    Returns
    -------
    MatrixWorkspace
    """
    if output_workspace is None:
        output_workspace = uwd()  # unique hidden name

    with amend_config({"instrumentName": "EQSANS"}, data_dir=data_dir):
        LoadEventNexus(Filename=run, OutputWorkspace=output_workspace)
        e_geometry.translate_detector_by_z(output_workspace)  # inplace
        correct_frame.correct_detector_frame(output_workspace)
        sdd = geometry.source_detector_distance(output_workspace, unit="m")
        bands = correct_frame.transmitted_bands_clipped(
            output_workspace, sdd, low_tof_clip, high_tof_clip, interior_clip=True
        )
        correct_frame.convert_to_wavelength(output_workspace, bands, dw, events=False)
        correct_frame.log_tof_structure(
            output_workspace, low_tof_clip, high_tof_clip, interior_clip=True
        )
        return mtd[output_workspace]
