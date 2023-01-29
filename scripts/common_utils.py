""" Common utility functions for all SANS """
import os
import numpy as np
import matplotlib.pyplot as plt
import drtsans  # noqa E402
import mantid.simpleapi as msapi  # noqa E402
from drtsans.iq import (
    BinningMethod,
    determine_1d_linear_bins,
    determine_1d_log_bins,
)  # noqa E402
from drtsans.plots import plot_IQmod, plot_IQazimuthal
from drtsans.save_ascii import save_ascii_binned_1D, save_ascii_binned_2D  # noqa E402
from drtsans.settings import unique_workspace_dundername as uwd  # noqa E402


def setup_configuration(json_params, instrument):
    """
    Extract configuration from json file passed through Shaman
    """
    # Currently unused
    # json_params['configuration']['sampleApertureSize']
    # json_params['configuration']['useDetectorTubeType']
    # json_params['configuration']['useThetaDepTransCorrection']
    # json_params['configuration']['nPixelsRadiusForTransmission']

    config = dict(
        is_wing=False,
        dark_current=None,
        sensitivity_file_path=None,
        center_x=None,
        center_y=None,
        center_y_wing=None,
        detector_offset=0,
        sample_offset=0,
        mask_detector=None,
        flux_method="time",
        solid_angle=True,
        mask=None,
        mask_panel=None,
        transmission_radius=None,
    )

    # Dark current
    useDarkFileName = json_params["configuration"]["useDarkFileName"]
    useDarkFileBlockedBeam = json_params["configuration"]["useDarkFileBlockedBeam"]
    if useDarkFileName and useDarkFileBlockedBeam:
        config["dark_current"] = json_params["configuration"]["darkFileName"]

    # Sensitivity
    if json_params["configuration"]["sensitivityFileName"] is not None:
        config["sensitivity_file_path"] = json_params["configuration"][
            "sensitivityFileName"
        ]

    # Normalization
    config["flux_method"] = json_params["configuration"]["normalization"].lower()
    offset = json_params["configuration"]["sampleOffset"]
    config["sample_offset"] = 0.0 if offset is None else offset

    if json_params["configuration"]["detectorOffset"] is None:
        config["detector_offset"] = 0.0

    # Solid angle
    config["solid_angle"] = json_params["configuration"]["useSolidAngleCorrection"]

    # Masking
    # TODO: get default mask from configuration
    if json_params["configuration"]["useDefaultMask"]:
        default_mask = []
        if instrument in ["GPSANS"]:
            default_mask = [{"Pixel": "1-12,245-256"}]
        w = msapi.LoadEmptyInstrument(InstrumentName=instrument, OutputWorkspace=uwd())
        for d in default_mask:
            msapi.MaskBTP(Workspace=w, **d)
        config["mask"] = msapi.ExtractMask(w, OutputWorkspace=uwd()).OutputWorkspace
    elif json_params["configuration"]["maskFileName"] is not None:
        config["mask"] = json_params["configuration"]["maskFileName"]

    if json_params["configuration"]["useMaskBackTubes"]:
        config["mask_panel"] = "back"

    try:
        tr_rad = float(json_params["configuration"]["mmRadiusForTransmission"])
        config["transmission_radius"] = tr_rad
    except ValueError:
        pass
    return config


def get_Iq(
    q_data,
    output_dir,
    output_file,
    label="",
    linear_binning=True,
    weighting=False,
    nbins=100,
):
    """
    Compute I(q) from corrected workspace
    """
    q_min = np.min(q_data.mod_q)
    q_max = np.max(q_data.mod_q)
    if weighting:
        bin_method = BinningMethod.WEIGHTED
    else:
        bin_method = BinningMethod.NOWEIGHT
    if linear_binning:
        linear_bins = determine_1d_linear_bins(q_min, q_max, nbins)
        iq_output = drtsans.iq.bin_intensity_into_q1d(
            q_data, linear_bins, bin_method=bin_method
        )
    else:
        log_bins = determine_1d_log_bins(q_min, q_max, nbins, even_decade=True)
        iq_output = drtsans.iq.bin_intensity_into_q1d(
            q_data, log_bins, bin_method=bin_method
        )
    filename = os.path.join(output_dir, output_file + label + "_Iq.txt")
    save_ascii_binned_1D(filename, "I(Q)", iq_output)

    filename = os.path.join(output_dir, output_file + label + "_Iq.png")
    plot_IQmod([iq_output], filename, backend="mpl")
    plt.clf()
    filename = os.path.join(output_dir, output_file + label + "_Iq.json")
    plot_IQmod([iq_output], filename, backend="d3")
    plt.close()
    return iq_output


def get_Iqxqy(q_data, output_dir, output_file, label="", weighting=False, nbins=100):
    """
    Compute I(qx,qy) from corrected workspace
    """
    if weighting:
        bin_method = BinningMethod.WEIGHTED
    else:
        bin_method = BinningMethod.NOWEIGHT
    qx_min = np.min(q_data.qx)
    qx_max = np.max(q_data.qx)
    linear_x_bins = determine_1d_linear_bins(qx_min, qx_max, nbins)
    qy_min = np.min(q_data.qy)
    qy_max = np.max(q_data.qy)
    linear_y_bins = determine_1d_linear_bins(qy_min, qy_max, nbins)

    iq_output = drtsans.iq.bin_intensity_into_q2d(
        q_data, linear_x_bins, linear_y_bins, method=bin_method
    )

    filename = os.path.join(output_dir, output_file + label + "_Iqxqy.txt")
    save_ascii_binned_2D(filename, "I(Qx,Qy)", iq_output)

    filename = os.path.join(output_dir, output_file + label + "_Iqxqy.png")
    plot_IQazimuthal(iq_output, filename, backend="mpl")
    plt.clf()
    filename = os.path.join(output_dir, output_file + label + "_Iqxqy.json")
    plot_IQazimuthal(iq_output, filename, backend="d3")
    plt.close()

    return iq_output
