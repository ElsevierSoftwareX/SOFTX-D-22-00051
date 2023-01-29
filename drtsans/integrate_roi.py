from mantid.api import AnalysisDataService as mtd
import numpy as np


def integrate_detector_roi(workspace, roi_det_list):
    """Integrate neutron counts in ROI of detectors (pixels)

    Integrate neutron counts and uncertainties in ROI of detectors

    Parameters
    ----------
    workspace: String or ~mantid.api.MatrixWorkspace
        Name of workspace or Workspace instance
    roi_det_list: List of integers
        Workspace indexes for the detectors in ROI

    Returns
    -------
    float, float
        Integrated intensity, Integrated uncertainties
    """
    # Get workspace instance if input is a string
    workspace = mtd[str(workspace)]

    # Get Y and E array
    counts_array = workspace.extractY()
    error_array = workspace.extractE()

    # Get the sum
    roi_intensity = counts_array[roi_det_list]
    roi_intensity = roi_intensity.sum()

    # ROI error: sqrt(sum(sigma(i)^2))
    roi_error_sq = (error_array[roi_det_list]) ** 2
    roi_error = np.sqrt(np.sum(roi_error_sq))

    return roi_intensity, roi_error
