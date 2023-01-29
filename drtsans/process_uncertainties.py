# https://docs.mantidproject.org/nightly/algorithms/CloneWorkspace-v1.html
# https://docs.mantidproject.org/nightly/algorithms/SetUncertainties-v1.html
from mantid.simpleapi import mtd, CloneWorkspace, SetUncertainties
from mantid.api import EventType, IEventWorkspace
import numpy


def set_init_uncertainties(input_workspace, output_workspace=None):
    """
    Set the initial uncertainty of a :py:obj:`~mantid.api.MatrixWorkspace`

    Mantid algorithm :ref:`SetUncertainties <algm-SetUncertainties-v1>` will be called to make sure
    1: set the uncertainty to square root of intensity
    2: make sure all zero uncertainties will be set to 1

    In case of output workspace is py:obj:`None`, the input workspace will be
    replaced by output workspace.

    :exception RuntimeError: output workspace (string) is empty

    **Mantid algorithms used:**
    :ref:`CloneWorkspace <algm-CloneWorkspace-v1>`
    :ref:`SetUncertainties <algm-SetUncertainties-v1>`

    Parameters
    ----------
    input_workspace: ~mantid.api.MatrixWorkspace
        Input workspace
    output_workspace: str
        Output workspace (workspace name or instance) or py:obj:`None` for in-place operation

    Returns
    -------
    ~mantid.api.MatrixWorkspace
    """
    input_workspace = str(input_workspace)
    if output_workspace is None:
        output_workspace = input_workspace  # in-place by default
    else:
        output_workspace = str(output_workspace)

    # in the case of event workspaces, don't do anything if they are RAW events (eventType=='TOF')
    # and the histogram representation doesn't have any zeros
    input_ws = mtd[input_workspace]
    if (
        isinstance(input_ws, IEventWorkspace)
        and input_ws.getSpectrum(0).getEventType() == EventType.TOF
        and input_ws.findY(0.0) == (-1, -1)
    ):
        # clone the input_workspace or return it
        if input_workspace == output_workspace:
            return mtd[input_workspace]
        else:
            return CloneWorkspace(
                InputWorkspace=input_workspace, OutputWorkspace=output_workspace
            )

    # Calculate uncertainties as square root and set 1 for 0 count
    # But SetUncertainties does not treat nan as SANS team desires
    SetUncertainties(
        InputWorkspace=input_workspace,
        OutputWorkspace=output_workspace,
        SetError="sqrtOrOne",
    )

    # get a handle to the workspace
    output_ws = mtd[output_workspace]

    # Set nan as the uncertainty for all nan-intensity - check that there are nans first
    if output_ws.findY(numpy.nan) != (-1, -1):
        for ws_index in range(output_ws.getNumberHistograms()):
            vec_y = output_ws.readY(ws_index)
            if numpy.count_nonzero(numpy.isnan(vec_y)):
                nan_indexes = numpy.argwhere(numpy.isnan(vec_y))

                # There existing nan
                if len(nan_indexes) > 0:
                    vec_e = output_ws.dataE(ws_index)
                    vec_e[nan_indexes] = numpy.nan
                # END-IF
        # END-FOR (spectra)

    return output_ws
