import numpy as np

# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/samplelogs.py
from drtsans.samplelogs import SampleLogs

# Functions exposed to the general user (public) API
__all__ = ["attenuation_factor"]


def attenuation_factor(input_workspace):
    r"""This gets the wavelength and attenuator value from the workspace
    logs then calculates the attenuation factor based on the fitted
    parameters for each different attenuator based on the equation

    .. math::

       attenuation factor = A e^{-B \lambda} + C

    The attenuation scale factor and the uncertainty for this is
    returned.

    The attenuator value pulled from the logs is mapped to the
    attenuator name by:

    0: "Undefined"
    1: "Close"
    2: "Open",
    3: "x3",
    4: "x30",
    5: "x300",
    6: "x2k",
    7: "x10k",
    8: "x100k"

    If the attenuator is one of Undefined, Close or Open then a
    attenuation factor of 1 with uncertainty 0 is returned

    Parameters
    ----------
    input_workspace: str, ~mantid.api.MatrixWorkspace
        Input workspace for which to calculate attenuation factor

    Returns
    -------
    float, float
        attenuation_factor, attenuation_factor_uncertainty

    """
    sample_logs = SampleLogs(input_workspace)

    # Get attenuator value
    attenuator = sample_logs.single_value("attenuator")

    if attenuator < 3:  # Undefined, Close, Open
        # return scale factor of 1 and error 0
        return 1, 0

    # The fitted attenuator parameters for the equation A * exp(-B * λ) + C
    # Provided by Lisa Debeer-Schmitt, 2020-02-26, original file
    # provided can be found at
    # https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/data/attenuator_fitting_parameters.txt
    # In the following format (Amp, Amp Err, exp const, exp const err,  bkgd, bkgd err)
    attenuators = {
        3: (  # x3
            0.3733459538730628,
            0.008609717163831113,
            0.08056544906925872,
            0.008241433507695071,
            0.0724341919138054,
            0.01125160779959418,
        ),
        4: (  # x30
            0.11696573514650677,
            0.006304060228295941,
            0.25014801934427583,
            0.01012469612884642,
            0.003696051816711061,
            0.0003197928933191539,
        ),
        5: (  # x300
            0.028719247985112162,
            0.0019190523738874328,
            0.3884993528348815,
            0.010600714703684273,
            0.00017081815634872129,
            9.055642884664314e-06,
        ),
        6: (  # x2k
            0.015510737042113254,
            0.0008301527045697745,
            0.5840982399579384,
            0.010252064767405953,
            6.966839283167031e-05,
            2.260503164358648e-06,
        ),
        7: (  # x10k
            0.00563013075327734,
            0.0005203265715819975,
            0.6961581698084675,
            0.01938010154115584,
            1.3123049075167468e-05,
            1.5266828654446554e-06,
        ),
        8: (  # x100k
            0.1439135754790426,
            0.005573924841205431,
            0.30824770207752383,
            0.011967728090637404,
            0.006739099792400909,
            0.0007076026868930973,
        ),
    }

    # Get wavelength from workspace
    wavelength = sample_logs.single_value("wavelength")

    # Call the attenuation function and return results
    return _attenuation_factor(*attenuators[attenuator], wavelength)


def _attenuation_factor(A, A_e, B, B_e, C, C_e, wavelength):
    """
    This calculates the function
        A * exp(-B * λ) + C
    along with the uncertainty
    """
    scale = A * np.exp(-B * wavelength) + C
    scale_error_Amp = np.exp(-B * wavelength)
    scale_error_exp_const = A * np.exp(-B * wavelength) * (-wavelength)
    scale_error_bkgd = 1
    scale_error = np.sqrt(
        (scale_error_Amp * A_e) ** 2
        + (scale_error_exp_const * B_e) ** 2
        + (scale_error_bkgd * C_e) ** 2
    )
    return scale, scale_error
