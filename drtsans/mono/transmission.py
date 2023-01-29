from mantid.kernel import logger
from drtsans.mono.geometry import beam_radius
from drtsans.transmission import calculate_transmission as raw_calculate_transmission
from drtsans.transmission import apply_transmission_correction

# Symbols to be exported to the eqsans namespace
__all__ = ["calculate_transmission", "apply_transmission_correction"]


def calculate_transmission(
    input_sample, input_reference, radius=None, radius_unit="mm", output_workspace=None
):

    if radius is None:
        logger.information("Calculating beam radius from sample logs")
        radius = beam_radius(input_reference, unit="mm")

    zero_angle_transmission_workspace = raw_calculate_transmission(
        input_sample,
        input_reference,
        radius,
        radius_unit=radius_unit,
        output_workspace=output_workspace,
    )

    return zero_angle_transmission_workspace
