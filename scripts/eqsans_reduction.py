import json
import os
import sys
import mantid.simpleapi as msapi  # noqa E402
from drtsans.tof.eqsans import validate_reduction_parameters
from drtsans.tof.eqsans.api import (
    load_all_files,
    reduce_single_configuration,
    plot_reduction_output,
)  # noqa E402


def reduce_eqsans_configuration(input_config):
    input_config = validate_reduction_parameters(input_config)

    # chekcing if output directory exists, if it doesn't, creates the folder
    output_dir = input_config["configuration"]["outputDir"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    loaded = load_all_files(input_config)
    msapi.logger.warning("...loading completed.")
    out = reduce_single_configuration(loaded, input_config)
    msapi.logger.warning("...single reduction completed.")
    plot_reduction_output(out, input_config)
    msapi.logger.warning("...plotting completed.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise RuntimeError(".!.reduction code requires a parameter json string or file")
    if os.path.isfile(sys.argv[1]):
        json_filename = sys.argv[1]
        msapi.logger.warning("...json file is used.")
        with open(sys.argv[1], "r") as fd:
            reduction_input = json.load(fd)
    else:
        msapi.logger.warning("...json string is used.")
        json_filename = " ".join(sys.argv[1:])
        reduction_input = json.loads(json_filename)

    reduce_eqsans_configuration(reduction_input)
