from pathlib import Path

from drtsans.instruments import InstrumentEnumName
from drtsans.mono.gpsans import load_all_files as load_all_gpsans_files
from drtsans.mono.gpsans import (
    reduce_single_configuration as reduce_single_gpsans_configuration,
)
from drtsans.mono.biosans import load_all_files as load_all_biosans_files
from drtsans.mono.biosans import (
    reduce_single_configuration as reduce_single_biosans_configuration,
)

__all__ = ["create_output_directory", "run_reduction"]


def create_output_directory(output_dir="", subfolder=None, hfir_sans=True):
    r"""Create output folder if not there already. If `subfolder` is provided with a list of string,
    those subfolders will be created inside the output_dir. If `hfir_sans` is set to True (default value)
    2 subfolders `1D` and `2D` will be automatically created

    :param output_dir: str
        name of the output folder
    :param subfolder: list
       list of subfolders to create inside that output folder
    :param hfir_sans: boolean
       if True, subfolders 1D and 2D will be created automatically

    :raise: ValueError
       if not output_dir has been provided
    """

    if output_dir == "":
        raise ValueError("Please provide an output_dir value!")

    subfolder_array = [] if subfolder is None else subfolder

    if hfir_sans:
        subfolder_array.extend(["1D", "2D"])

    if subfolder_array == []:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    else:
        for folder in subfolder_array:
            output_folder = Path(output_dir).joinpath(folder)
            Path(output_folder).mkdir(parents=True, exist_ok=True)


def run_reduction(
    reduction_input, q_type="", sample_name="", instrument=InstrumentEnumName.GPSANS
):
    r"""Run a full reduction for GP or BIO sans instruments

    :param reduction_input: dictionary
        dictionary of all the paremeters used to run reduction
    :param q_type: str (ex: "low", "mid", "high"
        Use to create uniq output file name
    :param sample_name: str
        name of the sample used. This is used to create a uniq output filename
    :param instrument: InstrumentEnumName
        name of the instrument as InstrumentEnumName. Default being GPSANS
    :return: tuple, (out object, name of file created)

    """
    q_type = q_type + "_" if q_type else ""

    if instrument == InstrumentEnumName.GPSANS:
        loaded = load_all_gpsans_files(reduction_input)
        out = reduce_single_gpsans_configuration(loaded, reduction_input)
    elif instrument == InstrumentEnumName.BIOSANS:
        loaded = load_all_biosans_files(reduction_input)
        out = reduce_single_biosans_configuration(loaded, reduction_input)
    else:
        raise NotImplementedError("instrument not implemented yet!")
    filename = (
        reduction_input["configuration"]["outputDir"]
        + "/2D/"
        + sample_name
        + "_"
        + q_type
        + "q.jpg"
    )
    return out, filename
