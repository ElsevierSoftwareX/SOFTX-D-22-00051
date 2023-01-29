import enum
import os
from mantid.kernel import ConfigService
from mantid.api import mtd

__all__ = [
    "InstrumentEnumName",
    "instrument_enum_name",
    "instrument_standard_name",
    "is_time_of_flight",
]

INSTRUMENT_LABELS = ["CG3", "BIOSANS", "EQ-SANS", "EQSANS", "CG2", "GPSANS"]


@enum.unique
class InstrumentEnumName(enum.Enum):
    @staticmethod
    def names():
        r"""Standard names for all instruments, in alphabetical order"""
        names_all = list(map(str, InstrumentEnumName))
        names_all.remove("UNDEFINED")
        return sorted(names_all)

    r"""Unique names labelling each instrument"""
    UNDEFINED = None  # usually the dummy instrument used for testing
    BIOSANS = ConfigService.getFacility("HFIR").instrument("BIOSANS")
    EQSANS = ConfigService.getFacility("SNS").instrument("EQSANS")
    GPSANS = ConfigService.getFacility("HFIR").instrument("GPSANS")

    def __str__(self):
        return self.name


def instrument_enum_name(input_query):
    r"""
    Resolve the instrument name as a unique enumeration.

    Parameters
    ----------
    input_query: str,  ~mantid.api.MatrixWorkspace, ~mantid.api.IEventsWorkspace
        string representing a filepath, a valid instrument name, or a Mantid workspace containing an instrument

    Returns
    -------
    InstrumentEnumName
        The name of the instrument as one of the InstrumentName enumerations
    """
    string_to_enum = {
        "CG3": InstrumentEnumName.BIOSANS,
        "BIOSANS": InstrumentEnumName.BIOSANS,
        "EQ-SANS": InstrumentEnumName.EQSANS,
        "EQSANS": InstrumentEnumName.EQSANS,
        "CG2": InstrumentEnumName.GPSANS,
        "GPSANS": InstrumentEnumName.GPSANS,
    }
    # convert to a string
    name = str(input_query)

    if name in mtd:  # convert mantid workspace into a instrument string
        name = mtd[str(name)].getInstrument().getName()
    else:  # see if `name` contains any of the instrument labels
        name = name.upper()
        for instrument_string_label in sorted(string_to_enum.keys()):
            if instrument_string_label in name:
                name = instrument_string_label
                break
    return string_to_enum.get(name.upper(), InstrumentEnumName.UNDEFINED)


def instrument_standard_name(input_query):
    r"""
    Resolve the standard instrument name.

    Parameters
    ----------
    input_query: str,  ~mantid.api.MatrixWorkspace, ~mantid.api.IEventsWorkspace
        string representing a filepath, a valid instrument name, or a Mantid workspace containing an instrument

    Returns
    -------
    str
        The name of the instrument as the string representation of one of the InstrumentName enumerations
    """
    return str(instrument_enum_name(input_query))


def instrument_standard_names():
    r"""Standard names for all instruments, in alphabetical order"""
    return InstrumentEnumName.names()


def instrument_filesystem_name(input_query):
    r"""
    Resolve the name of the instrument that is the subdirectory name under /SNS or /HFIR

    Parameters
    ----------
    input_query: str,  ~mantid.api.MatrixWorkspace, ~mantid.api.IEventsWorkspace
        string representing a filepath, a valid instrument name, or a Mantid workspace containing an instrument

    Returns
    -------
    str
    """
    filesystem_name = {"BIOSANS": "CG3", "EQSANS": "EQSANS", "GPSANS": "CG2"}
    return filesystem_name[instrument_standard_name(input_query)]


def instrument_label(input_query):
    r"""
    Resolve the instrument name.

    Parameters
    ----------
    input_query: str,  ~mantid.api.MatrixWorkspace, ~mantid.api.IEventsWorkspace
        string representing a filepath, a valid instrument name, or a Mantid workspace containing an instrument

    Returns
    -------
    str
    """
    # convert to a string
    name = str(input_query)

    if name in mtd:  # convert mantid workspace into a instrument string
        return mtd[str(name)].getInstrument().getName()
    else:  # see if `name` contains any of the instrument labels
        name = name.upper()
        for instrument_string_label in INSTRUMENT_LABELS:
            if instrument_string_label in name:
                return instrument_string_label
    raise RuntimeError(
        'Instrument name can not be resolved from "{}"'.format(input_query)
    )


def extract_run_number(input_query):
    r"""
    Extract the run number from string

    Example:
    input string '/HFIR/..../CG3_961.nxs.h5', 'CG3_961.nxs.h5', 'CG3961', and 'CG3_961' should all return run
    number 961

    Parameters
    ----------
    input_query: str

    Returns
    -------
    int
    """
    try:
        # see if `input_query` is an integer
        run_number = int(input_query)
    except ValueError:
        # name of the file without path
        run_number = os.path.basename(input_query)
        # everything up to the extension
        run_number = run_number.split(".")[0]
        # remove the instrument name
        for label in INSTRUMENT_LABELS:
            run_number = run_number.replace(label, "")
        # remove any remaining '_'
        if "_" in run_number:
            run_number = run_number.split("_")[1]
        # convert to an integer

    return int(run_number)


def is_time_of_flight(input_query):
    r"""
    Find if the instrument is a time-of-flight one

    Parameters
    ----------
    input_query: str, ~mantid.api.MatrixWorkspace, ~mantid.api.IEventsWorkspace, InstrumentEnumName
        string representing a valid instrument name, or a Mantid workspace containing an instrument

    Returns
    -------
    bool
    """
    return (
        instrument_enum_name(input_query) is InstrumentEnumName.EQSANS
    )  # we only have one, for the moment
