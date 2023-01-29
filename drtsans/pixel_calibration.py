import copy
import enum
import itertools
import json
import numpy as np
import numexpr
import os
import stat
import sys
import warnings


r""" Hyperlinks to mantid algorithms
ApplyCalibration <https://docs.mantidproject.org/nightly/algorithms/ApplyCalibration-v1.html>
CloneWorkspace <https://docs.mantidproject.org/nightly/algorithms/CloneWorkspace-v1.html>
CreateEmptyTableWorkspace <https://docs.mantidproject.org/nightly/algorithms/CreateEmptyTableWorkspace-v1.html>
CreateWorkspace <https://docs.mantidproject.org/nightly/algorithms/CreateWorkspace-v1.html>
DeleteWorkspaces <https://docs.mantidproject.org/nightly/algorithms/DeleteWorkspaces-v1.html>
FilterEvents <https://docs.mantidproject.org/nightly/algorithms/FilterEvents-v1.html>
GenerateEventsFilter <https://docs.mantidproject.org/nightly/algorithms/GenerateEventsFilter-v1.html>
Integration <https://docs.mantidproject.org/nightly/algorithms/Integration-v1.html>
LoadEmptyInstrument <https://docs.mantidproject.org/nightly/algorithms/LoadEmptyInstrument-v1.html>
LoadInstrument <https://docs.mantidproject.org/nightly/algorithms/LoadInstrument-v1.html>
LoadNexus <https://docs.mantidproject.org/nightly/algorithms/LoadNexus-v1.html>
MaskDetectors <https://docs.mantidproject.org/nightly/algorithms/MaskDetectors-v1.html>
MaskDetectorsIf <https://docs.mantidproject.org/nightly/algorithms/MaskDetectorsIf-v1.html>
ReplaceSpecialValues <https://docs.mantidproject.org/nightly/algorithms/ReplaceSpecialValues-v1.html>
SaveNexus <https://docs.mantidproject.org/nightly/algorithms/SaveNexus-v1.html>
"""
from mantid.simpleapi import (
    AddSampleLog,
    ApplyCalibration,
    ClearMaskFlag,
    CloneWorkspace,
    CreateEmptyTableWorkspace,
    DeleteWorkspaces,
    FilterEvents,
    GenerateEventsFilter,
    Integration,
    Load,
    LoadEventNexus,
    LoadNexus,
    LoadNexusProcessed,
    MaskDetectors,
    MaskDetectorsIf,
    Rebin,
    ReplaceSpecialValues,
    SaveNexus,
)
from mantid.api import FileLoaderRegistry, mtd

r"""
Hyperlinks to drtsans functions
namedtuplefy, unique_workspace_dundername <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/settings.py>
SampleLogs <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/samplelogs.py>
TubeCollection <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/tubecollection.py>
"""  # noqa: E501
from drtsans.instruments import (
    InstrumentEnumName,
    instrument_enum_name,
    instrument_standard_name,
)
from drtsans.mask_utils import apply_mask
from drtsans.path import exists as file_exists
from drtsans.settings import namedtuplefy, unique_workspace_dundername
from drtsans.samplelogs import SampleLogs
from drtsans.tubecollection import TubeCollection


__all__ = [
    "apply_calibrations",
    "as_intensities",
    "calculate_apparent_tube_width",
    "day_stamp",
    "calculate_barscan_calibration",
    "load_calibration",
    "split_barscan_run",
]


class CalibrationNotFound(Exception):
    """Custom exception to be raised when no appropriate calibration is found in the database"""

    pass


r"""Flags a problem when running the barscan algorithm that identifies the pixel corresponding
to the bottom of the shadow cast by the bar on the detector array."""
INCORRECT_PIXEL_ASSIGNMENT = -1


r"""Default files storing the metadata of the pixel calibrations. There's one file for each instrument."""
database_file = {
    InstrumentEnumName.BIOSANS: "/HFIR/CG3/shared/calibration/pixel_calibration.json",
    InstrumentEnumName.EQSANS: "/SNS/EQSANS/shared/calibration/pixel_calibration.json",
    InstrumentEnumName.GPSANS: "/HFIR/CG2/shared/calibration/pixel_calibration.json",
}


class CalType(enum.Enum):
    r"""Enumerate the possible types of pixel calibrations"""
    BARSCAN = "BARSCAN"
    TUBEWIDTH = "TUBEWIDTH"


def loader_algorithm(input_file):
    r"""
    Determine which Mantid algorithm to use to load a file.

    If a specialized loading algorithm can't be found, then `Load` algorithm is returned.

    devs - Jose Borreguero <borreguerojm@ornl.gov>

    Parameters
    ----------
    input_file: str

    Returns
    -------
    ~mantid.api.Algorithm
    """
    loaders = {
        "LoadNexusProcessed": LoadNexusProcessed,
        "LoadEventNexus": LoadEventNexus,
    }
    loader_name = FileLoaderRegistry.Instance().chooseLoader(input_file).name()
    return loaders.get(loader_name, Load)


def day_stamp(input_workspace):
    r"""
    Find the day stamp (e.g 20200311 for March 11, 2020) using the "start_time" metadata from the
    Nexus events file as input.

    devs - Jose Borreguero <borreguerojm@ornl.gov>

    Parameters
    ----------
    input_workspace: str, ~mantid.api.MatrixWorkspace, ~mantid.api.IEventsWorkspace
        Workspace from which the day stamp is to be retrieved.

    Returns
    -------
    int
    """
    return int(SampleLogs(input_workspace).start_time.value[0:10].replace("-", ""))


class BarPositionFormula:

    # Default formulae for each instrument and detector component
    _default_formula = "565 - {y} + 0.0 * {tube}"  # no dependency on the tube index
    _default_formulae = {
        ("BIOSANS", "detector1"): "565 - {y} + 0.0083115 * (191 - {tube})",
        ("BIOSANS", "wing_detector"): _default_formula,
        ("EQSANS", "detector1"): _default_formula,
        ("GPSANS", "detector1"): "565 - {y} - 0.0914267 * (191 - {tube})",
    }

    @staticmethod
    def _elucidate_formula(instrument_component):
        r"""
        Find the default formula for an instrument and detector array.

        Parameters
        ----------
        instrument_component: tuple
            A two-item tuple. The first item is the standard name of the instrument, e.g. 'BIOSANS'; the
            second item is the name of the detector array (e.g. 'detector1' or 'wing_detector').

        Returns
        -------
        str

        Warnings
        --------
        UserWarning
            When ```instrument_component``` is not found in the default instrument formulae.
        """
        default_formula = BarPositionFormula._default_formula
        default_formulae = BarPositionFormula._default_formulae
        if instrument_component not in default_formulae:
            warnings.warn(
                f"Unable to find a bar position formula for argument {instrument_component}.\n"
                f"Using default formula: {default_formula}.\n"
                f"Valid values are {default_formulae.keys()}\n"
            )
        return default_formulae.get(instrument_component, default_formula)

    @staticmethod
    def _validate_symbols(formula):
        r"""
        Assess if a formula provider by the user contain the required symbols. Symbols '{y}' is required and
        symbol '{tube}' is optional.

        If '{tube}' is not present in the formulat, a null term '0.0 * {tube}' is appended to the formula.

        Parameters
        ----------
        formula: str
            Formula to obtain the Y-coordinate of the bar in the frame of reference of the sample.

        Raises
        ------
        ValueError
            When the formula fails to contain symbols '{y}' and '{tube}'.
        """
        if "{y}" not in formula:
            raise ValueError(
                f'Formula does not contain "{{y}}", e.g. formula = "565-{{y}}+0.008*(191-{{tube}})"'
            )
        if "{tube}" not in formula:
            warnings.warn(
                f'Formula does not contain "{{tube}}", e.g. formula = "565-{{y}}+0.008*(191-{{tube}})"'
            )
            formula += " + 0.0 * {tube}"
        return formula

    def __init__(self, instrument_component=None, formula=None):
        r"""
        Formula to obtain the bar position in the frame of reference of the sample, in milimeters.

        There are default formulae for each instrument and detector array.

        Parameters
        ----------
        instrument_component: tuple
            A two-item tuple. The first item is the standard name of the instrument, e.g. 'BIOSANS'; the
            second item is the name of the detector array (e.g. 'detector1' or 'wing_detector').
        formula: str
            Formula

        Raises
        ------
        RuntimeError
            When neither ```instrument_component``` nor ```formula``` is supplied.
        """
        if instrument_component is not None:
            self._formula = self._elucidate_formula(instrument_component)
        elif formula is not None:
            self._formula = self._validate_symbols(formula)
        else:
            raise RuntimeError("Insufficient input to create a bar position formula")

    def __str__(self):
        return self._formula

    def evaluate(self, bar_position, tube_index):
        r"""
        Y-coordinate of the bar in the frame of reference of the sample.

        Parameters
        ----------
        bar_position: float
            Log value for the bar, in mili meters.
        tube_index: int
            Tube index. The first index (zero) corresponds to the leftmost tube when standing at the sample.
            position.

        Returns
        -------
        float
        """
        values_inserted = self._formula.format(y=bar_position, tube=tube_index)
        return float(numexpr.evaluate(values_inserted))

    def validate_top_position(self, bar_top_position):
        r"""
        Assert that the formula evaluates to a positive Y-coordinate when the bar is at the top position.

        Parameters
        ----------
        bar_top_position: float
            Log value when the bar is located at the top position.

        Raises
        ------
        RuntimeError
            When the formula evaluates to a non-positive Y-coordinate
        """
        if self.evaluate(bar_top_position, 0) <= 0:  # evaluate for the leftmost tube
            raise RuntimeError(
                f"The Y-coordinate of the bar in the frame of reference of the sample when\n"
                f"              the bar is placed at the top position has evaluated as non-negative.\n"
                f'              The formula used is "{self._formula}"'
            )


class Table:
    r"""Container for a table of pixel calibration item data, plus metadata

    The table object holds two attributes:
    - metadata, dict, informs about the calibration run, instrument, detector array.
    - table, ~mantid.api.TableWorkspace, containing the actual calibration data.

    devs - Jose Borreguero <borreguerojm@ornl.gov>

    Parameters
    ----------
    metadata: dict
        Dictionary with the following fields about the calibration:
        - caltype, str, the type of calibration (BARSCAN, TUBEWIDTH)
        - instrument, str, standard name of the instrument for which the calibration was carried out.
        - component, str, standard name of the double detector array for which the calibration was carried out.
        - daystamp, int, 8-digit integer whose digits are to be understood as YYYYMMDD.
        - run_numbers, list, list of run numbers that encompassed the calibration.
    detector_ids: list
        List of detector IDs for which a calibration has been carried out.
    positions: list
        List of Y-coordinates for each detector, in meters.
    heights: list
        List of detector heights (along the Y-), in meters.
    widths: list
        List of detector widths (along the X-axis), in meters.
    """

    @classmethod
    def compose_table_name(cls, metadata):
        r"""Standard workspace name for a calibration table, built as a composite name using the
        calibration type, instrument, component, and daystamp. (e.g. "barscan_gpsans_detector1_20200311")

        Parameters
        ----------
        metadata: dict
            Dictionary containing the metadata of one calibration

        Returns
        -------
        str
        """
        m = metadata  # handy shortcut
        return f'{m["caltype"].lower()}_{m["instrument"]}_{m["component"]}_{str(m["daystamp"])}'

    @classmethod
    def load(
        cls, database, caltype, instrument, component, daystamp, output_workspace=None
    ):
        r"""
        Loading a calibration requires loading the calibration metadata from a ```database``` file along
        with loading a Nexus file containing the actual data.

        Metadata and data are encapsulated in a ~drtsans.pixel_calibration.Table object. This object
        contains attribute ```table``` which holds the actual data in a ~mantid.api.TableWorkspace.

        **Mantid algorithms used:**
        :ref:`LoadNexus <algm-LoadNexus-v1>`,
        <https://docs.mantidproject.org/algorithms/LoadNexus-v1.html>

        Parameters
        ----------
        database: str
            Path to JSON file containing metadata for different calibrations.
        caltype: str
            Type of calibration (BARSCAN, TUBEWIDHT).
        instrument: str
            Standard name of the instrument for which the calibration was carried out.
        component: str
            Standard name of the double detector array for which the calibration was carried out.
        daystamp: int
            8-digit integer whose digits are to be understood as YYYYMMDD. The returned calibration
            will have a daystamp equal or more recent.
        output_workspace: str
            Name of the output ~mantid.api.TableWorkspace containing the calibration data. If
            :py:obj:`None`, a composite name is created using the calibration type, instrument, component,
            and daystamp. (e.g. "barscan_gpsans_detector1_20200311").

        Returns
        -------'
        ~drtsans.pixel_calibration.Table
        """
        # Search the database for a match to the required metadata
        not_found_message = f"No suitable {caltype}_{instrument}_{component} calibration found in {database}"
        with open(database, mode="r") as json_file:
            entries = json.load(
                json_file
            )  # list of metadata entries stored in `database`
        required = {
            caltype,
            instrument,
            component,
        }  # required metadata pieces of information
        # Filter the metadata entries, leaving out those not containing the required pieces of information
        candidates = [
            entry
            for entry in entries
            if required.issubset(set([str(v) for v in entry.values()]))
        ]
        if len(candidates) == 0:
            raise CalibrationNotFound(not_found_message)
        candidates.sort(
            key=lambda c: c["daystamp"], reverse=True
        )  # sort candidates by decreasing day stamp
        # Find the metadata entry with the closest (equal or smaller) day stamp to the input `daystamp`
        for candidate in candidates:
            if candidate["daystamp"] <= daystamp:
                if output_workspace is None:
                    output_workspace = Table.compose_table_name(candidate)
                table = LoadNexus(
                    candidate["tablefile"], OutputWorkspace=output_workspace
                )
                return Table(candidate, table=table)
        raise CalibrationNotFound(not_found_message)

    @classmethod
    def build_mantid_table(
        cls, output_workspace, detector_ids, positions=None, heights=None, widths=None
    ):
        r"""
        Instantiate a Table workspace with input calibration data.

        The table contains a series of columns, the first one always being the list of detector IDs
        for which a calibration has been made. A 'BARSCAN' calibration will contains two additional columns,
        namely 'Detector Y Coordinate' and 'Detector Height' to store respective pixel positions and heights,
        in meters. A 'TUBEWIDTH' calibration table will contain one additional column, namely
        'Detector Width' to store pixel widths, in meters.

        **Mantid algorithms used:**
        :ref:`CreateEmptyTableWorkspace <algm-CreateEmptyTableWorkspace-v1>`,
        <https://docs.mantidproject.org/algorithms/CreateEmptyTableWorkspace-v1.html>

        Parameters
        ----------
        output_workspace: str
            Name of the output table workspace.
        detector_ids: list
            List of detector IDs for which a calibration has been carried out.
        positions: list
            List of Y-coordinates for each detector, in meters.
        heights: list
            List of detector heights (along the Y-), in meters.
        widths: list
            List of detector widths (along the X-axis), in meters.

        Returns
        -------
        ~mantid.api.TableWorkspace
        """
        # All possible columns names (and associated items) for a calibration table
        columns_data = {
            "Detector Y Coordinate": positions,
            "Detector Height": heights,
            "Detector Width": widths,
        }
        # Remove names not appropriate for the calibration type. For example, a barscan calibration will
        # not contain data for pixel widths (widths==None), so item 'Detector Width': widths is removed
        [
            columns_data.pop(column)
            for column in list(columns_data.keys())
            if columns_data[column] is None
        ]
        table = CreateEmptyTableWorkspace(OutputWorkspace=output_workspace)
        table.addColumn(type="int", name="Detector ID")
        [table.addColumn(type="double", name=column) for column in columns_data]
        # Loop over each calibrated detector and add a new row in the table
        for i in range(len(detector_ids)):
            row = {"Detector ID": detector_ids[i]}
            row.update({column: data[i] for column, data in columns_data.items()})
            table.addRow(row)
        return table

    @classmethod
    def validate_metadata(cls, metadata):
        r"""
        Verify the metadata contains entries for: instrument, name of the double-detector-array, and day stamp.

        Parameters
        ----------
        metadata: dict

        Returns
        -------
        bool

        Raises
        ------
        ValueError
            The metadata is missing one of the required entries.
        """
        required_keys = {"instrument", "component", "daystamp"}
        if required_keys.issubset(metadata.keys()) is False:
            raise ValueError(
                f"Metadata is missing one or more of these entries: {required_keys}"
            )

    def __init__(
        self,
        metadata,
        table=None,
        detector_ids=None,
        positions=None,
        heights=None,
        widths=None,
    ):
        Table.validate_metadata(metadata)
        self.metadata = copy.copy(metadata)
        if table is None:
            # Create table using column data
            output_workspace = Table.compose_table_name(metadata)
            self.table = Table.build_mantid_table(
                output_workspace,
                detector_ids,
                positions=positions,
                heights=heights,
                widths=widths,
            )
        else:
            self.table = mtd[str(table)]

    def __getattr__(self, item):
        r"""
        Access metadata's keys as attributes of the ```Table``` object. For instance,
        self.metadata['instrument'] can be accessed as self.instrument
        """
        if item not in self.__dict__:
            return self.__dict__["metadata"][item]
        return self.__dict__[item]

    def column_values(self, name):
        r"""
        Return a list of values for the selected table column.

        Possible names are 'Detector ID', 'Detector Y Coordinate', 'Detector Height', and 'Detector Width'.

        Parameters
        ----------
        name: str
            Name of the column. Must match the name of one of the columns in the ~mantid.api.TableWorkspace
            ```table``` attribute.

        Returns
        -------
        list
        """
        column_names = self.table.getColumnNames()
        try:
            column_index = column_names.index(name)
        except ValueError:
            raise ValueError(f'"{name}" is not a column name of the calibration table')
        return self.table.column(column_index)

    @property
    def detector_ids(self):
        r"""List of pixel positions stored in the calibration table."""
        return self.column_values("Detector ID")

    @property
    def positions(self):
        r"""List of pixel positions stored in the calibration table."""
        return self.column_values("Detector Y Coordinate")

    @property
    def heights(self):
        r"""List of pixel heights stored in the calibration table."""
        return self.column_values("Detector Height")

    @property
    def widths(self):
        r"""List of pixel widths stored in the calibration table."""
        return self.column_values("Detector Width")

    def apply(self, input_workspace, output_workspace=None):
        r"""
        Apply a calibration to an input workspace, and return the calibrated workspace.

        **Mantid algorithms used:**
        :ref:`CloneWorkspace <algm-CloneWorkspace-v1>`,
        <https://docs.mantidproject.org/algorithms/CloneWorkspace-v1.html>
        :ref:`ApplyCalibration <algm-ApplyCalibration-v1>`,
        <https://docs.mantidproject.org/algorithms/ApplyCalibration-v1.html>

        Parameters
        ----------
        input_workspace: str, ~mantid.api.MatrixWorkspace, ~mantid.api.IEventsWorkspace
            Workspace to which calibration needs to be applied.
        output_workspace: str
            Name of the output workspace with calibrated pixels. If :py:obj:`None`, the pixels
            of the input workspace will be calibrated and no new workspace is generated.

        Returns
        -------
        ~mantid.api.MatrixWorkspace, ~mantid.api.IEventsWorkspace
        """
        if output_workspace is None:
            output_workspace = str(
                input_workspace
            )  # calibrate the pixels of the input workspace
        else:
            # Copy the input workspace and calibrate the pixels of the copy
            CloneWorkspace(
                InputWorkspace=input_workspace, OutputWorkspace=output_workspace
            )
        ApplyCalibration(Workspace=output_workspace, CalibrationTable=self.table)
        return mtd[output_workspace]

    def save(self, database=None, tablefile=None, overwrite=False):
        r"""
        Save the calibration metadata in a JSON file, and the calibration table workspace in a Nexus file.

        **Mantid algorithms used:**
        :ref:`SaveNexus <algm-SaveNexus-v1>`,
        <https://docs.mantidproject.org/algorithms/SaveNexus-v1.html>

        Parameters
        ----------
        database: str
            Path to the JSON file where the ```metadata``` dictionary will be appended. If :py:obj:`None`,
            then the appropriate default file from ~drtsans.pixel_calibration.database_file is used.
            Currently, these are the default files:
            - BIOSANS, '/HFIR/CG3/shared/calibration/pixel_calibration.json',
            - EQSANS, '/SNS/EQSANS/shared/calibration/pixel_calibration.json',
            - GPSANS, '/HFIR/CG2/shared/calibration/pixel_calibration.json'
        tablefile: str
            Path to the Nexus file storing the pixel calibration data. If :py:obj:`None`, then
            a composite name is created using the calibration type, instrument, component,
            and daystamp. (e.g. "barscan_gpsans_detector1_20200311"). The file is saved under
            subdirectory 'tables', located within the directory of the ```database``` file.
            For instance, '/HFIR/CG3/shared/calibration/tables/barscan_gpsans_detector1_20200311.nxs'
        overwrite: bool
            Substitute existing entry with same metadata.

        Raises
        ------
        ValueError
            If we save a calibration already in the database with option ```overwrite=False```.
        """
        if database is None:
            database = database_file[
                instrument_enum_name(self.instrument)
            ]  # default database file

        # Load existing calibrations
        entries = list()
        if os.path.exists(
            database
        ):  # the database may not exist if we're not saving to the default database
            with open(database, mode="r") as json_file:
                entries = json.load(json_file)  # list of metadata entries

        # Is there an entry with the same metadata?
        discard_index = None
        for i in range(len(entries)):
            if self.duplicate_metadata(entries[i]) is True:
                if overwrite is True:
                    discard_index = i  # index in the list of entries to replace
                    break
                else:
                    raise ValueError(
                        "A calibration with the same metadata already exists in the database."
                        'Use "overwrite=True" if you want to overwrite the existing calibration'
                    )

        # Save the table containing the actual data.
        if tablefile is None:
            cal_dir = os.path.join(
                os.path.dirname(database), "tables"
            )  # directory where to save the table file
            os.makedirs(
                cal_dir, exist_ok=True
            )  # Create directory, and don't complain if already exists
            tablefile = (
                os.path.join(cal_dir, Table.compose_table_name(self.metadata)) + ".nxs"
            )
        self.metadata[
            "tablefile"
        ] = tablefile  # store the location in the metadata, used later when loading.
        # save new table and overwrite existing one if having the same name
        if os.path.exists(tablefile):
            os.remove(tablefile)
        SaveNexus(InputWorkspace=self.table, Filename=tablefile)
        os.chmod(tablefile, 0o666)  # everybody can read and write

        # Save the metadata, and replace an existing duplicate entry if required to do so
        if discard_index is not None:
            entries[discard_index] = self.metadata  # replace the metadata
        else:
            entries.append(self.metadata)  # this is a new entry

        if os.path.exists(database):
            os.remove(database)  # delete the old database
        with open(database, mode="w") as json_file:
            json.dump(entries, json_file)  # save the new database
            os.chmod(database, 0o666)  # everybody can read and write

    def duplicate_metadata(self, metadata):
        r"""
        Find if the metadata coincides with a query metadata.

        Keys used for comparison are "caltype", "component", "daystamp", "instrument", and "runnumbers".

        Parameters
        ----------
        metadata: dict

        Returns
        -------
        bool
        """
        for key in ("caltype", "instrument", "component", "daystamp"):
            if self.metadata[key] != metadata[key]:
                return False
        # We don't require the run numbers to be sorted
        if set(self.metadata["runnumbers"]) != set(self.metadata["runnumbers"]):
            return False
        return True

    @namedtuplefy
    def as_intensities(self, reference_workspace):
        r"""
        Returns one workspace for each pixel property that is calibrated (e.g., pixel height),
        and the calibration datum is stored as the intensity value for that pixel. Intended to
        visualize the calibration in MantidPlot's instrument viewer. Not required for calibration
        generation or for data reduction.

        For example, a BARSCAN calibration will generate workspaces ```tablename_positions```
        and ```tablename_heights```, where ```tablename``` is the name of the ~mantid.api.TableWorkspace
        holding the calibration data.

        Note: Positions for visualization in Mantid's instrument view are shifted so that the
              lowest position (usually a negative number) becomes zero. The reason being that
              showing the instrument in Mantid will mask negative intensities, and we want to avoid this.

        **Mantid algorithms used:**
        :ref:`CreateWorkspace <algm-CreateWorkspace-v1>`,
        <https://docs.mantidproject.org/algorithms/CreateWorkspace-v1.html>
        :ref:`LoadEmptyInstrument <algm-LoadEmptyInstrument-v1>`,
        <https://docs.mantidproject.org/algorithms/LoadEmptyInstrument-v1.html>
        :ref:`LoadInstrument <algm-LoadInstrument-v1>`,
        <https://docs.mantidproject.org/algorithms/LoadInstrument-v1.html>

        Parameters
        ----------
        reference_workspace: str, ~mantid.api.MatrixWorkspace, , ~mantid.api.IEventsWorkspace
            Workspace having the same embedded instrument as the instrument used in the barscan calculation

        Returns
        -------
        namedtuple
            A namedtuple containing the ~mantid.api.MatrixWorkspace workspaces with the following fields
            - 'positions' and 'heights' for a BARSCAN calibration
            - 'widths' for a TUBEWIDTH calibration
        """

        # Create the template workspace for the views. It will contain a single intensity value per histogram
        reference = unique_workspace_dundername()
        # We only need one intensity value per histogram
        Rebin(
            InputWorkspace=reference_workspace,
            OutputWorkspace=reference,
            Params=[0, 1000000, 1000000],
            PreserveEvents=False,
        )
        # In the event that the reference workspace has any of the calibrated detector pixels masked, we have to
        # preemptively clear the mask of all pixels
        ClearMaskFlag(Workspace=reference)

        # Find workspace indexes having a calibrated detector, otherwise flag them to be masked
        detector_ids = self.detector_ids
        wi_to_ri = []  # from workspace index to table's row index
        workspace_indexes_to_mask = []
        get_detector = mtd[reference].getDetector
        for workspace_index in range(mtd[reference].getNumberHistograms()):
            detector = get_detector(workspace_index)  # associated detector pixel
            try:
                # Find the entry in the calibration table for the detector with a particular detector ID
                wi_to_ri.append((workspace_index, detector_ids.index(detector.getID())))
            except ValueError:  # This detector was not calibrated, thus is not in detector_ids
                workspace_indexes_to_mask.append(workspace_index)
                continue

        # Mask those uncalibrated detectors
        MaskDetectors(Workspace=reference, WorkspaceIndexList=workspace_indexes_to_mask)

        def transfer_values_to_workspace(property_values, property_name):
            r"""Create a workspace with the modified intensities, and overlay them in the appropriate instrument"""
            output_workspace = f"{self.table.name()}_{property_name}"
            workspace = CloneWorkspace(
                InputWorkspace=reference, OutputWorkspace=output_workspace
            )
            # substitute the intensity in this histogram with the calibration datum for the detector pixel
            for wi, ri in wi_to_ri:
                workspace.dataY(wi)[:] = property_values[ri]
            return mtd[output_workspace]

        calibration_properties = (
            ["positions", "heights"]
            if self.caltype == "BARSCAN"
            else [
                "widths",
            ]
        )
        returned_views = {}
        for cal_prop in calibration_properties:
            values = getattr(self, cal_prop)
            returned_views[cal_prop] = transfer_values_to_workspace(values, cal_prop)

        # Mantid cannot display negative intensities
        if "positions" in calibration_properties:
            values = np.array(self.positions)
            returned_views["positions_mantid"] = transfer_values_to_workspace(
                values - np.min(values), "positions_mantid"
            )

        DeleteWorkspaces(
            [
                reference,
            ]
        )
        return returned_views


def load_calibration(
    input_workspace,
    caltype,
    component="detector1",
    database=None,
    output_workspace=None,
):
    r"""
    Load a calibration into a ~drtsans.pixel_calibration.Table object.

    devs - Jose Borreguero <borreguerojm@ornl.gov>

    Parameters
    ----------
    input_workspace: str, ~mantid.api.MatrixWorkspace, ~mantid.api.IEventsWorkspace
        Workspace from which calibration session is to be retrieved.
    caltype: str
        Either 'BARSCAN' or 'TUBEWIDTH'. A saved calibration can only contain one of these, but not both.
    component: str
        Name of one of the double detector array panels. For BIOSANS we have 'detector1' or 'wing-detector'.
    database: str
        Path to database file containing the metadata for the calibrations. If :py:obj:`None`, the default database
        is used. Currently, these are the default files:
        - BIOSANS, '/HFIR/CG3/shared/calibration/pixel_calibration.json',
        - EQSANS, '/SNS/EQSANS/shared/calibration/pixel_calibration.json',
        - GPSANS, '/HFIR/CG2/shared/calibration/pixel_calibration.json'
    output_workspace: str
        Name of the table workspace containing the calibration session values. If :py:obj:`None`, then a composite
        name is created using the calibration type, instrument, component, and daystamp. (e.g.
        "barscan_gpsans_detector1_20200311")

    Returns
    -------
    ~drtsans.pixel_calibration.Table
    """
    enum_instrument = instrument_enum_name(input_workspace)

    if database is None:
        database = database_file[enum_instrument]  # default database name
    return Table.load(
        database,
        caltype,
        str(enum_instrument),
        component,
        day_stamp(input_workspace),
        output_workspace=output_workspace,
    )


def apply_calibrations(
    input_workspace,
    database=None,
    calibrations=[cal.name for cal in CalType],
    output_workspace=None,
):
    r"""
    Load and apply pixel calibrations to an input workspace.

    devs - Jose Borreguero <borreguerojm@ornl.gov>

    Parameters
    ----------
    input_workspace: str, ~mantid.api.MatrixWorkspace, ~mantid.api.IEventWorkspace
        Input workspace whose pixels are to be calibrated.
    database: str, None
        Path to JSON file containing metadata for different past calibrations. If :py:obj:`None`,
        the default database is used. Currently, these are the default files:
        - BIOSANS, '/HFIR/CG3/shared/calibration/pixel_calibration.json',
        - EQSANS, '/SNS/EQSANS/shared/calibration/pixel_calibration.json',
        - GPSANS, '/HFIR/CG2/shared/calibration/pixel_calibration.json'
    calibrations: str, list
        One or more of 'BARSCAN' and/or 'TUBEWIDTH'.
    output_workspace: str
         Name of the output workspace with calibrated pixels. If :py:obj:`None`, the pixels
        of the input workspace will be calibrated.

    Returns
    -------
        ~mantid.api.MatrixWorkspace, ~mantid.api.IEventsWorkspace
    """
    if output_workspace is None:
        output_workspace = str(input_workspace)
    else:
        CloneWorkspace(InputWorkspace=input_workspace, OutputWorkspace=output_workspace)

    if isinstance(calibrations, str):  # we passed only one calibration
        calibrations = [
            calibrations,
        ]  # convert `calibrations` into a list
    components = {
        InstrumentEnumName.BIOSANS: ["detector1", "wing_detector"],
        InstrumentEnumName.EQSANS: ["detector1"],
        InstrumentEnumName.GPSANS: ["detector1"],
    }

    for component in components[instrument_enum_name(input_workspace)]:
        for caltype in calibrations:
            try:
                calibration = load_calibration(
                    input_workspace, caltype, component, database=database
                )
            except CalibrationNotFound as e:
                calibration = None
                warnings.warn(str(e))
            if calibration is not None:
                calibration.apply(output_workspace)

    return mtd[output_workspace]


def _consecutive_true_values(values, how_many, reverse=False, raise_message=None):
    r"""
    Find first array index of consecutive `how_many` True values.

    devs - Andrei Savici <saviciat@ornl.gov>,
           Jose Borreguero <borreguerojm@ornl.gov>

    Parameters
    ----------
    values: list
        list of `True` and `False` items
    how_many: int
        Number of desired consecutive `True` values
    raise_message: str
        Exception message. No exception if :py:obj:`None`, but INCORRECT_PIXEL_ASSIGNMENT is returned

    Returns
    -------
    int

    Raises
    ------
    IndexError
        If no index is found for any of the edges
    RuntimeError
        If a faulty tube is found
    """
    # use the array or the reverse one
    truth_array = values[::-1] if reverse else values
    # create a sub-array of length how_many of True values that we want to find
    pattern = [True] * how_many
    # loop over the input data and return the first index where the next
    # how_many elements match the pattern
    for i in range(len(truth_array) - how_many):
        if truth_array[i : i + how_many] == pattern:
            return len(values) - i - 1 if reverse else i
    # raise an error if the pattern is not found
    else:
        if raise_message is not None:
            raise IndexError(raise_message)
        return INCORRECT_PIXEL_ASSIGNMENT  # signal for non-identified value


@namedtuplefy
def find_edges(
    intensities,
    tube_threshold=0.2,
    shadow_threshold=0.3,
    tube_edge_min_width=3,
    shadow_edge_min_width=4,
    min_illuminated_length=7,
):
    r"""
    Find the active length of the tube and the shadow region

    All pixel indexes start from the bottom of the tube, with the first
    index being zero.

    devs - Andrei Savici <saviciat@ornl.gov>,
           Jose Borreguero <borreguerojm@ornl.gov>

    Parameters
    ----------
    intensities: list
        pixel pixel_intensities along the tube.
    tube_threshold: float
        fraction of the average intensity to determine the tube edges.
    shadow_threshold: float
        fraction of the average intensity to determine the shadow region.
    tube_edge_min_width: int
        required minimum number of consecutive good pixels above
        the tube threshold
    shadow_edge_min_width: int
        required minimum number of consecutive shadowed pixels
    min_illuminated_length: int
        minimum number of illuminated pixels on the active length

    Returns
    -------
    namedtuple
        the fields of the name tuple are:
        - bottom_pixel: first illuminated pixel
        - top_pixel: last illuminated pixel
        - bottom_shadow_pixel: first shadowed pixel
        - above_shadow_pixel= first illuminated pixel above the shadow region
    """
    # calculate minimum intensity thresholds for tube ends and shadows
    average_intensity = np.average(intensities)
    end_threshold = tube_threshold * average_intensity
    shadow_threshold = shadow_threshold * average_intensity

    # Find edges of the tube: want at least tube_edge_min_width pixels
    # (starting from the top or bottom of a tube) that have pixel_intensities greater
    # than the threshold

    illuminated = [bool(i > end_threshold) for i in intensities]
    # The bottom pixel is the first illuminated pixel. It is required that the next tube_edge_min_width pixels are
    # also illuminated
    bottom_pixel = _consecutive_true_values(
        illuminated,
        tube_edge_min_width,
        raise_message="Could not find bottom tube edge",
    )
    # The top pixel is the last illuminated pixel. It is required that the previous tube_edge_min_width pixels are
    # also illuminated.
    top_pixel = _consecutive_true_values(
        illuminated,
        tube_edge_min_width,
        raise_message="Could not find top tube edge",
        reverse=True,
    )

    # Find the shadow region: similar to tube edges, but in this case
    # we want shadow_edge_min_width pixel_intensities less than the shadow threshold,
    # followed by at least one intensity greater than the threshold

    # The bottom pixel shadowed by the bar is the first pixel below the intensity threshold. We require that the
    # next shadow_edge_min_width are also shadowed.
    shadowed = [bool(i < shadow_threshold) for i in intensities[bottom_pixel:]]
    bottom_shadow_pixel = bottom_pixel + _consecutive_true_values(
        shadowed,
        shadow_edge_min_width,
        raise_message="Could not find bottom shadow edge",
    )

    # Find the first illuminated pixel above the bar.
    illuminated = [
        bool(i > shadow_threshold)
        for i in intensities[
            bottom_shadow_pixel + shadow_edge_min_width : top_pixel + 1
        ]
    ]
    # Don't raise if the pixel is not found
    above_shadow_pixel = (
        bottom_shadow_pixel
        + shadow_edge_min_width
        + _consecutive_true_values(illuminated, 1, raise_message=None)
    )

    # Check for a faulty tube: we want a certain number of pixels not in the bar shaddow
    active_tube_length = top_pixel - bottom_pixel + 1
    shadow_length = above_shadow_pixel - bottom_shadow_pixel
    if active_tube_length < min_illuminated_length + shadow_length:
        raise RuntimeError("Faulty tube found")

    return dict(
        bottom_pixel=bottom_pixel,
        top_pixel=top_pixel,
        bottom_shadow_pixel=bottom_shadow_pixel,
        above_shadow_pixel=above_shadow_pixel,
    )


@namedtuplefy
def fit_positions(
    edge_pixels,
    bar_positions,
    tube_pixels=256,
    order=5,
    ignore_value=INCORRECT_PIXEL_ASSIGNMENT,
    permissive=False,
):
    r"""
    Fit the position and heights of the pixels in a tube. The bar_positions as a function of
    edge pixels are fitted to a nth order polynomial (by default n=5). The positions of the pixels along the
    tube are the values of the polynomial at integer points, while the heights are the derivatives.

    Description from the master requirements document, section A2.1

    All pixel indexes start from the bottom of the tube, with the first
    index being zero.

    Uses :ref:`~numpy.polynomial.polynomial.polyfit`.

    devs - Andrei Savici <saviciat@ornl.gov>,

    Parameters
    ----------
    edge_pixels: list (or numpy array)
        the bottom pixel for each bar position, as found in `find_edges` function
    bar_positions: list (or numpy array)
        the bar position from the logs for each file in the bar scan
    tube_pixels: integer
        number of pixels for which to calculate positions and heights
    order: integer
        the order of polynomial to be used in the fit (default 5)
    ignore_value: int
        certain positions of the bar (close to the top and bottom of the tube) results in incorrect assignment of the
        edge pixel. In those cases it is expected that the edge pixel has a particular value that flags incorrect
        assignment. The default value is INCORRECT_PIXEL_ASSIGNMENT. These edge pixels will be
        ignored when carrying out the fit.
    permissive: bool
        If :py:obj:`True`, then fitted positions and heights are allowed to be non-physical. Only for debugging.

    Returns
    -------
    namedtuple
        the fields of the name tuple are:
        - calculated_positions: calculated positions of the pixels
        - calculated_heights: calculated pixel heights
    """
    message_len = "The positions of the bar and edge pixels have to be the same length"
    assert len(edge_pixels) == len(bar_positions), message_len

    # Ignore the incorrectly assigned edge pixels
    edge_pixels = np.array(edge_pixels)
    bar_positions = np.array(bar_positions)
    valid_edge_pixels = edge_pixels[np.where(edge_pixels != ignore_value)]
    valid_bar_positions = bar_positions[np.where(edge_pixels != ignore_value)]

    try:
        # fit the bar positions to a 5th degree polynomial in edge_pixels
        coefficients = np.polynomial.polynomial.polyfit(
            valid_edge_pixels, valid_bar_positions, int(order)
        )
        # calculate the coefficients of the derivative
        deriv_coefficients = np.polynomial.polynomial.polyder(coefficients)
        # evaluate the positions. Should be monotonically increasing
        calculated_positions = np.polynomial.polynomial.polyval(
            np.arange(tube_pixels), coefficients
        )
        position_jumps = np.diff(calculated_positions)
        if permissive is False and position_jumps[position_jumps <= 0.0].size > 0:
            raise ValueError(
                f"Pixel positions do not increase monotonically starting from the bottom of the tube\n"
                f"Positions = : {calculated_positions}"
            )
        # evaluate the heights. All should be positive
        calculated_heights = np.polynomial.polynomial.polyval(
            np.arange(tube_pixels), deriv_coefficients
        )
        if (
            permissive is False
            and calculated_heights[calculated_heights <= 0.0].size > 0
        ):
            raise ValueError(
                f"Some of the calculated heights are negative.\n"
                f"Heights = {calculated_heights}"
            )
    except Exception:
        coefficients = np.ones(int(order)) * np.nan
        calculated_positions = np.ones(tube_pixels) * np.nan
        calculated_heights = np.ones(tube_pixels) * np.nan

    return dict(
        calculated_positions=calculated_positions,
        calculated_heights=calculated_heights,
        coefficients=coefficients,
    )


def event_splitter(
    barscan_workspace,
    split_workspace=None,
    info_workspace=None,
    bar_position_log="dcal_Readback",
):
    r"""
    Split a Nexus events file containing a full bar scan, saving the splitting information into a 'split'
    and 'info' workspaces. This information is used later by ```barscan_workspace_generator```.

    It is assumed that the bar is shifted by a fixed amount every time we go on to the next scan.

    devs - Jose Borreguero <borreguerojm@ornl.gov>

    Parameters
    ----------
    barscan_workspace: str
        Path to barscan run file containing multiple positions of the bar.
    split_workspace: str
        Name of the table workspace to be used as event splitter workpsce in algorithm FilterEvents. If
        :py:obj:`None`, a random name will be provided.
    info_workspace: str
        Name of the table workspace to be used along ``split_workspace`` in algorithm FilterEvents. If
        :py:obj:`None`, a random name will be provided.
    bar_position_log: str
        Name of the log entry in the barscan run file containing the position of the bar (Y-coordinate, in 'mm')
        with respect to some particular frame of reference, not necessarily the one located at the sample.
    Returns
    -------
    list
        List of bar positions
    """
    if split_workspace is None:
        split_workspace = unique_workspace_dundername()
    if info_workspace is None:
        info_workspace = unique_workspace_dundername()

    # Find the amount by which the position of the bar is shifted every time we go on to the next scan.
    # It is assumed that this shift is a fixed amount
    bar_positions = SampleLogs(barscan_workspace)[bar_position_log].value
    bar_delta_positions = (
        bar_positions[1:] - bar_positions[:-1]
    )  # list of shifts in the position of the bar
    # Only retain shifts where the bar position increases
    bar_delta_positions = bar_delta_positions[bar_delta_positions > 0]
    # Find the most likely shift of the bar position, within two significant figures. Even thought the shift
    # is supposed to be constant, minute fluctuations over this value (<1%) are often encountered
    # We round-off bar shifts up to two decimal places, then create a histogram of these values, and pick
    # the shift having the largest count.
    bar_step = (
        float(np.bincount(np.round(100 * bar_delta_positions).astype("int")).argmax())
        / 100.0
    )
    # Mantid algorithm that creates the 'split' and 'info' workspaces using the bar positions stored in the
    # metadata
    GenerateEventsFilter(
        InputWorkspace=barscan_workspace,
        OutputWorkspace=split_workspace,
        InformationWorkspace=info_workspace,
        UnitOfTime="Nanoseconds",
        LogName=bar_position_log,
        LogValueInterval=bar_step,
        LogValueTolerance=bar_step / 2,
        MinimumLogValue=min(bar_positions),
        MaximumLogValue=max(bar_positions),
    )
    # Read bar positions from the generated info_workspace. This is a table workspace whose the second column
    # has entries as strings of the form: "Log.dcal_Readback.From.{min}.To.{max}.Value-change-direction:both"
    # We parse this string to fetch the 'min' and 'max' values of the bar
    bar_positions = list()
    for min_max_bar_position in mtd[info_workspace].column(1):
        min_bar_position = float(
            min_max_bar_position.split(".From.")[-1].split(".To.")[0]
        )
        max_bar_position = float(
            min_max_bar_position.split(".To.")[-1].split(".Value")[0]
        )
        bar_positions.append(
            (min_bar_position + max_bar_position) / 2
        )  # average between 'min' and 'max'
    return bar_positions


def barscan_workspace_generator(
    barscan_dataset, bar_position_log="dcal_Readback", mask=None, delete_workspaces=True
):
    r"""
    A python generator to be used when the user wants to iterate over the runs that hold the bar at a fixed
    position. Each iteration prompts this generator to return the position of the bar and a workspace containing the
    intensities for the run that held the bar at the returned position.

    devs - Jose Borreguero <borreguerojm@ornl.gov>

    Parameters
    ----------
    barscan_dataset: str, list
        Path(s) to barscan run file(s), or list of workspaces. If only one file, it should contain multiple
        positions of the bar. The generator will split the file into multiple workpaces, each one containing
        the scan of the bar at a particular position. If a list of files, then each file contains the
        pixel_intensities recorded with a constant position for the bar. If a list of workspaces, each workspace
        must contain the same information as when passing a list of files.
    bar_position_log: str
        Name of the log entry in the barscan run file containing the position of the bar (Y-coordinate, in 'mm')
        with respect to some particular frame of reference, not necessarily the one located at the sample.
    mask: mask file path, ~mantid.api.MaskWorkspace, :py:obj:`list`
        A mask to be applied. If :py:obj:`list`, it is a list of detector ID's.
    delete_workspaces: Bool
        Delete temporary workspaces one we have iterated over all the scans.

    Returns
    -------
    tuple
        A two-item tuple containing, in this order:
        - the position of the bar as stated in the logs the run currently being returned.
        - the name of the workspace containing the run currently being returned.
    """
    temporary_workspaces = list()  # store the names of the workspaces to be removed

    if isinstance(barscan_dataset, str):
        # the whole barscan is contained in a single file. Must be splitted into subruns. Each subrun will contain
        # intensities for a run with the bar held at a fixed position.
        spliter_workspace = unique_workspace_dundername()
        info_workspace = unique_workspace_dundername()
        barscan_workspace = unique_workspace_dundername()
        temporary_workspaces.extend(
            [spliter_workspace, info_workspace, barscan_workspace]
        )
        # BOTTLENECK
        LoadEventNexus(barscan_dataset, OutputWorkspace=barscan_workspace)
        # Create the splitting scheme and save it in table workspaces `spliter_workspace` and `info_workspace`.
        bar_positions = event_splitter(
            barscan_workspace,
            split_workspace=spliter_workspace,
            info_workspace=info_workspace,
            bar_position_log=bar_position_log,
        )
        splitted_workspace_group = unique_workspace_dundername()  # group of subruns
        # Mantid algorithm using the 'spliter' and 'info' tables to carry out the splitting of
        # `barscans_workspace into a set of subruns.
        # BOTTLENECK
        FilterEvents(
            InputWorkspace=barscan_workspace,
            SplitterWorkspace=spliter_workspace,
            InformationWorkspace=info_workspace,
            OutputWorkspaceBaseName=splitted_workspace_group,
            GroupWorkspaces=True,
            TimeSeriesPropertyLogs=[bar_position_log],
            ExcludeSpecifiedLogs=False,
        )
        temporary_workspaces.append(splitted_workspace_group)
        temporary_workspaces.append(
            "TOFCorrectWS"
        )  # spurious workspace spawned by FilterEvents
        barscan_workspaces = [
            splitted_workspace_group + "_" + str(i) for i in range(len(bar_positions))
        ]
    else:  # of a set of files or workspaces, each contains intensities for a scan with the bar fixed
        # determine if the list contains files or workspaces
        first_scan = barscan_dataset[0]
        if isinstance(first_scan, str) and os.path.exists(
            first_scan
        ):  # list of files, thus load into workspaces
            loader = loader_algorithm(barscan_dataset[0])
            barscan_workspaces = list()
            barscan_workspace_basename = unique_workspace_dundername()
            for scan_index, scan_data in enumerate(barscan_dataset):
                barscan_workspace = f"{barscan_workspace_basename}_{scan_index:03d}"
                temporary_workspaces.append(barscan_workspace)
                loader(scan_data, OutputWorkspace=barscan_workspace)
                barscan_workspaces.append(barscan_workspace)
        else:  # barscan_dataset is a set of workspaces
            barscan_workspaces = barscan_dataset
        # bar_positions = [SampleLogs(barscan_workspace).find_log_with_units(bar_position_log, 'mm')
        #                  for barscan_workspace in barscan_workspaces]
        bar_positions = list()
        for barscan_workspace in barscan_workspaces:
            try:
                bar_position = SampleLogs(barscan_workspace).find_log_with_units(
                    bar_position_log, "mm"
                )
            except RuntimeError as run_error:
                raise RuntimeError(f"Workspace {str(barscan_workspace)}: {run_error}")
            bar_positions.append(bar_position)

    # Serve bar positions and workspaces, one at a time
    for bar_position, barscan_workspace in zip(bar_positions, barscan_workspaces):
        if mask is not None:
            apply_mask(barscan_workspace, mask=mask)
        yield bar_position, barscan_workspace

    # Clean up workspaces we instantiated for each scan
    if delete_workspaces is True:
        DeleteWorkspaces(temporary_workspaces)


# flake8: noqa: C901
def calculate_barscan_calibration(
    barscan_dataset,
    component="detector1",
    bar_position_log="dcal_Readback",
    formula=None,
    order=5,
    mask=None,
    inspect_data=False,
    permissive_fit=False,
):
    r"""
    Calculate pixel positions (only Y-coordinae) as well as pixel heights from a barscan calibration session.

    **Mantid Algorithms used:**
    :ref:`Load <algm-Load-v1>`,

    devs - Andrei Savici <saviciat@ornl.gov>,
           Jose Borreguero <borreguerojm@ornl.gov>

    Parameters
    ----------
    barscan_dataset: str, list
        Path(s) to barscan run file(s), or list of workspaces. If only one file, it should contain multiple
        positions of the bar. If a list of files, then each file contains the pixel_intensities recorded with a
        constant position for the bar. If a list of workspaces, each workspace must contain the same information as
        when passing a list of files.
    component: str
        Name of the detector panel scanned with the bar. Usually, 'detector1`.
    bar_position_log: str
        Name of the log entry in the barscan run file containing the position of the bar (Y-coordinate, in 'mm')
        with respect to some particular frame of reference, not necessarily the one located at the sample.
    formula: str
        Formula to obtain the position of the bar (Y-coordinate) in the frame of reference located at the sample.
    order: int
        Highest degree for the polynomial that will fit the observed positions of the bar.
    mask: mask file path, ~mantid.api.MaskWorkspace, :py:obj:`list`
        A mask to be applied. If :py:obj:`list`, it is a list of detector ID's.
    inspect_data: Bool
        Additional pieces of data returned by this function in order to assess the correctness of the barscan
        calculation. These data are returned as a dictionary with the current entries:
        - bar_positions: list of Y-coordinates of the bar for each scan holding the bar at a particular position.
        - bar workspaces: list of ~mantid.api.MatrixWorkspace objects, each containing the bar at a particular
        position.
        - bottom_shadow_pixels: ~numpy.ndarray of shape (number of scans, number of tubes) listing the indexes for
        the pixels shadowed by the lower portion of the bar.
    permissive_fit: bool
        If :py:obj:`True`, then fitted positions and heights are allowed to be non-physical. Only for debugging.

    Returns
    -------
    ~drtsans.pixel_calbration.Table, dict
        If ```inspect_data``` is :py:obj:`False`, only a table object is returned. Otherwise a tube is returned
        where the first component is the table, and the second item is a dictionary with the additional pieces of data.
    """
    addons = dict(
        bar_positions=[], bar_workspaces=[]
    )  # for inspecting the result of the calibration
    instrument_name, number_pixels_in_tube, number_tubes = (
        None,
        None,
        None,
    )  # initialize some variables
    run_numbers, daystamp = set(), None  # initialize some variables
    bar_positions = []  # Y-coordinates of the bar for each tube and scan
    # `bottom_shadow_pixels` is a 2D array defining the position of the bar on the detector, in pixel coordinates
    # The first index corresponds to the Y-axis (along each tube), the second to the X-axis (across tubes)
    # Thus, bottom_shadow_pixels[:, 0] indicates bottom shadow pixel coordinates along the very first tube
    # bottom_shadow_pixels.shape = (number of scans, number of tubes)
    bottom_shadow_pixels = []
    delete_workspaces = (
        True if inspect_data is False else False
    )  # retain workspace per scan if we want to inspect

    for bar_position, barscan_workspace in barscan_workspace_generator(
        barscan_dataset,
        bar_position_log=bar_position_log,
        mask=mask,
        delete_workspaces=delete_workspaces,
    ):
        if instrument_name is None:  # retrieve some info from the first bar position
            instrument_name = instrument_standard_name(barscan_workspace)
            daystamp = day_stamp(barscan_workspace)
            # instantiate a bar formula using either the instrument default or a user's formula
            if formula is None:
                bar_formula = BarPositionFormula(
                    instrument_component=(instrument_name, component)
                )
            else:
                bar_formula = BarPositionFormula(formula=formula)
            bar_formula.validate_top_position(bar_position)
        run_numbers.add(int(SampleLogs(barscan_workspace).run_number.value))
        # Find out the Y-coordinates of the bar in the reference-of-frame located at the sample
        bottom_shadow_pixels_per_scan = (
            []
        )  # For the current scan, we have one bottom shadow pixel for each tube
        if number_pixels_in_tube is None:
            # We create a tube collection to figure out the pixel indexes for each tube.
            # A TubeCollection is a list of TubeSpectrum objects, each representing a physical tube. Here
            # we obtain the list of tubes for the selected double-detector-panel array.
            # The view 'decreasing X' sort the tubes by decreasing value of their corresponding X-coordinate.
            # In this view, a double detector panel looks like a single detector panel. When looking at
            # the panel standing at the sample, the leftmost tube has the highest X-coordinate, so the
            # 'decreasing X' view orders the tubes from left to right.
            collection = TubeCollection(barscan_workspace, component).sorted(
                view="decreasing X"
            )
            # pixel_indexes is a list of length equal the number of tubes. Each item in the list is itself a list,
            # containing the pixel spectrumInfo indexes for a particular tube.
            pixel_indexes = [tube.spectrum_info_index for tube in collection]
            number_tubes, number_pixels_in_tube = len(collection), len(collection[0])
            # Find the detector ID's for the tube collection, that is, the selected component
            # BOTTLENECK, but it's run one time
            detector_ids = list(
                itertools.chain.from_iterable(tube.detector_ids for tube in collection)
            )
        bar_positions.append(
            [bar_formula.evaluate(bar_position, i) for i in range(number_tubes)]
        )
        # pixel_intensities is a list, whose items are the integrated intensities for each pixel. The index of this
        # list coincides with the spectrumInfo index.
        pixel_intensities = np.sum(mtd[barscan_workspace].extractY(), axis=1)
        for (
            pixel_indexes_in_tube
        ) in pixel_indexes:  # iterate over each tube, retrieving its pixel indexes
            try:
                # Find the bottom shadow pixel for the current tube and current barscan run
                pixel_intensities_in_tube = pixel_intensities[pixel_indexes_in_tube]
                bottom_shadow_pixels_per_scan.append(
                    find_edges(pixel_intensities_in_tube).bottom_shadow_pixel
                )
            except IndexError:  # tube masked or malfunctioning
                bottom_shadow_pixels_per_scan.append(INCORRECT_PIXEL_ASSIGNMENT)
            except RuntimeError:  # tube masked or malfunctioning
                bottom_shadow_pixels_per_scan.append(INCORRECT_PIXEL_ASSIGNMENT)
        bottom_shadow_pixels.append(bottom_shadow_pixels_per_scan)
        # Add iteration info to the addons
        addons["bar_positions"].append(bar_position)
        addons["bar_workspaces"].append(barscan_workspace)

    bottom_shadow_pixels = np.array(bottom_shadow_pixels)
    bar_positions = np.array(bar_positions)

    # Deal with corner cases not resolved with the find_edges algorithm
    resolve_incorrect_pixel_assignments(bottom_shadow_pixels, bar_positions)
    addons["bottom_shadow_pixels"] = bottom_shadow_pixels

    if len(bottom_shadow_pixels) <= order:
        raise ValueError(
            f"There are not enough bar positions to fo a fit with a polynomyal of order {order}."
        )

    # fit pixel positions for each tube
    positions, heights = [], []
    for tube_index in range(number_tubes):  # iterate over the tubes in the collection
        # Fit the pixel numbers and Y-coordinates of the bar for the current tube with a polynomial
        try:
            fit_results = fit_positions(
                bottom_shadow_pixels[:, tube_index],
                bar_positions[:, tube_index],
                order=order,
                tube_pixels=number_pixels_in_tube,
                permissive=permissive_fit,
            )
        except ValueError as e:
            raise ValueError(f"In tube index {tube_index}: {e}")
        # Store the fitted Y-coordinates and heights of each pixel in the current tube
        # Store as lists so that they can be easily serializable
        positions.append(
            list(1.0e-03 * fit_results.calculated_positions)
        )  # store with units of meters
        heights.append(
            list(1.0e-03 * fit_results.calculated_heights)
        )  # store with units of meters

    # Find the average pixel positions and heights in a tube using all good tubes. Then apply these to
    # any bad tube
    positions = np.array(positions)  # shape = (number_tubes, number_pixels_in_tube)
    heights = np.array(heights)
    bad_tube_indexes = np.isnan(
        np.sum(positions, axis=1)
    )  # array containing True at the index of a bad tube
    if np.any(bad_tube_indexes):  # there's at least one bad tube
        average_positions = np.average(positions[~bad_tube_indexes], axis=0)
        average_heights = np.average(heights[~bad_tube_indexes], axis=0)
        for tube_index in np.where(bad_tube_indexes)[
            0
        ]:  # insert averages in the bad tubes
            positions[tube_index] = average_positions
            heights[tube_index] = average_heights
    positions = list(positions.ravel())
    heights = list(heights.ravel())

    metadata = dict(
        caltype=CalType.BARSCAN.name,
        instrument=instrument_name,
        component=component,
        daystamp=daystamp,
        runnumbers=sorted(list(run_numbers)),
    )
    calibration = Table(
        metadata, detector_ids=detector_ids, positions=positions, heights=heights
    )

    # decide on what to return
    if inspect_data is True:
        return calibration, addons
    else:
        return calibration


def bad_tube(bottom_shadow_pixels):
    r"""
    Check if all pixels for this tube have been flagged as ```INCORRECT_PIXEL_ASSIGNMENT```. Happens when the
    tube has been masked or it was malfunctioning when the scans were taken.

    Parameters
    ----------
    bottom_shadow_pixels: list
        Pixel indexes defining the positions of the bottom of the bar in the barscan

    Returns
    -------
    bool
    """
    return list(set(bottom_shadow_pixels)) == [
        INCORRECT_PIXEL_ASSIGNMENT,
    ]


def resolve_incorrect_pixel_assignments(bottom_shadow_pixels, bar_positions):
    r"""
    The algorithm finding the position of the bottom of the bar sometimes fails when the bar is close to the
    bottom of the detector, finding two corner cases.

    Corner case 1:
    When the bar approaches the bottom of the tubes, the bottom edge of its shadow blends with the non-functioning
    pixels at the bottom of the tubes. In this scenario, the bottom edge of the bar is assigned as the first
    non-functioning pixel at the top of the tube. Thus, we must find out for each tube a sudden jump in the
    identified bottom shadow pixel.
    Example: the bottom shadow pixels for the first tube as we change the position of the bar have been identified:
       249, 230, 209, 187, 165, 145, 129, 103, 82, 67, 52, 39, 21, 249, 249
    In the example, the last two bottom pixels are erroneous. They must be set to INCORRECT_PIXEL_ASSIGNMENT

    Corner case 2:
    When the identified bottom pixel in a tube is far from the boundary between the illuminated pixels and the non
    illuminated pixels, it leaves a few non-illuminated pixels as active pixels. The bottom shadow pixel can then
    be incorrectly assigned as one of these non-illuminated pixels. The results are outliers pixel indexes.
    Example: 249 248 246 4 244 242.... Here "4" is in incorrectly assigned pixel index as bottom of the bar
    We must find these outliers and assign them as incorrect. We use a linear fit to find out outliers.

    Fixes to corner cases are skippedd for bad tubes (see ~drtsans.pixel_calibration.bad_tube).

    devs - Jose Borreguero <borreguerojm@ornl.gov>

    Parameters
    ----------
    bottom_shadow_pixels: numpy.ndarray
        2D array defining the position of the bar on the detector, in pixel coordinates.
        The first array index corresponds to the Y-axis (along each tube), the second to the X-axis (across tubes).
        Thus, bottom_shadow_pixels[:, 0] indicates bottom shadow pixel coordinates along the very first tube
    bar_positions: numpy.ndarray
        2D array defining the position of the bar on the detector, in mili meters.
        The first array index corresponds to the Y-axis (along each tube), the second to the X-axis (across tubes).
        Thus, bar_positions[:, 0] indicates bottom shadow Y-coordinates along the very first tube
    """
    number_bar_positions, number_tubes = bottom_shadow_pixels.shape
    # Iterate over each tube
    for tube_index in range(number_tubes):
        if bad_tube(bottom_shadow_pixels[:, tube_index]):
            continue  # this tube has all pixels marked as incorrectly assigned. Don't attempt to fix corner issues.
        #
        # Correct corner case 1
        # We scan the end of the tube for possible missassignments
        jump_index = None
        y = bottom_shadow_pixels[:, tube_index]
        begin_index = number_bar_positions - 2  # start next to last index
        end_index = max(0, number_bar_positions - 12)  # look at the last 12 pixels
        for i in range(
            begin_index, end_index, -1
        ):  # start from the bottom of the tube, work upwards
            if abs(y[i] - y[i - 1]) > 10 * max(
                1, abs(y[i + 1] - y[i])
            ):  # value/factor of 10 selected as threshold
                jump_index = i  # The position of the bar jumped by more than 10 pixels. Flags an incorrect assigment
                break
        if jump_index is not None:
            y[jump_index:] = INCORRECT_PIXEL_ASSIGNMENT
        #
        # Correct corner case 2
        y = bottom_shadow_pixels[:, tube_index]  # bar positions in pixel coordinates
        x = bar_positions[:, tube_index]  # bar positions in mili meters
        # Filter out bad pixels
        y_correct = y[y != INCORRECT_PIXEL_ASSIGNMENT]
        x_correct = x[y != INCORRECT_PIXEL_ASSIGNMENT]  # bar positions, in mili-meters
        # Linear fit between bar positions in pixel coordinates and mili-meters
        try:
            coefficients = np.polynomial.polynomial.polyfit(x_correct, y_correct, 1)
        except TypeError:  # empty array `x_correct`, meaning this tube was masked or malfunctioning
            continue
        # Find residuals of correct pixels
        y_fitted = np.polynomial.polynomial.polyval(x_correct, coefficients)
        residuals = np.abs(
            y_correct - y_fitted
        )  # deviations between the linear fit and the actual positions
        large_residual = np.average(residuals) + 1.5 * np.std(residuals)
        # Find residuals of correct and incorrect pixels. Residuals for incorrect pixels are nonsense,
        # but we include them because we need array `residuals` and array `y` of same length.
        y_fitted = np.polynomial.polynomial.polyval(x, coefficients)
        residuals = np.abs(y - y_fitted)
        # We flag as incorrect assignments those correct pixels with bar positions largely deviating from the linear
        # fit. The incorrect pixels already have nonsense large residuals
        y[
            (residuals > large_residual) & (y != INCORRECT_PIXEL_ASSIGNMENT)
        ] = INCORRECT_PIXEL_ASSIGNMENT


def calculate_apparent_tube_width(
    flood_input, component="detector1", load_barscan_calibration=True, db_file=None
):
    r"""
    Determine the tube width most efficient for detecting neutrons. An effective tube (or pixel) diameter is
    determined for tubes in the front panel, and likewise for the tubes in the back panel.

    devs - Jose Borreguero <borreguerojm@ornl.gov>

    **Mantid algorithms used:**
    :ref:`DeleteWorkspaces <algm-DeleteWorkspaces-v1>`,
    :ref:`Integration <algm-Integration-v1>`,
    :ref:`MaskDetectors <algm-MaskDetectors-v1>`,
    :ref:`MaskDetectorsIf <algm-MaskDetectorsIf-v1>`,
    :ref:`ReplaceSpecialValues <algm-ReplaceSpecialValues-v1>`,

    Parameters
    ----------
    flood_input: str, ~mantid.api.IEventWorkspace, ~mantid.api.MatrixWorkspace
        Path to flood run, flood workspace name, or flood workspace object.
    component: str
        Name of the instrument component containing the detector array consisting of two parallel panels of tubes.
    load_barscan_calibration: bool
        Load pixel positions and heights from the pixel-calibrations database appropriate to ```input_workspace```. If
        :py:obj:`False`, then the pixel positions and heigths will be those of ```input_workspace```.
    db_file: str, None
        database file (json format).  None will load the default database for the selected instrument.
        Otherwise the combination load_barscan_calibration=True, db_file=None may come across as some data
        contradictory

    Returns
    -------
    dict
        Dictionary containing the following keys:
        - instrument, str, Standard name of the instrument.
        - component, str, name of the double detector array, usually "detector1".
        - run, int, run number of ``input_workspace``.
        - unit: str, the units for the tube widths. Set to 'mm' for mili-meters.
        - widths, list, A two-item list containing the apparent widths for the front and back tubes.
    """
    # Determine the type of input for the flood data
    if file_exists(flood_input):
        input_workspace = unique_workspace_dundername()
        Load(Filename=flood_input, OutputWorkspace=input_workspace)
    else:
        input_workspace = str(flood_input)  # workspace object or workspace name

    integrated_intensities = unique_workspace_dundername()
    Integration(InputWorkspace=input_workspace, OutputWorkspace=integrated_intensities)

    # Mask non-finite pixel_intensities (nan, inf). They can't be used in the calculation.
    #
    # Replace non-finite pixel_intensities with a value of -1
    ReplaceSpecialValues(
        InputWorkspace=integrated_intensities,
        OutputWorkspace=integrated_intensities,
        NanValue=-1,
        NanError=-1,
        InfinityValue=-1,
        InfinityError=-1,
    )
    # Mask detectors with negative pixel_intensities
    mask_workspace = unique_workspace_dundername()
    MaskDetectorsIf(
        InputWorkspace=integrated_intensities,
        Operator="Less",
        Value=0.0,
        OutputWorkspace=mask_workspace,
    )
    MaskDetectors(Workspace=integrated_intensities, MaskedWorkspace=mask_workspace)

    # Update pixel positions and heights with the appropriate calibration, if so requested.
    if load_barscan_calibration is True:
        calibration = load_calibration(
            input_workspace, "BARSCAN", component=component, database=db_file
        )
        calibration.apply(integrated_intensities)

    # Calculate the count density for each tube. Notice that if the whole tube is masked, then the associated
    # intensity is stored as nan.
    #
    # Sort the tubes according to the X-coordinate in decreasing value. This is the order when sitting on the
    # sample and iterating over the tubes "from left to right"
    collection = TubeCollection(integrated_intensities, component).sorted(
        view="fbfb"
    )  # BOTTLENECK
    detector_ids = list(
        itertools.chain.from_iterable(tube.detector_ids for tube in collection)
    )
    count_densities = list()
    for tube in collection:
        weighted_intensities = tube.readY.ravel() / tube.pixel_heights
        d = np.mean(weighted_intensities[~tube.isMasked])
        count_densities.append(d)
    count_densities = np.array(
        count_densities
    )  # is convenient to cast densities into a numpy array data structure.

    # Determine the count densities per panel and for the whole detector array.
    # We must be careful to pick only tubes with finite densities (avoid 'nan')
    average_count_density = np.mean(count_densities[np.isfinite(count_densities)])
    front_count_density = np.mean(
        count_densities[::2][np.isfinite(count_densities[::2])]
    )  # front tubes, even indexes
    back_count_density = np.mean(
        count_densities[1::2][np.isfinite(count_densities[1::2])]
    )  # back tubes, odd indexes

    # Determine the front and back pixel widths
    nominal_width = collection[0][0].width  # width of the first pixel in the first tube
    front_width = (front_count_density / average_count_density) * nominal_width
    back_width = (back_count_density / average_count_density) * nominal_width

    # Generate a list of pixel widths. It is assumed that front tubes have an even tube index
    widths = list()
    for tube_index, tube in enumerate(collection):
        pixel_width = front_width if tube_index % 2 == 0 else back_width
        widths.extend([pixel_width] * len(tube))

    DeleteWorkspaces(integrated_intensities, mask_workspace)

    metadata = dict(
        caltype=CalType.TUBEWIDTH.name,
        instrument=instrument_standard_name(input_workspace),
        component=component,
        daystamp=day_stamp(input_workspace),
        runnumbers=[
            int(SampleLogs(input_workspace).run_number.value),
        ],
    )
    return Table(metadata, detector_ids=detector_ids, widths=widths)


@namedtuplefy
def as_intensities(
    input_workspace, component="detector1", views=["positions", "heights", "widths"]
):
    r"""
    Returns one workspace for each pixel property that is calibrated (e.g., pixel height),
    and the calibration datum is stored as the intensity value for that pixel. Intended to
    visualize the calibration in MantidPlot's instrument viewer. Not required for calibration
    generation or for data reduction.

    Generated workspaces are ```input_name_positions```, ```input_name_heights```,
    and ```input_name_widths```, where ```input_name``` is the name of the input workspace.

    Note: Positions for visualization in Mantid's instrument view are shifted so that the
          lowest position (usually a negative number) becomes zero. The reason being that
          showing the instrument in Mantid will mask negative intensities, and we want to avoid this.

    **Mantid algorithms used:**
    :ref:`CreateWorkspace <algm-CreateWorkspace-v1>`,
    <https://docs.mantidproject.org/algorithms/CreateWorkspace-v1.html>
    :ref:`LoadEmptyInstrument <algm-LoadEmptyInstrument-v1>`,
    <https://docs.mantidproject.org/algorithms/LoadEmptyInstrument-v1.html>
    :ref:`LoadInstrument <algm-LoadInstrument-v1>`,
    <https://docs.mantidproject.org/algorithms/LoadInstrument-v1.html>

    Parameters
    ----------
    input_workspace: str, ~mantid.api.MatrixWorkspace, ~mantid.api.IEventsWorkspace
        Workspace from which pixel properties are retrieved.
    component: str, list
        Name or list of names for the double detector array panels. For BIOSANS we have 'detector1' or
        'wing-detector'.
    views: list
        Generate views for the pixel properties provided.

    Returns
    -------
    namedtuple
        A namedtuple containing the ~mantid.api.MatrixWorkspace workspaces
        with fields 'positions', 'positions_mantid', 'heights', and 'widths'
    """
    if isinstance(component, str):
        components = [
            component,
        ]
    else:
        components = component

    # methods TubeSpectrum.pixel_y, TubeSpectrum.pixel_heights, and TubeSpectrum.pixel_widths
    tube_properties = {
        "positions": "pixel_y",
        "heights": "pixel_heights",
        "widths": "pixel_widths",
    }

    # collect pixel properties and associated workspace indexes
    workspace_indexes = np.array([], dtype=int)
    pixel_props = dict()  # on entry for each view, each entry is a 1D numpy array
    for current_component in components:
        collection = TubeCollection(input_workspace, current_component).sorted(
            view="decreasing X"
        )
        for tube in collection:
            workspace_indexes = np.hstack(
                (workspace_indexes, tube.spectrum_info_index)
            )  # workspace indexes for the tube
            for view in views:  # 'positions', 'heights', 'widths'
                pixel_props_collected = pixel_props.get(
                    view, np.array([])
                )  # pixel properties up to the current tube
                addition = getattr(
                    tube, tube_properties[view]
                )  # pixel properties for the current tube
                pixel_props[view] = np.hstack((pixel_props_collected, addition))

    # Mantid can only show positive quantities in the instrument view
    if "positions" in pixel_props:
        pixel_props["positions_mantid"] = pixel_props["positions"] - np.min(
            pixel_props["positions"]
        )

    number_histograms = mtd[str(input_workspace)].getNumberHistograms()
    intensities = np.zeros(number_histograms)

    returned_views = {}
    for cal_prop in pixel_props:  # 'positions', 'heights', 'widths', 'positions_mantid'
        output_workspace = f"{str(input_workspace)}_{cal_prop}"  # Workspace containing the property as  intensity
        # intensties will be non-zero only for workpace indexes that have associated pixels of interests
        intensities[workspace_indexes] = pixel_props[cal_prop]
        workspace = Integration(
            InputWorkspace=input_workspace, OutputWorkspace=output_workspace
        )
        for index in range(number_histograms):
            workspace.dataY(index)[:] = intensities[index]
        returned_views[cal_prop] = mtd[output_workspace]

    return returned_views


def split_barscan_run(input_file, output_directory, bar_position_log="dcal_Readback"):
    r"""
    Split a barscan file containing many positions of the bar into a set of files each holding the bar at a
    unique position.

    The input file must be an events file. The output files contain only the total intensity per pixel.
    If input file is of the name 'INST_1234.nxs.h5', the output files' names are 'INST_1234_0.nxs',
    'INST_1234_1.nxs', and so on.

    Parameters
    ----------
    input_file: str
        Events Nexus file containing a full barscan (many positions of the bar)

    ouput_directory: str
        Path where the individual scans are saved
    """
    barscan_workspace = unique_workspace_dundername()
    spliter_workspace = unique_workspace_dundername()
    info_workspace = unique_workspace_dundername()
    splitted_workspace_group = unique_workspace_dundername()

    LoadEventNexus(input_file, OutputWorkspace=barscan_workspace)
    bar_positions = event_splitter(
        barscan_workspace,
        split_workspace=spliter_workspace,
        info_workspace=info_workspace,
        bar_position_log=bar_position_log,
    )
    FilterEvents(
        InputWorkspace=barscan_workspace,
        SplitterWorkspace=spliter_workspace,
        InformationWorkspace=info_workspace,
        OutputWorkspaceBaseName=splitted_workspace_group,
        GroupWorkspaces=True,
        TimeSeriesPropertyLogs=[bar_position_log],
        ExcludeSpecifiedLogs=False,
    )
    os.makedirs(
        output_directory, exist_ok=True
    )  # Create directory, and don't complain if already exists
    basename = os.path.basename(input_file).split(".nxs")[0]
    for i in range(len(bar_positions)):
        workspace = splitted_workspace_group + "_" + str(i)
        AddSampleLog(
            Workspace=workspace,
            LogName=bar_position_log,
            LogText=str(bar_positions[i]),
            LogType="Number Series",
            LogUnit="mm",
        )
        Rebin(
            workspace,
            Params=[0, 1000000, 1000000],
            PreserveEvents=False,
            OutputWorkspace=workspace,
        )
        out_file = os.path.join(output_directory, f"{basename}_{i}.nxs")
        SaveNexus(InputWorkspace=workspace, Filename=out_file)

    # Clean up all workspaces
    DeleteWorkspaces(
        [splitted_workspace_group + "_" + str(i) for i in range(len(bar_positions))]
    )
    DeleteWorkspaces(
        [splitted_workspace_group, info_workspace, spliter_workspace, barscan_workspace]
    )
