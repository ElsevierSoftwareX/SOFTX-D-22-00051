from datetime import datetime
import h5py
import glob
import os
import re
import json
import socket
from mantid import __version__ as mantid_version
from mantid.kernel import logger

# from mantid.simpleapi import mtd, SaveNexusProcessed
import numpy as np
from drtsans import __version__ as drtsans_version
from os import environ, path


__all__ = ["savereductionlog"]


def _createnxgroup(parent, name, klass):
    child = parent.create_group(name)
    child.attrs["NX_class"] = klass
    return child


def _savenxnote(nxentry, name, mimetype, file_name, data):
    """Write an NXnote inside an entry
    http://download.nexusformat.org/doc/html/classes/base_classes/NXnote.html

    nxentry: HDF handle
        NXentry to put the NXnote into
    name: str
        Name of the NXnote group
    mimetype: str
        Mimetype of the data
    file_name: str
        Name of the file that the note was taken from
    data: str
        The contents of the note
    """
    nxnote = _createnxgroup(nxentry, name, "NXnote")

    nxnote.create_dataset(name="file_name", data=[np.string_(file_name)])
    nxnote.create_dataset(name="type", data=[np.string_(mimetype)])
    nxnote.create_dataset(name="data", data=[np.string_(data)])

    return nxnote


def _savepythonscript(nxentry, pythonfile, pythonscript):
    """Write the python script as a NXnote

    nxentry: HDF handle
        NXentry to put the NXnote into
    pythonfile: str
        Filename that was supplied to reduction
    pythonscript: str
        The python script itself
    """
    # read the contents of the script
    if not pythonscript and path.exists(pythonfile):
        with open(pythonfile, "r") as script:
            pythonscript = script.read()

    return _savenxnote(
        nxentry, "reduction_script", "text/x-python", pythonfile, pythonscript
    )


def _savereductionjson(nxentry, parameters):
    """Save the all of the reduction parameters as an NXnote

    nxentry: HDF handle
        NXentry to put the NXnote into
    parameters: dict or str
        The parameters supplied to the reduction script. This will be converted
        to a json string if it isn't one already.
    """

    if "filename" in parameters.keys():
        filename = parameters["filename"]
    else:
        filename = ""

    # convert the parameters into a string to save
    if not isinstance(parameters["data"], str):
        parameters = json.dumps(parameters["data"])

    return _savenxnote(
        nxentry,
        "reduction_json",
        "application/json",
        file_name=filename,
        data=parameters,
    )


def _savereductionparams(nxentry, parameters, name_of_entry):
    """save the reduction parameters as a nice tree structure"""
    if parameters is None:
        return

    nxentry = _createnxgroup(nxentry, name_of_entry, "NXnote")

    for _key, _value in parameters.items():
        if isinstance(_value, dict):
            _savereductionparams(nxentry, _value, _key)
        elif isinstance(_value, list):
            _new_entry = nxentry.create_dataset(name=_key, data=str(_value))
            _new_entry.attrs["NX_class"] = "NXdata"
        else:
            if _value is None:
                _value = ""
            _new_entry = nxentry.create_dataset(name=_key, data=_value)
            _new_entry.attrs["NX_class"] = "NXdata"


def _savenxprocess(nxentry, program, version):
    """Create a NXprocess for the specified program

    Parameters
    ----------
    nxentry: HDF handle
        NXentry to put the NXprocess into
    program: str
        name of the program
    version: str
        program's version string
    """
    nxprocess = _createnxgroup(nxentry, program, "NXprocess")
    nxprocess.create_dataset(name="program", data=[np.string_(program)])
    nxprocess.create_dataset(name="version", data=[np.string_(version)])


def _savenxlog(nxcollection, property):
    """Create a NXlog from the supplied property

    Parameters
    ----------
    nxcollection: HDF handle
        NXcollection that the NXlog should be added to
    property: PropertyWithValue
        log item to inspect and write its values to disk
    """
    nxlog = _createnxgroup(nxcollection, property.name, "NXlog")

    try:
        if isinstance(property.value, str):
            value = nxlog.create_dataset(
                name="value", data=[np.string_(property.value)]
            )
        elif len(property.value) > 1:
            value = nxlog.create_dataset(name="value", data=property.value)
        else:
            raise RuntimeError("Should never have gotten here")
    except TypeError:
        value = nxlog.create_dataset(name="value", data=[property.value])

    if value and property.units:
        value.attrs["units"] = property.units

    # TODO should get time from the logs but current examples don't have that
    # this code should work
    try:
        times = property.times
        if len(times) > 0:
            # convert to float in seconds
            epoch = times[0].toISO8601String()
            times = (times - times[0]) / np.timedelta64(1, "s")
            times = nxlog.create_dataset(name="time", data=times)
            times.attrs["offset"] = np.string_(epoch)
            times.attrs["units"] = "second"
    except AttributeError:
        pass  # doesn't have times

    return nxlog


def _savespecialparameters(nxentry, dict_special_parameters, name_of_entry):
    """Save the special parameters

    Parameters
    ----------
    nxentry: HDF handle
        Entry group to put information in
    dict_special_parameters: dict
        dictionary where to get the parameters from
    name_of_entry: String
        Name of the top tree structure
    """
    nxentry = _createnxgroup(nxentry, name_of_entry, "NXnote")

    for _key, _value in dict_special_parameters.items():
        if isinstance(_value, dict):
            _savereductionparams(nxentry, _value, _key)
        else:
            if _value is None:
                _value = ""
            _new_entry = nxentry.create_dataset(name=_key, data=_value)
            _new_entry.attrs["NX_class"] = "NXdata"


def _savesamplelogs(nxentry, dict_sample_logs, name_of_entry):
    """Save all the DAS logs infos

    Parameters
    ----------
    nxentry: HDF handle
        Entry group to put information in
    dict_sample_logs: SampleLogs object
    name_of_entry: String
        Name of the top tree structure
    """
    nxentry = _createnxgroup(nxentry, name_of_entry, "NXnote")

    for _sample_key in dict_sample_logs.keys():
        nxentry_log = _createnxgroup(nxentry, _sample_key, "NXnote")
        local_dict_sample_logs = dict_sample_logs[_sample_key]
        for _key in local_dict_sample_logs.keys():
            _value = str(local_dict_sample_logs[_key].value)
            if _value is None:
                _value = ""
            _units = str(local_dict_sample_logs[_key].units)
            if _units is None:
                _units = ""
            _new_entry = nxentry_log.create_dataset(name=_key, data=_value)
            _new_entry.attrs["NX_class"] = "NXdata"
            _new_entry.attrs["units"] = _units


def _create_groupe(entry=None, name="Default", data=[], units=""):
    if data is not None:
        _entry_group = entry.create_dataset(name=name, data=data)
        _entry_group.attrs["units"] = units


def _save_logslicedata(logslicedata={}, index=0, topEntry=None):
    if logslicedata == {}:
        return

    nameGroup = logslicedata[str(index)]["name"]
    entry = topEntry.create_group(nameGroup)
    entry.attrs["NX_class"] = "NXdata"

    _create_groupe(
        entry=entry,
        name="data",
        data=logslicedata[str(index)]["data"],
        units=logslicedata[str(index)]["units"],
    )


def _save_iqxqy_to_log(iqxqy=None, topEntry=None):
    entry = topEntry.create_group("I(QxQy)")
    entry.attrs["NX_class"] = "NXdata"

    # intensity
    _create_groupe(entry=entry, name="I", data=iqxqy.intensity, units="1/A")

    # errors
    _create_groupe(entry=entry, name="Idev", data=iqxqy.error, units="1/cm")

    # qx
    if not (iqxqy.qx is None):
        _create_groupe(entry=entry, name="Qx", data=iqxqy.qx, units="1/A")

        _create_groupe(entry=entry, name="Qxdev", data=iqxqy.delta_qx, units="1/A")

    # qy
    if not (iqxqy.qy is None):
        _create_groupe(entry=entry, name="Qy", data=iqxqy.qy, units="1/A")

        _create_groupe(entry=entry, name="Qydev", data=iqxqy.delta_qy, units="1/A")
    # wavelength
    if not (iqxqy.wavelength is None):
        wavelength = "{}".format(iqxqy.wavelength)
        _create_groupe(entry=entry, name="Wavelength", data=wavelength, units="A")


def __save_individual_iq_to_log(iq=None, topEntry=None, entryNameExt=""):

    entry_name = "I(Q)"
    if entryNameExt:
        entry_name += "_" + entryNameExt
    entry = topEntry.create_group(entry_name)
    entry.attrs["NX_class"] = "NXdata"
    entry.attrs["signal"] = "I"
    entry.attrs["axes"] = "Q"

    # intensity
    _create_groupe(entry=entry, name="I", data=iq.intensity, units="1/cm")

    # errors
    _create_groupe(entry=entry, name="Idev", data=iq.error, units="1/cm")

    # mod_q
    if not (iq.mod_q is None):
        _create_groupe(entry=entry, name="Q", data=iq.mod_q, units="1/A")

        logger.debug(f"delta mod q: {iq.delta_mod_q}")
        _create_groupe(entry=entry, name="Qdev", data=iq.delta_mod_q, units="1/A")

    # wavelength
    if iq.wavelength:
        wavelength = "{}".format(iq.wavelength)
        _create_groupe(entry=entry, name="Wavelength", data=wavelength, units="A")


def _save_iq_to_log(iq=None, topEntry=None):

    if (type(iq) is list) and len(iq) > 1:
        for _index, _iq in enumerate(iq):
            __save_individual_iq_to_log(
                iq=_iq, topEntry=topEntry, entryNameExt="wedge{}".format(_index)
            )
    else:
        __save_individual_iq_to_log(iq=iq[0], topEntry=topEntry, entryNameExt="")


def _retrieve_beam_radius_from_out_file(outfolder=""):
    name_of_out_file = glob.glob(os.path.join(outfolder, "*.out"))
    if name_of_out_file == []:
        return ""

    with open(name_of_out_file[0], "r") as handler:
        file_contain = handler.readlines()
    string_to_look_for = "Radius calculated from the input workspace ="
    for _line in file_contain:
        if string_to_look_for in _line:
            regular_exp = r".*= (?P<radius>.*) mm\n"
            m = re.search(regular_exp, _line)
            if m:
                return m.group("radius")
    return ""


def _appendCalculatedBeamRadius(specialparameters=None, json=None, outfolder=""):
    if json is None:
        return specialparameters

    try:
        beam_radius_in_json = json["configuration"]["mmRadiusForTransmission"]
    except KeyError:
        return specialparameters
    except TypeError:
        return specialparameters

    if beam_radius_in_json == "":
        beam_radius_in_json = _retrieve_beam_radius_from_out_file(outfolder=outfolder)

    if specialparameters is None:
        specialparameters = {"transmission_radius_used (mm)": beam_radius_in_json}
    else:
        specialparameters = {
            **specialparameters,
            "transmission_radius_used (mm)": beam_radius_in_json,
        }
    return specialparameters


def savereductionlog(filename="", detectordata=None, **kwargs):
    r"""Save the reduction log

    There are three ``NXentry``. The first is for the 1d reduced data, second
    is for the 2d reduced data, and the third is for the extra information
    about how the data was processed.

    The options read from the ``kwargs`` parameter are listed with the other parameters.

    Parameters
    ----------
    detectordata: dict
        for each key (name of detector), will have iq: Iqmod and iqxqy: IQazimuthal
        where Iqmod is a tuple with the following informations: intensity, error, mod_q, delta_mode_q
        and IQazimuthal is a tuple with the following informations: intensity, error, qx, delta_qx, qy, delta_y
    python: string
        The script used to create everything (optional)
    pythonfile: string
        The name of the file containing the python script.
        Will be read into ``python`` argument if not already supplied (optional)
    reductionparams: str, dict
        The parameters supplied to the reduction script as either a nested :py:obj:`dict`
        or a json formatted :py:obj:`str` (optional)
    logslicedata: dict
        data corresponding to the various slices
    starttime: str
        When the original script was started (optional, default: now)
    hostname: str
        Name of the computer used. If not provided, will be gotten from the system
        environment ``HOSTNAME`` (optional)
    user: str
        User-id of who reduced the data (as in xcamms). If not provided will be
        gotten from the system environment ``USER`` (optional)
    username: str
        Username of who reduced the data (as in actual name). If not provided
        will be gotten from the system environment ``USERNAME`` (optional)
    specialparameters: dict
        dictionary of any other arguments you want to keep in the log file
    samplelogs: SampleLogs
        SampleLogs object of all the EPICS infos logged into the NeXus (and visible on ONCat)
    """
    if filename == "":
        filename = "_reduction_log.hdf"
        # raise RuntimeError('Cannot write to file "{}"'.format(filename))

    if detectordata is None:
        raise RuntimeError("Provide at least one detector data  {}".format(filename))

    if not type(detectordata) is dict:
        raise RuntimeError(
            "detectordata has the wrong type. It should be a dictionary "
            "and not a {}".format(type(detectordata))
        )
    for _slice_name in detectordata.keys():

        if not type(detectordata[_slice_name]) is dict:
            raise RuntimeError(
                "detectordata value has the wrong type. It should be a dictionary "
                "and not a {}".format(type(detectordata[_slice_name]))
            )

        for _detector_name in detectordata[_slice_name].keys():

            if not type(detectordata[_slice_name][_detector_name]) is dict:
                raise RuntimeError(
                    f"detectordata[{_slice_name}][{_detector_name}] value has the wrong type. It "
                    f"should be a dictionary "
                    f"and not a {type(detectordata[_slice_name][_detector_name])}"
                )

            if not ("iq" in detectordata[_slice_name][_detector_name].keys()) and not (
                "iqxqy" in detectordata[_slice_name][_detector_name].keys()
            ):
                raise KeyError(
                    "Provide at least a iq and/or iqxqy keys to {}".format(filename)
                )

    logslicedata = kwargs.get("logslicedata", {})
    if logslicedata:

        if not type(logslicedata) is dict:
            raise RuntimeError(
                "logslicedata has the wrong type. It should a dictionary "
                "and not a {}".format(type(logslicedata))
            )

        if len(logslicedata.keys()) > len(detectordata.keys()):
            raise ValueError(
                f"Can not have more logs slice data ({len(logslicedata.keys())}) than "
                f"slices ({len(detectordata.keys())})"
            )

    # end of testing inputs

    writing_flag = "w"
    for _index, _slice_name in enumerate(detectordata.keys()):

        if _index > 0:
            writing_flag = "a"

        with h5py.File(filename, writing_flag) as handle:
            topEntry = handle.create_group(_slice_name)
            topEntry.attrs["NX_class"] = "NXdata"

            _current_detectordata = detectordata[_slice_name]

            for _frame_index, _frame_name in enumerate(_current_detectordata.keys()):

                _current_frame = _current_detectordata[_frame_name]
                midEntry = _createnxgroup(topEntry, _frame_name, "NXdata")

                cfkeys = list(_current_frame.keys())
                logger.debug(f"current frame keys: {cfkeys}")

                if "iq" in _current_frame.keys() and "iqxqy" in _current_frame.keys():
                    logger.debug(str(_current_frame["iq"]))
                    _save_iq_to_log(iq=_current_frame["iq"], topEntry=midEntry)
                    logger.debug(str(_current_frame["iqxqy"]))
                    _save_iqxqy_to_log(iqxqy=_current_frame["iqxqy"], topEntry=midEntry)
                elif "iq" in _current_frame.keys():
                    _save_iq_to_log(iq=_current_frame["iq"], topEntry=midEntry)
                else:
                    _save_iqxqy_to_log(iqxqy=_current_frame["iqxqy"], topEntry=midEntry)

            _save_logslicedata(
                logslicedata=logslicedata, index=_index, topEntry=topEntry
            )

    # re-open the file to append other information
    with h5py.File(filename, "a") as handle:
        entry = _createnxgroup(handle, "reduction_information", "NXentry")

        # read the contents of the script
        _pythonfile = kwargs.get("pythonfile", None)
        if _pythonfile:
            _savepythonscript(
                entry,
                pythonfile=kwargs.get("pythonfile", None),
                pythonscript=kwargs.get("python", ""),
            )

        _reduction_parameters = kwargs.get("reductionparams", "")
        if _reduction_parameters:
            _savereductionjson(entry, parameters=_reduction_parameters)
            _savereductionparams(
                entry,
                parameters=_reduction_parameters["data"],
                name_of_entry="reduction_parameters",
            )

        # timestamp of when it happened - default to now
        starttime = kwargs.get("starttime", datetime.now().isoformat())
        entry.create_dataset(name="start_time", data=[np.string_(starttime)])

        # computer it was on
        hostname = kwargs.get("hostname", None)
        if not hostname:
            hostname = socket.gethostname()
        if hostname:
            entry.create_dataset(name="hostname", data=[np.string_(hostname)])

        # software involved
        _savenxprocess(entry, "mantid", mantid_version)
        _savenxprocess(entry, "drtsans", drtsans_version)

        # user information
        user = kwargs.get("user", environ.get("USER", ""))
        if user:
            nxuser = _createnxgroup(entry, "user", "NXuser")
            nxuser.create_dataset(name="facility_user_id", data=[np.string_(user)])

            username = kwargs.get("username", environ.get("USERNAME", ""))
            if username:
                nxuser.create_dataset(name="name", data=[np.string_(username)])

        specialparameters = kwargs.get("specialparameters", None)
        if specialparameters:
            # add calculated beam radius if beam radius is None
            if _reduction_parameters:
                json_entry = _reduction_parameters["data"]
            else:
                json_entry = None
            specialparameters = _appendCalculatedBeamRadius(
                specialparameters, json=json_entry, outfolder=os.path.dirname(filename)
            )

        if specialparameters:
            _savespecialparameters(entry, specialparameters, "special_parameters")

        samplelogs = kwargs.get("samplelogs", None)
        if samplelogs:
            _savesamplelogs(entry, samplelogs, "sample_logs")
