import numpy as np

from mantid.api import Run, MatrixWorkspace
from mantid.kernel import (
    BoolTimeSeriesProperty,
    DateAndTime,
    FloatTimeSeriesProperty,
    Int64TimeSeriesProperty,
    StringTimeSeriesProperty,
)
from mantid.simpleapi import mtd


class SampleLogs(object):
    r"""
    Log reader, a bit more pythonic

    source: PyObject
        Instrument object, MatrixWorkspace, workspace name, file name,
        run number
    """

    def __init__(self, source):
        self._ws = None
        self._run = self.find_run(source)

    def __getitem__(self, item):
        if item in self._run.keys():
            return self._run[item]
        raise KeyError('"{}" not found in sample logs'.format(item))

    def __getattr__(self, item):
        _run = self.__dict__["_run"]
        try:
            return getattr(_run, item)
        except AttributeError:
            if item in _run.keys():
                return _run.getProperty(item)
            else:
                raise AttributeError('"{}" not found in sample logs'.format(item))

    def __contains__(self, item):
        """Called when using python's ``in`` operation"""
        return item in self._run

    def insert(self, name, value, unit=None):
        r"""
        Wrapper to Mantid AddSampleLog algorithm

        The following properties of AddSampleLog are determined by
        inspection of `value`: LogText, LogType, NumberType

        Parameters
        ----------
        name: str
            log entry name
        value: str, int, double, list
            Value to insert
        unit: str
            Log unit
        """
        if not unit:
            unit = ""
        if isinstance(value, list):
            value = value[0]  # copies AddSampleLog behavior

        self._ws.mutableRun().addProperty(name, value, unit, True)

    def insert_time_series(
        self, name, elapsed_times, values, start_time="2000-01-01T00:00:00", unit=""
    ):
        r"""
        Insert a ~mantid.kernel.FloatTimeSeriesProperty in the logs

        Parameters
        ----------
        name: str
            log entry name
        start_time: str
            Starting time for the run
        elapsed_times: list
            List of elapsed times after ```start_time```, in seconds.
        values: list
            List of log values, same length as the list of times
        unit str
            Log unit
        """
        # Determine the type of the time series
        series_types = {
            bool: BoolTimeSeriesProperty,
            float: FloatTimeSeriesProperty,
            int: Int64TimeSeriesProperty,
            str: StringTimeSeriesProperty,
        }
        series_property = series_types.get(type(values[0]), FloatTimeSeriesProperty)(
            name
        )

        # Insert one pair of (time, elapsed_time) at a time
        seconds_to_nanoseconds = 1.0e09  # from seconds to nanoseconds
        start = DateAndTime(start_time)
        for (elapsed_time, value) in zip(elapsed_times, values):
            series_property.addValue(
                start + int(elapsed_time * seconds_to_nanoseconds), value
            )

        # include the whole time series property in the metadata
        self._ws.mutableRun().addProperty(name, series_property, unit, True)

    @property
    def mantid_logs(self):
        return self._run

    @property
    def workspace(self):
        return self._ws

    def single_value(self, log_key, operation=np.mean):
        _run = self.__dict__["_run"]
        return float(operation(_run[log_key].value))

    def find_run(self, other):
        r"""
        Retrieve the Run object

        Parameters
        ----------
        other: Run, str, MatrixWorkspace

        Returns
        -------
        Run
            Reference to the run object
        """

        def from_ws(ws):
            self._ws = ws
            return ws.getRun()

        def from_run(a_run):
            return a_run

        def from_string(s):
            # see if it is a file
            if s in mtd:
                return self.find_run(mtd[s])
            else:
                raise RuntimeError("{} is not a valid workspace name".format(s))

        dispatch = {Run: from_run, MatrixWorkspace: from_ws, str: from_string}

        # If others is not None: raise exception
        if other is None:
            a = "[DEBUG 187] dispatch: {}".format(dispatch.items())
            b = "[DEBUG 187] other: {}".format(other)
            raise NotImplementedError("{}\n{}".format(a, b))

        finders = [v for k, v in dispatch.items() if isinstance(other, k)]
        if len(finders) == 0:
            # In case no items found
            raise RuntimeError(
                'Input "other" of value {} is not supported to retrieve Mantid '
                '"run" object'.format(other)
            )
        finder = finders[0]
        return finder(other)

    def find_log_with_units(self, log_key, unit=None):
        r"""
        Find a log entry in the logs, and ensure it has the right units

        Parameters
        ----------
        log_key: string
                 key of the log to find
        unit: None or string
               units string to enforce

        Returns
        -------
            log value
        """
        if log_key in self.keys():
            if bool(unit) and not self[log_key].units == unit:
                error_msg = "Found %s with wrong units" % log_key
                error_msg += " [%s]" % self[log_key].units
                raise RuntimeError(error_msg)
            return np.average(self[log_key].value)
        raise RuntimeError(
            f"Could not find {log_key} with unit {unit} in logs: {self.keys()}"
        )
