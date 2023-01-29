"""
Reader for EQSANS configuration files in the old format
"""
import os
import re
from copy import deepcopy
from itertools import product as iproduct
import numpy as np
from contextlib import contextmanager
from drtsans.tof.eqsans.geometry import detector_id

__all__ = [
    "load_config",
]


# default directory for instrument configuration
cfg_dir = "/SNS/EQSANS/shared/instrument_configuration"


def closest_config(run, config_dir=cfg_dir):
    """
    Configuration file for a given run number.
    The appropriate configuration file is the one marked with a run number
    that is closest and smaller (or equal) than the input run number

    Parameters
    ----------
    config_dir: str
        Directory containing configuration files

    Returns
    -------
    str
        Absolute path to configuration file
    """
    pattern = re.compile(r"eqsans_configuration\.(\d+)")
    reference_runs = list()
    for root, dirs, files in os.walk(config_dir):
        for file_name in files:
            match = pattern.search(file_name)
            if match is not None:
                reference_runs.append(int(match.groups()[0]))
    reference_runs.sort()
    reference_runs = np.asarray(reference_runs)
    if not bool(reference_runs.size > 0):
        raise RuntimeError(
            'Failed to find any reference runs in "{}"'.format(config_dir)
        )
    maximum_index_below_run = np.where(run >= reference_runs)[0][-1]
    reference_run = reference_runs[maximum_index_below_run]
    return os.path.join(config_dir, "eqsans_configuration.{}".format(reference_run))


@contextmanager
def open_source(source, config_dir=cfg_dir):
    """
    Find the configuration file appropriate to the input source info

    Parameters
    ----------
    source: str
        One of the following: (1) absolute path or just filename to a
        configuration file; (2) run-number
    config_dir: str
        Directory containing configuration files.

    Yields
    ------
    file handle:
        Handle to the configuration file
    """
    src = str(source)
    if os.path.isfile(src):
        file_handle = open(src)
    else:
        file = os.path.join(config_dir, src)
        if os.path.isfile(file):
            file_handle = open(file)
        else:
            run = int(source)
            file_handle = open(closest_config(run, config_dir=config_dir))
    try:
        yield file_handle
    finally:
        file_handle.close()


class CfgItemValue(object):
    """
    Entry item in an EQSANS configuration file

    Parameters
    ----------
    data: string, or list of strings
        raw value of the entry
    note: str
        Description of the entry
    """

    def __init__(self, name="", data="", note=""):
        self.data = data
        self.note = note

    def __repr__(self):
        return 'CfgItemValue(data="{data}", note="{note}")'.format(**self.__dict__)

    def __eq__(self, other):
        """Discard note explanatory when comparing two value items"""
        return self.data == other.data

    def __iadd__(self, other):
        other_vals = [other.data]
        if isinstance(other.data, list):
            other_vals = other.data
        [self.insert(val) for val in other_vals]
        return self

    def insert(self, value):
        if isinstance(self.data, list):
            self.data.append(value)
        else:
            self.data = [self.data, value]

    @property
    def value(self):
        return self.data


class ItemMaskMixin(object):
    r"""Functionality common to all types of masks items in the config file"""

    @property
    def detectors(self):
        r"""List of masked detector ID's, sorted by increasing ID"""
        return sorted(detector_id(self.pixels))


class CfgItemRectangularMask(CfgItemValue, ItemMaskMixin):
    r"""Specialization for 'Rectangular Mask' entries

    Convention: a 'Rectangular Mask' is of the form 'xs, ys; xe, ye'
    X-axis for tube ID, from 0 to 191
    Y-axis for pixel ID, from 0 to 255
    (xs, ys) defines the lower-left corner of the rectangular mask
    (xe, ye) defines the upper-right corner of the rectangular mask
    """

    def __init__(self, *args, **kwargs):
        CfgItemValue.__init__(self, *args, **kwargs)

    def insert(self, value):
        r"""Additional validation"""
        if len(re.findall(r"\d+", value)) == 4:
            super().insert(value)

    @property
    def pixels(self):
        r"""
        List of pixels in (x, y) pixel coordinates, in no particular order

        Returns
        -------
        list
        """
        pxs = list()
        recs = (
            [
                self.data,
            ]
            if isinstance(self.data, str)
            else self.data
        )
        for rec in recs:
            # (xs,ys) left-lower corner; (xe,y) upper-right corner
            xs, ys, xe, ye = [int(n) for n in re.findall(r"\d+", rec)]
            pxs.extend(iproduct(range(xs, 1 + xe), range(ys, 1 + ye)))
        return list(set(pxs))  # remove possible duplicates

    @property
    def value(self):
        return self.detectors


CfgItemEllipticalMask = CfgItemRectangularMask


class CfgTofEdgeDiscard(CfgItemValue):
    r"""Specialization for entry 'TOF edge discard'"""

    def __init__(self, *args, **kwargs):
        CfgItemValue.__init__(self, *args, **kwargs)

    @property
    def value(self):
        return tuple([float(t) for t in self.data.split()])


class Cfg(object):
    """
    Read EQSANS configuration files
    """

    _item_types = {
        "rectangular mask": CfgItemRectangularMask,
        "elliptical mask": CfgItemEllipticalMask,
        "tof edge discard": CfgTofEdgeDiscard,
    }

    @staticmethod
    def load(source, config_dir=cfg_dir, comment_symbol="#"):
        """
        Load the configuration file appropriate to the input source info.

        Ignores commented lines.

        Parameters
        ----------
        source: str
            One of the following: (1) absolute path or just filename to a
            configuration file; (2) run-number
        config_dir: str
            Directory containing configuration files

        Returns
        -------
        dict
            A dictionary with CfgItemValue objects as values
        """
        cfg = dict()
        with open_source(source, config_dir=config_dir) as f:
            for line in f.readlines():
                if "=" not in line:
                    continue  # this line contains no valid entries
                key, val = [x.strip() for x in line.split("=")]
                if comment_symbol in key:
                    continue
                key = key.lower()  # ignore case
                description = ""
                if comment_symbol in val:
                    val, description = [x.strip() for x in val.split("#")]
                if key in cfg:
                    cfg[key].insert(val)
                    if description != "":
                        cfg[key].help = description
                else:
                    item_type = Cfg._item_types.get(key, CfgItemValue)
                    item = item_type(data=val, note=description)
                    cfg[key] = item

        # Old reduction combines the rectangular and elliptical masks
        cfg["combined mask"] = deepcopy(cfg["rectangular mask"])
        if "elliptical mask" in cfg:
            cfg["combined mask"] += cfg["elliptical mask"]

        return cfg

    def __init__(self, source=None, config_dir=cfg_dir):
        self._cfg = (
            dict() if source is None else Cfg.load(source, config_dir=config_dir)
        )

    def __getitem__(self, item):
        return self._cfg[item]

    def __setitem__(self, key, value):
        self._cfg[key] = value

    def keys(self):
        return self._cfg.keys()

    def as_dict(self):
        """Dictionary of `(key, val)` values. `val` is not the
        raw data but the reported value of `CfgItemValue.value`"""
        return {k: v.value for (k, v) in self._cfg.items()}

    def __repr__(self):
        fmt = '"{}" : {}'
        return "\n".join(fmt.format(k, v) for (k, v) in self._cfg.items())

    def logit(self, key, workspace, name=None, replace=False):
        """

        Parameters
        ----------
        key: str
            Key associated to a specific configuration entry.
        workspace: mantid.MatrixWorkspace
            Save the property in the logs of this workspace
        name: str
            Alternative log name to key
        replace: bool
            Overwrite existing log entry

        Raises
        ------
        ValueError
            If the log entry exists and replace is set to False
        """
        log_name = key if name is None else name
        run = workspace.getRun()
        if replace is False and run.hasProperty(log_name):
            msg = (
                'Property {} already exists. Set keyword "replace"'
                " to True if you wish to replace the existing property"
                " with the new value."
            )
            raise ValueError(msg.format(log_name))
        run.addProperty(log_name, self[key].data, replace=replace)


def load_config(source, config_dir=cfg_dir):
    r"""
    Load the configuration file appropriate to the input source info

    Parameters
    ----------
    source: str
        The filename (absolute path or relative) to a configuration file or a run-number
    config_dir: str
        Directory containing configuration files

    Returns
    -------
    dict
        keys are entries in the file
    """
    return Cfg(source, config_dir=config_dir).as_dict()
