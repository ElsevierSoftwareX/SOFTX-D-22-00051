from copy import deepcopy
from os.path import isdir
import random
import string
from collections import OrderedDict
import functools
from collections import namedtuple, Mapping
from contextlib import contextmanager

import mantid
from mantid.api import AnalysisDataService
from mantid.kernel import ConfigService

# import mantid's workspace types exposed to python
workspace_types = [
    getattr(mantid.dataobjects, w_type_name)
    for w_type_name in [s for s in dir(mantid.dataobjects) if "Workspace" in s]
]


class MultiOrderedDict(OrderedDict):
    def __setitem__(self, key, value):
        if isinstance(value, list) and key in self:
            self[key].extend(value)
        else:
            super(MultiOrderedDict, self).__setitem__(key, value)
            # super().__setitem__(key, value) # in Python 3


def namedtuplefy(func):
    r"""
    Decorator to transform the return dictionary of a function into
    a namedtuple

    Parameters
    ----------
    func: Function
        Function to be decorated
    name: str
        Class name for the namedtuple. If None, the name of the function
        will be used
    Returns
    -------
    Function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        if wrapper.nt is None:
            if isinstance(res, Mapping) is False:
                raise ValueError("Cannot namedtuplefy a non-dict")
            wrapper.nt = namedtuple(func.__name__ + "_nt", res.keys())
        return wrapper.nt(**res)

    wrapper.nt = None
    return wrapper


@contextmanager
def amend_config(new_config=None, data_dir=None):
    r"""
    Context manager to safely modify Mantid Configuration Service while
    the function is executed.

    Parameters
    ----------
    new_config: dict
        (key, value) pairs to substitute in the configuration service
    data_dir: str, list
        Append one (when passing a string) or more (when passing a list)
        directories to the list of data search directories.
    """
    modified_keys = list()
    backup = dict()
    config = ConfigService.Instance()
    if new_config is not None:
        SEARCH_ARCHIVE = "datasearch.searcharchive"
        if SEARCH_ARCHIVE not in new_config:
            new_config[SEARCH_ARCHIVE] = "hfir, sns"
        DEFAULT_FACILITY = "default.facility"
        if DEFAULT_FACILITY not in new_config:
            new_config[DEFAULT_FACILITY] = "SNS"
        for key, val in new_config.items():
            backup[key] = config[key]
            config[key] = val  # config does not have an 'update' method
            modified_keys.append(key)
    if data_dir is not None:
        data_dirs = (
            [
                data_dir,
            ]
            if isinstance(data_dir, str)
            else data_dir
        )
        key = "datasearch.directories"
        backup[key] = deepcopy(config[key])
        [config.appendDataSearchDir(dd) for dd in data_dirs if isdir(dd)]
        modified_keys.append(key)
    try:
        yield
    finally:
        for key in modified_keys:
            config[key] = backup[key]


def unique_workspace_name(n=5, prefix="", suffix=""):
    r"""
    Create a random sequence of `n` lowercase characters that is guaranteed
    not to collide with the name of any existing Mantid workspace

    uws stands for Unique Workspace Name

    Parameters
    ----------
    n: int
        Size of the sequence
    prefix: str
        String to prefix the randon sequence
    suffix: str
        String to suffix the randon sequence

    Returns
    -------
    string
    """

    ws_name = "".join(random.choice(string.ascii_lowercase) for _ in range(n))
    ws_name = "{}{}{}".format(str(prefix), ws_name, str(suffix))
    while ws_name in AnalysisDataService.getObjectNames():
        characters = [random.choice(string.ascii_lowercase) for _ in range(n)]
        ws_name = "".join(characters)
        ws_name = "{}{}{}".format(str(prefix), ws_name, str(suffix))
    return ws_name


def unique_workspace_dundername():
    return unique_workspace_name(n=5, prefix="__")


def unpack_v3d(functor, index):
    """Retain only the cartesian coordinates of the V3D object returned by ```functor```

    This function reduces the memory imprint, from a V3D object to a mere 3-component list.
    Speeds up execution by avoiding crowding the heap when interating over the detectors.
    e.g.
    x = [detectorInfo().position(i) for i in range(number_detectors)]  # number_detectors V3D objects in the heap
    x = [unpackV3D(detectorInfo.position, i) for i in range(number_detectors)]  # 100 times faster

    Parameters
    ----------
    functor: function
        Callable receiving argument ```index``` and returning a V3D object.
    index: int
        DetectorInfo, ComponentInfo, or SpectrumInfo index

    Returns
    -------
    list
    """
    v3d_vector = functor(index)
    return [v3d_vector.X(), v3d_vector.Y(), v3d_vector.Z()]
