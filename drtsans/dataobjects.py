from collections import namedtuple
from collections.abc import Iterable
import h5py
from enum import Enum
import numpy as np
from typing import Union

# https://docs.mantidproject.org/nightly/algorithms/CreateWorkspace-v1.html
from mantid.simpleapi import mtd, CreateWorkspace

from drtsans.settings import unique_workspace_dundername as uwd

__all__ = [
    "getDataType",
    "DataType",
    "IQmod",
    "IQazimuthal",
    "IQcrystal",
    "verify_same_q_bins",
]


class DataType(Enum):
    WORKSPACE2D = "Workspace2D"
    IQ_MOD = "IQmod"
    IQ_AZIMUTHAL = "IQazimuthal"
    IQ_CRYSTAL = "IQcrystal"


class HeaderType(Enum):
    MANTID_ASCII = "MantidAscii"
    PANDAS = "Pandas"


def getDataType(obj):
    try:
        return DataType(obj.id())
    except AttributeError:
        name = str(obj)
        if name not in mtd:
            raise ValueError("Do not know how to get id from: {}".format(obj))
        return DataType(mtd[name].id())


def _check_parallel(*args):
    """This makes sure that all input arrays are parallel to each
    other. It assumes that the inputs are ndarrays."""
    shape = args[0].shape
    for arg in args[1:]:
        if arg.shape != shape:
            raise TypeError("Shape mismatch ({} != {})".format(shape, arg.shape))


def _nary_operation(iq_objects, operation, unpack=True, **kwargs):
    r"""
    Carry out an operation on the component arrays for each of the IQ objects.

    Examples:
    - _nary_operation((iq_1, iq_2), numpy.append, unpack=True)
    - _nary_operation((iq_1, iq_2), numpy.concatenate, unpack=False)

    Parameters
    ----------
    iq_objects: list
        A list of ~drtsans.dataobjects.IQmod, ~drtsans.dataobjects.IQazimuthal, or ~drtsans.dataobjects.IQcrystal
        objects.
    operation: function
        A function operating on a list of :ref:`~numpy.ndarray` objects, e.g. numpy.concatenate((array1, array2))
    unpack: bool
        If set to :py:obj:`True`, then ``operation`` receives an unpacked list of arrays. If set to :py:obj:`False`,
        then ``operation`` receives the list of arrays as a single argument.
        Examples: numpy.append(*(array1, array2)) versus numpy.concatenate((array1, array2))
    kwargs: dict
        Additional options to be passed to ``operation``.

    Returns
    -------
    ~drtsans.dataobjects.IQmod, ~drtsans.dataobjects.IQazimuthal, or ~drtsans.dataobjects.IQcrystal
    """
    reference_object = iq_objects[0]
    assert (
        len(set([type(iq_object) for iq_object in iq_objects])) == 1
    )  # check all objects of same type
    new_components = list()
    for i in range(len(reference_object)):  # iterate over the IQ object components
        i_components = [
            iq_object[i] for iq_object in iq_objects
        ]  # collect the ith components of each object
        if True in [
            i_component is None for i_component in i_components
        ]:  # is any of these None?
            new_components.append(None)
        elif unpack is True:
            new_components.append(operation(*i_components, **kwargs))
        else:
            new_components.append(operation(i_components, **kwargs))
    return reference_object.__class__(*new_components)


def _extract(iq_object, selection):
    r"""
    Extract a subset of data points onto a new IQ object.

    Examples:
    - iq_object.extract(42)  # extract data point number 42
    - iq_object.extract(slice(None, None, 2))  # extract every other data point
    - iq_object.extract(IQmod().mod_q < 0.1)  # extract points with Q < 0.1

    Parameters
    ----------
    iq_object: ~drtsans.dataobjects.IQmod, ~drtsans.dataobjects.IQazimuthal, ~drtsans.dataobjects.IQcrystal
    selection: int, slice, :ref:`~numpy.ndarray`
        Any selection that can be passed onto a :ref:`~numpy.ndarray`

    Returns
    -------
    ~drtsans.dataobjects.IQmod, ~drtsans.dataobjects.IQazimuthal, ~drtsans.dataobjects.IQcrystal
    """
    component_fragments = list()
    for component in iq_object:
        if component is None:
            component_fragments.append(None)
        else:
            fragment = component.__getitem__(selection)
            if (
                isinstance(fragment, Iterable) is False
            ):  # selection extracts only one data point
                fragment = [
                    fragment,
                ]
            component_fragments.append(fragment)
    return iq_object.__class__(*component_fragments)


def scale_intensity(iq_object, scaling):
    r"""Rescale intensity and error for one IQ object.
    Relies on fields 'intensity' and 'error' being the first two components

    Parameters
    ----------
    iq_object: ~drtsans.dataobjects.IQmod, ~drtsans.dataobjects.IQazimuthal, ~drtsans.dataobjects.IQcrystal
    scaling: float

    Returns
    -------
    ~drtsans.dataobjects.IQmod, ~drtsans.dataobjects.IQazimuthal, ~drtsans.dataobjects.IQcrystal
    """
    intensity = scaling * iq_object.intensity
    error = scaling * iq_object.error
    return iq_object.__class__(
        intensity, error, *[iq_object[i] for i in range(2, len(iq_object))]
    )


def verify_same_q_bins(iq0, iq1, raise_exception_if_diffrent=False, tolerance=None):
    """Verify whether 2 I(Q) has the same range of Q

    Parameters
    ----------
    iq0: ~drtsans.dataobjects.IQmod, ~drtsans.dataobjects.IQazimuthal, ~drtsans.dataobjects.IQcrystal
    iq1: ~drtsans.dataobjects.IQmod, ~drtsans.dataobjects.IQazimuthal, ~drtsans.dataobjects.IQcrystal

    Returns
    -------
    bool
        True if they are same
    """
    # Same class
    if iq0.__class__ != iq1.__class__:
        raise RuntimeError(
            f"Input I(Q)s are of different type: {type(iq0)} and {type(iq1)}"
        )

    # IQmod
    if isinstance(iq0, IQmod):
        # Q1D
        if iq0.wavelength is None or iq1.wavelength is None:
            # no wave length
            q0vec = iq0.mod_q
            q1vec = iq1.mod_q
        else:
            # also comparing the wavelength bins if they do exist
            q0vec = np.array([iq0.mod_q, iq0.wavelength])
            q1vec = np.array([iq1.mod_q, iq1.wavelength])
    elif isinstance(iq0, IQazimuthal) or isinstance(iq0, IQcrystal):
        # Q2D
        if iq0.wavelength is None or iq1.wavelength is None:
            # no wavelength
            q0vec = np.array([iq0.qx, iq0.qy])
            q1vec = np.array([iq1.qx, iq1.qy])
        else:
            q0vec = np.array([iq0.qx, iq0.qy, iq0.wavelength])
            q1vec = np.array([iq1.qx, iq1.qy, iq0.wavelength])
    else:
        raise RuntimeError(
            f"I(Q) of type {type(iq0)} is not supported by verify same binning"
        )

    # Verify
    same = True
    try:
        if tolerance:
            np.testing.assert_allclose(q0vec, q1vec, atol=tolerance)
        else:
            np.testing.assert_allclose(q0vec, q1vec)
    except AssertionError as assert_error:
        same = False
        if raise_exception_if_diffrent:
            raise assert_error

    return same


def q_azimuthal_to_q_modulo(Iq):
    """this method converts Qazimuthal to Qmodulo using
        (1) Q = sqrt(Qx**2 + Qy**2)
        (2) sigmaQ = sqrt(sigmaQx**2 + sigmaQy**2)

    Parameters:
    ----------
    Iq: IQazimuthal object

    Returns:
    -------
    Iqmod: IQmod object
    """
    qx = Iq.qx
    qy = Iq.qy
    delta_qx = Iq.delta_qx
    delta_qy = Iq.delta_qy

    mod_q = np.sqrt(np.square(qx) + np.square(qy))
    delta_mode_q = np.sqrt(np.square(delta_qx) + np.square(delta_qy))

    q_azimuthal_to_q_modulo = namedtuple(
        "q_azimuthal_to_q_modulo", "mod_q, delta_mod_q"
    )
    q_azimuthal_to_q_modulo.mod_q = mod_q
    q_azimuthal_to_q_modulo.delta_mod_q = delta_mode_q

    iqmod = IQmod(
        intensity=Iq.intensity, error=Iq.error, mod_q=mod_q, delta_mod_q=delta_mode_q
    )

    return iqmod


def concatenate(iq_objects):
    r"""
    Join a sequence IQ objects; concatenate((iq1, iq2, iq3,...))

    Parameters
    ----------
    iq_objects: list
        A sequence of ~drtsans.dataobjects.IQmod, ~drtsans.dataobjects.IQazimuthal, or
        ~drtsans.dataobjects.IQcrystal objects. All objects must be of the same type

    Returns
    -------
    ~drtsans.dataobjects.IQmod, ~drtsans.dataobjects.IQazimuthal, ~drtsans.dataobjects.IQcrystal
    """
    return _nary_operation(iq_objects, np.concatenate, unpack=False)


class IQmod(namedtuple("IQmod", "intensity error mod_q delta_mod_q wavelength")):
    r"""This class holds the information for I(Q) scalar. All of the arrays must be 1-dimensional
    and parallel (same length). The ``delta_mod_q`` and ``wavelength`` fields are optional."""

    @staticmethod
    def read_csv(file, sep=" "):
        r"""
        Load an intensity profile into a ~drtsans.dataobjects.IQmod object.

        Required file format:
        The first row must include the names for the file columns. The order of the columns is irrelevant and
        the names of the columns must be:
        - 'intensity' for profile intensities. This column is required.
        - 'error' for uncertainties in the profile intensities. This column is required.
        - 'mod_q' for values of Q. This column is required.
        - 'delta_mod_q' for uncertainties in the Q values. This column is optional.
        - 'wavelength' This column is optional.

        Example of file contents:
            intensity error mod_q
            1000.0 89.0 0.001
            90.0 8.0 0.01
            4.7 0.9 0.1

        Usage example:
        ```
        from drtsans.mono.gpsans import IQmod
        iq = IQmod.read_csv(file_name)
        ```

        Parameters
        ----------
        file: str
            Path to input file
        sep: str
            String of length 1. Field delimiter in the input file.

        Returns
        -------
        ~drtsans.dataobjects.IQmod
        """
        from pandas import read_csv as pd_read_csv

        frame = pd_read_csv(
            file, sep=sep, dtype=np.float64, na_values="NAN", comment="#"
        )
        args = [frame[label].values for label in ["intensity", "error", "mod_q"]]
        kwargs = {
            label: frame[label].values
            for label in ["delta_mod_q", "wavelength"]
            if label in list(frame.columns)
        }
        return IQmod(*args, **kwargs)

    def __new__(cls, intensity, error, mod_q, delta_mod_q=None, wavelength=None):
        # these conversions do nothing if the supplied information is already a numpy.ndarray
        intensity = np.array(intensity)
        error = np.array(error)
        mod_q = np.array(mod_q)

        # if intensity is 1d, then everything else will be if they are parallel
        if len(intensity.shape) != 1:
            raise TypeError(
                '"intensity" must be a 1-dimensional array, found shape={}'.format(
                    intensity.shape
                )
            )

        # check that the manditory fields are parallel
        _check_parallel(intensity, error, mod_q)

        # work with optional fields
        if delta_mod_q is not None:
            delta_mod_q = np.array(delta_mod_q)
            _check_parallel(intensity, delta_mod_q)
        if wavelength is not None:
            wavelength = np.array(wavelength)
            _check_parallel(intensity, wavelength)

        # pass everything to namedtuple
        return super(IQmod, cls).__new__(
            cls, intensity, error, mod_q, delta_mod_q, wavelength
        )

    def __mul__(self, scaling):
        r"""Scale intensities and their uncertainties by a number"""
        return scale_intensity(self, scaling)

    def __rmul__(self, scaling):
        return self.__mul__(scaling)

    def __truediv__(self, divisor):
        r"""Divide intensities and their uncertainties by a number"""
        return self.__mul__(1.0 / divisor)

    def extract(self, selection):
        r"""
        Extract a subset of data points onto a new ~drtsans.dataobjects.IQmod object.

        Examples:
        - IQmod().extract(42)  # extract data point number 42
        - IQmod().extract(slice(None, None, 2))  # extract every other data point
        - IQmod().extract(IQmod().mod_q < 0.1)  # extract points with Q < 0.1

        Parameters
        ----------
        selection: int, slice, ~numpy.ndarray
            Any selection that can be passed onto a ~numpy.ndarray

        Returns
        -------
        ~drtsans.dataobjects.IQmod
        """
        return _extract(self, selection)

    def concatenate(self, other):
        r"""
        Append additional data points from another ~drtsans.dataobjects.IQmod object and return the composite as a
        new ~drtsans.dataobjects.IQmod object.

        Parameters
        ----------
        other: ~drtsans.dataobjects.IQmod
            Additional data points.

        Returns
        -------
        ~drtsans.dataobjects.IQmod
        """
        return _nary_operation((self, other), np.concatenate, unpack=False)

    def sort(self, key="mod_q"):
        r"""
        Sort the data points according to one of the components of the ~drtsans.dataobjects.IQmod object.

        Parameters
        ----------
        key: str
            Component prescribing the sorting order. Default sorting is by increasing Q value.

        Returns
        -------
        ~drtsans.dataobjects.IQmod
        """
        return _extract(self, np.argsort(getattr(self, key)))

    def id(self):
        return DataType.IQ_MOD

    def be_finite(self):
        #  Remove NaN
        finite_locations = np.isfinite(self.intensity)
        finite_delta_mod_q = (
            None if self.delta_mod_q is None else self.delta_mod_q[finite_locations]
        )
        finite_binned_iq_wl = IQmod(
            intensity=self.intensity[finite_locations],
            error=self.error[finite_locations],
            mod_q=self.mod_q[finite_locations],
            delta_mod_q=finite_delta_mod_q,
            wavelength=self.wavelength[finite_locations],
        )

        return finite_binned_iq_wl

    def to_workspace(self, name=None):
        # create a name if one isn't provided
        if name is None:
            name = uwd()

        dq = self.delta_mod_q
        if dq is None:
            dq = self.mod_q * 0.0
        return CreateWorkspace(
            DataX=self.mod_q,
            DataY=self.intensity,
            DataE=self.error,
            UnitX="momentumtransfer",
            OutputWorkspace=name,
            Dx=dq,
            EnableLogging=False,
        )

    def to_csv(self, file_name, sep=" ", float_format="%.6E", skip_nan=True):
        r"""
        Write the ~drtsans.dataobjects.IQmod object into an ASCII file.

        Parameters
        ----------
        file_name: str
            Path to output file
        sep: str
            String of length 1. Field delimiter for the output file.
        float_format: str
            Format string for floating point numbers.
        skip_nan: bool
            If true, any data point where intensity is NAN will not be written to file
        """
        # Convert to dictionary to construct a pandas DataFrame instance
        from pandas import DataFrame

        frame = DataFrame(
            {
                label: value
                for label, value in self._asdict().items()
                if value is not None
            }
        )

        #  Create the order of the columns
        i_q_mod_cols = ["mod_q", "intensity", "error"]  # 3 mandatory columns
        if "delta_mod_q" in frame.keys():
            i_q_mod_cols.append("delta_mod_q")
        if "wavelength" in frame.keys():
            i_q_mod_cols.append("wavelength")
        mode_nan = (
            "w"  # write mode for csv file. If we add a header first, it will be 'a'
        )
        # delete NANs if requested
        if skip_nan:
            finites = np.isfinite(frame["intensity"])
            if np.count_nonzero(finites) < len(frame):
                frame = frame[finites]
                mode_nan = "a"
                with open(file_name, "w") as f:
                    f.write("# NANs have been skipped\n")

        # Write to file
        frame.to_csv(
            file_name,
            columns=i_q_mod_cols,
            index=False,
            sep=sep,
            float_format=float_format,
            na_rep="NAN",
            mode=mode_nan,
        )


def load_iqmod(file, sep=" ", header_type=HeaderType.PANDAS.value):
    r"""
    Load an intensity profile into a ~drtsans.dataobjects.IQmod object.

    Required file format:
    The first row must include the names for the file columns. The order of the columns is irrelevant and
    the names of the columns must be:
    - 'intensity' for profile intensities. This column is required.
    - 'error' for uncertainties in the profile intensities. This column is required.
    - 'mod_q' for values of Q. This column is required.
    - 'delta_mod_q' for uncertainties in the Q values. This column is optional.
    - 'wavelength' This column is optional.

    Example of file contents:
        intensity error mod_q
        1000.0 89.0 0.001
        90.0 8.0 0.01
        4.7 0.9 0.1

    Usage example:
    ```
    from drtsans.mono.gpsans import load_iqmod
    iq = load_iqmod(file_name)
    ```

    Parameters
    ----------
    file: str
        Path to input file
    sep: str
        String of length 1. Field delimiter in the input file.

    Returns
    -------
    ~drtsans.dataobjects.IQmod
    """
    if header_type == HeaderType.MANTID_ASCII.value:
        csv_data = np.genfromtxt(file, comments="#", dtype=np.float64, skip_header=2)
        num_cols = len(csv_data[0])
        assert num_cols == 4, "Incompatible number of colums: {} should be 4".format(
            num_cols
        )

        return IQmod(csv_data[:, 1], csv_data[:, 2], csv_data[:, 0], csv_data[:, 3])
    else:
        return IQmod.read_csv(file, sep=sep)


def save_iqmod(
    iq,
    file,
    sep=" ",
    float_format="%.6E",
    skip_nan=True,
    header_type=HeaderType.MANTID_ASCII.value,
):
    r"""
    Write the ~drtsans.dataobjects.IQmod object into an ASCII file.

    Current output columns
    (Line 0: ) intensity error mod_q
    Expected
    (Line 0: ) mod_q intensity error mod_q_error

    Parameters
    ----------
    iq: ~drtsans.dataobjects.IQmod
        Profile to be saved
    file: str
        Path to output file
    sep: str
        String of length 1. Field delimiter for the output file.
    float_format: str
        Format string for floating point numbers.
    skip_nan: bool
        If true, any data point where intensity is NAN will not be written to file
    header: text
        Determine the header type to make 1D data compatible with panda or Mantid
        possible values:
        HeaderType.MANTID_ASCII.value
        HeaderType.PANDAS.value
    """
    if header_type == HeaderType.MANTID_ASCII.value:
        from drtsans.save_ascii import save_ascii_binned_1D

        save_ascii_binned_1D(file, "I(Q)", iq)
    else:
        iq.to_csv(file, sep=sep, float_format=float_format, skip_nan=skip_nan)


class IQazimuthal(
    namedtuple("IQazimuthal", "intensity error qx qy delta_qx delta_qy wavelength")
):
    r"""
    This class holds the information for the azimuthal projection, I(Qx, Qy). The resolution terms,
    (``delta_qx``, ``delta_qy``) and ``wavelength`` fields are optional.

    All of the arrays must be 1-dimensional or 2-dimensional and matching length. For the 1-dimensional
    case, all of the arrays must be parallel (same length). For the 2-dimensional case, (``intensity``,
    ``error``, ``delta_qx``, ``delta_qy``, ``wavelength``) must all be parallel. However, for (``qx``,
    ``qy``), they must either (both) be 2-dimensional and parallel, or (both) 1-dimensional with
    ``len(qx) == intensity.shape[0]`` and ``len(qy) == intensity.shape[1]``.

    if intensity is 2D, and qx and qy are 1D: In this constructor, it is assumed that intensity 2D array
    will match
    qx = [[qx0, qx1, ...], [qx0, qx1, ...], ...]
    qy = [[qy0, qy0, ...], [qy1, qy1, ...], ...]
    because qx and qy will be created in such style.
    """

    def __new__(
        cls, intensity, error, qx, qy, delta_qx=None, delta_qy=None, wavelength=None
    ):  # noqa: C901
        # these conversions do nothing if the supplied information is already a numpy.ndarray
        intensity = np.array(intensity)
        error = np.array(error)
        qx = np.array(qx)
        qy = np.array(qy)

        # check that the manditory fields are parallel
        if len(intensity.shape) == 1:
            _check_parallel(intensity, error, qx, qy)
        elif len(intensity.shape) == 2:
            if len(qx.shape) == 1:
                # Qx and Qy are given in 1D array (not meshed)
                _check_parallel(intensity, error)
                if intensity.shape[0] != qx.shape[0]:
                    raise TypeError(
                        "Incompatible dimensions intensity[{}] and qx[{}]".format(
                            intensity.shape, qx.shape[0]
                        )
                    )
                if intensity.shape[1] != qy.shape[0]:
                    raise TypeError(
                        "Incompatible dimensions intensity[{}] and qy[{}]".format(
                            intensity.shape, qy.shape[0]
                        )
                    )
            elif len(qx.shape) == 2:
                # Qx and Qy are given in meshed 2D
                _check_parallel(intensity, error, qx, qy)
            else:
                raise TypeError(
                    "Qx can only be of dimension 1 or 2, found {}".format(len(qx.shape))
                )
        else:
            raise TypeError(
                "intensity can only be of dimension 1 or 2, found {}".format(
                    len(intensity.shape)
                )
            )

        # work with optional fields
        if np.logical_xor(delta_qx is None, delta_qy is None):
            raise TypeError(
                "Must specify either both or neither of delta_qx and delta_qy"
            )
        if delta_qx is not None:
            delta_qx = np.array(delta_qx)
            delta_qy = np.array(delta_qy)
            _check_parallel(intensity, delta_qx, delta_qy)
        if wavelength is not None:
            wavelength = np.array(wavelength)
            _check_parallel(intensity, wavelength)

        # make the qx and qy have the same shape as the data
        if len(intensity.shape) == 2 and len(qx.shape) == 1 and len(qy.shape) == 1:
            # Using meshgrid to construct the Qx and Qy 2D arrays.  This is consistent with the algorithm
            # that is used in bin_iq_2d()
            qx, qy = np.meshgrid(qx, qy, indexing='ij')

            # Sanity check
            assert qx.shape == intensity.shape, f'qx and intensity must have same shapes.  ' \
                                                f'It is not now: {qx.shape} vs {intensity.shape}'
            assert qy.shape == intensity.shape, f'qy and intensity must have same shapes.  ' \
                                                f'It is not now: {qy.shape} vs {intensity.shape}'

        # pass everything to namedtuple
        return super(IQazimuthal, cls).__new__(
            cls, intensity, error, qx, qy, delta_qx, delta_qy, wavelength
        )

    def be_finite(self):
        """Remove NaN by flattening first

        Returns
        -------
        IQazimuthal
            I(qx, qy, wavelength) with NaN removed

        """
        # Check whether is any need to recontruct the IQazimuthal
        num_finite_points = len(np.where(np.isfinite(self.intensity))[0])
        if num_finite_points == self.intensity.size:
            return self

        # Flatten
        intensity = self.intensity.flatten()
        # finite values
        finite_locations = np.isfinite(intensity)

        # construct output
        intensity = intensity[finite_locations]
        error = self.error.flatten()[finite_locations]
        qx = self.qx.flatten()[finite_locations]
        qy = self.qy.flatten()[finite_locations]
        dqx = (
            None if self.delta_qx is None else self.delta_qx.flatten()[finite_locations]
        )
        dqy = (
            None if self.delta_qy is None else self.delta_qy.flatten()[finite_locations]
        )
        wavelength = (
            None
            if self.wavelength is None
            else self.wavelength.flatten()[finite_locations]
        )

        finite_iq2d = IQazimuthal(
            intensity=intensity,
            error=error,
            qx=qx,
            qy=qy,
            delta_qx=dqx,
            delta_qy=dqy,
            wavelength=wavelength,
        )

        return finite_iq2d

    def id(self):
        return DataType.IQ_AZIMUTHAL

    def concatenate(self, other):
        r"""
        Append additional data points from another ~drtsans.dataobjects.IQazimuthal object and
        return the composite as a new ~drtsans.dataobjects.IQazimuthal object.

        Parameters
        ----------
        other: ~drtsans.dataobjects.IQazimuthal
            Additional data points.

        Returns
        -------
        ~drtsans.dataobjects.IQazimuthal
        """
        return _nary_operation((self, other), np.concatenate, unpack=False)

    def ravel(self):
        """Create a new ~drtsans.dataobjects.IQazimuthal with all of the arrays flattened"""
        kwargs = dict()

        if self.delta_qx is not None:
            kwargs["delta_qx"] = self.delta_qx.ravel()
        if self.delta_qx is not None:
            kwargs["delta_qy"] = self.delta_qy.ravel()
        if self.wavelength is not None:
            kwargs["wavelength"] = self.wavelength.ravel()

        return IQazimuthal(
            intensity=self.intensity.ravel(),
            error=self.error.ravel(),
            qx=self.qx.ravel(),
            qy=self.qy.ravel(),
            **kwargs,
        )


class IQcrystal(
    namedtuple(
        "IQazimuthal", "intensity error qx qy qz delta_qx delta_qy delta_qz wavelength"
    )
):
    """This class holds the information for the crystallographic projection, I(Qx, Qy, Qz). All of the
    arrays must be 1-dimensional and parallel (same length). The resolution terms, (``delta_qx``,
    ``delta_qy``, ``delta_qz``) and ``wavelength`` fields are optional."""

    def __new__(
        cls,
        intensity,
        error,
        qx,
        qy,
        qz,
        delta_qx=None,
        delta_qy=None,
        delta_qz=None,
        wavelength=None,
    ):
        # these conversions do nothing if the supplied information is already a numpy.ndarray
        intensity = np.array(intensity)
        error = np.array(error)
        qx = np.array(qx)
        qy = np.array(qy)
        qz = np.array(qz)

        # check that the manditory fields are parallel
        if len(intensity.shape) != 1:
            raise NotImplementedError(
                "Do not currently support dimension != 1, found {}".format(
                    len(intensity.shape)
                )
            )
        _check_parallel(intensity, error)
        _check_parallel(intensity, qx, qy, qz)  # TODO make more generic

        # work with optional fields
        count = 0
        if delta_qx is not None:
            count += 1
        if delta_qy is not None:
            count += 1
        if delta_qz is not None:
            count += 1
        if not (count == 0 or count == 3):
            raise TypeError(
                "Must specify either all or none of delta_qx, delta_qy, delta_qz"
            )
        if delta_qx is not None:
            delta_qx = np.array(delta_qx)
            delta_qy = np.array(delta_qy)
            delta_qz = np.array(delta_qz)
            _check_parallel(intensity, delta_qx, delta_qy, delta_qz)
        if wavelength is not None:
            wavelength = np.array(wavelength)
            _check_parallel(intensity, wavelength)

        # pass everything to namedtuple
        return super(IQcrystal, cls).__new__(
            cls, intensity, error, qx, qy, qz, delta_qx, delta_qy, delta_qz, wavelength
        )

    def id(self):
        return DataType.IQ_CRYSTAL


def save_i_of_q_to_h5(iq: Union[IQmod, IQazimuthal], h5_name: str):
    """Export I of Q, in form of namedtuple, to an HDF5"""
    # assert isinstance(iq, namedtuple), f'I of Q must be of type namedtuple but not {type(iq)}'

    # Init h5
    iq_h5 = h5py.File(h5_name, "w")
    # create group
    data_group = iq_h5.create_group(iq.__class__.__name__)

    # Write field
    for index, field in enumerate(iq._fields):
        # add data
        data = iq[index]
        if data is not None:
            data_group.create_dataset(field, data=data)

    # Close
    iq_h5.close()


def load_iq1d_from_h5(h5_name: str) -> IQmod:
    """Load an HDF5 for I(Q1D)"""
    # Open file
    with h5py.File(h5_name, "r") as iq_h5:
        data_group = iq_h5["IQmod"]

        value_dict = dict()

        # get tuple element
        for field in ["intensity", "error", "mod_q", "delta_mod_q", "wavelength"]:
            try:
                value_dict[field] = data_group[field][()]
            except KeyError:
                value_dict[field] = None

        iqmod = IQmod(**value_dict)

    return iqmod


def load_iq2d_from_h5(h5_name: str) -> IQazimuthal:
    # Open file
    with h5py.File(h5_name, "r") as iq_h5:
        data_group = iq_h5["IQazimuthal"]

        value_dict = dict()

        # get tuple element
        print(f"DEBUG field: {IQazimuthal._fields}")
        for field in IQazimuthal._fields:
            try:
                value_dict[field] = data_group[field][()]
            except KeyError:
                value_dict[field] = None

        iq2d = IQazimuthal(**value_dict)

    return iq2d


class _Testing:
    r"""
    Mimic the numpy.testing module by applying functions of this module to the component arrays of the IQ objects
    """

    @staticmethod
    def _nary_assertion(iq_objects, assertion_function, unpack=True, **kwargs):
        r"""
        Carry out an assertion on the component arrays for each of the IQ objects.

        Examples:
        - _nary_assertion((iq_1, iq_2), numpy.append, unpack=True)
        - _nary_assertion((iq_1, iq_2), numpy.concatenate, unpack=False)

        Parameters
        ----------
        iq_objects: list
            A list of ~drtsans.dataobjects.IQmod, ~drtsans.dataobjects.IQazimuthal, or ~drtsans.dataobjects.IQcrystal
            objects.
        assertion_function: function
            A function operating on a list of :ref:`~numpy.ndarray` objects, e.g. numpy.concatenate((array1, array2))
        unpack: bool
            If set to :py:obj:`True`, then ``assertion_function`` receives an unpacked list of arrays.
            If set to :py:obj:`False`, then ``assertion_function`` receives the list of arrays as a single argument.
            Examples: numpy.append(*(array1, array2)) versus numpy.concatenate((array1, array2))
        kwargs: dict
            Additional options to be passed

        Raises
        ------
        AssertionError
        """
        reference_object = iq_objects[
            0
        ]  # pick the first of the list as reference object
        assert (
            len(set([type(iq_object) for iq_object in iq_objects])) == 1
        )  # check all objects of same type
        for i in range(len(reference_object)):  # iterate over the IQ object components
            component_name = reference_object._fields[i]
            print(f"all_close on {component_name}")
            i_components = [
                iq_object[i] for iq_object in iq_objects
            ]  # collect the ith components of each object
            if True in [
                i_component is None for i_component in i_components
            ]:  # is any of these None?
                if set(i_components) == set([None]):
                    continue  # all arrays are actually None, so they are identical
                else:
                    raise AssertionError(
                        f"field {component_name} is None for some of the iQ objects"
                    )
            elif unpack is True:
                assertion_function(*i_components, **kwargs)
            else:
                assertion_function(i_components, **kwargs)

    @staticmethod
    def assert_allclose(actual, desired, **kwargs):
        r"""Apply :ref:`~numpy.testing.assert_allclose on each component array"""
        _Testing._nary_assertion(
            (actual, desired),
            assertion_function=np.testing.assert_allclose,
            unpack=True,
            **kwargs,
        )


testing = (
    _Testing()
)  # use it as if it were a module, e.g. testing.assert_allclose(iq_1, iq_2, atol=1.e-6)
