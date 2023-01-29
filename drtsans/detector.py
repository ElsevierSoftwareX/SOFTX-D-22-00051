from collections import OrderedDict
import numpy as np
from mantid.kernel import Property


class Detector:
    r"""
    Auxiliary class that has all the information about a detector
    It allows to read tube by tube.
    """

    def __init__(self, workspace, component_name):
        self._workspace = workspace
        self._current_start_ws_index = (
            None  # first workspace index of the currently considered tube
        )
        self._current_stop_ws_index = (
            None  # last workspace index of the currently considered tube
        )
        self._tube_ws_indices = (
            None  # iterator for tube endpoints, given as workspace indexes
        )

        # Public variables
        self.component_name = (
            None  # name of the assembly of pixels making up the Detector
        )
        self.n_tubes = None  # number of tubes in the detector
        self.n_pixels_per_tube = None
        self.first_det_id = (
            None  # pixel ID for first pixel detector that is not a monitor
        )
        self.last_det_id = None  # pixel ID for last pixel detector
        self.detector_id_to_ws_index = None  # mapping from pixel ID to workspace index
        self.data_y = None
        self.data_e = None
        self.tube_ws_indices = None

        # Initialize attributes of the Detector
        self._detector_details(component_name)
        self._detector_id_to_ws_index()
        self._extract_data()
        self._set_tube_ws_indices()

    def _detector_details(self, component_name):
        r"""
        Initializes the following attributes of the Detector:
        component_name, n_tubes, n_pixels_per_tube, first_det_id, last_det_id

        Parameters
        ----------
        component_name : string
            Name of the pixel assembly, one component of the instrument.
        """
        self.component_name = component_name
        i = self._workspace.getInstrument()
        component = i.getComponentByName(component_name)
        num_pixels = 1
        # dive into subelements until get a detector
        while (
            component.type() != "DetectorComponent"
            and component.type() != "GridDetectorPixel"
        ):
            self.n_pixels_per_tube = component.nelements()
            num_pixels *= self.n_pixels_per_tube
            component = component[0]
        self.first_det_id = component.getID()
        self.last_det_id = self.first_det_id + num_pixels - 1
        self.n_tubes = int(num_pixels / self.n_pixels_per_tube)

    def _detector_id_to_ws_index(self):
        r"""
        Maps the detector ID of one pixel to one workspace index.
        Initializes attribute ``detector_id_to_ws_index``.
        """

        spectrum_info = self._workspace.spectrumInfo()
        detector_id_to_index = []
        for ws_index in range(self._workspace.getNumberHistograms()):
            if spectrum_info.isMonitor(ws_index) is True:
                continue
            detector_id_to_index.append(
                (self._workspace.getSpectrum(ws_index).getDetectorIDs()[0], ws_index)
            )
        self.detector_id_to_ws_index = OrderedDict(detector_id_to_index)

    def _extract_data(self):
        r"""
        Extract intensitites and associated uncertainties from the workspace.
        Initializes attributes ``data_y`` and ``data_e``
        """
        self.data_y = self._workspace.extractY()
        self.data_e = self._workspace.extractE()

    def _set_tube_ws_indices(self):
        r"""
        Initializes attribute ``_tube_ws_indices`` by assigning to it an iterator that yields pairs of the form
        ``(start, end)``, where ``start`` and ``end`` are the starting and ending workspace indexes for a given tube.
        The iterator yields as many pairs as tubes in Detector.
        """
        tube_ws_indices = []
        for tube_idx in range(self.n_tubes):
            first_det_id = (
                self.first_det_id + tube_idx * self.n_pixels_per_tube
            )  # tube starts with this pixel ID
            last_det_id = (
                first_det_id + self.n_pixels_per_tube - 1
            )  # tube ends with this pixel ID

            first_ws_index = self.detector_id_to_ws_index[
                first_det_id
            ]  # tube starts with this workspace index
            last_ws_index = self.detector_id_to_ws_index[
                last_det_id
            ]  # tube ends with this workspace index

            tube_ws_indices.append((first_ws_index, last_ws_index))
        self._tube_ws_indices = iter(tube_ws_indices)  # return an iterator

    def next_tube(self):
        r"""Initializes/ updates attributes ``_current_start_ws_index`` and ``_current_stop_ws_index``"""
        self._current_start_ws_index, self._current_stop_ws_index = next(
            self._tube_ws_indices
        )

    def get_current_ws_indices(self):
        r"""
        First and last workspace indices for the currently considered tube.

        Returns
        -------
        tuple
        """
        return self._current_start_ws_index, self._current_stop_ws_index

    def get_current_ws_indices_range(self):
        r"""
        Array of workspace indexes for the currently considered tube.

        Returns
        -------
        ~numpy.ndarray
        """

        return np.array(
            range(self._current_start_ws_index, self._current_stop_ws_index + 1)
        )

    def get_ws_data(self):
        r"""
        Intensities and associated uncertainties for the currently considered tube.

        Returns
        -------
        tuple
            A two-item tuple containing, in this order, intensites and uncertainties in the shape of ~numpy.ndarray.
        """
        return (
            self.data_y[
                self._current_start_ws_index : self._current_stop_ws_index + 1
            ].flatten(),
            self.data_e[
                self._current_start_ws_index : self._current_stop_ws_index + 1
            ].flatten(),
        )

    def get_pixels_masked(self):
        r"""
        Pixel masks for the currently considered tube.

        Returns
        -------
        ~numpy.ndarray
            Array of ``Bool`` values, with :py:obj:`True` for the masked pixels and :py:obj:`False` otherwise.
        """
        spectrum_info = self._workspace.spectrumInfo()
        return np.array(
            [
                spectrum_info.isMasked(int(idx))
                for idx in self.get_current_ws_indices_range()
            ]
        )

    def get_pixels_infinite(self):
        r"""
        Pixel mask for pixels with non-finite intensities in the currently considered tube.
        Returns an array of booleans for this tube
        where the pixel count is EMPTY_DBL

        Returns
        -------
        ~numpy.ndarray
            Array of ``Bool`` values, with :py:obj:`True` for the pixels with non-finite intensities, and
            :py:obj:`False` otherwise.
        """
        return np.array(
            [
                self._workspace.readY(int(idx))[0] == Property.EMPTY_DBL
                for idx in self.get_current_ws_indices_range()
            ]
        )

    def get_y_coordinates(self):
        r"""
        Y-coordinates of the pixels for the currently considered tube.

        Returns
        -------
        ~numpy.ndarray
        """
        detector_info = self._workspace.spectrumInfo()
        return np.array(
            [
                detector_info.position(int(idx))[1]
                for idx in self.get_current_ws_indices_range()
            ]
        )


class Component:
    """
    Stores information about the component
    """

    # class variables that will cache the component details
    dim_x = -1  # number of tubes
    dim_y = -1  # number of pixels per tube
    dims = -1  # total number of pixels
    first_index = -1  # workspace index of the smallest pixel id

    def __init__(self, workspace, component_name):
        self._component_name = component_name
        self._workspace = workspace
        self._initialization()

    def _initialization(self):
        first_det_id = self._detector_details()
        self._detector_first_ws_index(first_det_id)

    def _num_pixels_in_tube(self, info, component_index):
        """Recursive function that determines how many pixels are in a single
        tube (y-dimension). This assumes that things without grand-children
        are tubes"""
        component_index = int(component_index)
        children = info.children(component_index)

        grandchildren = info.children(int(children[0]))
        if len(grandchildren) == 0:
            return children[0], len(children)
        else:
            return self._num_pixels_in_tube(info, children[0])

    def _detector_details(self):
        """Private function that reads the instrument and get component_name
        dimensions and first detector id
        """
        component_info = self._workspace.componentInfo()
        detector_info = self._workspace.detectorInfo()
        component_index = component_info.indexOfAny(self._component_name)

        total_pixels = len(component_info.detectorsInSubtree(component_index))
        tube_index, self.dim_y = self._num_pixels_in_tube(
            component_info, component_index
        )
        self.dim_x = total_pixels // self.dim_y
        self.dims = total_pixels

        return detector_info.detectorIDs()[tube_index]

    def _detector_first_ws_index(self, first_det_id):
        """sets the first_index of this component"""
        for ws_index in range(self._workspace.getNumberHistograms()):
            if (
                self._workspace.getSpectrum(ws_index).getDetectorIDs()[0]
                == first_det_id
            ):
                self.first_index = ws_index
                break
        else:
            raise ValueError(
                "Iterared WS and did not find first det id = " "{}".format(first_det_id)
            )

    def masked_ws_indices(self):
        """
        return an array with True or False if a detector is either masked or
        not for all the component

        Returns
        -------
        bool np.array
            array with True or False if a detector is either masked or
            not for all the component
        """
        si = self._workspace.spectrumInfo()
        mask_array = [
            si.isMasked(i)
            for i in range(self.first_index, self.first_index + self.dim_x * self.dim_y)
        ]
        return np.array(mask_array)

    # TODO - Implement!
    def monitor_indices(self):
        """

        Returns
        -------

        """
        return np.array([])

    def __str__(self):
        return (
            "Component: {} with {} pixels (dim x={}, dim y={})."
            " First index = {}.".format(
                self._component_name,
                self.dims,
                self.dim_x,
                self.dim_y,
                self.first_index,
            )
        )
