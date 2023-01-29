from sortedcontainers import SortedList

sigma = 3.9560346e-03  # plank constant divided by neutron mass


def tof(wavelength, distance, pulse_width=0.0):
    r"""
    Convert neutron wavelength to time of flight

    Parameters
    ----------
    wavelength: float
        wavelength of the travelling neutron, in microseconds
    distance: float
        Distance travelled by the neutron, in meters
    pulse_width: float
        Neutrons emitted from the moderator with a certain wavelength
        :math:`\lambda` have a distribution of delayed emission times
        with :math:`FWHM(\lambda) \simeq pulsewidth \cdot \lambda`.
        Units are microseconds/Angstroms.

    Returns
    -------
    float
        time of flight (in micro seconds)
    """
    return wavelength * (distance + sigma * pulse_width) / sigma


def from_tof(tof, distance, pulse_width=0.0):
    r"""
    Convert time of flight of arriving neutron to wavelength.

    Parameters
    ----------
    tof: float
        time of flight of the traveling neutron, in Angstroms.
    distance: float
        Distance traveled by the neutron, in meters.
    pulse_width: float
        Neutrons emitted from the moderator with a certain wavelength
        :math:`\lambda` have a distribution of delayed emission times that depends on the wavelength,
        with :math:`FWHM(\lambda) \simeq pulsewidth \cdot \lambda`.
        Units are microseconds/Angstroms.

    Returns
    -------
    float
        wavelength (in Angstroms)
    """
    return tof * sigma / (distance + sigma * pulse_width)


class Wband(object):
    r"""
    A wavelength band, useful for defining one of the possible bands transmitted by a disk chopper.

    Parameters
    ----------
    w_min: float
        Lower boundary wavelength.
    w_max: float
        Upper boundary wavelength.

    Raises
    -------
    ValueError
        Negative input values or lower boundary is bigger than the upper one.
    """

    def __init__(self, w_min, w_max):
        if w_min < 0 or w_max < 0 or w_min > w_max:
            raise ValueError("Invalid wavelength band")
        self._min = w_min
        self._max = w_max

    @property
    def min(self):
        r"""Lower wavelength boundary."""
        return self._min

    @property
    def max(self):
        r"""Upper wavelength boundary."""
        return self._max

    @property
    def width(self):
        r"""Difference between the upper and lower wavelength boundaries."""
        return self._max - self._min

    def __mul__(self, other):
        """
        Find the intersection band between two bands.
        The intersection operation is applied to find out the wavelength bands transmitted by a set of
        two choppers. The bands transmitted by the first chopper will be clipped by the second chopper.

        For example, the intersection between `Wband(1, 6)` and `Wband(3, 4)` is `Wband(3, 4)`. The intersection
        between `Wband(1, 6)` and `Wband(3, 8)` is `Wband(3, 6)`.

        Parameters
        ----------
        other: ~drtsans.wavelength.Wband
            Intersecting band.

        Returns
        -------
        ~drtsans.wavelength.Wband or :py:obj:`None`
            :py:obj:`None` if no intersection, or if the intersection is not a band but just a point,
            as in the intersection between `Wband(0, 1)` and `Wband(1, 2)`.
        """
        # Corner case when we multiply by the null band
        if self is None or other is None:
            return None

        def mul_band(band):
            a = self._min if self._min > band._min else band._min
            b = self._max if self._max < band._max else band._max
            if a >= b:
                return None
            return Wband(a, b)

        def mul_bands(bands):
            return bands * self

        dispatcher = {Wband: mul_band, Wbands: mul_bands}
        mul = [v for k, v in dispatcher.items() if isinstance(other, k)][0]

        return mul(other)

    def __imul__(self, other):
        r"""
        In-place multiplication. The band is clipped by the intersection with another band.

        Parameters
        ----------
        band: ~drtsans.wavelength.Wband
            Intersecting band.
        Returns
        -------
        ~drtsans.wavelength.Wband or :py:obj:`None`
            :py:obj:`None` if no intersection, or if the intersection is not a band but just a point,
            as in the intersection between Wband(0, 1) and Wband(1, 2).
        """
        b = self * other
        self = b
        return self

    def __eq__(self, other):
        return self._min == other._min and self._max == other._max

    def __lt__(self, other):
        return self._min < other._min

    def __str__(self):
        return "Wband({:.3f}, {:.3f})".format(self._min, self._max)


class Wbands(object):
    r"""
    A list of *non overlapping* wavelength bands. Useful for defining a set of bands transmitted by one or
    a set of disk choppers.

    Parameters
    ----------
    args: ~drtsans.wavelength.Wband, ~drtsans.wavelength.Wbands, list of ~drtsans.wavelength.Wband objects,
    list of ~drtsans.wavelength.Wbands objects.
    """

    def __init__(self, *args):
        self._bands = SortedList()
        self += args

    def __len__(self):
        return len(self._bands)

    def __eq__(self, other):
        return self._bands == other._bands

    def __getitem__(self, item):
        return self._bands[item]

    def _valid_add(self, band, index):
        r"""
        Check if ```band`` intersect with any of the bands.

        Parameters
        ----------
        band: ~drtsans.wavelength.Wband
            Candidate band to be inserted.
        index: int
            index of list attribute ```_bands``` containing the band immediaty below ```band```.

        Returns
        -------
        bool
            Tue if ```band``` does not intersect. The band can be inserted in this case.
        """
        if len(self) == 0:
            return True  # fist element goes in always
        if index < len(self):
            low_band = self._bands[index]
            if low_band * band is not None:
                return False
        if index + 1 < len(self):
            high_band = self._bands[index + 1]
            if high_band * band is not None:
                return False
        return True

    def __iadd__(self, other):
        r"""
        Insert one or more wavelength bands in place

        Parameters
        ----------
        other: ~drtsans.wavelength.Wband, iterable.
            Wavelength band to be inserted, or iterable serving ~drtsans.wavelength.Wband objects
            (e.g. another ~drtsans.wavelength.Wbands object).
        """
        if isinstance(other, Wband):
            index = self._bands.bisect_right(other)
            if self._valid_add(other, index):
                self._bands.add(other)
        else:
            try:
                for band in other:
                    self += band
            except TypeError as te:
                print("Argument is not iterable", te)
        return self

    def __mul__(self, other):
        r"""
        Find the intersection with one or more bands.

        Parameters
        ----------
        other: ~drtsans.wavelength.Wband, ~drtsans.wavelength.Wbands
            Intersecting band(s).

        Returns
        -------
        ~drtsans.wavelength.Wbands, :py:obj:`None`
            :py:obj:`None` if no intersection, or if the intersection is not a band but just a point,
            as in the intersection between Wband(0, 1) and Wband(1, 2).
        """
        # Corner case when we multiply by the null bands object
        if self is None or other is None:
            return None

        def mul_band(other_band):
            ib = Wbands()
            for band in self._bands:
                i = band * other_band
                if i is not None:
                    ib += i  # add a Wband object
            return ib

        def mul_bands(other_bands):
            ib = Wbands()
            for band in other_bands:
                i = self * band
                if i is not None:
                    ib += i  # add from a Wbands object
            return ib

        dispatcher = {Wband: mul_band, Wbands: mul_bands}
        mul = [v for k, v in dispatcher.items() if isinstance(other, k)][0]
        r = mul(other)
        if len(r) == 0:
            return None
        return r

    def __imul__(self, other):
        r"""
        In-place intersection with one or more bands.

        Parameters
        ----------
        other: ~drtsans.wavelength.Wbands or ~drtsans.wavelength.Wbandss
            Intersecting band(s).

        Returns
        -------
        ~drtsans.wavelength.Wbands, :py:obj:`None`
            :py:obj:`None` if no intersection, or if the intersection is not a band but just a point,
            as in the intersection between Wband(0, 1) and Wband(1, 2).
        """
        wb = self * other
        self = wb
        return self

    def __str__(self):
        return "(" + ", ".join([str(band) for band in self]) + ")"
