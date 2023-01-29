r"""
This module provides class `DiskChopper` representing a rotating chopper with an aperture of certain width. The
main goal is to find the set of neutron wavelength bands transmitted by the chopper, given definite chopper
settings such as aperture and starting phase.
"""

from drtsans.wavelength import Wband, Wbands


class DiskChopper(object):
    r"""
    Rotating disk chopper with an aperture of a certain width letting neutrons through.

    The angular position of the middle of the chopper aperture at the moment a neutron pulse happens is given
    by a metadata entry (the sensor phase) and an additional angle (the offset). The offset server to calibrate
    the value reported by the metadata.

    Parameters
    ----------
    to_source: float
        Distance to the neutron source (moderator) in meters
    aperture: float
        Width of the opening window letting neutrons through, in degrees
    speed: float
        rotational frequency, in Hz
    sensor_phase: float
        phase reported by the installed sensor in the metadata. It's the time for the chopper to rotate by
        and angle created by the following three points: (1) the center of the chopper, (2) the middle of
        the aperture, and (3) the point of intersection of the chopper and the pulse prompt neutrons. Units
        are micro seconds
    offset: float
        Additional phase difference to be added to the `sensor_phase` due to miscalibrations. The offset calibrates
        the value `sensor_phase` reported by the metadata. Units are in micro seconds.
    """

    #: Neutrons of a given wavelength :math:`\lambda` emitted from the moderator follow a distribution of delayed
    #: emission times that depends on the wavelength, and is characterized by function
    #: :math:`FWHM(\lambda) \simeq pulsewidth \cdot \lambda`.
    #: This is the default :math:`pulsewidth` in micro-sec/Angstrom.
    _pulse_width = 20

    #: The number of wavelength bands transmitted by a disk chopper is determined by the slowest emitted neutron,
    #: expressed as the maximum wavelength. This is the default cut-off maximum wavelength, in Angstroms.
    _cutoff_wl = 35

    def __init__(self, to_source, aperture, speed, sensor_phase, offset=0):
        self.to_source = to_source
        self.aperture = float(aperture)
        self.speed = float(speed)
        self.sensor_phase = float(sensor_phase)
        self.offset = float(offset)

    @property
    def pulse_width(self):
        r"""
        Neutrons of a given wavelength :math:`\lambda` emitted from the
        moderator have a distribution of delayed times that depends on the wavelength, and is characterized by
        a :math:`FWHM(\lambda) \simeq pulsewidth \cdot \lambda`. This property can override the default
        pulse width :const:`~drtsans.chopper.DiskChopper._pulse_width`.
        """
        return self._pulse_width

    @pulse_width.setter
    def pulse_width(self, value):
        r"""
        Override the default pulse width :const:`~drtsans.chopper.DiskChopper._pulse_width`.
        """
        self._pulse_width = value

    @property
    def cutoff_wl(self):
        r"""
        Discard neutrons transmitted by the disk chopper having a wavelength above this quantity. This
        property can override the default cutoff wavelength :const:`~drtsans.chopper.DiskChopper._cutoff_wl`.
        """
        return self._cutoff_wl

    @cutoff_wl.setter
    def cutoff_wl(self, value):
        r"""
        Override default cutoff wavelength :const:`~drtsans.chopper.DiskChopper._cutoff_wl`.
        """
        self._cutoff_wl = value

    @property
    def phase(self):
        r"""
        Time (starting from the current pulse) when the middle of the chopper aperture will
        intersect with the neutron beam axis, in micro seconds.
        """
        return self.sensor_phase - self.offset

    @property
    def period(self):
        r"""
        Time span required by the chopper for a full spin, in micro seconds.
        """
        return 1.0e6 / self.speed

    @property
    def transmission_duration(self):
        r"""
        Time span taking the chopper to spin an angle equal to its aperture, in micro seconds.
        """
        return self.period * (self.aperture / 360.0)

    @property
    def opening_phase(self):
        r"""
        Time (starting from the current pulse) when the opening edge of the chopper aperture will
        intersect with the neutron beam axis, in micro seconds.
        """
        return self.phase - 0.5 * self.transmission_duration

    @property
    def closing_phase(self):
        r"""
        Time (starting from the current pulse) when the closing edge of the chopper aperture will
        intersect with the neutron beam axis, in micro seconds.
        """
        return self.phase + 0.5 * self.transmission_duration

    @property
    def rewind(self):
        r"""
        Spin the chopper backwards until the chopper aperture intersects with the neutron beam axis.
        At this point, the :const:`~drtsans.chopper.DiskChopper.opening_phase` will be negative, and the
        :const:`~drtsans.chopper.DiskChopper.closing_phase` will be positive.

        Returns
        -------
        float
            Opening phase, in micro seconds. The opening phase will be negative (most likely) or zero.
        """
        t_closing = self.closing_phase
        while t_closing < 0:
            t_closing += self.period
        return t_closing - self.transmission_duration

    def wavelength(self, tof, delay=0, pulsed=False):
        r"""
        Convert time-of-flight to neutron wavelength, for a neutron that has traveled the distance from the
        moderator to the chopper.

        The measured time of flight :math:`t_m` plus the additional delay :math:`d` is equal to the
        real time of flight :math:`tof` plus the delayed emission time from the moderator :math:`p \lambda`,
        where :math:`p` is constant :const:`~drtsans.chopper.DiskChopperSet._pulse_width`.

        .. math::

           t_m + d = tof + p \lambda

           D = tof / v

           v = \frac{h}{m\lambda}

        where :math:`D` is the distance from moderator to chopper and :math:`v` is the neutron velocity.
        Solving this system of equations for :math:`\lambda`, one obtains

        .. math::

            \lambda = \frac{h}{m} \frac{t_m + d}{D + hp/m}

        Parameters
        ----------
        tof: float
            time of flight, in micro seconds
        delay: float
            Additional time-of-flight to include in the calculations. For instance, this could be a multiple
            of the the pulse period.
        pulsed: bool
            Include a correction due to delayed emission of neutrons from the moderator. See
            :const:`~drtsans.chopper.DiskChopper._pulse_width` for a more detailed explanation.

        Returns
        -------
        float
            Neutron wavelength (in Angstroms). Returns zero for negative `tof`.
        """
        sigma = 3.9560346e-03  # plank constant divided by neutron mass
        loc = self.to_source
        if pulsed is True:
            loc += sigma * self._pulse_width
        wl = sigma * (tof + delay) / loc
        if wl < 0:
            wl = 0
        return wl

    def tof(self, wavelength, delay=0, pulsed=False):
        r"""
        Convert wavelength to *measured* time-of-flight, for a neutron that has traveled the distance from the
        moderator to the chopper.

        The measured time of flight :math:`t_m` plus the additional delay :math:`d` is equal to the
        real time of flight :math:`t_r` plus the delayed emission time from the moderator :math:`p \lambda`,
        where :math:`p` is constant :const:`~drtsans.chopper.DiskChopperSet._pulse_width`.

        .. math::

           t_m + d = t_r + p \lambda

           D = t_r / v

           v = \frac{h}{m\lambda}

        where :math:`D` is the distance from moderator to chopper and :math:`v` is the neutron velocity.
        Solving this system of equations for :math:`t_m`, one obtains

        .. math::

            t_m = \lambda \frac{D + hp/m}{h/m} - d

        Parameters
        ----------
        wavelength: float
            wavelength of the neutron, in micro seconds.
        delay: float
            Additional time-of-flight to include in the calculations. For instance, this could be a multiple
            of the the pulse period.
        pulsed: bool
            Include a correction due to delayed emission of neutrons from the moderator. See
            :const:`~drtsans.chopper.DiskChopper._pulse_width` for a more detailed explanation.

        Returns
        -------
        float
            time-of-flight, in micro seconds.
        """
        sigma = 3.9560346e-03  # plank constant divided by neutron mass
        loc = self.to_source
        if pulsed is True:
            loc += sigma * self._pulse_width
        return wavelength * loc / sigma - delay

    def transmission_bands(self, cutoff_wl=None, delay=0, pulsed=False):
        r"""
        Wavelength bands transmitted by the chopper aperture. The number of bands is determined by the
        slowest neutrons emitted from the moderator.

        Parameters
        ----------
        cutoff_wl: float
            maximum wavelength of incoming neutrons. Discard slower neutrons when finding the transmission bands.
        delay: float
            Additional time-of-flight to include in the calculations. For instance, this could be a multiple
            of the the pulse period.
        pulsed: bool
            Include a correction due to delayed emission of neutrons from the moderator. See
            :const:`~drtsans.chopper.DiskChopper._pulse_width` for a more detailed explanation.

        Returns
        -------
        ~drtsans.wavelength.Wbands
            Set of wavelength bands transmitted by the chopper.
        """
        if cutoff_wl is None:
            cutoff_wl = self.cutoff_wl
        wb = Wbands()
        t_opening = self.rewind
        # shortest wavelength, obtained with pulsed correction if needed
        opening_wl = self.wavelength(t_opening, delay, pulsed)
        while opening_wl < cutoff_wl:
            # slowest wavelength, obtained with no pulse correction
            t = t_opening + self.transmission_duration
            closing_wl = self.wavelength(t, delay, False)
            if closing_wl > cutoff_wl:
                closing_wl = cutoff_wl
            wb += Wband(opening_wl, closing_wl)
            t_opening += self.period
            opening_wl = self.wavelength(t_opening, delay, pulsed)
        return wb
