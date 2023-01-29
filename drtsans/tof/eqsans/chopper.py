r"""
This module provides class `EQSANSDiskChopperSet` representing the set of four disk choppers
(two of them paired as a double chopper).  The main goal is to find the set of neutron wavelength
bands transmitted by the chopper set, given definite choppers settings such as aperture and starting phase.
"""

from drtsans.chopper import DiskChopper
from drtsans.samplelogs import SampleLogs
from drtsans.frame_mode import FrameMode
from drtsans.path import exists
from mantid.api import Run
from mantid.simpleapi import LoadNexusProcessed, mtd


class EQSANSDiskChopperSet(object):
    r"""
    Set of disks choppers installed in EQSANS.

    Parameters
    ----------
    other: file name, workspace, Run object, run number
        Load the chopper settings from this object.
    """

    #: Neutrons of a given wavelength :math:`\lambda` emitted from the moderator follow a distribution of delayed
    #: emission times that depends on the wavelength, and is characterized by function
    #: :math:`FWHM(\lambda) \simeq pulsewidth \cdot \lambda`.
    #: This is the default :math:`pulsewidth` in micro-sec/Angstrom.
    _pulse_width = 20

    #: The number of wavelength bands transmitted by a disk chopper is determined by the slowest emitted neutron,
    #: expressed as the maximum wavelength. This is the default cut-off maximum wavelength, in Angstroms.
    _cutoff_wl = 35

    #: number of single-disk choppers in the set.
    _n_choppers = 4

    #: Transmission aperture of the choppers, in degrees.
    _aperture = [129.605, 179.989, 230.010, 230.007]

    #: Distance to neutron source (moderator), in meters.
    _to_source = [5.700, 7.800, 9.497, 9.507]

    #: Phase offsets, in micro-seconds. These values are required to calibrate the value reported in the
    #: metadata. The combination on the reported phase and this offset is the time (starting from the
    #: current pulse) at which the middle of the choppers apertures will intersect with the neutron beam axis.
    _offsets = {
        FrameMode.not_skip: [9507.0, 9471.0, 9829.7, 9584.3],
        FrameMode.skip: [19024.0, 18820.0, 19714.0, 19360.0],
    }

    def __init__(self, other):
        # Load choppers settings from the logs
        if isinstance(other, Run) or str(other) in mtd:
            sample_logs = SampleLogs(other)
        elif exists(other):
            ws = LoadNexusProcessed(other)
            sample_logs = SampleLogs(ws)
        else:
            raise RuntimeError(
                "{} is not a valid file name, workspace, Run object or run number".format(
                    other
                )
            )

        self._choppers = list()
        for chopper_index in range(self._n_choppers):
            aperture = self._aperture[chopper_index]
            to_source = self._to_source[chopper_index]
            speed = sample_logs["Speed{}".format(1 + chopper_index)].value.mean()
            sensor_phase = sample_logs["Phase{}".format(1 + chopper_index)].value.mean()
            ch = DiskChopper(to_source, aperture, speed, sensor_phase)
            ch.pulse_width = self._pulse_width
            ch.cutoff_wl = self._cutoff_wl
            self._choppers.append(ch)

        # Determine period and if frame skipping mode from the first chopper
        ch = self._choppers[0]
        condition = abs(ch.speed - sample_logs.frequency.value.mean()) / 2 > 1
        self.frame_mode = FrameMode.skip if condition else FrameMode.not_skip

        # Select appropriate offsets, based on the frame-skip mode.
        for chopper_index in range(self._n_choppers):
            ch = self._choppers[chopper_index]
            ch.offset = self._offsets[self.frame_mode][chopper_index]

    def transmission_bands(self, cutoff_wl=None, delay=0, pulsed=False):
        r"""
        Wavelength bands transmitted by the chopper apertures. The number of bands is determined by the
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
            :const:`~drtsans.tof.eqsans.chopper.EQSANSDiskChopperSet._pulse_width` for a
            more detailed explanation.

        Returns
        -------
        ~drtsans.wavelength.Wbands
            Set of wavelength bands transmitted by the chopper.
        """
        if cutoff_wl is None:
            cutoff_wl = self._cutoff_wl
        # Transmission bands of the first chopper
        ch = self._choppers[0]
        wb = ch.transmission_bands(cutoff_wl, delay, pulsed)
        # Find the common transmitted bands between the first chopper
        # and the ensuing choppers
        for ch in self._choppers[1:]:
            wb *= ch.transmission_bands(cutoff_wl, delay, pulsed)
        # We end up with the transmission bands of the chopper set
        return wb

    @property
    def period(self):
        return self._choppers[0].period

    def __getitem__(self, item):
        return self._choppers[item]

    @property
    def pulse_width(self):
        return self._pulse_width
