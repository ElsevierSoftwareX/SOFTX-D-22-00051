import pytest
from numpy.testing import assert_almost_equal

from drtsans.chopper import DiskChopper


@pytest.mark.offline
class TestDiskChopper(object):
    ch = DiskChopper(1.0, 45, 60, 2000, 850)

    def test_pulse_width(self):
        assert self.ch.pulse_width == DiskChopper._pulse_width
        self.ch.pulse_width = 0
        assert self.ch.pulse_width == 0
        assert self.ch.pulse_width != DiskChopper._pulse_width
        self.ch.pulse_width = DiskChopper._pulse_width  # restore state

    def test_cutoff_wl(self):
        assert self.ch.cutoff_wl == DiskChopper._cutoff_wl
        self.ch.cutoff_wl = 0
        assert self.ch.cutoff_wl == 0
        assert self.ch.cutoff_wl != DiskChopper._cutoff_wl
        self.ch.cutoff_wl = DiskChopper._cutoff_wl  # restore state

    def test_phase(self):
        assert self.ch.phase == 1150

    def test_period(self):
        assert_almost_equal(self.ch.period, 16666, decimal=0)

    def test_transmission_duration(self):
        assert_almost_equal(self.ch.transmission_duration, 2083, decimal=0)

    def test_opening_phase(self):
        assert_almost_equal(self.ch.opening_phase, 108, decimal=0)

    def test_closing_phase(self):
        assert_almost_equal(self.ch.closing_phase, 2191, decimal=0)

    def test_rewind(self):
        assert_almost_equal(self.ch.rewind, 108, decimal=0)
        self.ch.offset += 109
        assert_almost_equal(self.ch.rewind, -1, decimal=0)
        self.ch.offset -= 109  # restore state

    def test_wavelength(self):
        assert_almost_equal(self.ch.wavelength(1200), 4.7, decimal=1)
        assert_almost_equal(self.ch.wavelength(1200, pulsed=True), 4.3, decimal=1)

    def test_tof(self):
        assert_almost_equal(self.ch.tof(4.747), 1200, decimal=0)
        assert_almost_equal(self.ch.tof(4.399, pulsed=True), 1200, decimal=0)

    def test_transmission_bands(self):
        wb = self.ch.transmission_bands()
        assert len(wb) == 1
        assert_almost_equal((wb[0].min, wb[0].max), (0.42, 8.67), decimal=2)
        ch = DiskChopper(1.0, 30, 240, 0, 0)
        wb = ch.transmission_bands()
        assert len(wb) == 3
        assert_almost_equal((wb[0].min, wb[0].max), (0, 0.687), decimal=2)


if __name__ == "__main__":
    pytest.main([__file__])
