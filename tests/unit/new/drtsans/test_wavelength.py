import pytest

from drtsans.wavelength import Wband, Wbands


@pytest.mark.offline
class TestWband(object):
    def test_init(self):
        with pytest.raises(ValueError):
            Wband(-1, 0)
            assert False, 'Should have failed "Wband(-1, 0)"'
        with pytest.raises(ValueError):
            Wband(0, -1)
            assert False, 'Should have failed "Wband(0, 1)"'
        with pytest.raises(ValueError):
            Wband(1, 0)
            assert False, 'Should have failed "Wband(1, 0)"'

    def test_width(self):
        assert Wband(1, 2).width == 1
        assert Wband(1, 1).width == 0

    def test_intersect(self):
        b = Wband(1, 2)
        assert b * Wband(0, 0.5) is None
        assert b * Wband(0, 1) is None
        assert b * Wband(0, 1.5) == Wband(1, 1.5)
        assert b * Wband(0, 2) == b
        assert b * Wband(2, 3) is None
        assert b * Wband(2.5, 3) is None


@pytest.mark.offline
class TestWbands(object):
    def test_init(self):
        # initialize with a Wband object
        ws = Wbands(Wband(1, 2))
        assert len(ws) == 1
        # initialize with multiple arguments
        ws = Wbands(Wband(1, 2), Wband(0, 0.5))
        assert len(ws) == 2
        ref = Wbands(Wband(0, 0.5), Wband(1, 2))
        assert ws == ref
        # initialize from iterable
        ws = Wbands([Wband(1, 2), Wband(0, 0.5)])
        assert ws == ref
        # initialize from Wbands object
        bs = Wbands(ws)
        assert bs == ref
        # Mix object types in initializer
        bs = Wbands(Wband(3, 5), ws)
        assert bs == Wbands(Wband(0, 0.5), Wband(1, 2), Wband(3, 5))

    def test_mul(self):
        ws = Wbands(Wband(1, 2), Wband(3, 5))
        # Product of one Wband with one Wbands
        assert ws * Wband(0, 4) == Wbands(Wband(1, 2), Wband(3, 4))
        assert Wband(0, 4) * ws == Wbands(Wband(1, 2), Wband(3, 4))
        assert ws * Wband(1.5, 3.5) == Wbands(Wband(1.5, 2), Wband(3, 3.5))
        assert Wband(1.5, 3.5) * ws == Wbands(Wband(1.5, 2), Wband(3, 3.5))
        # Product of two Wbands
        vs = Wbands(Wband(0, 1.5), Wband(1.7, 4))
        assert ws * vs == Wbands(Wband(1.7, 2), Wband(3, 4), Wband(1, 1.5))
        assert vs * ws == Wbands(Wband(1, 1.5), Wband(3, 4), Wband(1.7, 2))
        # Product of two Wband and one Wbands
        assert Wband(2, 3.5) * ws * Wband(0, 4) == Wbands(Wband(3, 3.5))
        # Product of three Wbands
        intersection = Wbands(Wband(0, 1.5), Wband(2, 3.5)) * ws * vs
        assert intersection == Wbands(Wband(1, 1.5), Wband(3, 3.5))

    def test_getitem(self):
        ws = Wbands(Wband(1, 1.5), Wband(3, 4), Wband(1.7, 2))
        assert ws[1] == Wband(1.7, 2)


if __name__ == "__main__":
    pytest.main([__file__])
