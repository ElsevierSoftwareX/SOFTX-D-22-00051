import os
import pytest
import tempfile

from drtsans.dataobjects import (
    concatenate,
    IQazimuthal,
    IQmod,
    load_iqmod,
    save_iqmod,
    testing,
)
from drtsans.dataobjects import verify_same_q_bins
from tests.conftest import assert_wksp_equal


def test_concatenate():
    # test with IQmod objects
    iq1 = IQmod([1, 2, 3], [4, 5, 6], [7, 8, 9], delta_mod_q=[10, 11, 12])
    iq2 = IQmod([4, 5, 6], [4, 5, 6], [7, 8, 9], wavelength=[10, 11, 12])
    iq3 = IQmod([7, 8, 9], [4, 5, 6], [7, 8, 9])
    iq = concatenate((iq1, iq2, iq3))
    assert iq.intensity == pytest.approx([1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert iq.delta_mod_q is None
    assert iq.wavelength is None
    # test with IQazimuthal objects
    iq1 = IQazimuthal(
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        delta_qx=[13, 14, 15],
        delta_qy=[10, 11, 12],
    )
    iq2 = IQazimuthal(
        [4, 5, 6], [4, 5, 6], [7, 8, 9], [13, 14, 15], wavelength=[10, 11, 12]
    )
    iq3 = IQazimuthal([7, 8, 9], [4, 5, 6], [7, 8, 9], [16, 17, 18])
    iq = concatenate((iq1, iq2, iq3))
    assert iq.intensity == pytest.approx([1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert iq.qy == pytest.approx([10, 11, 12, 13, 14, 15, 16, 17, 18])
    assert iq.delta_qx is None
    assert iq.delta_qy is None
    assert iq.wavelength is None


class TestIQmod:
    def test_to_from_csv(self):
        iq = IQmod([1, 2, 3], [4, 5, 6], [7, 8, 9])
        filename = tempfile.NamedTemporaryFile("wb", suffix=".dat").name
        iq.to_csv(filename)
        iq_other = IQmod.read_csv(filename)
        testing.assert_allclose(iq, iq_other)
        os.remove(filename)

    def test_IQmod_creation(self):
        # these are expected to work
        IQmod([1, 2, 3], [4, 5, 6], [7, 8, 9])
        IQmod([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12])
        IQmod([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15])
        IQmod([1, 2, 3], [4, 5, 6], [7, 8, 9], wavelength=[13, 14, 15])

        # intensity isn't 1d
        with pytest.raises(TypeError):
            IQmod([[1, 2], [3, 4]], [4, 5, 6], [7, 8, 9])

        # arrays are not parallel
        with pytest.raises(TypeError):
            IQmod([1, 3], [4, 5, 6], [7, 8, 9])
        with pytest.raises(TypeError):
            IQmod([1, 2, 3], [4, 6], [7, 8, 9])
        with pytest.raises(TypeError):
            IQmod([1, 2, 3], [4, 5, 6], [7, 9])

        # not enough arguments
        with pytest.raises(TypeError):
            IQmod([1, 2, 3], [4, 5, 6])

    def test_mul(self):
        iqmod = IQmod([1, 2, 3], [4, 5, 6], [7, 8, 9])
        iqmod = 2.5 * iqmod
        assert iqmod.intensity == pytest.approx([2.5, 5, 7.5])
        iqmod = iqmod * 2
        assert iqmod.intensity == pytest.approx([5, 10, 15])

    def test_truediv(self):
        iqmod = IQmod([1, 2, 3], [4, 5, 6], [7, 8, 9])
        iqmod = iqmod / 2
        assert iqmod.error == pytest.approx([2, 2.5, 3])

    def test_extract(self):
        iqmod = IQmod([1, 2, 3], [4, 5, 6], [7, 8, 9])
        iqmod_2 = iqmod.extract(2)
        assert iqmod_2.mod_q == pytest.approx(9)
        iqmod_2 = iqmod.extract(slice(None, None, 2))
        assert iqmod_2.intensity == pytest.approx([1, 3])
        iqmod_2 = iqmod.extract(iqmod.mod_q < 9)
        assert iqmod_2.error == pytest.approx([4, 5])

    def test_concatenate(self):
        iqmod = IQmod([1, 2, 3], [4, 5, 6], [7, 8, 9])
        iqmod_2 = iqmod.concatenate(IQmod([4, 5], [7, 8], [10, 11]))
        assert iqmod_2.mod_q == pytest.approx([7, 8, 9, 10, 11])

    def test_sort(self):
        iqmod = IQmod([1, 2, 3], [4, 5, 6], [7, 9, 8])
        iqmod = iqmod.sort()
        assert iqmod.mod_q == pytest.approx([7, 8, 9])
        assert iqmod.intensity == pytest.approx([1, 3, 2])

    def test_IQmod_to_mtd(self):
        # create the data object
        iqmod = IQmod([1, 2, 3], [4, 5, 6], [7, 8, 9])
        # convert to mantid workspace
        wksp = iqmod.to_workspace()

        # verify results
        assert_wksp_equal(wksp, iqmod)


def test_save_load_iqmod_pandas():
    """Test save and load I(Q) to and from ASCII with pandas

    Returns
    -------

    """
    import numpy as np

    # Test on IQmod with Q, I, dI
    # I(Q) without delta Q
    iq = IQmod([1, 2, 3, np.nan], [4, 5, 6, 0], [7, 8, 9, 0])
    filename = tempfile.NamedTemporaryFile("wb", suffix=".dat").name
    #  Save
    save_iqmod(iq, filename, header_type="Pandas")
    # Load
    iq_other = load_iqmod(filename)
    # Verify
    iq_expected = IQmod([1, 2, 3], [4, 5, 6], [7, 8, 9])
    testing.assert_allclose(iq_expected, iq_other)

    # Check column order
    iq_file = open(filename, "r")
    line0 = iq_file.readline()
    line1 = iq_file.readline()
    iq_file.close()

    # Clean
    os.remove(filename)

    assert line0.strip() == "# NANs have been skipped"
    column_names = line1.split()
    assert column_names == ["mod_q", "intensity", "error"]


def test_save_load_iqmod_mantid_mantid_ascii():
    """Test save and load I(Q) to and from ASCII

    Returns
    -------

    """
    # Test on IQmod with Q, I, dI, dQ
    iq = IQmod([1, 2, 3, 3.5], [4, 5, 6, 0], [7, 8, 9, 0], [1.1, 1.2, 1.3, 1.4])
    filename = tempfile.NamedTemporaryFile("wb", suffix=".dat").name
    #  Save
    save_iqmod(iq, filename, header_type="MantidAscii")
    # Load
    iq_other = load_iqmod(filename, header_type="MantidAscii")

    testing.assert_allclose(iq, iq_other)

    # Check column order
    iq_file = open(filename, "r")
    line0 = iq_file.readline()
    line1 = iq_file.readline()
    iq_file.close()

    # Clean
    os.remove(filename)

    assert line0.strip() == "# I(Q)"
    assert line1 == "#Q (1/A)        I (1/cm)        dI (1/cm)       dQ (1/A)\n"


def test_verify_same_bins():
    """Test method verify_same_q_bins"""
    # Test IQmod without wavelength
    iq1d0 = IQmod([1, 2, 3], [4, 5, 6], [7, 8, 9])
    iq1d1 = IQmod([4, 9, 3], [4, 5, 6], [7, 8, 9])
    assert verify_same_q_bins(iq1d0, iq1d1)

    iq1d0 = IQmod([1, 2, 3], [4, 5.5, 6], [7, 8.8, 9])
    iq1d1 = IQmod([4, 9, 3], [4, 5, 6], [7, 8, 9])
    assert verify_same_q_bins(iq1d0, iq1d1) is False

    # Test IQmod with wavelength
    iq1d0 = IQmod(
        [1, 2, 3, 4, 6, 6],
        [4, 5, 6, 4, 5, 6],
        [7, 8, 9, 7, 8, 9],
        wavelength=[10, 10, 10, 11, 11, 11],
    )
    iq1d1 = IQmod(
        [4, 9, 3, 8, 7, 6],
        [4, 5, 6, 4, 5, 6],
        [7, 8, 9, 7, 8, 9],
        wavelength=[10, 10, 10, 11, 11, 11],
    )
    assert verify_same_q_bins(iq1d0, iq1d1)

    # Test IQmod with wavelength
    iq1d0 = IQmod(
        [1, 2, 3, 4, 6, 6],
        [4, 5, 6, 4, 5, 6],
        [7, 8, 9, 7, 8, 9],
        wavelength=[10.1, 10.1, 10.1, 11, 11, 11],
    )
    iq1d1 = IQmod(
        [4, 9, 3, 8, 7, 6],
        [4, 5, 6, 4, 5, 6],
        [7, 8, 9, 7, 8, 9],
        wavelength=[10, 10, 10, 11, 11, 11],
    )
    assert verify_same_q_bins(iq1d0, iq1d1) is False

    # Test 2D
    iq2d0 = IQazimuthal(
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18],
        [19, 20, 21],
    )
    iq2d1 = IQazimuthal(
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18],
        [19, 20, 21],
    )
    assert verify_same_q_bins(iq2d0, iq2d1)

    iq2d0 = IQazimuthal(
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12.2],
        [13, 14, 15],
        [16, 17, 18],
        [19, 20, 21],
    )
    iq2d1 = IQazimuthal(
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18],
        [19, 20, 21],
    )
    assert verify_same_q_bins(iq2d0, iq2d1) is False

    # Expect failure
    with pytest.raises(RuntimeError):
        verify_same_q_bins(iq1d0, iq2d0)


def test_save_load_iqmod_dq():
    """Test save and load I(Q) to and from ASCII with Q, I(Q), dI(Q) and delta Q

    Returns
    -------

    """
    # Test on IQmod with Q, I, dI
    # I(Q) without delta Q
    iq = IQmod([1, 2, 3], [4, 5, 6], [7, 8, 9], [0.1, 0.12, 0.13])
    filename = tempfile.NamedTemporaryFile("wb", suffix=".dat").name

    #  Save
    save_iqmod(iq, filename, header_type="Pandas")
    # Load
    iq_other = load_iqmod(filename)
    # Verify
    testing.assert_allclose(iq, iq_other)

    # Check column order
    iq_file = open(filename, "r")
    line0 = iq_file.readline()
    iq_file.close()

    # Clean
    os.remove(filename)

    column_names = line0.split()
    assert column_names == ["mod_q", "intensity", "error", "delta_mod_q"]


class TestIQazimuthal:
    def test_1d_creation(self):
        # these are expected to work
        IQazimuthal([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12])
        IQazimuthal(
            [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]
        )
        IQazimuthal(
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21],
        )

        # arrays are not parallel
        with pytest.raises(TypeError):
            IQazimuthal(
                [1, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
                [13, 14, 15],
                [16, 17, 18],
                [19, 20, 21],
            )
        with pytest.raises(TypeError):
            IQazimuthal(
                [1, 2, 3],
                [4, 6],
                [7, 8, 9],
                [10, 11, 12],
                [13, 14, 15],
                [16, 17, 18],
                [19, 20, 21],
            )
        with pytest.raises(TypeError):
            IQazimuthal(
                [1, 2, 3],
                [4, 5, 6],
                [7, 9],
                [10, 11, 12],
                [13, 14, 15],
                [16, 17, 18],
                [19, 20, 21],
            )
        with pytest.raises(TypeError):
            IQazimuthal(
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 12],
                [13, 14, 15],
                [16, 17, 18],
                [19, 20, 21],
            )
        with pytest.raises(TypeError):
            IQazimuthal(
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
                [13, 15],
                [16, 17, 18],
                [19, 20, 21],
            )
        with pytest.raises(TypeError):
            IQazimuthal(
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
                [13, 14, 15],
                [16, 18],
                [19, 20, 21],
            )
        with pytest.raises(TypeError):
            IQazimuthal(
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
                [13, 14, 15],
                [16, 17, 18],
                [19, 21],
            )

        # not enough arguments
        with pytest.raises(TypeError):
            IQazimuthal(
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
                [13, 14, 15],
                wavelength=[16, 17, 18],
            )
        with pytest.raises(TypeError):
            IQazimuthal([1, 2, 3], [4, 5, 6], [7, 8, 9])

    def test_2d_creation(self):
        # these are expected to work
        IQazimuthal(
            [[1, 2], [3, 4]], [[4, 5], [6, 7]], [[7, 8], [9, 10]], [[10, 11], [12, 13]]
        )
        IQazimuthal(
            [[1, 2], [3, 4]],
            [[4, 5], [6, 7]],
            [[7, 8], [9, 10]],
            [[10, 11], [12, 13]],
            [[13, 14], [15, 16]],
            [[16, 17], [18, 19]],
        )
        IQazimuthal(
            [[1, 2], [3, 4]],
            [[4, 5], [6, 7]],
            [[7, 8], [9, 10]],
            [[10, 11], [12, 13]],
            [[13, 14], [15, 16]],
            [[16, 17], [18, 19]],
            [[19, 20], [21, 22]],
        )

        # arrays are not parallel
        with pytest.raises(TypeError):
            IQazimuthal(
                [[1, 2]],
                [[4, 5], [6, 7]],
                [[7, 8], [9, 10]],
                [[10, 11], [12, 13]],
                [[13, 14], [15, 16]],
                [[16, 17], [18, 19]],
                [[19, 20], [21, 22]],
            )
        with pytest.raises(TypeError):
            IQazimuthal(
                [[1, 2], [3, 4]],
                [[4, 5]],
                [[7, 8], [9, 10]],
                [[10, 11], [12, 13]],
                [[13, 14], [15, 16]],
                [[16, 17], [18, 19]],
                [[19, 20], [21, 22]],
            )
        with pytest.raises(TypeError):
            IQazimuthal(
                [[1, 2], [3, 4]],
                [[4, 5], [6, 7]],
                [[7, 8]],
                [[10, 11], [12, 13]],
                [[13, 14], [15, 16]],
                [[16, 17], [18, 19]],
                [[19, 20], [21, 22]],
            )
        with pytest.raises(TypeError):
            IQazimuthal(
                [[1, 2], [3, 4]],
                [[4, 5], [6, 7]],
                [[7, 8], [9, 10]],
                [[10, 11]],
                [[13, 14], [15, 16]],
                [[16, 17], [18, 19]],
                [[19, 20], [21, 22]],
            )
        with pytest.raises(TypeError):
            IQazimuthal(
                [[1, 2], [3, 4]],
                [[4, 5], [6, 7]],
                [[7, 8], [9, 10]],
                [[10, 11], [12, 13]],
                [[13, 14]],
                [[16, 17], [18, 19]],
                [[19, 20], [21, 22]],
            )
        with pytest.raises(TypeError):
            IQazimuthal(
                [[1, 2], [3, 4]],
                [[4, 5], [6, 7]],
                [[7, 8], [9, 10]],
                [[10, 11], [12, 13]],
                [[13, 14], [15, 16]],
                [[16, 17]],
                [[19, 20], [21, 22]],
            )
        with pytest.raises(TypeError):
            IQazimuthal(
                [[1, 2], [3, 4]],
                [[4, 5], [6, 7]],
                [[7, 8], [9, 10]],
                [[10, 11], [12, 13]],
                [[13, 14], [15, 16]],
                [[16, 17], [18, 19]],
                [[19, 20]],
            )

        # not enough arguments
        with pytest.raises(TypeError):
            IQazimuthal([[1, 2], [3, 4]], [[4, 5], [6, 7]], [[7, 8], [9, 10]])

        # qx and qy are linear
        IQazimuthal(
            [[1, 2, 3], [3, 4, 5]], [[4, 5, 6], [6, 7, 8]], [7, 8], [10, 11, 12]
        )
        # qx and qy are linear and not right dimension
        with pytest.raises(TypeError):
            IQazimuthal(
                [[1, 2, 3], [3, 4, 5]], [[4, 5, 6], [6, 7, 8]], [7, 8, 9], [10, 11]
            )

    def test_concatenate(self):
        iq1 = IQazimuthal(
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            delta_qx=[13, 14, 15],
            delta_qy=[10, 11, 12],
        )
        iq2 = iq1.concatenate(
            IQazimuthal(
                [4, 5, 6], [4, 5, 6], [7, 8, 9], [13, 14, 15], wavelength=[10, 11, 12]
            )
        )
        assert iq2.intensity == pytest.approx([1, 2, 3, 4, 5, 6])
        assert iq2.qy == pytest.approx([10, 11, 12, 13, 14, 15])
        assert iq2.delta_qx is None
        assert iq2.delta_qy is None
        assert iq2.wavelength is None


class TestTesting:
    def test_assert_all_close(self):
        iqmod = IQmod([1, 2, 3], [4, 5, 6], [7, 8, 9])
        iqmod2 = IQmod([1, 2, 3], [4, 5.1, 6], [7, 8, 9.19])
        testing.assert_allclose(iqmod, iqmod)
        testing.assert_allclose(iqmod, iqmod2, atol=0.2)
        with pytest.raises(AssertionError):
            testing.assert_allclose(iqmod, iqmod2, atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__])
