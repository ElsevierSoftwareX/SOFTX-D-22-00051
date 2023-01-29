import numpy as np
from drtsans.dataobjects import IQmod, IQazimuthal
from drtsans.iq import bin_all, determine_1d_log_bins
import pytest


def generate_IQ():
    alpha = np.arange(0.001, 360, 22.5)
    intensity = np.arange(32) + 1.0
    error = np.sqrt(intensity)
    qx = np.cos(np.radians(alpha))
    qx = np.concatenate((qx, qx * 4.0))
    qy = np.sin(np.radians(alpha))
    qy = np.concatenate((qy, qy * 4.0))
    q = np.ones(16)
    q = np.concatenate((q, q * 4.0))
    dqx = np.ones(32) * 0.1
    dqy = np.ones(32) * 0.1
    dq = np.ones(32) * 0.1
    iq1d = IQmod(intensity, error, q, dq)
    iq2d = IQazimuthal(intensity, error, qx, qy, dqx, dqy)
    return iq1d, iq2d


def test_bin_2d():
    iq1d, iq2d = generate_IQ()

    binned2d, binned1d = bin_all(
        iq2d,
        iq1d,
        nxbins=4,
        nybins=4,
        n1dbins=4,
        bin1d_type="scalar",
        log_scale=False,
        qmin=None,
        qmax=None,
        annular_angle_bin=1.0,
        wedges=None,
        error_weighted=False,
    )

    # From standard bin 2d, 2D Qx changes from row to row
    # From standard bin 2d, 2D Qy changes from column to column
    assert binned2d.qx[::, 0] == pytest.approx([-3, -1, 1, 3])
    assert binned2d.qy[0] == pytest.approx([-3, -1, 1, 3])
    assert binned2d.intensity[2, 2] == pytest.approx(2.5)
    assert binned2d.intensity[1, 2] == pytest.approx(6.5)
    assert binned2d.intensity[1, 1] == pytest.approx(10.5)
    assert binned2d.intensity[2, 1] == pytest.approx(14.5)


def test_bin_modq():
    iq1d, iq2d = generate_IQ()

    # test linear scale, no weights
    binned2d, binned1d = bin_all(
        iq2d,
        iq1d,
        nxbins=4,
        nybins=4,
        n1dbins=4,
        bin1d_type="scalar",
        log_scale=False,
        qmin=None,
        qmax=None,
        annular_angle_bin=1.0,
        wedges=None,
        error_weighted=False,
    )
    binned1d = binned1d[0]
    stepq = 3.0 / 4
    expected_q = np.arange(0.5, 4) * stepq + 1
    expected_intensity = np.array([(1.0 + 16) / 2, np.nan, np.nan, (32 + 17.0) / 2])
    assert binned1d.mod_q == pytest.approx(expected_q)
    assert binned1d.intensity == pytest.approx(expected_intensity, nan_ok=True)

    # test linear scale with weights
    binned2d, binned1d = bin_all(
        iq2d,
        iq1d,
        nxbins=4,
        nybins=4,
        n1dbins=4,
        bin1d_type="scalar",
        log_scale=False,
        qmin=None,
        qmax=None,
        annular_angle_bin=1.0,
        wedges=None,
        error_weighted=True,
    )
    binned1d = binned1d[0]
    assert binned1d.mod_q == pytest.approx(expected_q)
    assert binned1d.intensity[0] < expected_intensity[0]
    assert np.isnan(binned1d.intensity[1])
    assert binned1d.intensity[3] < expected_intensity[3]

    # test external qmin/qmax
    binned2d, binned1d = bin_all(
        iq2d,
        iq1d,
        nxbins=4,
        nybins=4,
        n1dbins=4,
        bin1d_type="scalar",
        log_scale=False,
        qmin=0,
        qmax=8,
        annular_angle_bin=1.0,
        wedges=None,
        error_weighted=False,
    )
    binned1d = binned1d[0]
    assert binned1d.mod_q == pytest.approx([1, 3, 5, 7])

    # test log scale, decade on center = False, qmin and qmax are not specified
    binned2d, binned1d = bin_all(
        iq2d,
        iq1d,
        nxbins=4,
        nybins=4,
        n1dbins=4,
        bin1d_type="scalar",
        log_scale=True,
        decade_on_center=False,
        qmin=None,
        qmax=None,
        annular_angle_bin=1.0,
        wedges=None,
        error_weighted=False,
    )
    binned1d = binned1d[0]
    expected_q = determine_1d_log_bins(
        1.0, 4.0, decade_on_center=False, n_bins=4
    ).centers
    assert binned1d.mod_q == pytest.approx(expected_q)

    # test log scale: decade on center, qmin and qmax are not given
    binned2d, binned1d = bin_all(
        iq2d,
        iq1d,
        nxbins=4,
        nybins=4,
        n1dbins_per_decade=4,
        bin1d_type="scalar",
        log_scale=True,
        decade_on_center=True,
        qmin=None,
        qmax=None,
        annular_angle_bin=1.0,
        wedges=None,
        error_weighted=False,
    )
    binned1d = binned1d[0]
    expected_q = determine_1d_log_bins(
        1.0, 4.0, decade_on_center=True, n_bins_per_decade=4
    ).centers
    assert binned1d.mod_q == pytest.approx(expected_q)
    expected_intensity = np.array([(1.0 + 16) / 2, np.nan, (32 + 17.0) / 2])
    assert binned1d.intensity == pytest.approx(expected_intensity, nan_ok=True)

    # test log scale: decade on center is False, total bins is given, q_min and q_max are given
    binned2d, binned1d = bin_all(
        iq2d,
        iq1d,
        nxbins=4,
        nybins=4,
        n1dbins=4,
        bin1d_type="scalar",
        log_scale=True,
        decade_on_center=False,
        qmin=2,
        qmax=10,
        annular_angle_bin=1.0,
        wedges=None,
        error_weighted=False,
    )
    binned1d = binned1d[0]
    expected_q = determine_1d_log_bins(
        2.0, 10.0, decade_on_center=False, n_bins=4
    ).centers
    assert binned1d.mod_q == pytest.approx(expected_q)
    expected_intensity = np.array([np.nan, (32 + 17.0) / 2, np.nan, np.nan])
    assert binned1d.intensity == pytest.approx(expected_intensity, nan_ok=True)


def test_annular():
    iq1d, iq2d = generate_IQ()
    binned2d, binned1d = bin_all(
        iq2d,
        iq1d,
        nxbins=4,
        nybins=4,
        n1dbins=4,
        bin1d_type="annular",
        log_scale=False,
        qmin=0,
        qmax=8,
        annular_angle_bin=90.0,
        wedges=None,
        error_weighted=False,
    )
    binned1d = binned1d[0]
    assert binned1d.intensity == pytest.approx([10.5, 14.5, 18.5, 22.5])
    assert binned1d.mod_q == pytest.approx(np.arange(45, 360, 90))


def test_wedges():
    iq1d, iq2d = generate_IQ()
    binned2d, binned1d = bin_all(
        iq2d,
        iq1d,
        nxbins=4,
        nybins=4,
        n1dbins=4,
        bin1d_type="wedge",
        log_scale=False,
        qmin=None,
        qmax=None,
        annular_angle_bin=1.0,
        wedges=[(-30, 30), (60, 120)],
        error_weighted=False,
    )
    # in wedge 0, at low q 1, 2, 16, 8, 9, 10
    # in wedge 1, at low q 4, 5, 6, 12, 13, 14
    # at high q add 16
    expected_intensity_wedge0 = np.array([7.66666, np.nan, np.nan, 23.66666])
    expected_intensity_wedge1 = np.array([9, np.nan, np.nan, 25])
    assert len(binned1d) == 2
    assert binned1d[0].intensity == pytest.approx(
        expected_intensity_wedge0, nan_ok=True, abs=1e-5
    )
    assert binned1d[1].intensity == pytest.approx(
        expected_intensity_wedge1, nan_ok=True, abs=1e-5
    )


if __name__ == "__main__":
    pytest.main([__file__])
