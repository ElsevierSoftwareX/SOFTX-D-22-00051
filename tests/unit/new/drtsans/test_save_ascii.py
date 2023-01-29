from drtsans.dataobjects import IQazimuthal
from drtsans.save_ascii import load_ascii_binned_2D, save_ascii_binned_2D
import numpy as np
from pathlib import Path
import pytest
from tempfile import NamedTemporaryFile


def createIQAzimuthal(with_resolution=True):
    """Create :pyobj:`~drtsans.dataobjects.IQazimuthal`. The data has shape=(21,21)

    Parameters
    ----------
    with_resolution: bool
        Whether or not to add the resolution fields
    """
    resolutionArgs = {}
    if with_resolution:
        resolutionArgs["delta_qx"] = np.full((21, 21), 1.0, dtype=float)
        resolutionArgs["delta_qy"] = np.full((21, 21), 1.0, dtype=float)

    return IQazimuthal(
        intensity=np.arange(21 * 21, dtype=float).reshape((21, 21)),
        error=np.full((21, 21), 1.0, dtype=float),
        qx=np.arange(-10, 11, 1, dtype=float),
        qy=np.arange(-10, 11, 1, dtype=float),
        **resolutionArgs
    )


def assert_IQazi_allclose(data2d_exp, data2d_obs, err_msg=""):
    if err_msg:
        err_msg += " "
    for key in ["intensity", "error", "qx", "qy", "delta_qx", "delta_qy", "wavelength"]:
        exp = getattr(data2d_exp, key)
        obs = getattr(data2d_obs, key)
        if exp is None or obs is None:
            assert exp == obs, "{} are not both None".format(key)
        else:
            np.testing.assert_allclose(
                exp, obs, err_msg="{}{} doesn't match".format(err_msg, key)
            )


@pytest.mark.parametrize("with_resolution", [True, False])
def test_ascii_binned_2D_roundtrip(with_resolution):
    # create test data with -10 <= Qx/Qy <= 10
    data2d = createIQAzimuthal(with_resolution)

    # file to write to
    filename = Path(NamedTemporaryFile(suffix=".dat").name)

    # write out the data
    print("writing data to", filename)
    save_ascii_binned_2D(filename, "test data", data2d)
    assert filename.exists()

    # read the data back in
    print("reading data from", filename)
    data2d_reread = load_ascii_binned_2D(filename)
    assert data2d_reread

    # flatten the data objects for comparison
    data2d = data2d.ravel()
    data2d_reread = data2d_reread.ravel()

    # validate the data is unchanged
    assert_IQazi_allclose(data2d, data2d_reread)

    # cleanup - not using fixture so failures can be inspected
    filename.unlink()


if __name__ == "__main__":
    pytest.main([__file__])
