import pytest
import os, numpy as np  # noqa: E401
from numpy.testing import assert_allclose

r"""
Links to mantid algorithms
CreateWorkspace <https://docs.mantidproject.org/nightly/algorithms/CreateWorkspace-v1.html>
"""
from mantid.simpleapi import CreateWorkspace


r"""
Hyperlinks to drtsans functions
normalize_by_thickness <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans
/thickness_normalization.py>
"""  # noqa: E501
from drtsans.settings import unique_workspace_dundername
from drtsans.thickness_normalization import normalize_by_thickness

here = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture(scope="module")
def testdata():
    # create test data from csv file provided by LiLin He
    path = os.path.join(here, "Thickness_normalization_He.csv")
    data = np.genfromtxt(path, delimiter=",", skip_header=3)
    # intensity and errorbars, expected normalized intensity and errorbars
    I, E, normedI, normedE = data.T
    return I, E, normedI, normedE


@pytest.fixture(scope="module")
def workspaces(testdata):
    """create workspace using the data loaded from the csv file provided by LiLin He"""
    # load data from csv
    I, E, normedI, normedE = testdata
    nbins = I.size
    inputws = CreateWorkspace(
        NSpec=1,
        DataX=np.arange(nbins + 1),
        DataY=I,
        DataE=E,
        OutputWorkspace=unique_workspace_dundername(),
    )
    expected_output_ws = CreateWorkspace(
        NSpec=1,
        DataX=np.arange(nbins + 1),
        DataY=normedI,
        DataE=normedE,
        OutputWorkspace=unique_workspace_dundername(),
    )
    yield inputws, expected_output_ws
    # Delete upon closure of the fixture
    [workspace.delete() for workspace in (inputws, expected_output_ws)]


def test_thickness_normalization(workspaces):
    r"""
    Test thickness normalization using data from a cvs file.
    The normalized result is compared to expected result provided in a csv file.

    Function tested: drtsans.tof.eqsans.api.normalize_by_thickness
    Underlying Mantid algorithms:
        Divide https://docs.mantidproject.org/nightly/algorithms/Divide-v1.html

    dev - Jiao Lin <linjiao@ornl.gov>
    SME - LiLin He <hel3@ornl.gov>
    """
    inputws, expected_output_ws = workspaces
    thickness = 0.1
    normed = normalize_by_thickness(inputws, thickness)
    assert_allclose(normed.readY(0), expected_output_ws.readY(0), rtol=5e-3)
    assert_allclose(normed.readE(0), expected_output_ws.readE(0), rtol=1e-7)
    return


if __name__ == "__main__":
    pytest.main([__file__])
