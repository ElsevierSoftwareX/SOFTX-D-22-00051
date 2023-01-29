# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/215_transmission_test/drtsans/transmission.py
from drtsans.transmission import calculate_transmission
import numpy as np
import pytest

# https://docs.mantidproject.org/nightly/algorithms/SetUncertainties-v1.html
from mantid.simpleapi import SetUncertainties

# uncertainties is plain old sqrt(counts)
# Nx=15, Ny=22
Isam = np.asarray(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 3, 2, 3, 2, 3, 5, 8, 2, 3, 2, 2, 1, 0, 0, 1],
        [0, 0, 0, 1, 2, 7, 10, 12, 30, 51, 47, 42, 51, 40, 26, 25, 17, 8, 2, 0, 0, 0],
        [
            0,
            1,
            1,
            3,
            4,
            32,
            65,
            130,
            206,
            254,
            305,
            325,
            313,
            292,
            257,
            245,
            204,
            144,
            81,
            36,
            18,
            12,
        ],
        [
            3,
            7,
            8,
            27,
            47,
            90,
            123,
            211,
            271,
            263,
            303,
            263,
            252,
            192,
            184,
            156,
            114,
            73,
            33,
            11,
            3,
            1,
        ],
        [
            8,
            11,
            29,
            50,
            97,
            187,
            365,
            517,
            763,
            811,
            784,
            781,
            744,
            644,
            575,
            528,
            416,
            271,
            190,
            94,
            23,
            10,
        ],
        [
            2,
            5,
            8,
            23,
            46,
            104,
            185,
            271,
            320,
            514,
            465,
            504,
            411,
            454,
            400,
            313,
            303,
            211,
            153,
            82,
            34,
            13,
        ],
        [
            0,
            3,
            5,
            18,
            71,
            217,
            324,
            578,
            776,
            946,
            1087,
            1144,
            1072,
            1023,
            1014,
            879,
            712,
            599,
            366,
            210,
            103,
            44,
        ],
        [
            13,
            19,
            35,
            58,
            127,
            229,
            315,
            421,
            513,
            592,
            559,
            515,
            423,
            410,
            328,
            294,
            215,
            121,
            76,
            37,
            6,
            4,
        ],
        [
            3,
            19,
            27,
            63,
            143,
            258,
            408,
            609,
            806,
            857,
            905,
            882,
            793,
            725,
            586,
            522,
            406,
            287,
            160,
            62,
            27,
            12,
        ],
        [
            0,
            3,
            5,
            18,
            35,
            68,
            123,
            199,
            275,
            305,
            338,
            348,
            304,
            263,
            236,
            173,
            158,
            90,
            45,
            20,
            16,
            6,
        ],
        [
            2,
            0,
            3,
            6,
            24,
            74,
            119,
            181,
            266,
            305,
            307,
            317,
            326,
            254,
            224,
            202,
            115,
            112,
            46,
            15,
            10,
            3,
        ],
        [1, 1, 1, 2, 6, 3, 11, 23, 39, 64, 53, 59, 44, 49, 31, 23, 10, 5, 2, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 5, 8, 9, 14, 11, 14, 14, 6, 2, 3, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.float64,
)
pixel_size = 0.005  # meter


@pytest.mark.parametrize(
    "generic_workspace",
    [
        {
            "name": "Isam",
            "dx": pixel_size,
            "dy": pixel_size,  # data requires a square pixel
            "yc": pixel_size / 2.0,  # shift because the "detector" y-direction is even
            "axis_values": [5.925, 6.075],
            "intensities": Isam,
        }
    ],
    indirect=True,
)
def test_transmission(generic_workspace):
    """Test the calculation of the detector transmission in the master document section 7.2
    dev - Pete Peterson <petersonpf@ornl.gov>
    SME - Lilin He <hel3@ornl.gov>
    """
    Isam = generic_workspace  # convenient name
    assert Isam.extractY().sum() == 50212  # checksum

    # generate the reference data - uncertainties are set separately
    Iref = 1.8 * Isam
    Iref = SetUncertainties(InputWorkspace=Iref, OutputWorkspace=Iref, SetError="sqrt")
    assert Iref.extractY().sum() == 1.8 * 50212  # checksum
    assert 1.8 * Isam.extractE().sum() > Iref.extractE().sum()  # shouldn't match

    # run the algorithm
    result = calculate_transmission(Isam, Iref, 2.5 * pixel_size, "m")

    # it should be a single value workspace
    assert result.getNumberHistograms(), 1
    assert result.extractY().shape == (1, 1)
    # taken from the spreadsheet calculation
    assert result.extractY()[0][0] == pytest.approx(
        0.555555556
    )  # expected transmission value
    assert result.extractE()[0][0] == pytest.approx(
        0.005683898
    )  # expected transmission uncertainty


if __name__ == "__main__":
    pytest.main([__file__])
