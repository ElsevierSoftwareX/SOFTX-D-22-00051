"""The test data for all of these examples come from
https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/issues/177

Much of the spreadsheet is split into smaller tests to aid in verifying the intermediate results
"""
import pytest
import numpy as np
import os
import matplotlib.pyplot as plt
from drtsans.auto_wedge import _binInQAndAzimuthal, _fitQAndAzimuthal
from drtsans.dataobjects import IQazimuthal, IQmod
from drtsans.determine_bins import determine_1d_linear_bins
from drtsans import getWedgeSelection
from drtsans.mono import biosans
from drtsans.plots import plot_IQazimuthal

# test internal function _toQmodAndAzimuthal as well
from drtsans.iq import (
    _toQmodAndAzimuthal,
    BinningMethod,
    bin_intensity_into_q2d,
    select_i_of_q_by_wedge,
    bin_all,
)
from matplotlib.colors import LogNorm  # noqa E402
from mantid.simpleapi import LoadNexusProcessed
from tempfile import NamedTemporaryFile


def _create_2d_data():
    """This creates the IQazimuthal data from ``Raw_Data_Anisotropic_v2_new`` the pages are

    * "Anisotropic Data - Qy vs Qx" contains the intensities with the Q values labeled in grey
    * "Anisotropic Data - EB Qy vs Qx" contains the uncertainties
    * "Anisotropic Data - Q" contains the Q-magnitude
    * "Anisotropic Data - Phi" contains the azimuthal angle
    """
    intensities = np.array(
        [
            [200, 190, 150, 190, 220, 210, 230, 250, 210, 190, 180],
            [180, 150, 220, 230, 230, 290, 190, 300, 180, 280, 310],
            [200, 200, 300, 210, 430, 380, 280, 290, 400, 380, 330],
            [400, 550, 420, 200, 220, 320, 320, 240, 700, 600, 350],
            [500, 600, 1100, 1500, 200, 180, 220, 1500, 1000, 700, 400],
            [600, 700, 1500, 2200, 3100, 0, 3000, 2300, 1300, 800, 500],
            [500, 600, 1200, 1500, 400, 250, 240, 1500, 1000, 700, 400],
            [400, 550, 230, 280, 380, 240, 200, 220, 700, 600, 350],
            [250, 320, 220, 220, 250, 340, 340, 290, 220, 220, 300],
            [200, 220, 180, 200, 300, 330, 230, 300, 180, 280, 320],
            [150, 190, 220, 180, 190, 280, 220, 290, 220, 220, 150],
        ]
    )
    errors = np.array(
        [
            [14.1, 13.8, 12.2, 13.8, 14.8, 14.5, 15.2, 15.8, 14.5, 13.8, 13.4],
            [13.4, 12.2, 14.8, 15.2, 15.2, 17.0, 13.8, 17.3, 13.4, 16.7, 17.6],
            [14.1, 14.1, 17.3, 14.5, 20.7, 19.5, 16.7, 17.0, 20.0, 19.5, 18.2],
            [20.0, 23.5, 20.5, 14.1, 14.8, 17.9, 17.9, 15.5, 26.5, 24.5, 18.7],
            [22.4, 24.5, 33.2, 38.7, 14.1, 13.4, 14.8, 38.7, 31.6, 26.5, 20.0],
            [24.5, 26.5, 38.7, 46.9, 55.7, 01.0, 54.8, 48.0, 36.1, 28.3, 22.4],
            [22.4, 24.5, 34.6, 38.7, 20.0, 15.8, 15.5, 38.7, 31.6, 26.5, 20.0],
            [20.0, 23.5, 15.2, 16.7, 19.5, 15.5, 14.1, 14.8, 26.5, 24.5, 18.7],
            [15.8, 17.9, 14.8, 14.8, 15.8, 18.4, 18.4, 17.0, 14.8, 14.8, 17.3],
            [14.1, 14.8, 13.4, 14.1, 17.3, 18.2, 15.2, 17.3, 13.4, 16.7, 17.9],
            [12.2, 13.8, 14.8, 13.4, 13.8, 16.7, 14.8, 17.0, 14.8, 14.8, 12.2],
        ]
    )

    # IQazimuthal's constructor is corrected in
    # https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/-/merge_requests/951
    # Thus the test input data shall be tranposed accordingly
    intensities = intensities.T
    errors = errors.T

    data2d = IQazimuthal(
        intensity=intensities,
        error=errors,
        qx=np.linspace(-5.0, 5.0, 11, dtype=float),
        qy=np.linspace(5.0, -5.0, 11, dtype=float),
    )
    assert data2d.intensity.shape == (11, 11)
    return data2d


def _create_2d_histogram_data():
    """This creates the parallel arrays of the binned data from ``Raw_Data_Anisotropic_v2_new`` the pages are

    * "Anisotropic Data - Q vs Phi" which contains the intensity of each Q column and the rows are the
      azimuthal angle values labeled with bin boundary. Both of these express bin centers.
    * "Anisotropic Data - EB Q vs Phi" which contains the uncertainties
    """
    # numbers taken from the spreadsheet
    intensity = np.array(
        [
            [3000, 2300, 1300, 800, 500, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 400, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 700, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 1000, np.nan, 350, np.nan, np.nan],
            [np.nan, 1500, np.nan, 600, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 330, np.nan],
            [np.nan, np.nan, np.nan, 700, 380, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 310, np.nan],
            [220, np.nan, 240, 400, np.nan, 280, 180],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 190, np.nan],
            [np.nan, np.nan, np.nan, 290, 180, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 210, np.nan],
            [np.nan, 320, np.nan, 300, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 280, np.nan, 250, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 190, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 230, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [180, 320, 380, 290, 210, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 220, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 230, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 430, np.nan, 190, np.nan, np.nan],
            [np.nan, 220, np.nan, 230, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 150, np.nan],
            [np.nan, np.nan, np.nan, 210, 220, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 190, np.nan],
            [200, np.nan, 200, 300, np.nan, 150, 200],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 180, np.nan],
            [np.nan, np.nan, np.nan, 420, 200, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 200, np.nan],
            [np.nan, 1500, np.nan, 550, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 1100, np.nan, 400, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 600, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 500, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [3100, 2200, 1500, 700, 600, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 500, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 600, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 1200, np.nan, 400, np.nan, np.nan],
            [np.nan, 1500, np.nan, 550, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 250, np.nan],
            [np.nan, np.nan, np.nan, 230, 320, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 200, np.nan],
            [400, np.nan, 280, 220, np.nan, 220, 150],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 190, np.nan],
            [np.nan, np.nan, np.nan, 220, 180, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 220, np.nan],
            [np.nan, 380, np.nan, 200, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 250, np.nan, 180, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 300, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 190, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [250, 240, 340, 330, 280, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 220, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 230, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 340, np.nan, 290, np.nan, np.nan],
            [np.nan, 200, np.nan, 300, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 220, np.nan],
            [np.nan, np.nan, np.nan, 290, 180, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 220, np.nan],
            [240, np.nan, 220, 220, np.nan, 280, 150],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 320, np.nan],
            [np.nan, np.nan, np.nan, 700, 220, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 300, np.nan],
            [np.nan, 1500, np.nan, 600, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 1000, np.nan, 350, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 700, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 400, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [3000, 2300, 1300, 800, 500, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 400, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 700, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 1000, np.nan, 350, np.nan, np.nan],
            [np.nan, 1500, np.nan, 600, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 330, np.nan],
            [np.nan, np.nan, np.nan, 700, 380, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 310, np.nan],
            [220, np.nan, 240, 400, np.nan, 280, 180],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 190, np.nan],
            [np.nan, np.nan, np.nan, 290, 180, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 210, np.nan],
            [np.nan, 320, np.nan, 300, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 280, np.nan, 250, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 190, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 230, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [180, 320, 380, 290, 210, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 220, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 230, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 430, np.nan, 190, np.nan, np.nan],
            [np.nan, 220, np.nan, 230, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 150, np.nan],
            [np.nan, np.nan, np.nan, 210, 220, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 190, np.nan],
            [200, np.nan, 200, 300, np.nan, 150, 200],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 180, np.nan],
            [np.nan, np.nan, np.nan, 420, 200, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 200, np.nan],
            [np.nan, 1500, np.nan, 550, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 1100, np.nan, 400, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 600, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 500, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [3100, 2200, 1500, 700, 600, np.nan, np.nan],
        ],
        dtype=float,
    )

    error = np.array(
        [
            [54.8, 48.0, 36.1, 28.3, 22.4, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 20.0, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 26.5, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 31.6, np.nan, 18.7, np.nan, np.nan],
            [np.nan, 38.7, np.nan, 24.5, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 18.2, np.nan],
            [np.nan, np.nan, np.nan, 26.5, 19.5, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 17.6, np.nan],
            [14.8, np.nan, 15.5, 20.0, np.nan, 16.7, 13.4],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 13.8, np.nan],
            [np.nan, np.nan, np.nan, 17.0, 13.4, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 14.5, np.nan],
            [np.nan, 17.9, np.nan, 17.3, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 16.7, np.nan, 15.8, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 13.8, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 15.2, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [13.4, 17.9, 19.5, 17.0, 14.5, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 14.8, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 15.2, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 20.7, np.nan, 13.8, np.nan, np.nan],
            [np.nan, 14.8, np.nan, 15.2, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 12.2, np.nan],
            [np.nan, np.nan, np.nan, 14.5, 14.8, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 13.8, np.nan],
            [14.1, np.nan, 14.1, 17.3, np.nan, 12.2, 14.1],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 13.4, np.nan],
            [np.nan, np.nan, np.nan, 20.5, 14.1, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 14.1, np.nan],
            [np.nan, 38.7, np.nan, 23.5, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 33.2, np.nan, 20.0, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 24.5, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 22.4, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [55.7, 46.9, 38.7, 26.5, 24.5, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 22.4, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 24.5, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 34.6, np.nan, 20.0, np.nan, np.nan],
            [np.nan, 38.7, np.nan, 23.5, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 15.8, np.nan],
            [np.nan, np.nan, np.nan, 15.2, 17.9, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 14.1, np.nan],
            [20.0, np.nan, 16.7, 14.8, np.nan, 14.8, 12.2],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 13.8, np.nan],
            [np.nan, np.nan, np.nan, 14.8, 13.4, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 14.8, np.nan],
            [np.nan, 19.5, np.nan, 14.1, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 15.8, np.nan, 13.4, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 17.3, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 13.8, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [15.8, 15.5, 18.4, 18.2, 16.7, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 14.8, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 15.2, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 18.4, np.nan, 17.0, np.nan, np.nan],
            [np.nan, 14.1, np.nan, 17.3, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 14.8, np.nan],
            [np.nan, np.nan, np.nan, 17.0, 13.4, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 14.8, np.nan],
            [15.5, np.nan, 14.8, 14.8, np.nan, 16.7, 12.2],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 17.9, np.nan],
            [np.nan, np.nan, np.nan, 26.5, 14.8, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 17.3, np.nan],
            [np.nan, 38.7, np.nan, 24.5, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 31.6, np.nan, 18.7, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 26.5, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 20.0, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [54.8, 48.0, 36.1, 28.3, 22.4, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 20.0, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 26.5, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 31.6, np.nan, 18.7, np.nan, np.nan],
            [np.nan, 38.7, np.nan, 24.5, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 18.2, np.nan],
            [np.nan, np.nan, np.nan, 26.5, 19.5, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 17.6, np.nan],
            [14.8, np.nan, 15.5, 20.0, np.nan, 16.7, 13.4],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 13.8, np.nan],
            [np.nan, np.nan, np.nan, 17.0, 13.4, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 14.5, np.nan],
            [np.nan, 17.9, np.nan, 17.3, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 16.7, np.nan, 15.8, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 13.8, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 15.2, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [13.4, 17.9, 19.5, 17.0, 14.5, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 14.8, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 15.2, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 20.7, np.nan, 13.8, np.nan, np.nan],
            [np.nan, 14.8, np.nan, 15.2, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 12.2, np.nan],
            [np.nan, np.nan, np.nan, 14.5, 14.8, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 13.8, np.nan],
            [14.1, np.nan, 14.1, 17.3, np.nan, 12.2, 14.1],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 13.4, np.nan],
            [np.nan, np.nan, np.nan, 20.5, 14.1, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 14.1, np.nan],
            [np.nan, 38.7, np.nan, 23.5, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 33.2, np.nan, 20.0, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 24.5, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 22.4, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [55.7, 46.9, 38.7, 26.5, 24.5, np.nan, np.nan],
        ],
        dtype=float,
    )
    assert intensity.shape == error.shape

    # put together binning parameters to reproduce spreadsheet
    q_min = 0.5
    q_max = 7.5
    q_delta = 1.0
    q_bins = np.arange(q_min, q_max + q_delta, q_delta, dtype=float)
    # verify the first and last values
    assert q_bins[0] == 0.5, "q_bins[0]"
    assert q_bins[-1] == 7.5, "q_bins[-1]"

    # create azimuthal angles as bin centers
    azimuthal_delta = 5.0
    azimuthal_max = 540.0
    azimuthal_bins = np.arange(
        start=0.0,
        stop=azimuthal_max + azimuthal_delta,
        step=azimuthal_delta,
        dtype=float,
    )
    # verify the first and last values
    assert azimuthal_bins[0] == 0.0, "azimuthal_bins[0]"
    assert azimuthal_bins[-1] == 540.0, "azimuthal_bins[-1]"
    azimuthal_bins += 2.5

    assert intensity.shape == (len(azimuthal_bins), len(q_bins) - 1)

    azimuthal_rings = []
    for i in range(intensity.shape[1]):
        azimuthal_rings.append(
            IQmod(intensity=intensity.T[i], error=error.T[i], mod_q=azimuthal_bins)
        )

    return q_bins, azimuthal_rings


def test_calc_qmod_and_azimuthal():
    """Test the conversion of data into 2d arrays of qmod and azimuthal angle. The results are
    checked against "Anisotropic Data -Q" and "Anisotropic Data - Phi"
    """
    data2d = _create_2d_data()

    # convert to q and azimuthal
    intensity, error, qmod, delta_qmod, azimuthal = _toQmodAndAzimuthal(data2d)
    assert qmod.shape == intensity.shape
    assert delta_qmod is None
    assert azimuthal.shape == intensity.shape

    from drtsans.plots.api import plot_IQazimuthal
    plot_IQazimuthal(data2d, 'input_wedge.png', backend='mpl')

    # numbers taken from the spreadsheet
    q_exp = np.array(
        [
            [7.07, 6.40, 5.83, 5.39, 5.10, 5.00, 5.10, 5.39, 5.83, 6.40, 7.07],
            [6.40, 5.66, 5.00, 4.47, 4.12, 4.00, 4.12, 4.47, 5.00, 5.66, 6.40],
            [5.83, 5.00, 4.24, 3.61, 3.16, 3.00, 3.16, 3.61, 4.24, 5.00, 5.83],
            [5.39, 4.47, 3.61, 2.83, 2.24, 2.00, 2.24, 2.83, 3.61, 4.47, 5.39],
            [5.10, 4.12, 3.16, 2.24, 1.41, 1.00, 1.41, 2.24, 3.16, 4.12, 5.10],
            [5.00, 4.00, 3.00, 2.00, 1.00, 0.00, 1.00, 2.00, 3.00, 4.00, 5.00],
            [5.10, 4.12, 3.16, 2.24, 1.41, 1.00, 1.41, 2.24, 3.16, 4.12, 5.10],
            [5.39, 4.47, 3.61, 2.83, 2.24, 2.00, 2.24, 2.83, 3.61, 4.47, 5.39],
            [5.83, 5.00, 4.24, 3.61, 3.16, 3.00, 3.16, 3.61, 4.24, 5.00, 5.83],
            [6.40, 5.66, 5.00, 4.47, 4.12, 4.00, 4.12, 4.47, 5.00, 5.66, 6.40],
            [7.07, 6.40, 5.83, 5.39, 5.10, 5.00, 5.10, 5.39, 5.83, 6.40, 7.07],
        ],
        dtype=float,
    )

    azimuthal_exp = np.array(
        [
            [135, 129, 121, 112, 101, 90, 79, 68, 59, 51, 45],
            [141, 135, 127, 117, 104, 90, 76, 63, 53, 45, 39],
            [149, 143, 135, 124, 108, 90, 72, 56, 45, 37, 31],
            [158, 153, 146, 135, 117, 90, 63, 45, 34, 27, 22],
            [169, 166, 162, 153, 135, 90, 45, 27, 18, 14, 11],
            [180, 180, 180, 180, 180, 0, 0, 0, 0, 0, 0],
            [191, 194, 198, 207, 225, 270, 315, 333, 342, 346, 349],
            [202, 207, 214, 225, 243, 270, 297, 315, 326, 333, 338],
            [211, 217, 225, 236, 252, 270, 288, 304, 315, 323, 329],
            [219, 225, 233, 243, 256, 270, 284, 297, 307, 315, 321],
            [225, 231, 239, 248, 259, 270, 281, 292, 301, 309, 315],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(qmod, q_exp.ravel(), atol=0.005)
    # IQazimuthal's constructor is corrected in
    # https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/-/merge_requests/951
    # Thus the expected test result shall be tranposed accordingly
    np.testing.assert_allclose(azimuthal, azimuthal_exp.T.ravel(), atol=0.5)


def test_bin_into_q_and_azimuthal():
    '''Test binning into Q and azimuthal matches the results from "Anisotropic Data - Q vs Phi"'''
    # get the test data
    data2d = _create_2d_data()
    q_exp, azimuthal_rings_exp = _create_2d_histogram_data()

    # parameters for azimuthal
    azimuthal_delta = 5.0

    # parameters for q
    q_min = 0.5
    q_max = 7.5
    q_delta = 1.0

    # get the histogrammed data
    q, azimuthal_rings = _binInQAndAzimuthal(
        data2d,
        q_min=q_min,
        q_max=q_max,
        q_delta=q_delta,
        azimuthal_delta=azimuthal_delta,
    )

    # verify the q-binning
    assert q.min() == q_min == q_exp.min()
    assert q.max() == q_max == q_exp.max()  # using bin boundaries

    for spectrum, spectrum_exp in zip(azimuthal_rings, azimuthal_rings_exp):
        assert spectrum.intensity.shape == spectrum_exp.intensity.shape
        np.testing.assert_allclose(
            spectrum.mod_q, spectrum_exp.mod_q, atol=0.05, equal_nan=True
        )
        assert spectrum.delta_mod_q is None

        np.testing.assert_allclose(
            spectrum.intensity, spectrum_exp.intensity, atol=0.05, equal_nan=True
        )
        np.testing.assert_allclose(
            spectrum.error, spectrum_exp.error, atol=0.05, equal_nan=True
        )


def test_fitting():
    """Test that the fitting generates reasonable results for fitting the peaks"""
    q, azimuthal_rings = _create_2d_histogram_data()
    # this calling forces there to be two found peaks
    ring_fit_tuple = _fitQAndAzimuthal(
        azimuthal_rings,
        q,
        signal_to_noise_min=2.0,
        azimuthal_start=110.0,
        maxchisq=1000.0,
        peak_search_window_size_factor=0.6,
    )
    center_list = ring_fit_tuple[0]
    fwhm_list = ring_fit_tuple[1]

    assert center_list[0] == pytest.approx(180.0, abs=3.0)
    assert center_list[1] == pytest.approx(360.0, abs=4.0)
    assert fwhm_list[0] == pytest.approx(fwhm_list[1], abs=2.0)


def test_integration():
    """Test the full workflow of the algorithm"""
    data2d = _create_2d_data()
    # parameters for azimuthal
    azimuthal_delta = 5.0

    # parameters for q
    q_min = 0.5
    q_max = 7.5
    q_delta = 1.0

    # run the function
    wedges = getWedgeSelection(
        data2d,
        q_min=q_min,
        q_max=q_max,
        q_delta=q_delta,
        azimuthal_delta=azimuthal_delta,
    )

    # from drtsans.plots.api import plot_IQazimuthal
    # plot_IQazimuthal(data2d, 'wedge_int_test_debug.png', backend='mpl')
    # assert False, f'Wedges: {wedges}'

    # tests
    assert len(wedges) == 2
    for wedge in wedges:
        for min_val, max_val in wedge:
            print(min_val, "<", max_val)
            assert -90.0 < min_val < 270.0
            assert -90.0 < max_val < 270.0
    peak_wedge, back_wedge = wedges

    # verify peaks
    # first peak
    assert 0.5 * (peak_wedge[0][0] + peak_wedge[0][1]) == pytest.approx(
        180.0, abs=3.0
    ), "First peak center is at 180."
    assert peak_wedge[0][0] == pytest.approx(171.0, abs=0.5)
    assert peak_wedge[0][1] == pytest.approx(195.0, abs=0.5)
    # second peak
    assert 0.5 * (peak_wedge[1][0] + peak_wedge[1][1]) == pytest.approx(
        3.0, abs=1.0
    ), "Second peak center is at 0."
    assert peak_wedge[1][0] == pytest.approx(-9.0, abs=0.5)
    assert peak_wedge[1][1] == pytest.approx(16.0, abs=0.5)

    # verify background
    # first background - the extra 360 is to get around the circle
    assert 0.5 * (back_wedge[0][0] + back_wedge[0][1] + 360) == pytest.approx(
        272.0, abs=1.2
    ), "First background center is at 270."
    assert back_wedge[0][0] == pytest.approx(255.0, abs=0.5)
    assert back_wedge[0][1] == pytest.approx(-69.0, abs=0.5)
    # second background
    assert 0.5 * (back_wedge[1][0] + back_wedge[1][1]) == pytest.approx(
        90.0, abs=5.0
    ), "Second background center is at 90."
    assert back_wedge[1][0] == pytest.approx(76.0, abs=0.5)
    assert back_wedge[1][1] == pytest.approx(111.0, abs=0.5)


def test_real_data_biosans(reference_dir):
    MSamp_fn = os.path.join(reference_dir.new.biosans, "CG3_127_5532_mBSub.h5")
    MBuff_fn = os.path.join(reference_dir.new.biosans, "CG3_127_5562_mBSub.h5")

    ws_ms = LoadNexusProcessed(
        Filename=MSamp_fn, OutputWorkspace="sample", LoadHistory=False
    )
    ws_mb = LoadNexusProcessed(
        Filename=MBuff_fn, OutputWorkspace="Main_buffer", LoadHistory=False
    )
    ws_ms -= ws_mb  # subtract the buffer
    ws_mb.delete()

    # convert to I(qx,qy)
    q1d_data = biosans.convert_to_q(ws_ms, mode="scalar")
    q2d_data = biosans.convert_to_q(ws_ms, mode="azimuthal")
    ws_ms.delete()

    # calculate the wedge angles to use
    wedge_angles = getWedgeSelection(
        q2d_data,
        0.00,
        0.001,
        0.02,
        0.5,
        peak_width=0.25,
        background_width=1.5,
        signal_to_noise_min=1.2,
    )
    assert len(wedge_angles) == 2, "Expect 2 separate wedges"

    # use these to integrate the wedges
    for wedges in wedge_angles:
        for azi_min, azi_max in wedges:
            print("integrating from {}deg to {} deg".format(azi_min, azi_max))
            iq_wedge = select_i_of_q_by_wedge(q2d_data, azi_min, azi_max)
            assert iq_wedge

    # test bin_all
    nbins = 100.0
    iq2d_rebinned, iq1d_rebinned = bin_all(
        q2d_data,
        q1d_data,
        nxbins=nbins,
        nybins=nbins,
        n1dbins=nbins,
        bin1d_type="wedge",
        wedges=wedge_angles,
        symmetric_wedges=False,
        error_weighted=False,
    )
    assert len(iq1d_rebinned) == 2, "Expect exactly 2 output 1d spectra"

    # rebin the data onto a regular grid for plotting
    linear_x_bins = determine_1d_linear_bins(
        q2d_data.qx.min(), q2d_data.qx.max(), nbins
    )
    linear_y_bins = determine_1d_linear_bins(
        q2d_data.qy.min(), q2d_data.qy.max(), nbins
    )
    q2d_data = bin_intensity_into_q2d(
        q2d_data, linear_x_bins, linear_y_bins, BinningMethod.NOWEIGHT
    )

    # save an image
    filename = NamedTemporaryFile(
        delete=False, prefix="CG3_127_5532_Iqxqy", suffix=".png"
    ).name
    plot_IQazimuthal(
        q2d_data,
        filename,
        backend="mpl",
        wedges=wedge_angles,
        symmetric_wedges=False,
        imshow_kwargs={"norm": LogNorm()},
    )
    print("saved image to", filename)
    plt.close()
    # verify the plot was created and remove the file
    assert os.path.exists(filename), '"{}" does not exist'.format(filename)
    os.remove(filename)


def test_real_data_biosans_manual(reference_dir):
    """Test asymmetric manual wedge binning on BIOSANS data"""
    MSamp_fn = os.path.join(reference_dir.new.biosans, "CG3_127_5532_mBSub.h5")
    MBuff_fn = os.path.join(reference_dir.new.biosans, "CG3_127_5562_mBSub.h5")

    ws_ms = LoadNexusProcessed(
        Filename=MSamp_fn, OutputWorkspace="sample", LoadHistory=False
    )
    ws_mb = LoadNexusProcessed(
        Filename=MBuff_fn, OutputWorkspace="Main_buffer", LoadHistory=False
    )
    ws_ms -= ws_mb  # subtract the buffer
    ws_mb.delete()

    # convert to I(qx,qy)
    q1d_data = biosans.convert_to_q(ws_ms, mode="scalar")
    q2d_data = biosans.convert_to_q(ws_ms, mode="azimuthal")
    ws_ms.delete()

    # calculate the wedge angles to use
    wedge_angles = [(60.0, 120)]

    # test bin_all
    nbins = 100.0
    iq2d_rebinned, iq1d_rebinned = bin_all(
        q2d_data,
        q1d_data,
        nxbins=nbins,
        nybins=nbins,
        n1dbins=nbins,
        bin1d_type="wedge",
        wedges=wedge_angles,
        symmetric_wedges=False,
        error_weighted=False,
    )
    assert (
        len(iq1d_rebinned) == 1
    ), "Expect exactly 1 output 1d spectra for manual wedge"


if __name__ == "__main__":
    pytest.main([__file__])
