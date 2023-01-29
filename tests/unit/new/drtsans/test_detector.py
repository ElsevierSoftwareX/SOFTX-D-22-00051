import pytest
from mantid.simpleapi import LoadEmptyInstrument, MaskBTP
from drtsans.detector import Component
from drtsans.settings import unique_workspace_name

import numpy as np


def test_detector_biosans():

    ws = LoadEmptyInstrument(
        InstrumentName="biosans", OutputWorkspace=unique_workspace_name()
    )

    d = Component(ws, "detector1")
    assert 192 == d.dim_x
    assert 256 == d.dim_y
    assert 192 * 256 == d.dims
    assert 2 == d.first_index

    d = Component(ws, "wing_detector")
    assert 160 == d.dim_x
    assert 256 == d.dim_y
    assert 160 * 256 == d.dims
    assert 192 * 256 + 2 == d.first_index


def test_detector_gpsans():

    ws = LoadEmptyInstrument(
        InstrumentName="cg2", OutputWorkspace=unique_workspace_name()
    )

    d = Component(ws, "detector1")
    assert 192 == d.dim_x
    assert 256 == d.dim_y
    assert 192 * 256 == d.dims
    assert 2 == d.first_index


def test_detector_eqsans():

    ws = LoadEmptyInstrument(
        InstrumentName="eqsans", OutputWorkspace=unique_workspace_name()
    )

    d = Component(ws, "detector1")
    assert 192 == d.dim_x
    assert 256 == d.dim_y
    assert 192 * 256 == d.dims
    assert 1 == d.first_index


def test_detector_masked_gpsans():
    # flake8: noqa E712
    ws = LoadEmptyInstrument(
        InstrumentName="cg2", OutputWorkspace=unique_workspace_name()
    )
    d = Component(ws, "detector1")
    masked_array = d.masked_ws_indices()
    # No Masks applied yet, all should be false
    assert (masked_array == False).all()

    # Let's mask the detector ends
    MaskBTP(ws, Tube="1-4", Pixel="0-19,236-255")

    masked_array = d.masked_ws_indices()
    assert (masked_array == False).all() == False

    n_pixels_masked = np.count_nonzero(masked_array == True)
    # 40 pixels * 192 tubes
    assert n_pixels_masked == 40 * 192
