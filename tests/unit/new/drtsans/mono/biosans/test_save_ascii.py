import re

# import tempfile

import pytest
from pytest import approx

from mantid.simpleapi import LoadHFIRSANS

# from drtsans.save_ascii import save_ascii_1D, save_xml_1D
from drtsans.settings import unique_workspace_name


def numbers_in_line(line, numbers):
    xyz = [float(s) for s in re.findall(r"\d+\.\d+", line)]
    return all([x == approx(n, rel=1.0e-03) or x < 0.02 for x, n in zip(xyz, numbers)])


def test_save_ascii(biosans_sensitivity_dataset):

    ws = LoadHFIRSANS(
        Filename=biosans_sensitivity_dataset["flood"],
        OutputWorkspace=unique_workspace_name(),
    )
    assert ws is not None

    # TODO - will review and rewrite with I(Q) binning rewrite

    # mt = MomentumTransfer(ws)
    # _, ws_iqxqy = mt.bin_into_q2d()
    # assert ws_iqxqy.extractY().shape == (256, 192)
    # assert ws_iqxqy.extractX().shape == (256, 193)
    #
    # _, ws_iq = mt.bin_into_q1d()
    # assert ws_iq.extractY().shape == (1, 100)
    # assert ws_iq.extractX().shape == (1, 101)
    #
    # with tempfile.NamedTemporaryFile('r+') as tmp:
    #     save_ascii_1D(ws_iq, 'Test BioSANS', tmp.name)
    #     output_lines = tmp.readlines()
    #     numbers = (0.113011, 93965.916955, 2435.952921, 0.021613)
    #     assert numbers_in_line(output_lines[101], numbers) is True
    #
    # with tempfile.NamedTemporaryFile('r+') as tmp:
    #     save_xml_1D(ws_iq, 'Test BioSANS', tmp.name)
    #     output_lines = tmp.readlines()
    #     numbers = (0.113011, 106.286, 2.75533, 0.0216135)
    #     assert numbers_in_line(output_lines[110], numbers) is True

    # with tempfile.NamedTemporaryFile('r+') as tmp:
    #     save_ascii_2D(ws_iqxqy, ws_dqx, ws_dqy, 'Test BioSANS 2D', tmp.name)
    #     output_lines = tmp.readlines()
    #     numbers = (0.077098, 0.081494, 73.000000, 8.544004, 0.015055,
    #                0.001741)
    #     assert numbers_in_line(output_lines[48388], numbers) is True


if __name__ == "__main__":
    pytest.main([__file__])
