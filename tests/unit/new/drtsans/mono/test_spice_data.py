# Unit test for drtsans.mono.spice_data
import pytest
import os
from drtsans.mono.spice_data import SpiceRun


def test_spice_data_constructor():
    """Test constructor and properties access of SpiceRun"""
    # regular constructor
    cg2_run = SpiceRun(
        beam_line="CG2", ipts_number=828, exp_number=280, scan_number=5, pt_number=1
    )
    assert cg2_run.beam_line == "CG2"
    assert cg2_run.ipts_number == 828
    assert cg2_run.exp_number == 280
    assert cg2_run.scan_number == 5
    assert cg2_run.pt_number == 1

    # check unique run number generator
    run_number = cg2_run.unique_run_number
    assert run_number == 28000050001


def test_locate_file(reference_dir):
    """Test method to locate file"""
    cg2_run = SpiceRun(
        beam_line="CG2", ipts_number=828, exp_number=280, scan_number=5, pt_number=1
    )

    # Test file location on server
    if os.path.exists("/HFIR/CG2/IPTS-828"):
        # Access with server: can raise exception
        spice_path = cg2_run.locate_spice_file(raise_if_not_exist=True)
    else:
        spice_path = cg2_run.locate_spice_file(raise_if_not_exist=False)
    assert (
        spice_path == "/HFIR/CG2/IPTS-828/exp280/Datafiles/CG2_exp280_scan0005_0001.xml"
    )

    # Test for specified data file directory
    cg2_test_run = SpiceRun("CG2", 0, 245, 10, 8)
    spice_path = cg2_test_run.locate_spice_file(
        data_dir=reference_dir.new.gpsans, raise_if_not_exist=True
    )
    assert os.path.basename(spice_path) == "CG2_exp245_scan0010_0008.xml"


if __name__ == "__main__":
    pytest.main(__file__)
