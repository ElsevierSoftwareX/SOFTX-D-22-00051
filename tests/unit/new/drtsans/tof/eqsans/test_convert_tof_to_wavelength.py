import numpy as np
import pytest

# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/samplelogs.py
from drtsans.samplelogs import SampleLogs

# https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/tof/eqsans/correct_frame.py
from drtsans.tof.eqsans.correct_frame import convert_to_wavelength


def add_frame_skipping_log(ws):
    samplelog = SampleLogs(ws)
    samplelog.insert("is_frame_skipping", False)


@pytest.mark.parametrize(
    "generic_workspace",
    [
        {
            "dx": 0.005,
            "dy": 0.004,
            "zc": 5.0,
            "l1": 14.0,
            "axis_units": "tof",
            "axis_values": [15432.0],
        }
    ],
    indirect=True,
)
def test_william(generic_workspace):
    """Test the conversion of time-of-flight to wavelength
    in master document section 3.3
    dev - Pete Peterson <petersonpf@ornl.gov>
    SME - William Heller <hellerwt@ornl.gov>"""
    # set up input workspace in the first frame
    ws = generic_workspace  # friendly name
    add_frame_skipping_log(ws)

    # run the drt-sans version
    ws = convert_to_wavelength(input_workspace=generic_workspace)

    # make sure the unit is wavelength
    assert ws.getAxis(0).getUnit().caption() == "Wavelength"

    # get information for detector pixel positions
    specInfo = ws.spectrumInfo()
    source_sample = specInfo.l1()  # in meters

    # verify the individual wavelength values
    for i in range(4):
        # distance to detector pixel in meters
        sample_detector = specInfo.l2(i)
        # equation supplied by SME applied to time-of-flight
        lambda_exp = (
            3.9560346e-3 * np.array([15432.0]) / (source_sample + sample_detector)
        )

        # verify the results
        assert ws.dataX(i)[0] == pytest.approx(lambda_exp[0])
        assert ws.dataX(i)[0] == pytest.approx(3.2131329446)


TOF = [12345.0, 12346.0]


@pytest.mark.parametrize(
    "generic_workspace",
    [
        {
            "dx": 0.005,
            "dy": 0.004,
            "zc": 2.5,
            "l1": 10.1,
            "axis_units": "tof",
            "axis_values": TOF,
        }
    ],
    indirect=True,
)
def test_shuo(generic_workspace):
    """Test the conversion of time-of-flight to wavelength
    in master document section 3.3
    dev - Pete Peterson <petersonpf@ornl.gov>
    SME - Shuo Qian"""
    # set up input workspace in the first frame
    ws = generic_workspace  # friendly name
    add_frame_skipping_log(ws)

    # run the drt-sans version
    ws = convert_to_wavelength(input_workspace=generic_workspace)

    # make sure the unit is wavelength
    assert ws.getAxis(0).getUnit().caption() == "Wavelength"

    # get information for detector pixel positions
    specInfo = ws.spectrumInfo()
    source_sample = specInfo.l1()  # in meters

    # verify the individual wavelength values
    for i in range(4):
        # distance to detector pixel in meters
        sample_detector = specInfo.l2(i)
        # equation supplied by SME applied to time-of-flight
        lambda_exp = 3.9560346e-3 * np.array(TOF) / (source_sample + sample_detector)

        # verify the results
        assert ws.dataX(i)[0] == pytest.approx(lambda_exp[0])
        assert ws.dataX(i)[1] == pytest.approx(lambda_exp[1])
        assert ws.dataX(i)[0] == pytest.approx(3.875969)  # Shuo asked for 3.8760


if __name__ == "__main__":
    pytest.main([__file__])
