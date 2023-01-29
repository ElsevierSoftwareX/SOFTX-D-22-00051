import pytest
import numpy as np
from pytest import approx
from drtsans.tof.eqsans import convert_to_q
from drtsans.tof.eqsans.momentum_transfer import retrieve_instrument_setup
from mantid.simpleapi import AddSampleLog

intensities_array = np.array(
    [
        [
            [93, 60, 89, 32, 97],
            [43, 61, 82, 97, 55],
            [78, 34, 50, 54, 67],
            [98, 88, 37, 92, 97],
            [72, 97, 100, 71, 39],
        ],
        [
            [76, 39, 51, 70, 61],
            [64, 54, 78, 35, 30],
            [67, 98, 100, 56, 79],
            [97, 35, 41, 90, 45],
            [30, 41, 68, 34, 51],
        ],
        [
            [78, 36, 46, 75, 91],
            [64, 56, 92, 73, 60],
            [74, 72, 69, 84, 87],
            [36, 78, 40, 68, 72],
            [59, 40, 39, 34, 85],
        ],
    ]
)
intensities_array = np.transpose(intensities_array, axes=(1, 2, 0))
expected_intensities = np.transpose(intensities_array, axes=(1, 0, 2))[:, ::-1, :]

uncertainties_array = np.sqrt(intensities_array)

wavelength_array = np.zeros((5, 5, 4))  # bin boundaries
wavelength_array[:, :, 0] = 2.95
wavelength_array[:, :, 1] = 3.05
wavelength_array[:, :, 2] = 3.15
wavelength_array[:, :, 3] = 3.25

expected_qx = np.array(
    [
        [
            [-0.006134, -0.003254, -0.000374, 0.002505, 0.005385],
            [-0.006134, -0.003254, -0.000374, 0.002505, 0.005385],
            [-0.006134, -0.003254, -0.000374, 0.002505, 0.005385],
            [-0.006134, -0.003254, -0.000374, 0.002505, 0.005385],
            [-0.006134, -0.003254, -0.000374, 0.002505, 0.005385],
        ],
        [
            [-0.005936, -0.003149, -0.000362, 0.002425, 0.005211],
            [-0.005936, -0.003149, -0.000362, 0.002425, 0.005211],
            [-0.005936, -0.003149, -0.000362, 0.002425, 0.005211],
            [-0.005936, -0.003149, -0.000362, 0.002425, 0.005211],
            [-0.005936, -0.003149, -0.000362, 0.002425, 0.005211],
        ],
        [
            [-0.005751, -0.003051, -0.000351, 0.002349, 0.005049],
            [-0.005751, -0.003051, -0.000351, 0.002349, 0.005049],
            [-0.005751, -0.003051, -0.000351, 0.002349, 0.005049],
            [-0.005751, -0.003051, -0.000351, 0.002349, 0.005049],
            [-0.005751, -0.003051, -0.000351, 0.002349, 0.005049],
        ],
    ]
)
expected_qx = np.transpose(expected_qx, axes=(2, 1, 0))[:, ::-1, :]

expected_qy = np.array(
    [
        [
            [0.004962, 0.004962, 0.004962, 0.004962, 0.004962],
            [0.002737, 0.002737, 0.002737, 0.002737, 0.002737],
            [0.000512, 0.000512, 0.000512, 0.000512, 0.000512],
            [-0.001713, -0.001713, -0.001713, -0.001713, -0.001713],
            [-0.003939, -0.003939, -0.003939, -0.003939, -0.003939],
        ],
        [
            [0.004802, 0.004802, 0.004802, 0.004802, 0.004802],
            [0.002649, 0.002649, 0.002649, 0.002649, 0.002649],
            [0.000495, 0.000495, 0.000495, 0.000495, 0.000495],
            [-0.001658, -0.001658, -0.001658, -0.001658, -0.001658],
            [-0.003812, -0.003812, -0.003812, -0.003812, -0.003812],
        ],
        [
            [0.004652, 0.004652, 0.004652, 0.004652, 0.004652],
            [0.002566, 0.002566, 0.002566, 0.002566, 0.002566],
            [0.000480, 0.000480, 0.000480, 0.000480, 0.000480],
            [-0.001606, -0.001606, -0.001606, -0.001606, -0.001606],
            [-0.003693, -0.003693, -0.003693, -0.003693, -0.003693],
        ],
    ]
)
expected_qy = np.transpose(expected_qy, axes=(2, 1, 0))[:, ::-1, :]

expected_dqx = np.array(
    [
        [
            [0.000066, 0.000035, 0.000005, 0.000027, 0.000058],
            [0.000066, 0.000035, 0.000005, 0.000027, 0.000058],
            [0.000066, 0.000035, 0.000005, 0.000027, 0.000058],
            [0.000066, 0.000035, 0.000005, 0.000027, 0.000058],
            [0.000066, 0.000035, 0.000005, 0.000027, 0.000058],
        ],
        [
            [0.000062, 0.000033, 0.000004, 0.000025, 0.000055],
            [0.000062, 0.000033, 0.000004, 0.000025, 0.000055],
            [0.000062, 0.000033, 0.000004, 0.000025, 0.000055],
            [0.000062, 0.000033, 0.000004, 0.000025, 0.000055],
            [0.000062, 0.000033, 0.000004, 0.000025, 0.000055],
        ],
        [
            [0.000058, 0.000031, 0.000004, 0.000024, 0.000051],
            [0.000058, 0.000031, 0.000004, 0.000024, 0.000051],
            [0.000058, 0.000031, 0.000004, 0.000024, 0.000051],
            [0.000058, 0.000031, 0.000004, 0.000024, 0.000051],
            [0.000058, 0.000031, 0.000004, 0.000024, 0.000051],
        ],
    ]
)
expected_dqx = np.transpose(expected_dqx, axes=(2, 1, 0))[:, ::-1, :]

expected_dqy = np.array(
    [
        [
            [0.000054, 0.000054, 0.000054, 0.000054, 0.000054],
            [0.000030, 0.000030, 0.000030, 0.000030, 0.000030],
            [0.000006, 0.000006, 0.000006, 0.000006, 0.000006],
            [0.000019, 0.000019, 0.000019, 0.000019, 0.000019],
            [0.000043, 0.000043, 0.000043, 0.000043, 0.000043],
        ],
        [
            [0.000050, 0.000050, 0.000050, 0.000050, 0.000050],
            [0.000028, 0.000028, 0.000028, 0.000028, 0.000028],
            [0.000006, 0.000006, 0.000006, 0.000006, 0.000006],
            [0.000017, 0.000017, 0.000017, 0.000017, 0.000017],
            [0.000040, 0.000040, 0.000040, 0.000040, 0.000040],
        ],
        [
            [0.000047, 0.000047, 0.000047, 0.000047, 0.000047],
            [0.000026, 0.000026, 0.000026, 0.000026, 0.000026],
            [0.000005, 0.000005, 0.000005, 0.000005, 0.000005],
            [0.000016, 0.000016, 0.000016, 0.000016, 0.000016],
            [0.000038, 0.000038, 0.000038, 0.000038, 0.000038],
        ],
    ]
)
expected_dqy = np.transpose(expected_dqy, axes=(2, 1, 0))[:, ::-1, :]


@pytest.mark.parametrize(
    "workspace_with_instrument",
    [
        {
            "Nx": 5,
            "Ny": 5,
            "name": "EQSANS",
            "dx": 0.0055,
            "dy": 0.00425,
            "xc": 0.000715,
            "yc": 0.0009775,
            "zc": 4.0,
            "l1": 4.0,
        }
    ],
    indirect=True,
)
def test_convert_to_q_eqsans(workspace_with_instrument):
    ws = workspace_with_instrument(
        axis_values=[2.95, 3.05, 3.15, 3.25],
        intensities=intensities_array,
        uncertainties=np.sqrt(intensities_array),
        view="array",
    )
    AddSampleLog(
        Workspace=ws,
        LogName="source_aperture_diameter",
        LogText="50.",
        LogUnit="mm",
        LogType="Number",
    )
    AddSampleLog(
        Workspace=ws,
        LogName="sample_aperture_diameter",
        LogText="20.",
        LogUnit="mm",
        LogType="Number",
    )
    AddSampleLog(
        Workspace=ws,
        LogName="source_aperture_sample_distance",
        LogText="4000.",
        LogUnit="mm",
        LogType="Number",
    )
    # make sure we created the workspace right
    assert ws.readX(0) == approx([2.95, 3.05, 3.15, 3.25], abs=1e-8)
    assert ws.readY(0) == approx([72, 30, 59], abs=1e-8)
    assert ws.readY(1) == approx([98, 97, 36], abs=1e-8)
    assert ws.readY(5) == approx([97, 41, 40], abs=1e-8)
    assert ws.readE(0) == approx(np.sqrt(np.array([72, 30, 59])), abs=1e-8)
    assert ws.readE(1) == approx(np.sqrt(np.array([98, 97, 36])), abs=1e-8)
    assert ws.readE(5) == approx(np.sqrt(np.array([97, 41, 40])), abs=1e-8)
    det_info = ws.detectorInfo()
    assert det_info.position(0) == approx([11.715e-3, -7.5225e-3, 4], abs=1e-8)
    assert det_info.position(1) == approx([11.715e-3, -3.2725e-3, 4], abs=1e-8)
    assert det_info.position(5) == approx([6.215e-3, -7.5225e-3, 4], abs=1e-8)
    # check instrument info for resolution calculation
    instrument_setup = retrieve_instrument_setup(ws)
    assert instrument_setup.l1 == approx(4, 1e-8)
    assert instrument_setup.sample_det_center_distance == approx(4, abs=1e-6)
    assert instrument_setup.source_aperture_radius == approx(0.025, abs=1e-6)
    assert instrument_setup.sample_aperture_radius == approx(0.010, abs=1e-6)

    # calculate q and resolution, and check the output
    result = convert_to_q(ws, mode="azimuthal")
    intensity = result.intensity.reshape((5, 5, 3))
    assert intensity == approx(expected_intensities, abs=1e-6)
    error = result.error.reshape((5, 5, 3))
    assert error == approx(np.sqrt(expected_intensities), abs=1e-6)
    qx = result.qx.reshape((5, 5, 3))
    assert qx == approx(expected_qx, abs=1e-6)
    qy = result.qy.reshape((5, 5, 3))
    assert qy == approx(expected_qy, abs=1e-6)
    dqx = result.delta_qx.reshape((5, 5, 3))
    assert dqx == approx(expected_dqx, abs=1e6)
    dqy = result.delta_qy.reshape((5, 5, 3))
    assert dqy == approx(expected_dqy, abs=1e6)


@pytest.mark.parametrize(
    "generic_workspace",
    [
        {
            "name": "EQSANS",
            "Nx": 5,
            "Ny": 5,
            "dx": 0.00425,
            "dy": 0.0055,
            "xc": 0.0,
            "yc": 0.0,
            "zc": 15.5,
            "l1": 15,
            "axis_values": [5.925, 6.075],
        }
    ],
    indirect=True,
)
def test_retrieve_instrument_setup(generic_workspace):
    workspace = generic_workspace

    # Insert logs
    names = (
        "wavelength wavelength-spread source_aperture_diameter sample_aperture_diameter "
        "smearingPixelSizeX smearingPixelSizeY source_aperture_sample_distance sample-detector-distance".split()
    )
    values = [6.0, 0.15, 0.02, 0.007, 0.0085, 0.011, 15.0, 15.5]
    units = "A A mm mm m m m m".split()
    for name, value, unit in zip(names, values, units):
        AddSampleLog(
            Workspace=workspace,
            LogName=name,
            LogText="{}".format(value),
            LogType="Number",
            LogUnit=unit,
        )

    params = retrieve_instrument_setup(workspace)
    assert [
        params.smearing_pixel_width_ratio,
        params.smearing_pixel_height_ratio,
    ] == pytest.approx([2.0, 2.0])
