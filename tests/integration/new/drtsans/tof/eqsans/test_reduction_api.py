import pytest
import os
import tempfile
from drtsans.tof.eqsans import reduction_parameters, update_reduction_parameters
from drtsans.tof.eqsans.api import (
    load_all_files,
    reduce_single_configuration,
)  # noqa E402
from mantid.simpleapi import LoadNexusProcessed, CheckWorkspacesMatch
from mantid.simpleapi import mtd, DeleteWorkspace
import numpy as np
from drtsans.dataobjects import save_i_of_q_to_h5, load_iq1d_from_h5, load_iq2d_from_h5
from typing import List, Any, Union, Tuple, Dict
from drtsans.dataobjects import _Testing
from matplotlib import pyplot as plt
from drtsans.dataobjects import IQmod
from drtsans.settings import amend_config


# EQSANS reduction
specs_eqsans = {
    "EQSANS_88980": {
        "iptsNumber": 19800,
        "sample": {
            "runNumber": 88980,
            "thickness": 0.1,
            "transmission": {"runNumber": 88980},
        },
        "background": {"runNumber": 88978, "transmission": {"runNumber": 88974}},
        "beamCenter": {"runNumber": 88973},
        "emptyTransmission": {"runNumber": 88973},
        "configuration": {
            "sampleApertureSize": 30,
            "darkFileName": "/SNS/EQSANS/shared/NeXusFiles/EQSANS/2017B_mp/EQSANS_86275.nxs.h5",
            "StandardAbsoluteScale": 0.0208641883,
            "sampleOffset": 0,
        },
        "dataDirectories": [
            "/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/sns/eqsans/",
        ],
    }
}


@pytest.mark.parametrize(
    "run_config, basename",
    [(specs_eqsans["EQSANS_88980"], "EQSANS_88980")],
    ids=["88980"],
)
def test_regular_setup(run_config, basename, generatecleanfile, reference_dir):
    """Same reduction from Shaman test with regular non-correction and no-weighted binning"""
    # set flag to use weighted binning
    weighted_binning = False

    common_config = {
        "configuration": {
            "maskFileName": "/SNS/EQSANS/shared/NeXusFiles/EQSANS/2017B_mp/beamstop60_mask_4m.nxs",
            "useDefaultMask": True,
            "normalization": "Total charge",
            "fluxMonitorRatioFile": "/SNS/EQSANS/IPTS-24769/shared/EQSANS_110943.out",
            "beamFluxFileName": "/SNS/EQSANS/shared/instrument_configuration/bl6_flux_at_sample",
            "absoluteScaleMethod": "standard",
            "detectorOffset": 0,
            "mmRadiusForTransmission": 25,
            "numQxQyBins": 80,
            "1DQbinType": "scalar",
            "QbinType": "linear",
            "useErrorWeighting": weighted_binning,
            "numQBins": 120,
            "AnnularAngleBin": 5,
            "wavelengthStepType": "constant Delta lambda",
            "wavelengthStep": 0.1,
        }
    }
    # defaults and common options
    input_config = reduction_parameters(common_config, "EQSANS", validate=False)
    # final changes and validation
    input_config = update_reduction_parameters(input_config, run_config, validate=False)
    output_dir = generatecleanfile()
    amendments = {
        "outputFileName": basename,
        "configuration": {"outputDir": output_dir},
    }
    input_config = update_reduction_parameters(input_config, amendments, validate=True)

    # expected output Nexus file
    reduced_data_nexus = os.path.join(output_dir, f"{basename}.nxs")
    # remove files
    if os.path.exists(reduced_data_nexus):
        os.remove(reduced_data_nexus)

    # Load and reduce
    with amend_config(data_dir=run_config["dataDirectories"]):
        loaded = load_all_files(input_config)
    reduction_output = reduce_single_configuration(loaded, input_config)

    # Check reduced workspace
    assert os.path.exists(
        reduced_data_nexus
    ), f"Expected {reduced_data_nexus} does not exist"
    # verify with gold data and clean
    gold_file = os.path.join(
        reference_dir.new.eqsans, "test_integration_api/EQSANS_88980_reduced_m6.nxs"
    )
    verify_processed_workspace(
        test_file=reduced_data_nexus, gold_file=gold_file, ws_prefix="no_wl"
    )

    # Load data and compare
    gold_dir = reference_dir.new.eqsans
    gold_file_dict = dict()
    for frame_index in range(2):
        iq1d_h5_name = os.path.join(
            gold_dir, f"test_integration_api/88980_iq1d_{frame_index}_0_m6.h5"
        )
        gold_file_dict[1, frame_index, 0] = iq1d_h5_name
        iq2d_h5_name = os.path.join(
            gold_dir, f"test_integration_api/88980_iq2d_{frame_index}_m6.h5"
        )
        gold_file_dict[2, frame_index] = iq2d_h5_name
    verify_binned_iq(gold_file_dict, reduction_output)

    # clean up
    os.remove(reduced_data_nexus)
    # NOTE: similar to other tests, the design of load_all_files requires
    #       us to clean up the leftover workspaces in ADS by hand
    # _bkgd_trans:	123.262977 MB
    # _empty:	123.261169 MB
    # _EQSANS_86275_raw_histo:	64.764329 MB
    # _EQSANS_88973_raw_events:	33.996129 MB
    # _EQSANS_88973_raw_histo:	123.260961 MB
    # _EQSANS_88974_raw_histo:	123.262769 MB
    # _EQSANS_88978_raw_histo:	123.261585 MB
    # _EQSANS_88980_raw_histo:	124.588369 MB
    # _mask:	33.532216 MB
    # _sample_trans:	124.588577 MB
    # _sensitivity:	51.262916 MB
    # no_wl_gold:	124.589201 MB
    # no_wl_test:	124.589201 MB
    # processed_data_main:	124.589201 MB
    DeleteWorkspace("_bkgd_trans")
    DeleteWorkspace("_empty")
    DeleteWorkspace("_mask")
    DeleteWorkspace("_sample_trans")
    DeleteWorkspace("_sensitivity")
    DeleteWorkspace("no_wl_gold")
    DeleteWorkspace("no_wl_test")
    DeleteWorkspace("processed_data_main")
    for ws in mtd.getObjectNames():
        if str(ws).startswith("_EQSANS_8"):
            DeleteWorkspace(ws)


@pytest.mark.parametrize(
    "run_config, basename",
    [(specs_eqsans["EQSANS_88980"], "EQSANS_88980")],
    ids=["88980"],
)
def test_weighted_binning_setup(run_config, basename, generatecleanfile, reference_dir):
    """Same reduction from Shaman test but using weighted binning

    A previous integration test has approved that the 2-step binning
    (binning on Q, and then binning on wavelength) with weighted binning algorithm
    is able to generate same result as 1-step binning (binning on Q and wavelength together).

    - weighted binning must be used
    """
    common_config = {
        "configuration": {
            "maskFileName": "/SNS/EQSANS/shared/NeXusFiles/EQSANS/2017B_mp/beamstop60_mask_4m.nxs",
            "useDefaultMask": True,
            "normalization": "Total charge",
            "fluxMonitorRatioFile": "/SNS/EQSANS/IPTS-24769/shared/EQSANS_110943.out",
            "beamFluxFileName": "/SNS/EQSANS/shared/instrument_configuration/bl6_flux_at_sample",
            "absoluteScaleMethod": "standard",
            "detectorOffset": 0,
            "mmRadiusForTransmission": 25,
            "numQxQyBins": 80,
            "1DQbinType": "scalar",
            "QbinType": "linear",
            "numQBins": 120,
            "AnnularAngleBin": 5,
            "wavelengthStepType": "constant Delta lambda",
            "wavelengthStep": 0.1,
            "useErrorWeighting": True,
        }
    }
    input_config = reduction_parameters(
        common_config, "EQSANS", validate=False
    )  # defaults and common options
    input_config = update_reduction_parameters(input_config, run_config, validate=False)
    output_dir = generatecleanfile()
    amendments = {
        "outputFileName": f"{basename}_corr",
        "configuration": {"outputDir": output_dir},
    }
    input_config = update_reduction_parameters(
        input_config, amendments, validate=True
    )  # final changes and validation

    # expected output Nexus file
    reduced_data_nexus = os.path.join(output_dir, f"{basename}_corr.nxs")
    # clean output directory
    if os.path.exists(reduced_data_nexus):
        os.remove(reduced_data_nexus)

    # Load and reduce
    with amend_config(data_dir=run_config["dataDirectories"]):
        loaded = load_all_files(input_config)
    reduction_output = reduce_single_configuration(loaded, input_config)

    # Verify reduced workspace
    gold_ws_nexus = os.path.join(
        reference_dir.new.eqsans, "test_integration_api/EQSANS_88980_reduced_m6.nxs"
    )
    print(
        f"[TEST] Verify correction workflow reduction: {reduced_data_nexus} vs. {gold_ws_nexus}"
    )
    verify_processed_workspace(
        test_file=reduced_data_nexus,
        gold_file=gold_ws_nexus,
        ws_prefix="no_wl",
        ignore_error=False,
    )

    # Verify binned I(Q)
    gold_file_dict = dict()
    gold_dir = os.path.join(reference_dir.new.eqsans, "gold_data")

    # FIXME: The gold data are not stored inside the repository so when
    # gold data are changed a version prefix is added with date and developer
    # information. The old data will be kept as it is.
    version = "20220321_rys_"

    for frame_index in range(2):
        iq1d_h5_name = os.path.join(
            gold_dir, f"{version}gold_88980_weighted_1d_{frame_index}.h5"
        )
        gold_file_dict[1, frame_index, 0] = iq1d_h5_name
        iq2d_h5_name = os.path.join(
            gold_dir, f"gold_88980_weighted_2d_{frame_index}.h5"
        )
        gold_file_dict[2, frame_index] = iq2d_h5_name
    verify_binned_iq(gold_file_dict, reduction_output)

    # clean up
    os.remove(reduced_data_nexus)
    # NOTE: similar to other tests, the design of load_all_files requires
    #       us to clean up the leftover workspaces in ADS by hand
    # _bkgd_trans:	123.262977 MB
    # _empty:	123.261169 MB
    # _EQSANS_86275_raw_histo:	64.764329 MB
    # _EQSANS_88973_raw_events:	33.996129 MB
    # _EQSANS_88973_raw_histo:	123.260961 MB
    # _EQSANS_88974_raw_histo:	123.262769 MB
    # _EQSANS_88978_raw_histo:	123.261585 MB
    # _EQSANS_88980_raw_histo:	124.588369 MB
    # _mask:	33.532216 MB
    # _sample_trans:	124.588577 MB
    # _sensitivity:	51.262916 MB
    # no_wl_gold:	124.589201 MB
    # no_wl_test:	124.589201 MB
    # processed_data_main:	124.589201 MB
    DeleteWorkspace("_bkgd_trans")
    DeleteWorkspace("_empty")
    DeleteWorkspace("_mask")
    DeleteWorkspace("_sample_trans")
    DeleteWorkspace("_sensitivity")
    DeleteWorkspace("no_wl_gold")
    DeleteWorkspace("no_wl_test")
    DeleteWorkspace("processed_data_main")
    for ws in mtd.getObjectNames():
        if str(ws).startswith("_EQSANS_8"):
            DeleteWorkspace(ws)


def verify_binned_iq(gold_file_dict: Dict[Tuple, str], reduction_output):
    """Verify reduced I(Q1D) and I(qx, qy) by expected/gold data

    Parameters
    ----------
    gold_file_dict: ~dict
        dictionary for gold files
    reduction_output: ~list
        list of binned I(Q1D) and I(qx, qy)

    """
    for frame_index in range(2):
        # 1D
        iq1d_h5_name = gold_file_dict[1, frame_index, 0]
        gold_iq1d = load_iq1d_from_h5(iq1d_h5_name)
        _Testing.assert_allclose(reduction_output[frame_index].I1D_main[0], gold_iq1d, rtol=0.1)

        # 2D
        iq2d_h5_name = gold_file_dict[2, frame_index]
        gold_iq2d = load_iq2d_from_h5(iq2d_h5_name)
        _Testing.assert_allclose(reduction_output[frame_index].I2D_main, gold_iq2d, rtol=0.1)


def export_iq_comparison(iq1d_tuple_list: List[Tuple[str, IQmod, str]], png_name: str):
    """Export a list of IQmod to plot"""

    plt.figure(figsize=(18, 9))
    for iq1d_tuple in iq1d_tuple_list:
        label, iq1d, color = iq1d_tuple
        plt.plot(iq1d.mod_q, iq1d.intensity, color=color, label=label)

    # legend
    plt.legend()

    # save
    plt.savefig(png_name)
    # close
    plt.close()

    plt.figure(figsize=(18, 12))
    plt.yscale("log")

    # plot error bar to compare
    for iq1d_tuple in iq1d_tuple_list:
        label, iq1d, color = iq1d_tuple
        plt.plot(
            iq1d.mod_q,
            iq1d.error,
            color=color,
            label=label,
            marker=".",
            linestyle="None",
        )

    # legend
    plt.legend()

    # save
    plt.savefig(f'{png_name.split(".")[0]}_error_bar.png')
    # close
    plt.close()


def export_reduction_output(
    reduction_output: List[Any], output_dir: Union[None, str] = None, prefix: str = ""
):
    """Export the reduced I(Q) and I(Qx, Qy) to  hdf5 files"""
    # output of reduce_single_configuration: list of list, containing
    if output_dir is None:
        output_dir = os.getcwd()

    for section_index, section_output in enumerate(reduction_output):
        # 1D (list of IQmod)
        iq1ds = section_output.I1D_main
        for j_index, iq1d in enumerate(iq1ds):
            h5_file_name = os.path.join(
                output_dir, f"{prefix}iq1d_{section_index}_{j_index}.h5"
            )
            save_i_of_q_to_h5(iq1d, h5_file_name)
            print(f"Save frame {section_index} {j_index}-th I(Q1D) to {h5_file_name}")
        # 2D (IQazimuthal)
        iq2d = section_output.I2D_main
        h5_file_name = os.path.join(output_dir, f"{prefix}iq2d_{section_index}.h5")
        save_i_of_q_to_h5(iq2d, h5_file_name)
        print(f"Save frame {section_index} I(Q2D) to {h5_file_name}")


def test_wavelength_step(reference_dir):

    # Set up the configuration dict
    configuration = {
        "instrumentName": "EQSANS",
        "iptsNumber": 26015,
        "sample": {"runNumber": 115363, "thickness": 1.0},
        "background": {
            "runNumber": 115361,
            "transmission": {"runNumber": 115357, "value": ""},
        },
        "emptyTransmission": {"runNumber": 115356, "value": ""},
        "beamCenter": {"runNumber": 115356},
        "outputFileName": "test_wavelength_step",
        "configuration": {
            "cutTOFmax": "1500",
            "wavelengthStepType": "constant Delta lambda/lambda",
            "sampleApertureSize": "10",
            "fluxMonitorRatioFile": (
                "/SNS/EQSANS/" "IPTS-24769/shared/EQSANS_110943.out"
            ),
            "sensitivityFileName": (
                "/SNS/EQSANS/shared/NeXusFiles/"
                "EQSANS/2020A_mp/"
                "Sensitivity_patched_thinPMMA_2o5m_113514_mantid.nxs"
            ),
            "numQBins": 100,
            "WedgeMinAngles": "-30, 60",
            "WedgeMaxAngles": "30, 120",
            "AnnularAngleBin": "5",
            "useSliceIDxAsSuffix": True,
        },
    }

    # Specify gold dir
    gold_dir = reference_dir.new.eqsans

    # Test 1 with regular setup
    with tempfile.TemporaryDirectory() as test_dir:
        # continue to configure
        configuration["configuration"]["outputDir"] = test_dir
        configuration["outputFileName"] = "test_wavelength_step_reg"
        configuration["dataDirectories"] = test_dir
        # validate and clean configuration
        input_config = reduction_parameters(configuration)
        # reduce
        loaded = load_all_files(input_config)
        reduction_output = reduce_single_configuration(loaded, input_config)

        # verify output file existence
        output_file_name = os.path.join(test_dir, "test_wavelength_step_reg.nxs")
        assert os.path.isfile(
            output_file_name
        ), f"Expected output file {output_file_name} does not exists"
        # verify reduced worksapce
        gold_file = os.path.join(
            gold_dir, "test_integration_api/EQSANS_88980_wl_reduced_reg_m6.nxs"
        )
        verify_processed_workspace(
            output_file_name, gold_file, "reg", ignore_error=False
        )
        # verify binned reduced I(Q)
        gold_iq1d_h5 = os.path.join(
            gold_dir, "test_integration_api/88980_iq1d_wl_0_0_m6.h5"
        )
        gold_iq1d = load_iq1d_from_h5(gold_iq1d_h5)
        _Testing.assert_allclose(reduction_output[0].I1D_main[0], gold_iq1d)
        # verify binned reduced I(Qx, Qy)
        # TODO skip as no knowing what the user's requirement with wavelength kept
        # iq2d_h5_name = os.path.join(gold_dir, f'gold_iq2d_wave_0.h5')
        # gold_iq2d = load_iq2d_from_h5(iq2d_h5_name)
        # test_iq2d = reduction_output[0].I2D_main
        # _Testing.assert_allclose(reduction_output[0].I2D_main, gold_iq2d)

    # Test 2 with c.o.m beam center
    with tempfile.TemporaryDirectory() as test_dir:
        # continue to configure
        configuration["configuration"]["outputDir"] = test_dir
        configuration["outputFileName"] = "test_wavelength_step_com"
        configuration["dataDirectories"] = test_dir
        # validate and clean configuration
        input_config = reduction_parameters(configuration)
        input_config["beamCenter"]["method"] = "center_of_mass"
        input_config["beamCenter"]["com_centering_options"] = {
            "CenterX": 0.0,
            "CenterY": 0.0,
            "Tolerance": 0.00125,
        }
        # reduce
        loaded = load_all_files(input_config)
        reduce_single_configuration(loaded, input_config)
        # verify output file existence
        output_file_name = os.path.join(f"{test_dir}", "test_wavelength_step_com.nxs")
        assert os.path.isfile(
            output_file_name
        ), f"Expected output file {output_file_name} does not exists"
        # verify reduced worksapce
        gold_file = os.path.join(
            gold_dir, "test_integration_api/EQSANS_88980_wl_reduced_com_m6.nxs"
        )
        verify_processed_workspace(
            output_file_name, gold_file, "com", ignore_error=False
        )

    # Test 3 with gaussian beam center
    with tempfile.TemporaryDirectory() as test_dir:
        # continue to configure
        configuration["configuration"]["outputDir"] = test_dir
        configuration["outputFileName"] = "test_wavelength_step_gauss"
        configuration["dataDirectories"] = test_dir
        # validate and clean configuration
        input_config = reduction_parameters(configuration)
        input_config["beamCenter"]["method"] = "gaussian"
        input_config["beamCenter"]["gaussian_centering_options"] = {
            "theta": {"value": 0.0, "vary": False}
        }
        # reduce
        loaded = load_all_files(input_config)
        reduce_single_configuration(loaded, input_config)

        # verify output file existence
        output_file_name = os.path.join(f"{test_dir}", "test_wavelength_step_gauss.nxs")
        assert os.path.isfile(
            output_file_name
        ), f"Expected output file {output_file_name} does not exist."
        # verify_reduced_data
        # Difference from mantid5 result
        # E   AssertionError:
        # E   Not equal to tolerance rtol=1e-07, atol=0   Y is not same
        # E   Mismatched elements: 1 / 442368 (0.000226%)
        # E   Max absolute difference: 2.96006469e-09  Max relative difference: 1.7555871e-07
        gold_file = os.path.join(
            gold_dir, "test_integration_api/EQSANS_88980_wl_reduced_gauss_m6.nxs"
        )
        # This tolerance: 3E-7 comes from the different result between Ubuntu and REL7
        verify_processed_workspace(
            output_file_name,
            gold_file,
            "gauss",
            ignore_error=False,
            y_rel_tol=3.0e-7,
            e_rel_tol=1.36e-7,
        )

    # clean up
    # _empty:	11.331733 MB
    # _EQSANS_113569_raw_histo:	72.325189 MB
    # _EQSANS_115356_raw_events:	44.045813 MB
    # _EQSANS_115356_raw_histo:	11.331525 MB
    # _EQSANS_115357_raw_histo:	11.329493 MB
    # _EQSANS_115361_raw_histo:	12.371909 MB
    # _EQSANS_115363_raw_histo:	11.679333 MB
    # _mask:	143.871004 MB
    # _sensitivity:	55.210004 MB
    # com_gold:	11.680165 MB
    # com_test:	11.680165 MB
    # gauss_gold:	11.680165 MB
    # gauss_test:	11.680165 MB
    # processed_data_main:	11.680165 MB
    # reg_gold:	11.680165 MB
    # reg_test:	11.680165 MB
    DeleteWorkspace("_bkgd_trans")
    DeleteWorkspace("_empty")
    DeleteWorkspace("_mask")
    DeleteWorkspace("_sensitivity")
    DeleteWorkspace("com_gold")
    DeleteWorkspace("com_test")
    DeleteWorkspace("gauss_gold")
    DeleteWorkspace("gauss_test")
    DeleteWorkspace("processed_data_main")
    DeleteWorkspace("reg_gold")
    DeleteWorkspace("reg_test")
    for ws in mtd.getObjectNames():
        if str(ws).startswith("_EQSANS_11"):
            DeleteWorkspace(ws)


def verify_processed_workspace(
    test_file, gold_file, ws_prefix, ignore_error=False, y_rel_tol=None, e_rel_tol=0.1,
):
    """Verify pre-processed workspace by verified expected result (workspace)

    Parameters
    ----------
    test_file: str
        NeXus file from test to verify
    gold_file: str
        NeXus file containing the expected reduced result to verify against
    ws_prefix: str
        prefix for Mantid workspace that the
    ignore_error: bool
        flag to ignore the checking on intensity error
    y_rel_tol: float, None
        allowed maximum tolerance on Y
    e_rel_tol: float, None
        allowed maximum tolerance on E

    """
    assert os.path.exists(gold_file), f"Gold file {gold_file} cannot be found"
    assert os.path.exists(test_file), f"Test file {test_file} cannot be found"

    gold_ws = LoadNexusProcessed(
        Filename=gold_file, OutputWorkspace=f"{ws_prefix}_gold"
    )
    test_ws = LoadNexusProcessed(
        Filename=test_file, OutputWorkspace=f"{ws_prefix}_test"
    )
    r = CheckWorkspacesMatch(Workspace1=gold_ws, Workspace2=test_ws)
    print(
        f"[INT-TEST] Verify reduced workspace {test_ws} match expected/gold {gold_ws}: {r}"
    )
    if r != "Success":
        assert (
            gold_ws.getNumberHistograms() == test_ws.getNumberHistograms()
        ), f"Histograms: {gold_ws.getNumberHistograms()} != {test_ws.getNumberHistograms()}"
        assert (
            gold_ws.readY(0).shape == test_ws.readY(0).shape
        ), f"Number of wavelength: {gold_ws.readY(0).shape} != {test_ws.readY(0).shape}"
        assert (
            gold_ws.readX(0).shape == test_ws.readX(0).shape
        ), f"Histogram or point data: {gold_ws.readX(0).shape} != {test_ws.readX(0).shape}"
        gold_x_array = gold_ws.extractX()
        test_x_array = test_ws.extractX()
        assert gold_x_array.shape == test_x_array.shape, "Q bins sizes are different"
        np.testing.assert_allclose(
            gold_ws.extractX(), test_ws.extractX(), err_msg="X is not same"
        )
        if y_rel_tol is not None:
            y_dict = {"rtol": y_rel_tol}
        else:
            y_dict = dict()
        np.testing.assert_allclose(
            gold_ws.extractY(), test_ws.extractY(), err_msg="Y is not same", **y_dict
        )
        if not ignore_error:
            if e_rel_tol is None:
                e_dict = dict()
            else:
                e_dict = {"rtol": e_rel_tol}
            np.testing.assert_allclose(
                gold_ws.extractE(),
                test_ws.extractE(),
                err_msg="E is not same",
                **e_dict,
            )


if __name__ == "__main__":
    pytest.main([__file__])
