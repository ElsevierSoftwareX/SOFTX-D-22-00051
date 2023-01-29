# Move part of the methods from api.py to avoid importing in loops
from mantid.simpleapi import (
    mtd,
    SaveAscii,
)  # noqa E402

# Import rolled up to complete a single top-level API
from drtsans import apply_sensitivity_correction, solid_angle_correction  # noqa E402
from drtsans import subtract_background  # noqa E402
from drtsans.transmission import apply_transmission_correction  # noqa E402
from drtsans.tof.eqsans.transmission import calculate_transmission  # noqa E402
from drtsans.thickness_normalization import normalize_by_thickness  # noqa E402
from drtsans.tof.eqsans.dark_current import subtract_dark_current  # noqa E402
from drtsans.mask_utils import apply_mask  # noqa E402
from drtsans.tof.eqsans.normalization import normalize_by_flux  # noqa E402
import numpy as np
from drtsans.iq import bin_all  # noqa E402
from drtsans.tof.eqsans.correction_api import (
    do_inelastic_incoherence_correction_q1d,
    do_inelastic_incoherence_correction_q2d,
    save_k_vector,
)
from drtsans.tof.eqsans.elastic_reference_normalization import (
    normalize_by_elastic_reference,
    normalize_by_elastic_reference2D,
)
import os
from collections import namedtuple
from typing import Dict, List

# Binning parameters
BinningSetup = namedtuple(
    "binning_setup",
    "nxbins_main nybins_main n1dbins n1dbins_per_decade "
    "decade_on_center bin1d_type log_scale qmin, qmax, qxrange, qyrange",
)


def prepare_data_workspaces(
    data: namedtuple,
    dark_current=None,
    flux_method=None,  # normalization (proton charge/time/monitor)
    flux=None,  # additional file for normalization
    mask_ws=None,  # apply a custom mask from workspace
    mask_panel=None,  # mask back or front panel
    mask_btp=None,  # mask bank/tube/pixel
    solid_angle=True,
    sensitivity_workspace=None,
    output_workspace=None,
):

    r"""
    Given a " raw"data workspace, this function provides the following:

        - subtracts dark current
        - normalize by time or monitor
        - applies masks
        - corrects for solid angle
        - corrects for sensitivity

    All steps are optional. data, mask_ws, dark_current are either None
    or histogram workspaces. This function does not load any file.

    Parameters
    ----------
    data: namedtuple
        (~mantid.dataobjects.Workspace2D, ~mantid.dataobjects.Workspace2D)
        raw workspace (histogram) for data and monitor
    dark_current: ~mantid.dataobjects.Workspace2D
        histogram workspace containing the dark current measurement
    flux_method: str
        Method for flux normalization. Either 'monitor', or 'time'.
    flux: str
        if ``flux_method`` is proton charge, then path to file containing the
        wavelength distribution of the neutron flux. If ``flux method`` is
        monitor, then path to file containing the flux-to-monitor ratios.
        if ``flux_method`` is time, then pass one log entry name such
        as ``duration`` or leave it as :py:obj:`None` for automatic log search.
    mask_ws: ~mantid.dataobjects.Workspace2D
        Mask workspace
    mask_panel: str
        Either 'front' or 'back' to mask whole front or back panel.
    mask_btp: dict
        Additional properties to Mantid's MaskBTP algorithm
    solid_angle: bool
        Apply the solid angle correction
    sensitivity_workspace: str, ~mantid.api.MatrixWorkspace
        workspace containing previously calculated sensitivity correction. This
        overrides the sensitivity_filename if both are provided.
    output_workspace: str
        The output workspace name. If None will create data.name()+output_suffix

    Returns
    -------
    ~mantid.dataobjects.Workspace2D
        Reference to the processed workspace
    """
    if not output_workspace:
        output_workspace = str(data.data)
        output_workspace = (
            output_workspace.replace("_raw_histo", "") + "_processed_histo"
        )

    mtd[str(data.data)].clone(
        OutputWorkspace=output_workspace
    )  # name gets into workspace

    # Dark current
    if dark_current is not None and dark_current.data is not None:
        subtract_dark_current(output_workspace, dark_current.data)

    # Normalization
    if flux_method is not None:
        kw = dict(method=flux_method, output_workspace=output_workspace)
        if flux_method == "monitor":
            kw["monitor_workspace"] = data.monitor
        normalize_by_flux(output_workspace, flux, **kw)

    # Additional masks
    if mask_btp is None:
        mask_btp = dict()
    apply_mask(output_workspace, panel=mask_panel, mask=mask_ws, **mask_btp)

    # Solid angle
    if solid_angle:
        solid_angle_correction(output_workspace)

    # Sensitivity
    if sensitivity_workspace is not None:
        apply_sensitivity_correction(
            output_workspace, sensitivity_workspace=sensitivity_workspace
        )

    return mtd[output_workspace]


# NOTE: transformed from block of codes inside reduce_single_configuration
#       for calculating transmission
def process_transmission(
    transmission_ws,
    empty_trans_ws,
    transmission_radius,
    sensitivity_ws,
    flux_method,
    flux,
    prefix,
    type_name,
    output_dir,
    output_file_name,
):
    # sample transmission
    processed_transmission_dict = {}  # for output log
    raw_transmission_dict = {}  # for output log

    if transmission_ws.data is not None and empty_trans_ws is not None:
        # process transition workspace from raw
        processed_trans_ws_name = (
            f"{prefix}_{type_name}_trans"  # type_name: sample/background
        )
        processed_trans_ws = prepare_data_workspaces(
            transmission_ws,
            flux_method=flux_method,
            flux=flux,
            solid_angle=False,
            sensitivity_workspace=sensitivity_ws,
            output_workspace=processed_trans_ws_name,
        )
        # calculate transmission with fit function (default) Formula=a*x+b'
        calculated_trans_ws = calculate_transmission(
            processed_trans_ws,
            empty_trans_ws,
            radius=transmission_radius,
            radius_unit="mm",
        )
        print(f"{type_name} transmission =", calculated_trans_ws.extractY()[0, 0])

        # optionally save
        if output_dir:
            # save calculated transmission
            transmission_filename = os.path.join(
                output_dir, f"{output_file_name}_trans.txt"
            )
            SaveAscii(calculated_trans_ws, Filename=transmission_filename)
            # Prepare result for drtsans.savereductionlog
            processed_transmission_dict["value"] = calculated_trans_ws.extractY()
            processed_transmission_dict["error"] = calculated_trans_ws.extractE()
            processed_transmission_dict["wavelengths"] = calculated_trans_ws.extractX()

            # Prepare result for drtsans.savereductionlog including raw sample transmission
            sample_trans_raw_ws = calculate_transmission(
                processed_trans_ws,
                empty_trans_ws,
                radius=transmission_radius,
                radius_unit="mm",
                fit_function="",
            )

            raw_tr_fn = os.path.join(output_dir, f"{output_file_name}_raw_trans.txt")
            SaveAscii(sample_trans_raw_ws, Filename=raw_tr_fn)
            # Prepare result for drtsans.savereductionlog
            raw_transmission_dict["value"] = sample_trans_raw_ws.extractY()
            raw_transmission_dict["error"] = sample_trans_raw_ws.extractE()
            raw_transmission_dict["wavelengths"] = sample_trans_raw_ws.extractX()
    else:
        calculated_trans_ws = None

    return calculated_trans_ws, processed_transmission_dict, raw_transmission_dict


def bin_i_with_correction(
    iq1d_in_frames,
    iq2d_in_frames,
    wl_frame,
    weighted_errors,
    user_qmin,
    user_qmax,
    num_x_bins,
    num_y_bins,
    num_q1d_bins,
    num_q1d_bins_per_decade,
    decade_on_center,
    bin1d_type,
    log_binning,
    annular_bin,
    wedges,
    symmetric_wedges,
    incoherence_correction_setup,
    iq1d_elastic_ref_fr,
    iq2d_elastic_ref_fr,
    raw_name,
    output_dir,
    output_filename="",
):
    """Bin I(Q) in 1D and 2D with the option to do inelastic incoherent correction"""

    if incoherence_correction_setup.do_correction:
        # Define qmin and qmax for this frame
        if user_qmin is None:
            qmin = iq1d_in_frames[wl_frame].mod_q.min()
        else:
            qmin = user_qmin
        if user_qmax is None:
            qmax = iq1d_in_frames[wl_frame].mod_q.max()
        else:
            qmax = user_qmax

        # Determine qxrange and qyrange for this frame
        qx_min = np.min(iq2d_in_frames[wl_frame].qx)
        qx_max = np.max(iq2d_in_frames[wl_frame].qx)
        qxrange = qx_min, qx_max

        qy_min = np.min(iq2d_in_frames[wl_frame].qy)
        qy_max = np.max(iq2d_in_frames[wl_frame].qy)
        qyrange = qy_min, qy_max

        # Bin I(Q1D, wl) and I(Q2D, wl) in Q and (Qx, Qy) space respectively but not wavelength
        iq2d_main_wl, iq1d_main_wl = bin_all(
            iq2d_in_frames[wl_frame],
            iq1d_in_frames[wl_frame],
            num_x_bins,
            num_y_bins,
            n1dbins=num_q1d_bins,
            n1dbins_per_decade=num_q1d_bins_per_decade,
            decade_on_center=decade_on_center,
            bin1d_type=bin1d_type,
            log_scale=log_binning,
            qmin=qmin,
            qmax=qmax,
            qxrange=qxrange,
            qyrange=qyrange,
            annular_angle_bin=annular_bin,
            wedges=wedges,
            symmetric_wedges=symmetric_wedges,
            error_weighted=weighted_errors,
            n_wavelength_bin=None,
        )
        # Check due to functional limitation
        assert isinstance(
            iq1d_main_wl, list
        ), f"Output I(Q) must be a list but not a {type(iq1d_main_wl)}"
        if len(iq1d_main_wl) != 1:
            raise NotImplementedError(
                f"Not expected that there are more than 1 IQmod main but "
                f"{len(iq1d_main_wl)}"
            )

        # Bin elastic reference run
        if iq1d_elastic_ref_fr or iq2d_elastic_ref_fr:
            # bin the reference elastic runs of the current frame
            iq2d_elastic_wl, iq1d_elastic_wl = bin_all(
                iq2d_elastic_ref_fr[wl_frame],
                iq1d_elastic_ref_fr[wl_frame],
                num_x_bins,
                num_y_bins,
                n1dbins=num_q1d_bins,
                n1dbins_per_decade=num_q1d_bins_per_decade,
                decade_on_center=decade_on_center,
                bin1d_type=bin1d_type,
                log_scale=log_binning,
                qmin=qmin,
                qmax=qmax,
                qxrange=qxrange,
                qyrange=qyrange,
                annular_angle_bin=annular_bin,
                wedges=wedges,
                symmetric_wedges=symmetric_wedges,
                error_weighted=weighted_errors,
                n_wavelength_bin=None,
            )
            if (iq1d_elastic_ref_fr):
                if len(iq1d_elastic_wl) != 1:
                    raise NotImplementedError(
                        "Not expected that there are more than 1 IQmod of "
                        "elastic reference run."
                    )
                # normalization
                iq1d_wl, k_vec, k_error_vec = normalize_by_elastic_reference(
                    iq1d_main_wl[0], iq1d_elastic_wl[0]
                )
                iq1d_main_wl[0] = iq1d_wl
                # write
                run_number = os.path.basename(
                    str(incoherence_correction_setup.elastic_reference.run_number)
                ).split(".")[0]
                save_k_vector(
                    iq1d_wl.wavelength,
                    k_vec,
                    k_error_vec,
                    path=os.path.join(output_dir, f"k_{run_number}.dat"),
                )
            else:
                # 2D normalization
                iq2d_wl, k_vec, k_error_vec = normalize_by_elastic_reference2D(
                    iq2d_main_wl[0], iq2d_elastic_wl[0]
                )
                iq2d_main_wl[0] = iq2d_wl

                save_k_vector(
                    iq2d_wl.wavelength,
                    k_vec,
                    k_error_vec,
                    path=os.path.join(output_dir, f"k_{run_number}.dat"),
                )

        # 1D correction
        b_file_prefix = f"{raw_name}_frame_{wl_frame}"
        corrected_iq1d = do_inelastic_incoherence_correction_q1d(
            iq1d_main_wl[0],
            incoherence_correction_setup,
            b_file_prefix,
            output_dir,
            output_filename,
        )

        # 2D correction
        corrected_iq2d = do_inelastic_incoherence_correction_q2d(
            iq2d_main_wl,
            incoherence_correction_setup,
            b_file_prefix,
            output_dir,
            output_filename,
        )

        # Be finite
        finite_iq1d = corrected_iq1d.be_finite()
        finite_iq2d = corrected_iq2d.be_finite()
        # Bin binned I(Q1D, wl) and and binned I(Q2D, wl) in wavelength space
        assert len(iq1d_main_wl) == 1, (
            f"It is assumed that output I(Q) list contains 1 I(Q)"
            f" but not {len(iq1d_main_wl)}"
        )
    else:
        finite_iq2d = iq2d_in_frames[wl_frame]
        finite_iq1d = iq1d_in_frames[wl_frame]
        qmin = user_qmin
        qmax = user_qmax
    # END-IF-ELSE

    iq2d_main_out, iq1d_main_out = bin_all(
        finite_iq2d,
        finite_iq1d,
        num_x_bins,
        num_y_bins,
        n1dbins=num_q1d_bins,
        n1dbins_per_decade=num_q1d_bins_per_decade,
        decade_on_center=decade_on_center,
        bin1d_type=bin1d_type,
        log_scale=log_binning,
        qmin=qmin,
        qmax=qmax,
        qxrange=None,
        qyrange=None,
        annular_angle_bin=annular_bin,
        wedges=wedges,
        symmetric_wedges=symmetric_wedges,
        error_weighted=weighted_errors,
    )

    return iq2d_main_out, iq1d_main_out


def remove_workspaces(
    reduction_config: Dict,
    instrument_name: str,
    prefix: str,
    sample_run_number,
    center_run_number,
    extra_run_numbers: List,
):
    """Helping method to remove existing workspaces"""
    from drtsans.instruments import extract_run_number  # noqa E402
    from drtsans.path import registered_workspace  # noqa E402

    # In the future this should be made optional
    ws_to_remove = [
        f"{prefix}_{instrument_name}_{run_number}_raw_histo"
        for run_number in extra_run_numbers
    ]
    # List special workspaces and workspace groups
    ws_to_remove.append(
        f"{prefix}_{instrument_name}_{sample_run_number}_raw_histo_slice_group"
    )
    ws_to_remove.append(f"{prefix}_{instrument_name}_{center_run_number}_raw_events")
    ws_to_remove.append(f"{prefix}_sensitivity")
    ws_to_remove.append(f"{prefix}_mask")
    if reduction_config["darkFileName"]:
        run_number = extract_run_number(reduction_config["darkFileName"])
        ws_to_remove.append(f"{prefix}_{instrument_name}_{run_number}_raw_histo")
    for ws_name in ws_to_remove:
        # Remove existing workspaces, this is to guarantee that all the data is loaded correctly
        if registered_workspace(ws_name):
            mtd.remove(ws_name)
