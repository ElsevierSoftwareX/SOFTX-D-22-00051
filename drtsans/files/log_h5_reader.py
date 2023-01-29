import h5py
import numpy as np
import os
from matplotlib import pyplot as plt

__all__ = ["verify_cg2_reduction_results"]


def get_iq1d(log_file_name):
    """Get I(Q) from output GP-SANS log file

    Parameters
    ----------
    log_file_name: str
        log file name

    Returns
    -------
    tuple
        numpy 1D array for Q, numpy 1D array for intensity

    """
    # Open file and entry
    log_h5 = h5py.File(log_file_name, "r")

    if "_slice_1" in log_h5:
        data_entry = log_h5["_slice_1"]["main"]
    else:
        data_entry = log_h5["main"]

    # Get data
    iq1d_entry = data_entry["I(Q)"]

    # Get data with a copy
    vec_q = np.copy(iq1d_entry["Q"][()])
    vec_i = np.copy(iq1d_entry["I"][()])

    # close file
    log_h5.close()

    return vec_q, vec_i


def compare_reduced_iq(test_log_file, gold_log_file, title: str, prefix: str):
    """Compare I(Q) from reduced file and gold file for GPSANS

    Parameters
    ----------
    test_log_file: str
        log file from test
    gold_log_file: str
        log file for expected result
    title: str
        title of output figure
    prefix: str
        prefix to all the generated output file
    """
    # Get the main data
    test_q_vec, test_intensity_vec = get_iq1d(test_log_file)
    gold_q_vec, gold_intensity_vec = get_iq1d(gold_log_file)

    # Verify result
    test_exception = None
    diff_q = False
    for vec_name, test_vec, gold_vec, abs_tol in [
        ("q", test_q_vec, gold_q_vec, 1e-4),
        ("intensity", test_intensity_vec, gold_intensity_vec, 1e-7),
    ]:
        try:
            np.testing.assert_allclose(test_vec, gold_vec, atol=abs_tol)
        except AssertionError as assert_err:
            test_exception = assert_err
            if vec_name == "q":
                diff_q = True
            break

    # Output error message
    if test_exception:
        base_name = f'{prefix}{os.path.basename(test_log_file).split(".")[0]}'
        report_difference(
            (test_q_vec, test_intensity_vec),
            (gold_q_vec, gold_intensity_vec),
            title,
            diff_q,
            base_name,
        )

    if test_exception:
        raise test_exception


def report_difference(
    test_data, gold_data, title: str, is_q_diff: bool, base_file_name: str
):
    test_q_vec, test_intensity_vec = test_data
    gold_q_vec, gold_intensity_vec = gold_data

    # plot I(Q)
    png_name = f"{base_file_name}_compare_iq.png"
    plot_data(
        [
            (test_q_vec, test_intensity_vec, "red", "Test I(Q)"),
            (gold_q_vec, gold_intensity_vec, "black", "Gold I(Q)"),
        ],
        "log",
        title,
        png_name,
    )

    # plot difference of I(Q)
    if is_q_diff:
        # different Q
        png_name = f"{base_file_name}_diff_q.png"
        if test_q_vec.shape == gold_q_vec.shape:
            # same Q shape
            diff_q_array = test_q_vec - gold_q_vec
            plot_data(
                [(np.arange(diff_q_array.shape[0]), diff_q_array, "red", "diff(Q)")],
                None,
                "Difference in Q",
                png_name,
            )
        else:
            # different Q shape
            plot_data(
                [
                    (np.arange(test_q_vec.shape[0]), test_q_vec, "red", "Q (test)"),
                    (
                        np.arange(gold_q_vec.shape[0]),
                        gold_q_vec,
                        "black",
                        "Q (expected)",
                    ),
                ],
                None,
                "Difference in Q",
                png_name,
            )
    else:
        # different I(Q)
        diff_iq_array = test_intensity_vec - gold_intensity_vec
        png_name = f"{base_file_name}_diff_iq.png"
        plot_data(
            [(test_q_vec, diff_iq_array, "red", "I(Q)_test - I(Q)_gold")],
            None,
            "Difference in I(Q)",
            png_name,
        )


def plot_data(plot_info_list, y_scale, title, fig_name):

    # clear the canvas
    plt.cla()

    # plot
    for x_array, y_array, color, label in plot_info_list:
        plt.plot(x_array, y_array, color=color, label=label)

    # other setup
    plt.legend()
    plt.title(title)
    # scale
    if y_scale is not None:
        plt.yscale(y_scale)

    plt.savefig(fig_name)
    plt.close()
    return
    #
    # # Plot I(Q)
    # plt.cla()
    # plt.plot(test_q_vec, test_intensity_vec, color='red', label='Test I(Q)')
    # plt.plot(gold_q_vec, gold_intensity_vec, color='black', label='Gold I(Q)')
    # plt.legend()
    # plt.title(title)
    # plt.yscale('log')
    # # defaults and set outut png file name
    # if base_name is None:
    #     base_name = 'compare'
    # if test_log_file is None:
    #     test_log_file = 'iq'
    # out_name = base_name + '_' + os.path.basename(test_log_file).split('.')[0] + '.png'
    # plt.savefig(out_name)


def verify_cg2_reduction_results(sample_names, output_dir, gold_path, title, prefix):
    """Verify reduction result for GPSANS

    Parameters
    ----------
    sample_names: list
        list of names of samples
    output_dir: str
        output directory
    gold_path: str
        path to the gold file
    title: str
        title of output figure
    prefix: str
        prefix for output png file in order not to confusing various error outputs

    """
    unmatched_errors = ""

    for sample_name in sample_names:
        # output log file name
        output_log_file = os.path.join(
            output_dir, "{}_reduction_log.hdf".format(sample_name)
        )
        assert os.path.exists(output_log_file), "Output {} cannot be found".format(
            output_log_file
        )
        # gold file
        gold_log_file = os.path.join(
            gold_path, "{}_reduction_log.hdf".format(sample_name)
        )
        assert os.path.exists(gold_path), "Gold file {} cannot be found".format(
            gold_log_file
        )
        # compare
        title_i = "{}: {}".format(sample_name, title)
        try:
            compare_reduced_iq(output_log_file, gold_log_file, title_i, prefix)
        except AssertionError as unmatched_error:
            unmatched_errors = (
                "Testing output {} is different from gold result {}:\n{}"
                "".format(output_log_file, gold_log_file, unmatched_error)
            )
    # END-FOR

    # raise error for all
    if unmatched_errors != "":
        raise AssertionError(unmatched_errors)
