# This module contains workflow algorithms and methods correct intensity and error of
# sample and background data accounting wavelength-dependent incoherent inelastic scattering.
# The workflow algorithms will be directly called by eqsans.api.
import os.path
from drtsans.dataobjects import IQmod, IQazimuthal
from collections import namedtuple
from drtsans.iq import bin_all  # noqa E402
from typing import Union
from drtsans.tof.eqsans.incoherence_correction_1d import (
    correct_incoherence_inelastic_1d,
    CorrectedIQ1D,
)
from drtsans.tof.eqsans.incoherence_correction_2d import (
    correct_incoherence_inelastic_2d,
    CorrectedIQ2D,
)

"""
Workflow to correct intensities and errors accounting wavelength dependent
incoherent inelastic scattering

Case 1:  If the elastic reference run is NOT specified, the
 - sample
 - bkgd
will be
 * not be corrected in step 2 described in story #689.
 * corrected in step 3 and step 4 described in story #689.

Case 2: If an elastic reference run is specified.
        Step 2 in story #689 will be executed.

  Step 2:
    a) parse additional configuration

        elastic_ref = reduction_input["elasticReference"]["runNumber"]
        elastic_ref_trans = reduction_input["elasticReferenceTrans"]["runNumber"]
        elastic_ref_trans_value = reduction_input["elasticReferenceTrans"]["value"]
        elastic_ref_bkgd = None  # in story 689
        elastic_ref_bkgd_trans = None
        elastic_ref_bkgd_trans_value = None

    b) load elastic reference run, reference transition run, background run and background transition run
       This shall be implemented in eqsans.api.load_all_files()

    c) calculate K and delta K

    d) before bin_all() called in method reduce_single_configuration
       > # Bin 1D and 2D
       > iq2d_main_out, iq1d_main_out = bin_all(iq2d_main_in_fr[wl_frame], iq1d_main_in_fr[wl_frame],
       > ... ...

       1. normalize sample and bkgd run
       2. execute step 3 and step 4
"""


# TODO - when python is upgraded to 3.7+, this class shall be wrapped as dataclass
class CorrectionConfiguration:
    """
    A data class/structure to hold the parameters configured to do incoherence/inelastic
    scattering correction

    Parameters
    ----------
    do_correction: bool
        if to do correction or not
    select_minimum_incoherence: bool
        flag to determine correction B by minimum incoherence
    intensity_weighted: bool
       flag to determine if the b factor is calculated in a weighted function by intensity
    qmin_index: float
        optional, manually set the qmin used for incoherent calculation
    qmax_index: float
        optional, manually set the qmax used for incoherent calculation
    factor: float
        optional, automatically determine the qmin qmax by checking the intensity profile
    """

    def __init__(self, do_correction=False, select_min_incoherence=False,
                 select_intensityweighted=False, qmin=None, qmax=None, factor=None):
        self._do_correction = do_correction
        self._select_min_incoherence = select_min_incoherence
        self._select_intensityweighted = select_intensityweighted
        self._qmin = qmin
        self._qmax = qmax
        self._factor = factor
        self._elastic_ref_run_setup = None
        self._sample_thickness = 1  # mm

    def __str__(self):
        if self._do_correction:
            output = (
                f"Do correction: select min incoherence = {self._select_min_incoherence}, "
                f"thickness = {self._sample_thickness}, "
                f"select_intensityweighted = {self._select_intensityweighted}, "
                f"qmin = {self._qmin}, q_max = {self._qmax}, factor = {self._factor}"
            )
        else:
            output = "No correction"

        return output

    @property
    def do_correction(self):
        return self._do_correction

    @property
    def select_min_incoherence(self):
        return self._select_min_incoherence

    @select_min_incoherence.setter
    def select_min_incoherence(self, flag):
        self._select_min_incoherence = flag

    @property
    def select_intensityweighted(self):
        return self._select_intensityweighted

    @property
    def qmin(self):
        return self._qmin

    @property
    def qmax(self):
        return self._qmax

    @property
    def factor(self):
        return self._factor

    @property
    def elastic_reference(self):
        """elastic scattering normalization reference run and background run"""
        return self._elastic_ref_run_setup

    @property
    def sample_thickness(self):
        return self._sample_thickness

    def set_elastic_reference(self, reference_run_setup):
        """Set elastic reference run reduction setup

        Parameters
        ----------
        reference_run_setup: ElasticReferenceRunSetup
            reduction setup

        """
        assert isinstance(reference_run_setup, ElasticReferenceRunSetup)
        self._elastic_ref_run_setup = reference_run_setup


# TODO - when python is upgraded to 3.7+, this class shall be wrapped as dataclass
class ElasticReferenceRunSetup:
    """
    A data class/structure to hold the reference run
    """

    def __init__(
        self,
        ref_run_number: Union[int, str],
        thickness: float,
        trans_run_number: Union[None, Union[str, int]] = None,
        trans_value: Union[None, float] = None,
    ):
        self.run_number = ref_run_number
        self.thickness = thickness
        self.transmission_run_number = trans_run_number
        self.transmission_value = trans_value

        # sanity check
        if trans_run_number is None and trans_value is None:
            raise RuntimeError(
                "Either transmission run or transmission value shall be given."
            )
        elif trans_run_number and trans_value:
            raise RuntimeError(
                "Either transmission run or transmission value can be given, but "
                "not both"
            )

        # Background
        self.background_run_number = None
        self.background_transmission_run_number = None
        self.background_transmission_value = None

    def set_background(
        self,
        run_number: Union[int, str],
        trans_run_number: Union[None, Union[int, str]] = None,
        trans_value: Union[None, float] = None,
    ):
        """Set elastic reference background run setup"""
        self.background_run_number = run_number
        self.background_transmission_run_number = trans_run_number
        self.background_transmission_value = trans_value

        if trans_run_number is None and trans_value is None:
            raise RuntimeError(
                "Either background transmission run or transmission value shall be given."
            )
        elif trans_run_number and trans_value:
            raise RuntimeError(
                "Either background transmission run or transmission value can be given, but "
                "not both"
            )


def parse_correction_config(reduction_config):
    """Parse correction configuration from reduction configuration (top level)

    Parameters
    ----------
    reduction_config: ~dict
        reduction configuration from JSON

    Returns
    -------
    CorrectionConfiguration
        incoherence/inelastic scattering correction configuration

    """
    # an exception case
    if "configuration" not in reduction_config:
        _config = CorrectionConfiguration(False)
    else:
        # properly configured
        run_config = reduction_config["configuration"]

        # incoherence inelastic correction setup: basic
        do_correction = run_config.get("fitInelasticIncoh", False)
        select_min_incoherence = run_config.get("selectMinIncoh", False)
        select_intensityweighted = run_config.get("incohfit_intensityweighted", False)
        qmin = run_config.get("incohfit_qmin")
        qmax = run_config.get("incohfit_qmax")
        factor = run_config.get("incohfit_factor")

        _config = CorrectionConfiguration(do_correction, select_min_incoherence,
                                          select_intensityweighted, qmin, qmax, factor)

        # Optional elastic normalization
        elastic_ref_json = run_config.get("elasticReference")
        if elastic_ref_json:
            elastic_ref_run = elastic_ref_json.get("runNumber")
            if elastic_ref_run is not None and elastic_ref_run != "":
                # only set up elastic reference after checking run number
                try:
                    elastic_ref_trans_run = elastic_ref_json["transmission"].get(
                        "runNumber"
                    )
                    elastic_ref_trans_value = elastic_ref_json["transmission"].get(
                        "value"
                    )
                    elastic_ref_thickness = float(elastic_ref_json.get("thickness"))
                    elastic_ref_config = ElasticReferenceRunSetup(
                        elastic_ref_run,
                        elastic_ref_thickness,
                        elastic_ref_trans_run,
                        elastic_ref_trans_value,
                    )
                    # background runs
                    elastic_ref_bkgd = run_config.get("elasticReferenceBkgd")
                    if elastic_ref_bkgd:
                        elastic_bkgd_run = elastic_ref_bkgd.get("runNumber")
                        # only set up elastic reference background after checking run number
                        if elastic_bkgd_run is not None and elastic_bkgd_run != "":
                            elastic_bkgd_trans_run = elastic_ref_bkgd[
                                "transmission"
                            ].get("runNumber")
                            elastic_bkgd_trans_value = elastic_ref_bkgd[
                                "transmission"
                            ].get("value")
                            elastic_ref_config.set_background(
                                elastic_bkgd_run,
                                elastic_bkgd_trans_run,
                                elastic_bkgd_trans_value,
                            )

                    # Set to configuration
                    _config.set_elastic_reference(elastic_ref_config)
                except IndexError as index_err:
                    raise RuntimeError(
                        f"Invalid JSON for elastic reference run setup: {index_err}"
                    )

    return _config


# Define named tuple for elastic scattering normalization factor
NormFactor = namedtuple("NormFactor", "k k_error p s")


def do_inelastic_incoherence_correction_q1d(
    iq1d: IQmod,
    correction_setup: CorrectionConfiguration,
    prefix: str,
    output_dir: str,
    output_filename: str = "",
) -> IQmod:
    """Do inelastic incoherence correction on 1D data (Q1d)

    Parameters
    ----------
    iq1d: IQmod
        I(Q1D)
    correction_setup: CorrectionConfiguration
        correction configuration
    prefix: str
        prefix for b factor file
    output_dir: str
        output directory for b1d(lambda)
    output_filename: str
        output filename parsed from input configuration file (JSON)

    Returns
    -------
    IQmod

    """
    # type check
    assert isinstance(
        iq1d, IQmod
    ), f"Assuming each element in input is IQmod but not {type(iq1d)}"

    # do inelastic/incoherent correction
    corrected = correct_incoherence_inelastic_1d(
        iq1d, correction_setup.select_min_incoherence,
        correction_setup.select_intensityweighted,
        correction_setup.qmin,
        correction_setup.qmax,
        correction_setup.factor
    )

    # save file
    save_b_factor(
        corrected,
        os.path.join(output_dir, f"{output_filename}_inelastic_b1d_{prefix}.dat"),
    )

    return corrected.iq1d


def do_inelastic_incoherence_correction_q2d(
    iq2d: IQazimuthal,
    correction_setup: CorrectionConfiguration,
    prefix: Union[int, str],
    output_dir: str,
    output_filename: str = "",
) -> IQazimuthal:
    # type check
    assert isinstance(
        iq2d, IQazimuthal
    ), f"iq2d must be IQazimuthal but not {type(iq2d)}"

    # apply the correction to each
    corrected = correct_incoherence_inelastic_2d(
        iq2d, correction_setup.select_min_incoherence
    )

    # save file
    save_b_factor(
        corrected,
        os.path.join(output_dir, f"{output_filename}_inelastic_b2d_{prefix}.dat"),
    )

    return corrected.iq2d


def save_b_factor(i_of_q: Union[CorrectedIQ1D, CorrectedIQ2D], path: str) -> None:
    header = "lambda,b,delta_b\n"
    # grab the IQmod or IQazimuthal wavelength
    wavelength = i_of_q[0].wavelength
    wave_str = map(str, wavelength)
    b_str = map(str, i_of_q.b_factor)
    b_e_str = map(str, i_of_q.b_error)
    # merge items (all are appropriately ordered, so zip is usable)
    output = "\n".join(map(",".join, zip(wave_str, b_str, b_e_str)))
    with open(path, "w", encoding="utf-8") as save_file:
        save_file.write(header)
        save_file.write(output)


def save_k_vector(wavelength_vec, k_vec, delta_k_vec, path: str) -> None:
    """Save K vector from elastic scattering normalization"""
    header = "lambda,k,delta_k\n"
    # grab the IQmod or IQazimuthal wavelength
    wave_str = map(str, wavelength_vec)
    k_str = map(str, k_vec)
    k_e_str = map(str, delta_k_vec)
    # merge items (all are appropriately ordered, so zip is usable)
    output = "\n".join(map(",".join, zip(wave_str, k_str, k_e_str)))
    with open(path, "w", encoding="utf-8") as save_file:
        save_file.write(header)
        save_file.write(output)
