import pytest
import numpy as np

r""" Links to Mantid algorithms
CreateSingleValuedWorkspace <https://docs.mantidproject.org/nightly/algorithms/CreateSingleValuedWorkspace-v1.html>
"""
from mantid.simpleapi import CreateSingleValuedWorkspace
from mantid.simpleapi import DeleteWorkspace
from mantid.api import AnalysisDataService

r""" Links to drtsans imports
standard_sample_scaling <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/absolute_units.py>
unique_workspace_name <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/settings.py>
"""
from drtsans.absolute_units import standard_sample_scaling  # noqa: E402
from drtsans.settings import unique_workspace_name  # noqa: E402


def test_standard_sample_measurement():
    r"""
    Tests normalization with a calibrated standard sample as described in the master document
    section 12.3
    dev - Steven Hahn <hahnse@ornl.gov>
    SME - Changwoo Do <doc1@ornl.gov>

    **Mantid algorithms used:**
    :ref:`CreateSingleValuedWorkspace <algm-CreateSingleValuedWorkspace-v1>`,
    <https://docs.mantidproject.org/nightly/algorithms/CreateSingleValuedWorkspace-v1.html>
    :ref:`DeleteWorkspace <algm-DeleteWorksapce-v1>`,
    <https://docs.mantidproject.org/nightly/algorithms/DeleteWorkspace-v1.html>
    :ref:`Divide <algm-Divide-v1>`,
    <https://docs.mantidproject.org/nightly/algorithms/Divide-v1.html>
    :ref:`Multiply <algm-Multiply-v1>`,
    <https://docs.mantidproject.org/nightly/algorithms/Multiply-v1.html>

    **drtsans functions used:**
    ~drtsans.absolute_units.standard_sample_scaling,
    <https://code.ornl.gov/sns-hfir-scse/sans/sans-backend/blob/next/drtsans/absolute_units.py>
    """
    # record temporary worksapce names so that we can clean them up at the end
    wsn_list = []
    #
    F_std = 450.0  # value from supplied test
    F_std_err = 10.0  # value from supplied test
    tmp_wsn = unique_workspace_name()
    wsn_list.append(tmp_wsn)
    F_std_ws = CreateSingleValuedWorkspace(
        DataValue=F_std, ErrorValue=F_std_err, OutputWorkspace=tmp_wsn
    )
    #
    F = 10.0  # value from supplied test
    F_err = 2.0  # value from supplied test
    tmp_wsn = unique_workspace_name()
    wsn_list.append(tmp_wsn)
    F_ws = CreateSingleValuedWorkspace(
        DataValue=F, ErrorValue=F_err, OutputWorkspace=tmp_wsn
    )
    #
    Iq = 100.0  # value from supplied test
    Iq_err = np.sqrt(Iq)
    tmp_wsn = unique_workspace_name()
    wsn_list.append(tmp_wsn)
    Iq_ws = CreateSingleValuedWorkspace(
        DataValue=Iq, ErrorValue=Iq_err, OutputWorkspace=tmp_wsn
    )
    # perform calculation done by function standard_sample_scaling
    Iq_abs = Iq / F * F_std
    # calculate uncertainty as described in the supplied test. Symbolically this is identical to the calculation below,
    # but, due to numerical error, differs both the calculation below and Mantid's result by ~10%
    # err1 = F_std**2 / F**4 * F_err**2 * Iq**2
    # err2 = F_std_err**2 / F**2 * Iq**2
    # err3 = F_std**2 / F**2 * Iq_err*2
    # Iq_abs_err = np.sqrt(err1 + err2 + err3)
    err1 = F_err ** 2 / F ** 2
    err2 = F_std_err ** 2 / F_std ** 2
    err3 = Iq_err ** 2 / Iq ** 2
    Iq_abs_err = Iq / F * F_std * np.sqrt(err1 + err2 + err3)
    # run absolute_units.standard_sample_scaling
    Iq_abs_ws = standard_sample_scaling(Iq_ws, F_ws, F_std_ws)
    # NOTE:
    # the function standard_sample_scaling has a side effect workspace, output_workspace
    # by design, adding it to list
    wsn_list.append("output_workspace")
    # check results
    assert Iq_ws.dataY(0)[0] == pytest.approx(Iq)
    assert Iq_abs_ws.dataY(0)[0] == pytest.approx(Iq_abs)
    assert Iq_abs_ws.dataE(0)[0] == pytest.approx(Iq_abs_err)
    # clean up
    for wsn in wsn_list:
        if wsn in AnalysisDataService.getObjectNames():
            DeleteWorkspace(wsn)
