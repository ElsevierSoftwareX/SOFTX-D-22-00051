import pytest
from os.path import join as pj
from mantid.simpleapi import LoadNexus, SumSpectra, CompareWorkspaces

from drtsans.tof.eqsans import reduce
from drtsans.settings import amend_config, unique_workspace_dundername as uwd


def test_load_w(reference_dir):
    with amend_config(data_dir=reference_dir.new.eqsans):
        _w0 = reduce.load_w(
            "EQSANS_92353",
            output_workspace=uwd(),
            low_tof_clip=500,
            high_tof_clip=2000,
            dw=0.1,
        )
        _w1 = SumSpectra(_w0, OutputWorkspace=_w0.name())
        fn = pj(reference_dir.new.eqsans, "test_reduce", "compare", "ref_load_w.nxs")
        _w2 = LoadNexus(fn, OutputWorkspace=uwd())
        assert CompareWorkspaces(_w1, _w2)


if __name__ == "__main__":
    pytest.main([__file__])
