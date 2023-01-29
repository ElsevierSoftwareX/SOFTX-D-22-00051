import pytest
from tempfile import NamedTemporaryFile
from mantid.dataobjects import MaskWorkspace
from mantid.simpleapi import (
    LoadEmptyInstrument,
    ClearMaskFlag,
    MaskBTP,
    CompareWorkspaces,
    SaveMask,
    ExtractMask,
)
from drtsans.settings import unique_workspace_dundername as uwd
from drtsans.mask_utils import apply_mask


def test_apply_mask():
    w = LoadEmptyInstrument(InstrumentName="EQ-SANS", OutputWorkspace=uwd())
    #
    # using MaskBPT with a Component
    #
    apply_mask(w, Components="back-panel")
    m = ExtractMask(w, OutputWorkspace=uwd()).OutputWorkspace
    assert isinstance(m, MaskWorkspace)
    ClearMaskFlag(w)
    MaskBTP(Workspace=w, Components="back-panel")
    m2 = ExtractMask(w, OutputWorkspace=uwd()).OutputWorkspace
    assert CompareWorkspaces(m, m2).Result
    ClearMaskFlag(w)
    apply_mask(w, Bank="1-24", Pixel="1-10")
    m = ExtractMask(w, OutputWorkspace=uwd()).OutputWorkspace
    ClearMaskFlag(w)
    MaskBTP(Workspace=w, Bank="1-24", Pixel="1-10")
    m2 = ExtractMask(w, OutputWorkspace=m2.name()).OutputWorkspace
    assert CompareWorkspaces(m, m2).Result
    #
    # using a MaskWorkspace
    #
    ClearMaskFlag(w)
    apply_mask(w, mask=m)
    m2 = ExtractMask(w, OutputWorkspace=m2.name()).OutputWorkspace
    assert CompareWorkspaces(m, m2).Result
    #
    # using a File
    #
    ClearMaskFlag(w)
    MaskBTP(Workspace=w, Bank="1-24", Pixel="1-10")
    m = ExtractMask(w, OutputWorkspace=m.name()).OutputWorkspace
    with NamedTemporaryFile(suffix=".xml") as f:
        SaveMask(m, f.name)
        apply_mask(w, mask=f.name)
        m2 = ExtractMask(w, OutputWorkspace=m2.name()).OutputWorkspace
        assert CompareWorkspaces(m, m2).Result
    #
    # using a MaskWorkspace and MaskBTP
    #
    ClearMaskFlag(w)
    MaskBTP(Workspace=w, Components="front-panel")
    m2 = ExtractMask(w, OutputWorkspace=m2.name()).OutputWorkspace
    apply_mask(w, mask=m2, Bank="25-48", Pixel="1-10")
    m = ExtractMask(w, OutputWorkspace=m.name()).OutputWorkspace
    ClearMaskFlag(w)
    MaskBTP(Workspace=w, Components="front-panel")
    MaskBTP(Workspace=w, Bank="25-48", Pixel="1-10")
    m2 = ExtractMask(w, OutputWorkspace=m2.name()).OutputWorkspace
    assert CompareWorkspaces(m, m2).Result


if __name__ == "__main__":
    pytest.main([__file__])
