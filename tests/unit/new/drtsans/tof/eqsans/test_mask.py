import pytest
from mantid.dataobjects import MaskWorkspace
from mantid.simpleapi import (
    LoadEmptyInstrument,
    ClearMaskFlag,
    MaskBTP,
    CompareWorkspaces,
    ExtractMask,
)
from drtsans.settings import unique_workspace_dundername as uwd
from drtsans.mask_utils import apply_mask


def test_apply_mask():
    w = LoadEmptyInstrument(InstrumentName="EQ-SANS", OutputWorkspace=uwd())
    apply_mask(w, panel="front", Bank="25-48", Pixel="1-10")
    m = ExtractMask(w, OutputWorkspace=uwd()).OutputWorkspace
    assert isinstance(m, MaskWorkspace)
    ClearMaskFlag(w)
    MaskBTP(Workspace=w, Components="front-panel")
    MaskBTP(Workspace=w, Bank="25-48", Pixel="1-10")
    m2 = ExtractMask(w, OutputWorkspace=uwd()).OutputWorkspace
    assert CompareWorkspaces(m, m2).Result
    #
    # Mask back panel
    #
    ClearMaskFlag(w)
    apply_mask(w, panel="back")
    m = ExtractMask(w, OutputWorkspace=uwd()).OutputWorkspace
    ClearMaskFlag(w)
    MaskBTP(Workspace=w, Components="back-panel")
    m2 = ExtractMask(w, OutputWorkspace=uwd()).OutputWorkspace
    assert CompareWorkspaces(m, m2).Result


if __name__ == "__main__":
    pytest.main([__file__])
