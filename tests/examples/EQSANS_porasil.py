"""
  EQSANS example for the legacy reduction
"""
# flake8: noqa
import os

from mantid.simpleapi import mtd, Load, ExtractMask, SaveAscii
from reduction_workflow.instruments.sans.sns_command_interface import *  # noqa: F403


mtd.clear()
mask60_ws4m = Load(
    Filename="/SNS/EQSANS/shared/NeXusFiles/EQSANS/2017B_mp/beamstop60_mask_4m.nxs"
)
ws604m, masked60_detectors4m = ExtractMask(
    InputWorkspace=mask60_ws4m, OutputWorkspace="__edited_mask604m"
)
detector_ids604m = [int(i) for i in masked60_detectors4m]

EQSANS()
UseConfig(True)
UseConfigTOFTailsCutoff(False)
UseConfigMask(True)

SetTOFTailsCutoff(low_cut=500.0, high_cut=2000.0)

SolidAngle(detector_tubes=True)
DarkCurrent("/SNS/EQSANS/shared/NeXusFiles/EQSANS/2017B_mp/EQSANS_86275.nxs.h5")
TotalChargeNormalization(
    beam_file="/SNS/EQSANS/shared/instrument_configuration/bl6_flux_at_sample"
)
SetAbsoluteScale(0.0208641883)
AzimuthalAverage(n_bins=100, n_subpix=1, log_binning=True)
MaskDetectors(detector_ids604m)
OutputPath("/tmp")

ReductionSingleton().reduction_properties["SampleOffset"] = 340
ReductionSingleton().reduction_properties["DetectorOffset"] = 0
Resolution(sample_aperture_diameter=10)
PerformFlightPathCorrection(True)
DirectBeamCenter("88973")
SensitivityCorrection(
    "/SNS/EQSANS/shared/NeXusFiles/EQSANS/2017A_mp/Sensitivity_patched_thinPMMA_4m_79165_event.nxs",
    min_sensitivity=0.5,
    max_sensitivity=1.5,
    use_sample_dc=True,
)
DivideByThickness(0.1)
DirectBeamTransmission("88975", "88973", beam_radius=5)
ThetaDependentTransmission(True)
AppendDataFile(["88980"])
CombineTransmissionFits(False)
Background(["88979"])
BckDirectBeamTransmission("88974", "88973", beam_radius=5)
BckThetaDependentTransmission(True)
BckCombineTransmissionFits(False)
Reduce()

SaveAscii(
    InputWorkspace="88980_frame1_Iq",
    WriteSpectrumID=False,
    Filename=os.path.join(os.path.expanduser("~"), "EQSANS_88980_frame1_iq_ref.txt"),
)
SaveAscii(
    InputWorkspace="88980_frame2_Iq",
    WriteSpectrumID=False,
    Filename=os.path.join(os.path.expanduser("~"), "EQSANS_88980_frame2_iq_ref.txt"),
)
