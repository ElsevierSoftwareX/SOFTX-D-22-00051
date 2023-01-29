# flake8: noqa
"""
    The following is a real-life example of an EQSANS reduction script.
    It uses the current Mantid reduction for EQSANS.
"""
# EQSANS reduction script
# Script automatically generated on Fri Mar  3 12:00:50 2017

import mantid
from mantid.simpleapi import *
from reduction_workflow.instruments.sans.sns_command_interface import *

config = ConfigService.Instance()
config["instrumentName"] = "EQSANS"

mask60_ws4m = Load(
    Filename="/SNS/EQSANS/shared/NeXusFiles/EQSANS/2017B_mp/beamstop60_mask_4m.nxs"
)
ws604m, masked60_detectors4m = ExtractMask(
    InputWorkspace=mask60_ws4m, OutputWorkspace="__edited_mask604m"
)
detector_ids604m = [int(i) for i in masked60_detectors4m]
Mask_BS604m = detector_ids604m

# The various run numbers
# Sample scattering
Sca1 = 88980
# Sample transmission
Tra = 88975

# Background scattering
Bkg1 = 88979
# Background transmission
Btr = 88974

# the last run to reduce
Last = 88980

while Sca1 <= Last:
    FSca1 = str(Sca1)
    FTra = str(Tra)
    FBkg1 = str(Bkg1)
    FBtr = str(Btr)

    EQSANS()
    SolidAngle(detector_tubes=True)
    DarkCurrent("/SNS/EQSANS/shared/NeXusFiles/EQSANS/2017B_mp/EQSANS_86275.nxs.h5")
    TotalChargeNormalization()
    SetAbsoluteScale(0.0208641883)
    AzimuthalAverage(n_bins=100, n_subpix=1, log_binning=True)
    IQxQy(nbins=75)
    MaskDetectors(Mask_BS604m)
    OutputPath("/SNS/EQSANS/IPTS-19800/shared")
    UseConfigTOFTailsCutoff(True)
    UseConfigMask(True)
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
    DirectBeamTransmission(FTra, "88973", beam_radius=5)
    ThetaDependentTransmission(True)
    AppendDataFile([FSca1])
    CombineTransmissionFits(False)
    Background([FBkg1])
    BckDirectBeamTransmission(FBtr, "88973", beam_radius=5)
    BckThetaDependentTransmission(True)
    BckCombineTransmissionFits(False)
    SaveIq(process="None")
    Reduce()

    # this is the code for stitching the data sets, I think
    q_min = [0.04]
    q_max = [0.07]
    IPTS = "19800"
    Filename_lowq = "/SNS/EQSANS/IPTS-" + IPTS + "/shared/%s_frame2_Iq.xml" % str(Sca1)
    Filename_hiq = "/SNS/EQSANS/IPTS-" + IPTS + "/shared/%s_frame1_Iq.xml" % str(Sca1)

    WkSpMerge = "%s_merge" % str(Sca1)
    FileNameOut1 = "/SNS/EQSANS/IPTS-" + IPTS + "/shared/%s_merge.xml" % str(Sca1)
    FileNameOut2 = "/SNS/EQSANS/IPTS-" + IPTS + "/shared/%s_merge.txt" % str(Sca1)

    if os.path.exists(Filename_lowq) and os.path.exists(Filename_hiq):
        try:
            data_list = [str(Filename_lowq), str(Filename_hiq)]
            Stitch(data_list, q_min, q_max, scale=None, save_output=True)

        except (RuntimeError):
            print ""
            print "RuntimeError - the data file " + Filename_lowq + " may not exist, but other issues certainly exist. Check this error message:"
            print "RuntimeError - the data file " + Filename_hiq + " may not exist, but other issues certainly exist. Check this error message:"
            print sys.exc_info()

        try:
            RenameWorkspace(
                InputWorkspace="combined_scaled_Iq", OutputWorkspace=WkSpMerge
            )
            SaveAscii(
                InputWorkspace=WkSpMerge,
                Filename=FileNameOut2,
                WriteXError=True,
                Separator="Tab",
                CommentIndicator="#",
                WriteSpectrumID=False,
            )
            SaveCanSAS1D(InputWorkspace=WkSpMerge, Filename=FileNameOut1)
            Filename_lowq = str(Sca1) + "_frame2_iq.xml"
            DeleteWorkspace(Filename_lowq)
            Filename_hiq = str(Sca1) + "_frame1_iq.xml"
            DeleteWorkspace(Filename_hiq)
            junk = Filename_lowq + "_scaled"
            DeleteWorkspace(junk)
            junk = Filename_hiq + "_scaled"
            DeleteWorkspace(junk)

        except (RuntimeError):
            print ""
            print "RuntimeError - the data file " + FileNameOut1 + " may not exist.  Check this error message:"
            print "RuntimeError - the data file " + FileNameOut2 + " may not exist.  Check this error message:"
            print sys.exc_info()

    Sca1 = Sca1 + 1
    Tra = Tra + 1
