# Convert BIOSANS SPICE file to event NeXus
# All the output file is supposed to be written to
# /HFIR/CG3/IPTS-{ipts_number}/shared/spice_nexus/CG3_{exp}{scan}{pt}'
# and make up a unique run number from experiment number, scan number and pt number


# Set SPICE files information
# The following example is for sensitivities preparation

"""
Sample - /HFIR/CG3/IPTS-17240/exp318/Datafiles/BioSANS_exp318_scan0217_0001.xml (Scattering/Transmission)
Empty Beam - /HFIR/CG3/IPTS-17240/exp318/Datafiles/BioSANS_exp318_scan0220_0001.xml (For Transmission)
Beam Center - /HFIR/CG3/IPTS-17240/exp318/Datafiles/BioSANS_exp318_scan0220_0001.xml (Transmission Measurement)
Dark - /HFIR/CG3/IPTS-17240/exp318/Datafiles/BioSANS_exp318_scan0044_0001.xml (for both main and wing detectors)
"""

ipts = 17240
exp = 318
scan_pt_list = zip([44, 220, 217], [1] * 3)

# ----------------------------------------------------------------------------------
# TRY NOT TO TOUCH THIS PART
# ----------------------------------------------------------------------------------
# Template event nexus file for instrument geometry and etc
TEMPLATE_EVENT_NEXUS = (
    "/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/biosans/CG3_5705.nxs.h5"
)

# ----------------------------------------------------------------------------------
# DON'T TOUCH ANYTHING BELOW THIS LINE
# ----------------------------------------------------------------------------------
from drtsans.mono.biosans.cg3_spice_to_nexus import convert_spice_to_nexus  # noqa: E401


bad_pixels = [70911]
nexus_names = set()

# Output directory between standard and drtsans integration test
if True:
    nexus_dir = f"/HFIR/CG3/IPTS-{ipts}/shared/Exp{exp}"  # standard converted nexus
else:
    nexus_dir = "/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/biosans/"  # reference dir
for scan_num, pt_num in scan_pt_list:
    nexus = convert_spice_to_nexus(
        ipts,
        exp,
        scan_num,
        pt_num,
        TEMPLATE_EVENT_NEXUS,
        masked_detector_pixels=bad_pixels,
        output_dir=nexus_dir,
    )
    nexus_names.add(nexus)

# Check
from mantid.simpleapi import LoadEventNexus  # noqa: E401

print("\n\n-----  Verification -------\n\n")
for nexus_name in nexus_names:
    try:
        print(f"Loading {nexus_name}")
        ws = LoadEventNexus(Filename=nexus_name, LoadNexusInstrumentXML=True)
        print(f"{nexus_name}: {ws.getNumberHistograms()}, {ws.getNumberEvents()}")
    except RuntimeError as run_err:
        print(f"Failed to laod {nexus_name} due to {run_err}")
