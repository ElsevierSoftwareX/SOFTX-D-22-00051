# Convert GPSANS SPICE file to event NeXus

# set
ipts = 828
exp = 280
scan_pt_list = zip([12, 31, 27], [1] * 3)

# ----------------------------------------------------------------------------------
# TRY NOT TO TOUCH THIS PART
# ----------------------------------------------------------------------------------
# Template event nexus file for instrument geometry and etc
TEMPLATE_EVENT_NEXUS = (
    "/SNS/EQSANS/shared/sans-backend/data/new/ornl/sans/hfir/gpsans/CG2_9177.nxs.h5"
)

# ----------------------------------------------------------------------------------
# DON'T TOUCH ANYTHING BELOW THIS LINE
# ----------------------------------------------------------------------------------
from drtsans.mono.gpsans.cg2_spice_to_nexus import convert_spice_to_nexus  # noqa: E401

for scan_num, pt_num in scan_pt_list:
    convert_spice_to_nexus(
        ipts,
        exp,
        scan_num,
        pt_num,
        TEMPLATE_EVENT_NEXUS,
        output_dir=f"/HFIR/CG2/IPTS-{ipts}/shared/Exp{exp}",
    )
