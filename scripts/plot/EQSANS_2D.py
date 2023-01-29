import numpy as np
import json
import sys
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa E402
from matplotlib.colors import LogNorm  # noqa E402
import mpld3  # noqa E402
from mpld3 import plugins  # noqa E402

# Shut up Mantid, send everything to Null
# Create file to suppress output, this overrides the settings in /etc/mantid.local.properties
if not os.path.exists(os.path.expanduser("~/.mantid")):
    os.makedirs(os.path.expanduser("~/.mantid"))
with open(os.path.expanduser("~/.mantid/Mantid.user.properties"), "a") as f:
    f.write("\nlogging.channels.consoleChannel.class=NullChannel\n")
from mantid.simpleapi import Load, DeleteWorkspace  # noqa E402


# Load data using Mantid
def load_data(filename):
    ws = Load(filename)
    data = ws.extractY().reshape(-1, 8, 256).T
    DeleteWorkspace(ws)
    data2 = data[:, [0, 4, 1, 5, 2, 6, 3, 7], :]
    data2 = data2.transpose().reshape(-1, 256)

    # Get mask from detector info
    di = ws.detectorInfo()
    offset = int(di.detectorIDs().searchsorted(0))
    mask = (
        np.array([di.isMasked(i + offset) for i in range(data.size)])
        .reshape(-1, 8, 256)
        .T
    )
    mask2 = mask[:, [0, 4, 1, 5, 2, 6, 3, 7], :]
    mask2 = mask2.transpose().reshape(-1, 256)
    return np.ma.masked_where(mask2, data2).T, ws.getRunNumber(), ws.getTitle()


data, run_number, title = load_data(sys.argv[1])

fig, ax = plt.subplots()
current_cmap = matplotlib.cm.get_cmap()
current_cmap.set_bad(color="grey")
plot = ax.imshow(data, extent=(0.5, 192.5, 0.5, 256.5), origin="lower", aspect="auto")
ax.set_title("EQSANS_{} - {}".format(run_number, title))
ax.set_xlabel("Tube")
ax.set_ylabel("Pixel")

plugins.connect(fig, plugins.MousePosition(fontsize=14, fmt=".0f"))

print(json.dumps(mpld3.fig_to_dict(fig)))

os.remove(os.path.expanduser("~/.mantid/Mantid.user.properties"))
