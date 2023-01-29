import numpy as np
import json
import sys
import xml.etree.ElementTree as ET
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa E402
from matplotlib.colors import LogNorm  # noqa E402
import mpld3  # noqa E402
from mpld3 import plugins  # noqa E402


# Quick loader for CD3
def load_data(filename):
    root = ET.parse(filename).getroot()
    title = root.find("Header").findtext("Scan_Title")
    data1 = np.rot90(
        np.array(root.find("Data").findtext("Detector").split(), dtype=int).reshape(
            (192, 256)
        )
    )
    data2 = np.rot90(
        np.array(root.find("Data").findtext("DetectorWing").split(), dtype=int).reshape(
            (160, 256)
        )
    )
    return data1, data2, title


data1, data2, title = load_data(sys.argv[1])

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
plot = ax1.imshow(
    data1, norm=LogNorm(), extent=(0.5, 192.5, 0.5, 256.5), origin="lower"
)
plot = ax2.imshow(
    data2, norm=LogNorm(), extent=(0.5, 160.5, 0.5, 256.5), origin="lower"
)

ax1.set_xlabel("Tube")
ax1.set_ylabel("Pixel")
ax1.set_title("Detector")
ax2.set_xlabel("Tube")
ax2.set_title("WingDetector")

fig.suptitle(title)

plugins.connect(fig, plugins.MousePosition(fontsize=14, fmt=".0f"))

print(json.dumps(mpld3.fig_to_dict(fig)))
