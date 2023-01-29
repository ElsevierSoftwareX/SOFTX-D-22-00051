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


# Quick loader for CD2
def load_data(filename):
    root = ET.parse(filename).getroot()
    title = root.find("Header").findtext("Scan_Title")
    data = np.rot90(
        np.array(root.find("Data").findtext("Detector").split(), dtype=int).reshape(
            (192, 256)
        )
    )
    return data, title


data, title = load_data(sys.argv[1])

fig, ax = plt.subplots()
plot = ax.imshow(
    data, norm=LogNorm(), extent=(0.5, 192.5, 0.5, 256.5), origin="lower", aspect="auto"
)
ax.set_title(title)
ax.set_xlabel("Tube")
ax.set_ylabel("Pixel")

plugins.connect(fig, plugins.MousePosition(fontsize=14, fmt=".0f"))

print(json.dumps(mpld3.fig_to_dict(fig)))
