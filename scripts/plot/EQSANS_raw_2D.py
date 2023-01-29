import numpy as np
import h5py
import json
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa E402
from matplotlib.colors import LogNorm  # noqa E402
import mpld3  # noqa E402
from mpld3 import plugins  # noqa E402


# Quick event integrating loader for EQSANS
def load_data(filename):
    bc = np.zeros((256 * 192))
    with h5py.File(filename, "r") as f:
        run_number = f["/entry/run_number"].value[0]
        title = f["/entry/title"].value[0]
        for b in range(48):
            bc += np.bincount(
                f["/entry/bank" + str(b + 1) + "_events/event_id"].value,
                minlength=256 * 192,
            )
    data = bc.reshape(-1, 8, 256).T
    data2 = data[:, [0, 4, 1, 5, 2, 6, 3, 7], :]
    data2 = data2.transpose().reshape(-1, 256)
    return data2.T, run_number, title


data, run_number, title = load_data(sys.argv[1])

fig, ax = plt.subplots()
plot = ax.imshow(
    data, norm=LogNorm(), extent=(0.5, 192.5, 0.5, 256.5), origin="lower", aspect="auto"
)
ax.set_title("EQSANS_{} - {}".format(run_number, title))
ax.set_xlabel("Tube")
ax.set_ylabel("Pixel")

plugins.connect(fig, plugins.MousePosition(fontsize=14, fmt=".0f"))

print(json.dumps(mpld3.fig_to_dict(fig)))
