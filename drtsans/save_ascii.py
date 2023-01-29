from drtsans.dataobjects import IQazimuthal

try:
    from mantid.plots.datafunctions import get_spectrum  # mantid >4.2
except ImportError:
    from mantid.plots.helperfunctions import get_spectrum  # mantid <=4.2
from mantid.simpleapi import SaveCanSAS1D
import numpy as np


def save_ascii_binned_1D(filename, title, *args, **kwargs):
    """Save I(q) data in Ascii format

    Parameters
    ----------
    filename: str
        output filename
    title: str
        title to be added on the first line
    args: drtsans.dataobjects.IQmod
        output from 1D binning
    kwargs:
        intensity, error, mod_q, delta_mod_q - 1D numpy arrays of the same length, output from 1D binning
    """
    try:
        kwargs.update(args[0]._asdict())
    except AttributeError:
        pass
    # Read the value of skip_nan from kwargs or True as default
    skip_nan = kwargs.get('skip_nan', True)
    # Delete NaNs if requested
    if skip_nan:
        finites = np.isfinite(kwargs['intensity'])
        q = kwargs['mod_q'][finites]
        intensity = kwargs["intensity"][finites]
        error = kwargs["error"][finites]
        dq = kwargs["delta_mod_q"][finites]
    else:
        q = kwargs["mod_q"]
        intensity = kwargs["intensity"]
        error = kwargs["error"]
        dq = kwargs["delta_mod_q"]

    with open(filename, "w+") as f:
        f.write("# " + title + "\n")
        f.write("#Q (1/A)        I (1/cm)        dI (1/cm)       dQ (1/A)\n")
        for i in range(len(intensity)):
            f.write("{:.6E}\t".format(q[i]))
            f.write("{:.6E}\t".format(intensity[i]))
            f.write("{:.6E}\t".format(error[i]))
            f.write("{:.6E}\n".format(dq[i]))


def save_ascii_1D(wksp, title, filename):
    """Save the I(q) workspace in Ascii format

    Parameters
    ----------
    wksp : ~mantid.api.MatrixWorkspace
        Workspace containing only one spectrum (the I(q) curve)
    title : string
        first line of the ascii file
    filename : string
        The output filename
    """
    q, intensity, sigma_i, dq = get_spectrum(wksp, 0, True, True, True)
    f = open(filename, "w+")
    f.write("# " + title + "\n")
    f.write("#Q (1/A)        I (1/cm)        dI (1/cm)       dQ (1/A)\n")

    for i in range(len(intensity)):
        f.write("{:.6E}\t".format(q[i]))
        f.write("{:.6E}\t".format(intensity[i]))
        f.write("{:.6E}\t".format(sigma_i[i]))
        f.write("{:.6E}\n".format(dq[i]))
    f.close()


def save_xml_1D(wksp, title, filename):
    """Save the I(q) workspace in SaveCanSAS (XML) format

    Parameters
    ----------
    wksp : ~mantid.api.MatrixWorkspace
        Workspace containing only one spectrum (the I(q) curve)
    title : string
        Text to append to Process section
    filename : string
        The output filename
    """
    SaveCanSAS1D(InputWorkspace=wksp, Process=title, Filename=filename)


def save_ascii_binned_2D(filename, title, *args, **kwargs):
    r"""Save I(qx, qy) data in Ascii format

    Parameters
    ----------
    filename: str
        output filename
    title: str
        title to be added on the first line
    args: ~drtsans.dataobjects.IQazimuthal
        output from 2D binning
    kwargs:
        intensity, error, qx, qy, delta_qx, delta_qy - 1D numpy arrays of the same length, output from 1D binning
    """
    try:
        kwargs = args[0]._asdict()
    except AttributeError:
        pass
    # make everything a 1d array
    # this does nothing if the array is already 1d
    qx = kwargs["qx"].ravel()
    qy = kwargs["qy"].ravel()
    intensity = kwargs["intensity"].ravel()
    error = kwargs["error"].ravel()
    dqx = kwargs["delta_qx"]
    dqy = kwargs["delta_qy"]
    if dqx is not None and dqy is not None:
        dqx = dqx.ravel()
        dqy = dqy.ravel()

    with open(filename, "w+") as f:
        f.write("# " + title + "\n")
        f.write("#Qx (1/A)       Qy (1/A)        I (1/cm)        dI (1/cm)")
        if dqx is not None:
            f.write("       dQx (1/A)       dQy (1/A)")
        f.write("\n")
        f.write("ASCII data\n\n")

        for i in range(len(intensity)):
            line = "{:.6E}\t{:.6E}\t{:.6E}\t{:.6E}".format(
                qx[i], qy[i], intensity[i], error[i]
            )
            if dqx is not None:
                line += "\t{:.6E}\t{:.6E}".format(dqx[i], dqy[i])
            f.write(line + "\n")


def load_ascii_binned_2D(filename):
    """Load the format produced by save_ascii_binned_2D

    Parameters
    ----------
    filename: str
        Input filename

    Returns
    -------
    drtsans.dataobjects.IQazimuthal
    """
    csv_data = np.genfromtxt(filename, comments="#", dtype=np.float64, skip_header=3)
    num_cols = len(csv_data[0])
    assert (
        num_cols == 4 or num_cols == 6
    ), "Incompatible number of colums: {} should be 4 or 6".format(num_cols)

    if num_cols == 4:
        delta_qx = None
        delta_qy = None
    elif num_cols == 6:
        delta_qx = csv_data[:, 4]
        delta_qy = csv_data[:, 5]
    iq_azi = IQazimuthal(
        qx=csv_data[:, 0],
        qy=csv_data[:, 1],
        intensity=csv_data[:, 2],
        error=csv_data[:, 3],
        delta_qx=delta_qx,
        delta_qy=delta_qy,
    )  # wavelength isn't in the file
    del csv_data

    return iq_azi


def save_ascii_2D(q2, q2x, q2y, title, filename):
    """Save the I(qx,qy) workspace in Ascii format

    Parameters
    ----------
    q2 : Workspace2D
        Workspace containing the 2D I(qx,qy)
    q2x : Workspace2D
        Workspace containing the 2D dqx(qx,qy)
    q2y : Workspace2D
        Workspace containing the 2D dqy(qx,qy)
    title : string
        first line of the ascii file
    filename : string
        The output filename
    """

    f = open(filename, "w+")
    f.write("# " + title + "\n")
    f.write(
        "#Qx (1/A)       Qy (1/A)        I (1/cm)        dI (1/cm)"
        + "       dQx (1/A)       dQy (1/A)\n"
    )
    f.write("#ASCII data\n\n")
    for i in range(len(q2.readY(0))):
        for j in range(q2.getNumberHistograms()):
            qy = float(q2.getAxis(1).label(j))
            x = 0.5 * (q2.readX(j)[i] + q2.readX(j)[i + 1])
            f.write("{:.6E}\t".format(x))
            f.write("{:.6E}\t".format(qy))
            f.write("{:.6E}\t".format(q2.readY(j)[i]))
            f.write("{:.6E}\t".format(q2.readE(j)[i]))
            f.write("{:.6E}\t".format(q2x.readY(j)[i]))
            f.write("{:.6E}\n".format(q2y.readY(j)[i]))
    f.close()
