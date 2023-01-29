from mantid.simpleapi import SaveNISTDAT, SaveNexus


def save_nist_dat(input_workspace, filename):
    """Save I(Qx, Qy) data to a text file compatible with NIST and DANSE readers

    Parameters
    ----------
    input_workspace : ~mantid.api.MatrixWorkspace
        Workspace to be saved
    filename : string
        Filename of the output text file. Allowed extensions: [``.dat``]
    """
    SaveNISTDAT(InputWorkspace=input_workspace, Filename=filename)


def save_nexus(input_workspace, title, filename):
    """Write the given Mantid workspace to a NeXus file.

    Parameters
    ----------
    input_workspace : ~mantid.api.MatrixWorkspace
        Name of the workspace to be saved
    title : string
        Title to describe the saved worksapce
    filename : string
        The bame of the NeXus file to write, as a full or relative path.
        Allowed extensions: [``.nxs``, ``.nx5``, ``.xml``]
    """
    SaveNexus(InputWorkspace=input_workspace, Title=title, Filename=filename)
