import os
import pytest
import pathlib
import stat
from mantid.simpleapi import CreateWorkspace
from drtsans.path import abspath, exists, registered_workspace, allow_overwrite
from drtsans.settings import amend_config, unique_workspace_dundername as uwd
from os.path import exists as os_exists
from tempfile import gettempdir, NamedTemporaryFile

# arbitrarily selected IPTS to see if the mount is in place
HAVE_EQSANS_MOUNT = os_exists("/SNS/EQSANS/IPTS-23732/nexus/EQSANS_106055.nxs.h5")

SEARCH_ON = {}
SEARCH_OFF = {"datasearch.searcharchive": "off"}

IPTS_23732 = "/SNS/EQSANS/IPTS-23732/nexus/"
IPTS_19800 = "/SNS/EQSANS/IPTS-19800/nexus/"
IPTS_22699 = "/HFIR/CG3/IPTS-22699/nexus/"


@pytest.mark.skipif(
    not HAVE_EQSANS_MOUNT,
    reason="Do not have /SNS/EQSANS properly " "mounted on this system",
)
@pytest.mark.parametrize(
    "hint, fullpath",
    [
        ("EQSANS_106026", IPTS_23732 + "EQSANS_106026.nxs.h5"),
        ("EQSANS106027", IPTS_23732 + "EQSANS_106027.nxs.h5"),
        ("EQSANS_88974.nxs.h5", IPTS_19800 + "EQSANS_88974.nxs.h5"),
    ],
    ids=("EQSANS_106026", "EQSANS_106026", "EQSANS_88974"),
)
def test_abspath_with_archivesearch(hint, fullpath, reference_dir):
    # set the data directory in the result using the test fixture
    pytest.skip(
        f"Search {hint} inside archive is skipped as build server cannot query through ONCAT."
    )
    assert abspath(hint, search_archive=True) == fullpath


@pytest.mark.parametrize(
    "hint",
    ["randomname", "EQSANS_106026", "EQSANS_106026"],
    ids=("randomname", "EQSANS_106026", "EQSANS_106026"),
)
def test_abspath_without_archivesearch(hint):
    with pytest.raises(RuntimeError):
        found = abspath(hint, search_archive=False)
        assert False, 'found "{}" at "{}"'.format(hint, found)


@pytest.mark.parametrize(
    "hint, instr, ipts, fullpath",
    [
        ("EQSANS_106026", "", 23732, IPTS_23732 + "EQSANS_106026.nxs.h5"),
        ("EQSANS106027", "", 23732, IPTS_23732 + "EQSANS_106027.nxs.h5"),
        ("EQSANS_88974.nxs.h5", "", 19800, IPTS_19800 + "EQSANS_88974.nxs.h5"),
        ("5709", "CG3", 22699, IPTS_22699 + "CG3_5709.nxs.h5"),
        ("5709", "CG3", 24740, IPTS_22699 + "CG3_5709.nxs.h5"),  # wrong proposal
    ],
    ids=(
        "EQSANS_106026",
        "EQSANS_106026",
        "EQSANS_88974",
        "CG3_5709",
        "CG3_5709_bad_proposal",
    ),
)
def test_abspath_with_ipts(hint, instr, ipts, fullpath):
    if not os.path.exists(fullpath):
        pytest.skip("{} does not exist".format(fullpath))

    # do not turn on archive search
    assert abspath(hint, instrument=instr, ipts=ipts) == fullpath


def test_abspath_with_directory(reference_dir):
    filename = os.path.join(reference_dir.new.biosans, "CG3_5709.nxs.h5")
    abspath(
        "CG3_5709", directory=reference_dir.new.biosans, search_archive=False
    ) == filename
    abspath(
        "5709",
        instrument="CG3",
        directory=reference_dir.new.biosans,
        search_archive=False,
    ) == filename


@pytest.mark.skipif(
    not HAVE_EQSANS_MOUNT,
    reason="Do not have /SNS/EQSANS properly " "mounted on this system",
)
@pytest.mark.parametrize(
    "hint, found",
    [("EQSANS_106026", True), ("EQSANS106027", True), ("EQSANS_88974.nxs.h5", True)],
)
def test_exists_with_archivesearch(hint, found, reference_dir):
    with amend_config(SEARCH_ON, data_dir=reference_dir.new.eqsans):
        assert exists(hint) == found  # allows verifying against True and False


@pytest.mark.parametrize(
    "hint, found",
    [("EQSANS_106026", True), ("EQSANS106028", False), ("EQSANS_88974.nxs.h5", True)],
)
def test_exists_without_archivesearch(hint, found, reference_dir):
    with amend_config(SEARCH_OFF, data_dir=reference_dir.new.eqsans):
        assert exists(hint) == found  # allows verifying against True and False


def test_registered_workspace():
    w_name = uwd()
    assert registered_workspace(w_name) is False
    w = CreateWorkspace(DataX=[1], Datay=[1], OutputWorkspace=w_name)
    assert registered_workspace(w_name) is True
    assert registered_workspace(w) is True


def test_allow_overwrite(cleanfile):
    tmpdir = gettempdir()
    # create an empty file
    tmpfile = NamedTemporaryFile(dir=tmpdir, delete=False)
    tmpfile.close()
    cleanfile(tmpfile.name)  # remove the file when test finishes

    # check if others write permission is false
    path = pathlib.Path(tmpfile.name)
    assert not bool(path.stat().st_mode & stat.S_IWOTH)
    allow_overwrite(tmpdir)
    # check permissions
    assert bool(path.stat().st_mode & stat.S_IWUSR), "user writable"
    assert bool(path.stat().st_mode & stat.S_IWGRP), "group writable"
    assert bool(path.stat().st_mode & stat.S_IWOTH), "world writable"
    # delete file


if __name__ == "__main__":
    pytest.main([__file__])
