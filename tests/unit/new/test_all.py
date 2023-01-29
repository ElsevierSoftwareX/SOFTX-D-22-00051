r"""
This is a collection of tests to verify that the wild imports
(e.g. __all__) don't define things that don't exist

See http://xion.io/post/code/python-all-wild-imports.html for more information
"""
import pytest


def find_missing(package):
    missing = set(n for n in package.__all__ if getattr(package, n, None) is None)
    assert not missing, "__all__ contains unresolved names: {}".format(
        ", ".join(missing)
    )


def test_drtsans():
    import drtsans

    find_missing(drtsans)


def test_drtsans_dataobjects():
    import drtsans.dataobjects

    find_missing(drtsans.dataobjects)


def test_drtsans_mono():
    import drtsans.mono

    find_missing(drtsans.mono)


def test_drtsans_mono_biosans():
    import drtsans.mono.biosans

    find_missing(drtsans.mono.biosans)


def test_drtsans_mono_gpsans():
    import drtsans.mono.gpsans

    find_missing(drtsans.mono.gpsans)


def test_drtsans_tof():
    import drtsans.tof

    find_missing(drtsans.tof)


def test_drtsans_tof_eqsans():
    import drtsans.tof.eqsans

    find_missing(drtsans.tof.eqsans)


if __name__ == "__main__":
    pytest.main([__file__])
