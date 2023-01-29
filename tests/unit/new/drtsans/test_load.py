# package imports
from drtsans.load import __monitor_counts

# standard imports
import h5py
import os
import pytest
import tempfile


def test_monitor_counts():
    # Create HDF5 file with empty 'entry' group
    _, filename = tempfile.mkstemp(suffix=".h5")
    f = h5py.File(filename, "w")
    f.create_group("entry")
    f.close()
    # Assert we cannot read monitor counts
    with pytest.raises(RuntimeError) as except_info:
        __monitor_counts(filename)
    assert "does not contain /entry/" in str(except_info.value)
    # Append a monitor entry to the file
    f = h5py.File(filename, "a")
    group = f["entry"].create_group("monitor1")
    data_set = group.create_dataset("total_counts", (1,), dtype="i")
    total_counts = 42
    data_set[0] = total_counts
    f.close()
    # Assert the monitor counts
    assert __monitor_counts(filename) == total_counts
    os.remove(filename)


if __name__ == "__main__":
    pytest.main([__file__])
