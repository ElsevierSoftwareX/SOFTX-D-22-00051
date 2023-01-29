import pytest

from mantid.simpleapi import CreateWorkspace

from drtsans.settings import unique_workspace_dundername
from drtsans.instruments import (
    InstrumentEnumName,
    extract_run_number,
    instrument_enum_name,
    is_time_of_flight,
)


class TestInstrumentEnumName:
    def test_names(self):
        assert InstrumentEnumName.names() == ["BIOSANS", "EQSANS", "GPSANS"]


def test_instrument_name(serve_events_workspace):
    assert instrument_enum_name("EQ-SANS") == InstrumentEnumName.EQSANS
    assert str(instrument_enum_name("EQ-SANS")) == "EQSANS"
    assert str(instrument_enum_name("CG3")) == "BIOSANS"
    assert str(instrument_enum_name("somepath/CG3_961.nxs.h5")) == "BIOSANS"
    assert (
        instrument_enum_name("nonexistantsansinstrument")
        == InstrumentEnumName.UNDEFINED
    )
    assert (
        instrument_enum_name(serve_events_workspace("EQSANS_92353.nxs.h5"))
        == InstrumentEnumName.EQSANS
    )
    workspace = CreateWorkspace(
        DataX=range(42), DataY=range(42), OutputWorkspace=unique_workspace_dundername()
    )
    assert instrument_enum_name(workspace) == InstrumentEnumName.UNDEFINED
    workspace.delete()


def test_is_time_of_flight(serve_events_workspace):
    for query in ("EQSANS", "EQ-SANS", serve_events_workspace("EQSANS_92353.nxs.h5")):
        assert is_time_of_flight(query) is True
    for query in ("GPSANS", "CG2", "BIOSANS", "CG3"):
        assert is_time_of_flight(query) is False
    for query in [InstrumentEnumName.BIOSANS, InstrumentEnumName.GPSANS]:
        assert is_time_of_flight(query) is False
    for query in [InstrumentEnumName.EQSANS]:
        assert is_time_of_flight(query) is True


def test_extract_run_number():
    for query in (
        "/HFIR/CG3/CG3_961.nxs.h5",
        "CG3_961.nxs.h5",
        "CG3961",
        "CG3_961",
        "961",
        961,
    ):
        assert extract_run_number(query) == 961


if __name__ == "__main__":
    pytest.main([__file__])
