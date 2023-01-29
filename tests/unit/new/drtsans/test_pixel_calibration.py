import os
import pytest
import tempfile
import shutil

from mantid.api import AnalysisDataService

from drtsans.pixel_calibration import loader_algorithm, BarPositionFormula, Table
from drtsans.settings import namedtuplefy


@pytest.fixture(scope="session")
@namedtuplefy
def helper(reference_dir):
    database_file = os.path.join(
        reference_dir.new.sans, "pixel_calibration", "calibrations.json"
    )
    return {"database": database_file}


@pytest.fixture(scope="function")
def clone_database(helper):
    r"""Serve all contents of helper.database in a temporary database"""
    database_directory = os.path.dirname(helper.database)
    cloned_database_directory = tempfile.mkdtemp()
    os.rmdir(
        cloned_database_directory
    )  # shutil.copytree requires non-existing directory!
    shutil.copytree(database_directory, cloned_database_directory)
    cloned_database_file = os.path.join(
        cloned_database_directory, os.path.basename(helper.database)
    )
    yield cloned_database_file
    # Tear down the temporary database
    shutil.rmtree(cloned_database_directory)


@pytest.mark.parametrize(
    "input_file, loader_name",
    [
        ("CG3_960.nxs.h5", "LoadEventNexus"),
        ("CG3_838.nxs", "LoadNexusProcessed"),
        ("BioSANS_exp327_scan0066_0001_mask.xml", "Load"),
    ],
)
def test_loader_algorithm(input_file, loader_name, reference_dir):
    input_file = os.path.join(
        reference_dir.new.biosans,
        "pixel_calibration",
        "test_loader_algorithm",
        input_file,
    )
    assert loader_algorithm(input_file).__name__ == loader_name


class TestBarPositionFormula:
    def test_elucidate_formula(self):
        formula = BarPositionFormula._elucidate_formula(("BIOSANS", "detector1"))
        assert formula == "565 - {y} + 0.0083115 * (191 - {tube})"
        formula = BarPositionFormula._elucidate_formula("Mary Poppings")
        assert formula == "565 - {y} + 0.0 * {tube}"

    def test_validate_symbols(self):
        BarPositionFormula._validate_symbols("{y} {tube}")
        for invalid_formula in ("{tube}", "{dcal}", "y"):
            with pytest.raises(ValueError):
                BarPositionFormula._validate_symbols(invalid_formula)
        assert (
            BarPositionFormula._validate_symbols("565 - {y}")
            == "565 - {y} + 0.0 * {tube}"
        )

    def test_str(self):
        assert (
            str(BarPositionFormula(instrument_component="unknown"))
            == BarPositionFormula._default_formula
        )

    def test_evaluate(self):
        formula = BarPositionFormula(
            instrument_component=("GPSANS", "detector1")
        )  # use default formula
        assert formula.evaluate(565, 191) == pytest.approx(0.0)

    def test_validate_top_position(self):
        for formula in BarPositionFormula._default_formulae.values():
            BarPositionFormula(formula=formula).validate_top_position(0.0)
        with pytest.raises(RuntimeError):
            BarPositionFormula(("BIOSANS", "wing_detector")).validate_top_position(
                1150.0
            )
        BarPositionFormula(formula="{y} - 565").validate_top_position(1150.0)


class TestTable:
    def test_load(self, helper):
        r"""test method 'load'"""
        calibration = Table.load(
            helper.database, "BARSCAN", "GPSANS", "detector1", 20200104
        )
        assert calibration.daystamp == 20200103
        assert AnalysisDataService.doesExist("barscan_GPSANS_detector1_20200103")

    def test_save(self, helper, clone_database):
        r"""test method 'save'"""
        calibration = Table.load(
            clone_database, "BARSCAN", "GPSANS", "detector1", 20200104
        )
        assert os.path.dirname(calibration.tablefile) == os.path.join(
            os.path.dirname(helper.database), "tables"
        )
        with pytest.raises(ValueError):
            calibration.save(database=clone_database)  # we cannot save a duplicate
        calibration.save(
            database=clone_database, overwrite=True
        )  # force saving a duplicate
        calibration = Table.load(
            clone_database, "BARSCAN", "GPSANS", "detector1", 20200104
        )
        assert os.path.dirname(calibration.tablefile) == os.path.join(
            os.path.dirname(clone_database), "tables"
        )


if __name__ == "__main__":
    pytest.main([__file__])
