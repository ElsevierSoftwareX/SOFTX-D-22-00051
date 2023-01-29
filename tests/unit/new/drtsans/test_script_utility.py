import pytest
from pathlib import Path

from drtsans import script_utility
from tempfile import TemporaryDirectory


class TestScriptUtility:
    def setup_method(self):
        self.output_dir = TemporaryDirectory()
        self.output_dir_name = self.output_dir.name

    def teardown_method(self):
        self.output_dir.cleanup()

    def get_list_folders(self, input_folder=""):
        full_list = Path(input_folder).glob("**/*")
        list_folder = [_file for _file in full_list if _file.is_dir()]
        return list_folder

    def test_create_default_output_directory(self):
        """test if the HFIR SANS folder are created correctly"""
        script_utility.create_output_directory(output_dir=self.output_dir_name)
        assert Path(self.output_dir_name).exists()
        assert Path(self.output_dir_name).joinpath("1D").exists()
        assert Path(self.output_dir_name).joinpath("2D").exists()

    def test_error_raised_if_not_output_directory(self):
        """make sure a folder is provided"""
        with pytest.raises(ValueError):
            script_utility.create_output_directory()

    def test_recreate_output_directory(self):
        """make sure the default folder are created even when call several times"""
        script_utility.create_output_directory(output_dir=self.output_dir_name)
        script_utility.create_output_directory(output_dir=self.output_dir_name)
        assert Path(self.output_dir_name).exists()
        assert Path(self.output_dir_name).joinpath("1D").exists()
        assert Path(self.output_dir_name).joinpath("2D").exists()

    def test_right_number_of_folders_created(self):
        """make sure only 2 folders are created"""
        script_utility.create_output_directory(output_dir=self.output_dir_name)
        list_folder = self.get_list_folders(input_folder=self.output_dir_name)
        assert len(list_folder) == 2

    def test_create_non_default_output_directory(self):
        """make sure non default folders are created"""
        script_utility.create_output_directory(
            output_dir=self.output_dir_name, hfir_sans=False
        )
        assert Path(self.output_dir_name).exists()

    def test_create_custom_subfolders_for_hfir(self):
        """make sure subfolder list of folders are created for hfir sans"""
        script_utility.create_output_directory(
            output_dir=self.output_dir_name, subfolder=["folder1", "folder2"]
        )
        assert Path(self.output_dir_name).exists()
        assert Path(self.output_dir_name).joinpath("folder1").exists()
        assert Path(self.output_dir_name).joinpath("folder2").exists()
        list_folder = self.get_list_folders(input_folder=self.output_dir_name)
        assert len(list_folder) == 4

    def test_create_custom_subfolders_for_non_hfir(self):
        """make sure subfolder list of folders are created not for hfir sans"""
        script_utility.create_output_directory(
            output_dir=self.output_dir_name,
            subfolder=["folder1", "folder2"],
            hfir_sans=False,
        )
        assert Path(self.output_dir_name).exists()
        assert Path(self.output_dir_name).joinpath("folder1").exists()
        assert Path(self.output_dir_name).joinpath("folder2").exists()
        list_folder = self.get_list_folders(input_folder=self.output_dir_name)
        assert len(list_folder) == 2
