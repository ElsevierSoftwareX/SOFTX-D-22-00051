import pytest
import os
from drtsans.mono.spice_xml_parser import SpiceXMLParser
from mantid.simpleapi import LoadHFIRSANS, mtd


def test_get_das_logs(reference_dir):
    """

    Returns
    -------

    """
    # Init the parser
    # test_xml = '/Users/wzz/Projects/SANS/sans-backend/temp/FromXML/CG2_exp315_scan0005_0060.xml'
    test_xml = os.path.join(reference_dir.new.gpsans, "CG2_exp315_scan0005_0060.xml")
    assert os.path.exists(test_xml), f"Test XML {test_xml} cannot be found"
    xml_parser = SpiceXMLParser(test_xml)

    # Test method to retrieve the XML node
    nodes = [
        "sample_to_flange",
        "sdd",
        "source_aperture_size",
        "sample_aperture_size",
        "detector_trans",
        "source_distance",
    ]
    for node_name in nodes:
        xml_parser.get_xml_node(node_name)
        # assert xml_node

    # Test method to retrieve the data value and unit
    nexus_spice_log_map = {
        "CG2:CS:SampleToSi": ("sample_to_flange", "mm"),
        "sample_detector_distance": ("sdd", "m"),
        "wavelength": ("lambda", "angstroms"),
        "wavelength_spread": ("dlambda", None),
        "source_aperture_diameter": ("source_aperture_size", "mm"),
        "sample_aperture_diameter": ("sample_aperture_size", "mm"),
        "detector_trans_Readback": ("detector_trans", None),
        "source_distance": ("source_distance", "m"),
        "beamtrap_diameter": ("beamtrap_diameter", "mm"),
    }

    das_log_values = dict()
    for nexus_name, spice_tuple in nexus_spice_log_map.items():
        spice_name, default_unit = spice_tuple
        value, unit = xml_parser.get_node_value(spice_name, float)
        if unit is None:
            unit = default_unit
        # print(f'{nexus_name}: value = {value}, unit = {unit}')
        das_log_values[nexus_name] = value, unit

    # Attenuator: special case
    atten_value, atten_unit = xml_parser._read_attenuator()
    assert atten_value == pytest.approx(
        52.997100, 0.00001
    ), f"Attenuator value {atten_value} shall be 52.997100"
    assert atten_unit == "mm", f"Attenuator unit {atten_unit} shall be mm"

    # close
    xml_parser.close()

    # Verify other values
    LoadHFIRSANS(Filename=test_xml, OutputWorkspace="SpiceXMLTest")
    spice_ws = mtd["SpiceXMLTest"]

    for das_log_name in das_log_values:
        log_value, log_unit = das_log_values[das_log_name]
        print(f"{das_log_name}: {log_value}, {log_unit}")
        if das_log_name in ["sample_detector_distance", "wavelength_spread"]:
            continue
        try:
            run_property = spice_ws.run().getProperty(das_log_name)
            assert run_property.value == log_value
            # assert run_property.units.lower() == log_unit.lower()
        except RuntimeError as key_error:
            if das_log_name not in [
                "CG2:CS:SampleToSi",
                "source_distance",
                "beamtrap_diameter",
                "detector_trans_Readback",
            ]:
                raise key_error
        except AssertionError as a_error:
            if das_log_name not in ["sample_detector_distancce"]:
                raise a_error


if __name__ == "__main__":
    pytest.main(__file__)
