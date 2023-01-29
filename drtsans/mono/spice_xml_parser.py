"""
Module contains class and method to parse SPICE SANS XML file for DAS sample logs.
"""
from xml.etree import ElementTree
import numpy as np


class SpiceXMLParser(object):
    """
    Class to parse SPICE SANS data file in XML format
    """

    def __init__(self, spice_xml_name):
        """

        Parameters
        ----------
        spice_xml_name: str
            SPICE XML file
        """
        # Store file name for reference
        self._spice_name = spice_xml_name

        # open the file
        self._xml_file = open(spice_xml_name, "r")
        # parse
        self._xml_root = ElementTree.parse(self._xml_file).getroot()

    def close(self):
        """Close the opened XML file

        Returns
        -------

        """
        if self._xml_file is not None:
            self._xml_file.close()

    def get_xml_node(self, node_name, required_attribs=None):
        """Get an XML node by its name regardless its level in XML tree

        Parameters
        ----------
        node_name: str
            Name of the XML node to retrieve
        required_attribs: dict, None
            required attributes for the node

        Returns
        -------
        xlm.etree.Element

        """
        # Most of the node in SPICE XML file are in 2nd level
        xml_node_list = self._xml_root.findall(f".//{node_name}")

        # Only allow unique solution
        if len(xml_node_list) == 0:
            raise KeyError(
                f"SPICE file {self._spice_name}: XML node {node_name} does not exist."
            )

        # Check required attributes
        if required_attribs is not None:
            filtered_nodes = list()
            # filter
            for xml_node in xml_node_list:
                is_good = True
                for attr_name, attr_value in required_attribs.items():
                    if xml_node.attrib[attr_name] != attr_value:
                        is_good = False
                        break
                if is_good:
                    filtered_nodes.append(xml_node)

            # replace
            xml_node_list = filtered_nodes

        # check whether it is unique solution
        if len(xml_node_list) > 1:
            raise RuntimeError(f"XML node {node_name} is not unique")

        return xml_node_list[0]

    def get_node_value(self, node_name, value_type):
        """Get an XML node value regardless of the level of the node in the XML tree

        Parameters
        ----------
        node_name: str
            Name of the XML node to retrieve
        value_type: type
            type to cast the value

        Returns
        -------
        tuple
            value, unit

        """
        value_type = (
            value_type
            if type(value_type) is type
            else {
                "str": np.string_,
                "string": np.string_,
                "float": float,
                "double": float,
                "int": int,
                "integer": int,
            }[value_type.lower()]
        )

        if node_name == "attenuator":
            # Attenuator is special
            value, units = self._read_attenuator()

        else:
            # regular node
            # Get node
            xml_node = self.get_xml_node(node_name)

            # Get value
            str_value = xml_node.text

            # Get unit
            if "units" in xml_node.attrib:
                units = xml_node.attrib["units"]
            else:
                units = None

            value = value_type(str_value)

        return value, units

    def _read_attenuator(self):
        """Use this attenuator_pos and the attribute pos="open"

        Returns
        -------

        """
        # get attenuator position node
        xml_node = self.get_xml_node("attenuator_pos", required_attribs={"pos": "open"})

        value = float(xml_node.text)
        unit = xml_node.attrib["units"]

        return value, unit
