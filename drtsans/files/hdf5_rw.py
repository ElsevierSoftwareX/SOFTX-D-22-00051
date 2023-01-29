# Read an arbitrary h5 file in order to study its structure
# Goal: reveal and duplicate a CANSAS file such that it can be imported by SASVIEW
import h5py
import numpy as np


__all__ = ["HDFNode", "GroupNode", "FileNode", "DataSetNode"]


def parse_h5_entry(h5_entry):
    """Parse an HDF5 entry and generate an HDFNode object including all the sub entries

    Parameters
    ----------
    h5_entry: ~h5py._hl.dataset.Dataset, ~h5py._hl.group.Group, ~h5py._hl.files.File
        h5py entries including data set, group and file

    Returns
    -------
    HDFNode
        an HDFNode

    """
    # Create entry node instance
    entry_node = None
    for h5_entry_type, buffer_node_class in [
        (h5py._hl.files.File, FileNode),
        (h5py._hl.group.Group, GroupNode),
        (h5py._hl.dataset.Dataset, DataSetNode),
    ]:
        if isinstance(h5_entry, h5_entry_type):
            # generate node
            entry_node = buffer_node_class()
            # parse
            entry_node.parse_h5_entry(h5_entry)
            break
    # Check
    if entry_node is None:
        raise RuntimeError(
            "HDF entry of type {} is not supported".format(type(h5_entry))
        )

    return entry_node


class HDFNode(object):
    """
    an HDF node with more information
    """

    def __init__(self, name=None):
        """initialization

        Parameters
        ----------
        name: str, None
            entry name
        """
        self._name = name

        # Set attributes and etc
        self._attributes = dict()

        return

    def match(self, other_node):
        """Compare 2 HDFNode to see whether they are same

        If mismatch, an exception will be raised including
        - TypeError: for nodes are nto same type
        - ValueError: attribute or node name value mismatch
        - KeyError: some attribute does not exist in both node

        Parameters
        ----------
        other_node: HDFNode
            other node to compare

        Returns
        -------

        """
        # compare class type
        if not isinstance(other_node, type(self)):
            raise TypeError(
                "Try to match instance of class {} (other) to {} (self)"
                "".format(type(other_node), type(self))
            )

        # compare name
        if self._name != other_node.name:
            raise ValueError(
                "self.name = {}; other.name = {}".format(self.name, other_node.name)
            )

        # compare attributes
        if set(self._attributes.keys()) != set(other_node.attributes.keys()):
            print(
                "Data node {} Attributes are not same:\nself - other = {}]\nother - self = {}"
                "".format(
                    self.name,
                    set(self._attributes.keys()) - set(other_node.attributes.keys()),
                    set(other_node.attributes.keys()) - set(self._attributes.keys()),
                )
            )
            raise KeyError(
                "Data node {} Attributes are not same:\nself - other = {}]\nother - self = {}"
                "".format(
                    self.name,
                    set(self._attributes.keys()) - set(other_node.attributes.keys()),
                    set(other_node.attributes.keys()) - set(self._attributes.keys()),
                )
            )

        # compare attribute values
        error_msg = ""
        for attr_name in self._attributes.keys():
            if self._attributes[attr_name] != other_node.attributes[attr_name]:
                error_msg += (
                    "Mismatch attribute {} value: self = {}, other = {}"
                    "".format(
                        attr_name,
                        self._attributes[attr_name],
                        other_node.attributes[attr_name],
                    )
                )
        if error_msg != "":
            raise ValueError(error_msg)

    def parse_h5_entry(self, h5_entry):
        """Parse an HDF5 entry

        Parameters
        ----------
        h5_entry

        Returns
        -------

        """
        # Name
        self._name = h5_entry.name

        # Parse attributes
        # Reset data structure
        self._attributes = dict()

        # parse attributes
        for attr_name in h5_entry.attrs:
            # returned h5_attribute in fact is attribute name
            self._attributes[attr_name] = h5_entry.attrs[attr_name]

    @property
    def name(self):
        return self._name

    @property
    def attributes(self):
        return self._attributes

    def add_attributes(self, attributes):
        """Add a list of attributes to the HDF5 node

        Parameters
        ----------
        attributes: ~dict
            Attributes to add.  key = attribute name, value = attribute value

        Returns
        -------

        """
        for attr_name in attributes.keys():
            self._attributes[attr_name] = attributes[attr_name]

    def write(self, inputs):
        """

        Parameters
        ----------
        inputs: str, ~h5py._hl.group.Group, ~h5py._hl.files.File
            Node to input

        Returns
        -------

        """
        raise NotImplementedError("Virtual method to write {}".format(inputs))

    def write_attributes(self, curr_entry):

        # attributes
        for attr_name in self._attributes:
            # ignore if an attribute is None (might be missing)
            if self._attributes[attr_name] is None:
                continue
            try:
                curr_entry.attrs[attr_name] = self._attributes[attr_name]
            except TypeError as type_error:
                print(
                    f"[ERROR] {self._name}-node attribute {attr_name} is of type {type(attr_name)}"
                )
                raise TypeError(
                    f"[ERROR] {self._name}-node attribute {attr_name} is of type "
                    f"{type(attr_name)}: {type_error}"
                )


class GroupNode(HDFNode):
    """
    Node for an HDF Group
    """

    def __init__(self, name=None):
        """
        Initialization
        """
        super(GroupNode, self).__init__(name)

        self._children = list()

    @property
    def children(self):
        return self._children[:]

    def _create_child_name(self, short_name):
        """Create name of a child with full path from a short name

        For example:
            self._name = '/entry/DASlogs/Sampleid'
            child short name = 'time'
            child name with full path = '/entry/DASlogs/Sampleid/time'

        Parameters
        ----------
        short_name: str
            short name of child without full path

        Returns
        -------

        """
        return f"{self._name}/{short_name}"

    def match(self, other_node):
        """Compare this node with another node

        Parameters
        ----------
        other_node

        Returns
        -------

        """
        # call base class
        super(GroupNode, self).match(other_node)

        # compare child
        for child in self._children:
            child_name = child.name
            other_child = other_node.get_child(child_name)
            child.match(other_child)

    def get_child(self, child_name, is_short_name=False):
        """Get a child

        Parameters
        ----------
        child_name: str
        is_short_name: bool
            If True, concatenate the child name with current self._name

        Returns
        -------
        GroupNode, DataSetNode
            Child HDFNode

        """
        # process name
        if is_short_name:
            if self._name.endswith("/"):
                child_name = f"{self._name}{child_name}"
            else:
                child_name = f"{self._name}/{child_name}"

        child_node = None
        for child_node_i in self._children:
            if child_node_i.name == child_name:
                child_node = child_node_i
                break

        if child_node is None:
            raise RuntimeError(
                f"There is no child node with name {child_name} for node {self.name})"
            )

        return child_node

    def remove_child(self, child_node_name):
        for child_node in self._children:
            if child_node.name == child_node_name:
                self._children.remove(child_node)

    def set_child(self, child_node):
        """

        Parameters
        ----------
        child_node: GroupNode, DataSetNode
            child node to append

        Returns
        -------

        """
        # Check whether a child with same name exists
        for child_node_i in self._children:
            if child_node_i.name == child_node.name:
                raise RuntimeError(
                    f"Node {self.name} has child with name {child_node.name} already!"
                )

        # Attach
        self._children.append(child_node)

    def parse_h5_entry(self, h5_entry):
        """Parse HDF5 entry

        Parameters
        ----------
        h5_entry: ~h5py._hl.group.Group
            hdf5 entry

        Returns
        -------
        None

        """
        # Parse in general way
        super(GroupNode, self).parse_h5_entry(h5_entry)

        # parse children
        children_names = h5_entry.keys()
        for child_name in children_names:
            child_node = parse_h5_entry(h5_entry[child_name])
            self._children.append(child_node)

    def write(self, parent_entry):
        """Write buffer node to an HDF entry

        Parameters
        ----------
        parent_entry:  ~h5py._hl.dataset.Dataset, ~h5py._hl.group.Group, ~h5py._hl.files.File
            parent HDF node

        Returns
        -------
        """
        # create group or data set
        # h5py._hl.group.Group only
        curr_entry = parent_entry.create_group(self._name)
        # write
        self.write_content(curr_entry)

    def write_content(self, curr_entry):
        # write child
        for child in self._children:
            child.write(curr_entry)

        # attributes
        self.write_attributes(curr_entry)


class FileNode(GroupNode):
    """
    Node for an HDF file
    """

    def __init__(self):
        """
        Initialization
        """
        super(FileNode, self).__init__("/")

    def write(self, file_name):
        """Write to a file

        Parameters
        ----------
        file_name: str
            Name of file to write to

        Returns
        -------

        """
        # create file node
        h5 = h5py.File(file_name, "w")
        # write
        self.write_content(h5)

        # close
        h5.close()


class DataSetNode(HDFNode):
    """
    Node for data set
    """

    def __init__(self, name=None):
        """
        Initialization
        """
        super(DataSetNode, self).__init__(name)

        self._value = None

    def match(self, other_node):
        """Match this node with other

        Parameters
        ----------
        other_node: DataSetNode
            another node to match against

        Returns
        -------

        """
        # call base class's match
        super(DataSetNode, self).match(other_node)

        # compare this one
        try:
            np.testing.assert_allclose(self._value, other_node.value)
        except AssertionError as ass_err:
            raise ValueError(ass_err)
        except TypeError:
            # in case value is not float or integer
            if self._value.shape != other_node.value.shape:
                raise ValueError(
                    f"Node {self._name}: Value have different shape: self = {self.value.shape}, "
                    f"other = {other_node.value.shape}"
                )
            this_value = self._value.flatten()
            that_value = other_node.value.flatten()
            for i in range(this_value.shape[0]):
                if this_value[i] != that_value[i]:
                    raise ValueError(
                        "Different values:\n 1: {}\n 2: {}".format(
                            self._value, other_node.value
                        )
                    )

    def parse_h5_entry(self, h5_entry):
        """Parse HDF5 entry

        Parameters
        ----------
        h5_entry: ~h5py._hl.group.Group
            hdf5 entry

        Returns
        -------
        None

        """
        # Parse in general way
        super(DataSetNode, self).parse_h5_entry(h5_entry)

        # Parse value
        self._value = h5_entry[()]

    @property
    def value(self):
        return self._value

    def set_value(self, data_array):
        """Set data value (as numpy array)

        Parameters
        ----------
        data_array: np.ndarray
            data value

        Returns
        -------

        """
        self._value = data_array

    def set_string_value(self, str_value):
        """Set value from a single string (object)

        Parameters
        ----------
        str_value: str
            string to be written to an entry's only value

        Returns
        -------

        """
        # it is possible that input string is of type as unicode.  so it is better
        # to enforce it to be a string (encoded string) that can be accepted by h5py
        self._value = np.array([np.string_(str_value)])

    def write(self, parent_entry):
        """Write buffer node to an HDF entry

        Parameters
        ----------
        parent_entry:  ~h5py._hl.group.Group, ~h5py._hl.files.File
            parent HDF node

        Returns
        -------

        """
        # Generate current entry and set the data
        curr_entry = parent_entry.create_dataset(self._name, data=self._value)

        self.write_attributes(curr_entry)
