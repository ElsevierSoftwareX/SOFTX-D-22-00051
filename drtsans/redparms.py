import ast
from copy import deepcopy
import itertools
import json
import jsonschema
from jsonschema.exceptions import relevance
import os
import re
import warnings

warnings.filterwarnings("ignore")

from mantid.kernel import IntArrayProperty  # noqa: E404

from drtsans import configdir  # noqa: E404, non-source files
from drtsans.instruments import (
    instrument_filesystem_name,
    instrument_standard_name,
    instrument_standard_names,
)  # noqa: E404
from drtsans.path import abspath, abspaths  # noqa: E404


__all__ = [
    "reduction_parameters",
    "update_reduction_parameters",
    "validate_reduction_parameters",
]


def type_selector(preferred_type):  # noqa: C901
    r"""
    Callable the will cast an object to a preferred type.

    Example: type_enforcer('str')(42) returns '42'

    Parameters
    ----------
    preferred_type: str
        One of 'str', 'int', 'float', '[str]', '[float]', '[dict]'

    Returns
    -------
    :py:obj:`function`
    """

    def list_comprehension_exec(item_type):
        r"""
        Select the list below comprehension executor based on the type of the list item

        Parameters
        ----------
        item_type: type
            The preferred type of the list items

        Returns
        -------
        function
            a list comprehension executor in charge of casting the items in a list to a particular type.
        """

        def comprehension_executor(a_list):
            r"""
            Conversion of the items in a list to a selected type. The list can be in a string representation,
            or a number

            Examples:
            - [1, 2], ['1', '2'], '1, 2', '1 2' are all cast into [item_type('1'), item_type('2')
            - 1 is cast into [item_type(1)]
            - 1.0 is cast into [item_type(1.0)]

            Parameters
            ----------
            a_list: int, float, str, list
                If list, then list items must be either of int, float, str.

            Returns
            -------
            list
                items in this list are of type `item_type`
            """
            if isinstance(a_list, (int, float)):
                a_list = [a_list]
            elif (
                isinstance(a_list, str) is True
            ):  # a_list is the string representation of a list
                for brace in ("[", "]", "(", ")"):
                    a_list = a_list.replace(brace, "")
                if "," in a_list:
                    a_list = a_list.split(",")
                else:
                    a_list = a_list.split(None)
            return [item_type(x) for x in a_list]

        return comprehension_executor

    def list_comprehension_str2dict(instance):
        r"""
        Convert the items of a list into dictionaries. It is understood that each list items is the
        string representation of a dictionary.

        Example: ["{'Pixel':'1-18,239-256'}", "{'Bank':'18-24,42-48'}"]

        Parameters
        ----------
        instance: list

        Returns
        -------
        dict
        """
        list_of_dictionaries = list()
        for item in instance:
            if isinstance(item, str):
                try:
                    list_of_dictionaries.append(ast.literal_eval(item))
                except ValueError:
                    raise ValueError(
                        f'Could not translate "{item}" into a python dictionary'
                    )
            elif isinstance(item, dict):
                list_of_dictionaries.append(item)
            else:
                raise ValueError(
                    f"{instance} must be a list of strings or a list of dictionaries"
                )
        return list_of_dictionaries

    def run_str(instance):
        r"""
        Cast to a list of run numbers into a string, iterating over items when the object to cast is a list

        Examples: [1, '2', '3 - 5', '6:8'] and '1, 2, 3-5, 6 : 8' both  become '1, 2, 3, 4, 5, 6, 7, 8'

        Parameters
        ----------
        instance:str, list

        Returns
        -------
        str
        """
        if isinstance(instance, (list, tuple)):
            return ", ".join([run_str(item) for item in instance])
        # expand any run range, such as 12345-12349
        instance = str(instance)
        # Expand
        for run_range in re.findall(r"(\d+\s*[-:]\s*\d+)", instance):
            run_range_no_whitespaces = run_range.replace(" ", "")
            try:
                all_runs = IntArrayProperty("_", run_range_no_whitespaces).valueAsStr
            except RuntimeError as e:
                raise ValueError(
                    "Invalid range format: {}".format(run_range_no_whitespaces)
                ) from e
            instance = instance.replace(run_range, all_runs)
        return instance

    # the type_selector cast the input instance to desired data-type
    dispatcher = {
        "int": int,
        "float": float,
        "str": str,
        "runstr": run_str,
        "[float]": list_comprehension_exec(float),
        "[str]": list_comprehension_exec(str),
        "[dict]": list_comprehension_str2dict,
    }
    return dispatcher[preferred_type]


class ReferenceResolver:
    r"""
    Uncompress all {'$ref': URI} items with the value of the keyword pointed to by the URI.

    Recursive uncompression when the URI value contains itself more {'$ref': URI} items.

    Parameters
    ----------
    referred: str
        Absolute path to the JSON schema containing the definitions for the references. Default
        is common.json
    """

    def __init__(self, referred):
        schema_dir = os.path.dirname(referred) + "/"
        schema_file = os.path.basename(referred)
        self._resolver = jsonschema.RefResolver(f"file://{schema_dir}", schema_file)

    def dereference(self, to_resolve):
        r"""
        Resolve all '$ref' keys found in a dictionary. Do a recursive resolution for nested
        dictionaries containing '$ref' keys

        Parameters
        ----------
        to_resolve: dict
            property dictionary containing '$ref' keys

        Returns
        -------
        dict
            property dictionary where items containing '$ref' keys have been resolved
        """
        while "$ref" in to_resolve:
            # resolving a $ref introduces new $ref items when the URI points to
            # a property in common.json which contains $ref entries
            resolved = self._resolve_uri(to_resolve["$ref"])
            to_resolve.pop("$ref")
            to_resolve.update(resolved)
        for name, value in to_resolve.items():
            if isinstance(value, dict):
                value = self.dereference(
                    value
                )  # nested dictionary must be resolved, too
                to_resolve[name] = value
        return to_resolve

    def _resolve_uri(self, reference):
        r"""
        Resolve a reference as its corresponding dictionary.

        No resolution of nested references is performed.

        Parameters
        ----------
        reference: str
            The value associated to a '$ref' property name

        Returns
        -------
        dict
            resolved reference value
        """
        _, fragment = self._resolver.resolve(reference)
        return fragment


class Suggest:
    r"""
    Set of property names in a schema. The purpose of this class if to offer property name suggestions
    when a user enters a property name that is not in the set.

    Parameters
    ----------
    entries: list
        A list of strings representing the property names in a schema
    """

    @staticmethod
    def levenshtein(query, entry):
        r"""
        Calculate similarity between two strings.

        Credit to <https://www.python-course.eu/levenshtein_distance.php>

        Parameters
        ----------
        query: str
            potential entry typed in by the user.
        entry: str
            One of the entries in the JSON file/object.

        Returns
        -------
        int
        """
        rows, cols = len(query) + 1, len(entry) + 1
        dist = [[0 for x in range(cols)] for x in range(rows)]

        # source prefixes can be transformed into empty strings by deletions:
        for i in range(1, rows):
            dist[i][0] = i

        # entry prefixes can be created from an empty source string by inserting the characters
        for i in range(1, cols):
            dist[0][i] = i

        for col in range(1, cols):
            for row in range(1, rows):
                if query[row - 1] == entry[col - 1]:
                    cost = 0
                else:
                    cost = 1
                dist[row][col] = min(
                    dist[row - 1][col] + 1,  # deletion
                    dist[row][col - 1] + 1,  # insertion
                    dist[row - 1][col - 1] + cost,
                )  # substitution
        return dist[row][col]

    def __init__(self, entries):
        self.entries = set(entries)

    def __len__(self):
        return len(self.entries)

    def top_match(self, query):
        r"""
        Find the similarity between a query word and the reduction parameter entries, then return the
        most similar entry

        Parameters
        ----------
        query: str
            Potential entry typed in by the user

        Returns
        -------
        str
        """
        scores = [(Suggest.levenshtein(query, entry), entry) for entry in self.entries]
        return sorted(scores)[0][1]


resolver_common = ReferenceResolver(os.path.join(configdir, "schema", "common.json"))


class DefaultJson:
    r"""
    JSON dictionary containing only keywords related to physical properties (i.e., the domain
    of the business)

    This version of the schema is easy to review by the instrument scientists because it contains only
    keywords related to the instrument. Keywords specifying the data-types and the validators are
    omitted.

    Parameters
    ----------
    schema: dict
    field: str
        entry in the parameter's schema that we want to use as default value for the parameter.
    """

    # public class/static methods:

    @staticmethod
    def trim_schema(schema, field="default"):
        r"""
        Trim the (key, val) items from a full schema so that keys are assigned their default values,
        or :py:obj:`None`

        Parameters
        ----------
        schema: dict
        field: str
            entry in the schema for each parameter that we want to use as value

        Returns
        -------
        dict
         Property names and default values
        """
        # Bring items under 'properties' as items of the current schema
        if "properties" in schema:
            dict_slim = dict()
            for name, value in schema["properties"].items():
                dict_slim[name] = DefaultJson.trim_schema(value, field=field)
            return dict_slim
        return schema.get(field, None)

    # public bound methods:

    @property
    def parameters(self):
        return self._json

    @property
    def property_names(self):
        return set(self._property_names())

    def dumps(self, **kwargs):
        r"""
        JSON-dump schema to string after removing certain names from the dictionary.

        The goal is to yield a dictionary containing only instrument-related property names and values.

        Parameters
        ----------
        output_json: str, io.StringIO
            File name or File-like object where to save the reduction parameters. If passing file name,
            the file extension  must by ".json"
        kwargs: dict
            Optional arguments for json.dump

        Returns
        -------
        str
        """
        # default formatting
        if "indent" not in kwargs:
            kwargs["indent"] = 2
        return json.dumps(self._json, **kwargs)

    def dump(self, output_json, **kwargs):
        r"""
        JSON dump

        Parameters
        ----------
        output_json: str, io.StringIO
            File name or File-like object where to save the reduction parameters. If passing file name,
            the file extension  must by ".json"
        kwargs: dict
            Optional arguments for json.dump
        """
        file_handle = (
            output_json  # initialize assuming the file handle is a file-like object
        )
        if isinstance(output_json, str):
            if output_json.endswith(".json") is False:
                raise RuntimeError(f'"{output_json}" extension is not ".json"')
            file_handle = open(output_json, "w")
        # default formatting
        if "indent" not in kwargs:
            kwargs["indent"] = 2
        json.dump(self._json, file_handle, **kwargs)
        file_handle.close()

    def to_rest(self, line_length_max=75):
        r"""
        Represent the default parameters as a restructuredtext string.

        Returns
        -------
        str
        """

        def quote_in(lines):
            r"""Fix the quote marks when splitting a long line into a list of shorter lines"""
            if len(lines) == 1:
                return lines  # the line was not splited, so nothing to do
            quoted_lines = [lines[0] + '"']  # first line needs ending '"'
            for line in lines[1:-1]:
                quoted_lines.append('"' + line + '"')  # middle lines must be quoted
            quoted_lines.append('"' + lines[-1])  # last line needs beginning '"'
            return quoted_lines

        name = self.parameters["instrumentName"]
        doc = f"{name}\n" + "".join(["="] * len(name)) + "\n\n"  # instrument header
        doc += ".. code-block:: python\n\n"
        def_dict = list()
        # Split long lines
        for line in self.dumps(indent=2).split("\n"):
            def_dict.extend(
                quote_in(
                    [
                        line[i : i + line_length_max]
                        for i in range(0, len(line), line_length_max)
                    ]
                )
            )
        # Three white spaces for proper indentation of restructuredtext blockquotes
        def_dict = "\n   ".join(def_dict)
        # Change back some keywords from JSON representation to Python
        for json_key, python_key in {
            ": true": ": True",
            ": false": ": False",
            ": null": ": None",
        }.items():
            def_dict = def_dict.replace(json_key, python_key)
        doc += "   " + def_dict + "\n\n"
        return r"{}".format(doc)

    # private bound methods:

    def __init__(self, schema, field="default"):
        self._field = field
        self._json = self.trim_schema(schema, field=field)

    def __getitem__(self, item):
        return self._json.__getitem__(item)

    def __setitem__(self, key, value):
        self._json.__setitem__(key, value)

    def __str__(self):
        return self._to_string()

    def _to_string(self, parent_dictionary=None, n=0):
        r"""
        Pretty print the schema dictionary. Includes indentation and default values.

        Parameters
        ----------
        parent_dictionary: dict
            format this dictionary as a string
        n: int
            indentation level

        Returns
        -------
        str
        """
        if parent_dictionary is None:
            s = f"#\n# property-name ({self._field} value)\n#\n"
            parent_dictionary = self._json
        else:
            s = ""
        for name, value in parent_dictionary.items():
            s += "    " * n + f"{name}"
            if isinstance(value, dict) is False:
                print_value = "" if value is None else f" = {value}"
                s += f"{print_value}\n"
            else:
                s += ":\n"
                s += self._to_string(parent_dictionary=value, n=n + 1)
        return s

    def _iteritems_recursive(self, parent_dictionary=None):
        r"""
        Iterate over the schema's items, descending recursively into lower-level dictionaries"""
        if parent_dictionary is None:
            parent_dictionary = self._json
        for name, value in parent_dictionary.items():
            if isinstance(value, dict):
                self._iteritems_recursive(parent_dictionary=value)
            yield (name, value)

    def _property_names(self, parent_dictionary=None):
        r"""List of all property names, as we navigate the schema dictionary. Duplicates in the list are OK."""
        if parent_dictionary is None:
            parent_dictionary = self._json
        names = list(parent_dictionary.keys())
        for value in parent_dictionary.values():
            if isinstance(value, dict):
                names.extend(self._property_names(parent_dictionary=value))
        return names


class ReductionParameters:
    r"""
    Validation and sanity checks for all parameters defining a reduction.

    Broadly speaking, two types of validation against the JSON schema are enforced:
    1. validators built-in jsonschema package
    2. method of this class that start with "_validate". This methods are inserted in the
    JSON schema via dedicated keywords ('dataSource', 'useEntry',..)

    Parameters
    ----------
    parameters: dict
        The dictionary of reduction parameters to be validated.
    schema_instrument: dict
        One of the instrument schemae under subdirectory 'schema' of ~drtsans.configdir
    """

    # Routines organization
    # 1. private class variables
    # 2. public class methods and static functions
    # 3. member initialization
    # 4. public bound methods
    # 5. private bound methods

    # 1. private class variables

    _validators = {
        "dataSource": "_validate_data_source",
        "evaluateCondition": "_validate_evaluate_condition",
        "lessThan": "_validate_less_than",
        "exclusiveOr": "_validate_exclusive_or",
        "fluxFileTOF": "_validate_flux_file_tof",
        "pairedTo": "_validate_is_paired_to",
        "onlyOneTrue": "_validate_only_one_true",
        "sameLen": "_validate_equal_len",
        "useEntry": "_validate_use_entry",
        "wedgeSources": "_validate_wedge_sources",
    }

    # 2. public class methods and static functions

    @staticmethod
    def initialize_suggestions(schema_instrument):
        schema_resolved = resolver_common.dereference(schema_instrument)
        default_json = DefaultJson(schema_resolved)
        return Suggest(default_json.property_names)

    # 3. member initialization

    def __init__(self, parameters, schema_instrument):
        self._parameters = parameters
        self._schema = schema_instrument
        # object in charge of offering a suggestion when the user enters the wrong property name
        self._entries = ReductionParameters.initialize_suggestions(schema_instrument)
        # object called when executing any of the validators
        self._json_validator = self._initialize_json_validator()
        # object that resolves all {'$ref', URI} items
        self._reference_resolver = jsonschema.RefResolver(
            f'file://{os.path.join(configdir, "schema")}/', "common.json"
        )
        self._initialize_parameters()

    # 4. public bound methods

    def validate(self):
        r"""
        Run all the validators on the reduction parameters

        This code reproduces jsonschema.validate omitting error selection when more
        than one error is found

        Raises
        ------
        jsonschema.ValidationError
        """
        self._json_validator.check_schema(self._schema)
        validator = self._json_validator(
            self._schema, resolver=self._reference_resolver
        )
        errors = iter(validator.iter_errors(self._parameters))
        best = next(errors, None)
        if best is None:
            return
        error = max(itertools.chain([best], errors), key=relevance)
        if error is not None:
            raise error

    def dump(self, output_json, target="parameters", **kwargs):
        r"""
        Save the reduction parameters dictionary to a JSON object

        Parameters
        ----------
        output_json: str, io.StringIO
            File name or File-like object where to save the reduction parameters. If passing file name,
            the file extension  must by ".json"
        target: str
            One of 'parameters' or 'schema', depending on what dictionary we want to dump
        kwargs: dict
            Optional arguments for json.dump
        """
        target_selector = dict(parameters=self.parameters, schema=self._schema)
        file_handle = (
            output_json  # initialize assuming the file handle is a file-like object
        )
        if isinstance(output_json, str):
            if output_json.endswith(".json") is False:
                raise RuntimeError(f'"{output_json}" extension is not ".json"')
            file_handle = open(output_json, "w")
        # default formatting
        if "indent" not in kwargs:
            kwargs["indent"] = 2
        json.dump(target_selector[target], file_handle, **kwargs)
        file_handle.close()

    def dumps(self, target="parameters", **kwargs):
        r"""
        print the reduction parameters dictionary to a string

        Parameters
        ----------
        target: str
            One of 'parameters' or 'schema', depending on what dictionary we want to dump
        kwargs: dict
            Optional arguments for json.dump

        Returns
        -------
        str
        """
        target_selector = dict(parameters=self.parameters, schema=self._schema)
        # default formatting
        if "indent" not in kwargs:
            kwargs["indent"] = 2
        return json.dumps(target_selector[target], **kwargs)

    def get_parameter_value(self, composite_key):
        r"""
        Find the value for a composite key uniquely identifying one of the reduction parameter properties.

        Composite keys (optionally) start with character '#', and keys are joined with backslash '/' character.
        Example: '#configuration/Qmin' will search for `_parameters['configuration']['Qmin']`

        Parameters
        ----------
        composite_key: str

        Returns
        -------
        bool, str, int, float
        """
        unrooted_composite_key = composite_key.replace("#", "")
        value = self._parameters
        for name in unrooted_composite_key.split("/"):
            value = value[name]
        return value

    @property
    def parameters(self):
        r"""Accessor-only to private _parameters"""
        return self._parameters

    # 5. private bound methods

    def __getitem__(self, item):
        try:
            return self._parameters.__getitem__(item)
        except KeyError as e:
            raise KeyError(f'{e}. Did you mean "{self._entries.top_match(item)}"?')

    def __setitem__(self, key, value):
        top_match = self._entries.top_match(key)
        if top_match != key:
            warnings.warn(
                f'{key} not found in the parameters dictionary. Closest match is "{top_match}"'
            )
        self._parameters.__setitem__(key, value)

    def _initialize_json_validator(self):
        # Elucidate the draft version for the meta-schema
        meta_schemas = {"draft-07": jsonschema.Draft7Validator}
        # Find which schema-draft version in self._schema
        meta_schema_key = re.search("(draft-[0-9]+)", self._schema["$schema"]).groups()[
            0
        ]
        meta_schema = meta_schemas[
            meta_schema_key
        ]  # select schema appropriate to the schema-draft version
        #
        all_validators = dict(meta_schema.VALIDATORS)
        for keyword, function_name in self._validators.items():
            function = getattr(self, function_name)
            all_validators[keyword] = function
        return jsonschema.validators.create(
            meta_schema=meta_schema.META_SCHEMA, validators=all_validators
        )

    def _initialize_parameters(self, schema=None, parameters=None):
        r"""
        1. If value is empty string, set to None
        2. If value is None and has a default, set to the default value
        2. If value is a non-empty string but can be a number, set to integer or float
        """

        if schema is None:
            schema = self._schema
            parameters = self._parameters
        # We need list() in order to capture the initial state of parameter values
        for name, parameter_value in list(parameters.items()):
            try:
                schema_value = schema["properties"][
                    name
                ]  # schema dictionary associated to parameter_value
            except KeyError:
                try:
                    schema_value = schema["additionalProperties"][name]
                except KeyError as key_err:
                    properties_keys = schema["properties"].keys()
                    if "additionalProperties" in schema.keys():
                        properties_keys.extend(schema["additionalProperties"].keys())
                    errmsg = "Available properties: {}".format(properties_keys)
                    raise KeyError(errmsg + ".  " + str(key_err))
            if isinstance(parameter_value, dict) is True:
                # recursive call for nested dictionaries. We pass references to the child dictionaries
                # for the schema and the reduction parameters
                self._initialize_parameters(
                    schema=schema_value, parameters=parameter_value
                )
            else:
                # initialization of parameters[name begins
                if parameter_value in ("", None):
                    parameters[name] = schema_value.get(
                        "default", None
                    )  # is there a default value?
                elif "preferredType" in schema_value:  # parameter_value is not empty
                    cast = type_selector(schema_value["preferredType"])
                    parameters[name] = cast(parameter_value)

    def _validate_data_source(self, validator, value, instance, schema):
        r"""
        Check for the existence of the data source. Typically applied to find out if a file exists.

        If looking for event data, look also in the list of directories given by entry "eventDataDirectories".

        Parameters
        ----------
        validator: ~jsonschema.IValidator
        value: str
            One of 'file' or 'events'.
            - 'file' triggers a call to os.path.exists for a search of `instance` in the local file system.
            - 'events' triggers a call to ~drtsans.path.abspath to search the nexus events file associated
            to `instance`. Preconditions are that the JSON files contains entries 'iptsNumber' and 'instrumentName'.
        instance: str
            file path or run number to be validated.
        schema: dict
            schema related to `instance`

        Raises
        ------
        ~jsonschema.ValidationError
            when the validator fails or when `value` is not one of the allowed values
        """
        if instance is not None:
            if isinstance(instance, str) is False:
                yield jsonschema.ValidationError(f"{instance} is not a string")
            data_directories = self.get_parameter_value("#dataDirectories")
            if value == "file":
                try:
                    abspath(instance, directory=data_directories)
                except RuntimeError:
                    yield jsonschema.ValidationError(f"Cannot find file {instance}")
            elif value == "events":  # run number(s)
                instrument_name = instrument_filesystem_name(self["instrumentName"])
                try:
                    # the runNumber preferredType is a list of strings
                    abspaths(
                        instance,
                        instrument=instrument_name,
                        ipts=self["iptsNumber"],
                        directory=data_directories,
                        search_archive=True,
                    )
                except RuntimeError:
                    yield jsonschema.ValidationError(
                        f"Cannot find events file associated to {instance}"
                    )
            else:
                sources = ("file", "events")
                yield jsonschema.ValidationError(
                    f"{value} is not valid data source. Try one of {sources}"
                )

    def _validate_evaluate_condition(self, validator, value, instance, schema):
        r"""
        Checks a condition evaluates to :py:obj:`True`

        Example: len({this}) == len(#configuration/WedgeMinAngles) checks that the length of
        `instance` and the length of the value associated to
        #configuration/WedgeMinAngles is :py:obj:`True`

        Parameters
        ----------
        validator: ~jsonschema.IValidator
        value: str
            condition to be evaluated
        instance: float, list
            current reduction parameter to be compared to other reduction parameters in the condition.
        schema: dict
            schema related to `instance`

        Raises
        ------
        ~jsonschema.ValidationError
            when the condition in `value` evaluates to :py:obj:`False`
        """
        condition = value.replace("{this}", str(instance))
        # replace keywords of other reduction parameters with their corresponding values
        composite_keys = re.findall(r"{#([\w,/]+)}", value)
        condition = condition.replace(
            "#", ""
        )  # python has trouble formatting strings containing '#'
        for composite_key in composite_keys:
            other_instance = self.get_parameter_value(composite_key)
            other_instance_key = (
                f"{{{composite_key}}}"  # the key is enclosed by curly braces
            )
            condition = condition.replace(other_instance_key, str(other_instance))
        if eval(condition) is False:
            yield jsonschema.ValidationError(
                f"{value} condition has evaluated to False"
            )

    def _validate_less_than(self, validator, value, instance, schema):
        r"""
        Check the parameter value is smaller than the value of other parameters

        Example: Qmin should be smaller than Qmax

        Parameters
        ----------
        validator: ~jsonschema.IValidator
        value: str, list
            entry path(s) of the other parameters to compare to (e.g. '#configuration/Qmax')
        instance: str
            file path or run number to be validated.
        schema: dict
            schema related to `instance`

        Raises
        ------
        ~jsonschema.ValidationError
            when the validator fails or when `value` is not one of the allowed values
        """
        if instance is not None and isinstance(instance, (int, float)) is False:
            yield jsonschema.ValidationError(f"{instance} is not a number")
        entry_paths = value
        if isinstance(value, str):
            entry_paths = [value]
        for entry_path in entry_paths:
            other_instance = self.get_parameter_value(entry_path)
            if other_instance is not None:
                if isinstance(other_instance, (int, float)) is False:
                    yield jsonschema.ValidationError(f"{entry_path} is not a number")
                if instance >= other_instance:
                    yield jsonschema.ValidationError(
                        f"{instance} is not smaller than {entry_path}"
                    )

    def _validate_exclusive_or(self, validator, value, instance, schema):
        r"""
        Check that only one of two related entries entries is not :py:obj:`None`.

        Parameters
        ----------
        validator: ~jsonschema.IValidator
        value: str
            entry name for the datum entry associated with the boolean entry.
        instance: str
            file path or run number to be validated.
        schema: dict
            schema related to `instance`

        Raises
        ------
        ~jsonschema.ValidationError
            when the validator fails or when `value` is not one of the allowed values
        """
        other_instance = self.get_parameter_value(value)
        both_true = instance is not None and other_instance is not None
        both_false = instance is None and other_instance is None
        if both_true or both_false:
            yield jsonschema.ValidationError(f"{value}")

    def _validate_use_entry(self, validator, value, instance, schema):
        r"""
        Verify that parameters associated to a 'use' boolean entry is not-empty when the boolean entry evalues
        to :py:obj:`True`.

        Example: if 'useDefaultMask' is True, check entry 'defaultMask' exists and is not empty.

        Parameters
        ----------
        validator: ~jsonschema.IValidator
        value: str, list
            entry(ies) name(s) for the datum entry associated with the boolean entry.
        instance: bool
            only check for the datum entry if :py:obj:`True`
        schema: dict
            schema related to `instance`

        Raises
        ------
        ~jsonschema.ValidationError
            when the validator fails or when `value` is not one of the allowed values
        """
        if not isinstance(instance, bool):
            yield jsonschema.ValidationError(f"{instance} is not a boolean")
        if instance is True:
            composite_keys = (
                [
                    value,
                ]
                if isinstance(value, str)
                else value
            )  # either a string or a list
            for composite_key in composite_keys:
                other_instance = self.get_parameter_value(composite_key)
                if other_instance is None:
                    yield jsonschema.ValidationError(f"{composite_key} is empty")

    def _validate_is_paired_to(self, validator, value, instance, schema):
        r"""
        Verify two parameters must be both either empty or not-empty

        Example: if 'wavelength' parameter is not :py:obj:`None`, check 'wavelengthSpread' parameter is also
        not :py:obj:`None`

        Parameters
        ----------
        validator: ~jsonschema.IValidator
        value: str, list
            entry(ies) associated with this `instance`.
        instance: object
            value for the entry being validated.
        schema: dict
            schema related to `instance`

        Raises
        ------
        ~jsonschema.ValidationError
            when the validator fails or when `value` is not one of the allowed values
        """
        if instance is not None:
            composite_key = value
            other_instance = self.get_parameter_value(composite_key)
            if other_instance is None:
                yield jsonschema.ValidationError(f"{composite_key} is empty")

    def _validate_only_one_true(self, validator, value, instance, schema):
        r"""
        Check that only one boolean entry is :py:obj:`True`. All can be :py:obj:`False`.

        Example: either 'configuration/useTimeSlice' or 'configuration/useLogSlice' can be True,
        but not both.

        Parameters
        ----------
        validator: ~jsonschema.IValidator
        value: str, list
            entry(ies) associated with this `instance`.
        instance: object
            value for the entry being validated.
        schema: dict
            schema related to `instance`

        Raises
        ------
        ~jsonschema.ValidationError
            when the the type of `instance` is not Bool or when more than one entry in `value` plus
            `instance` evaluates to :py:obj:`True`.
        """
        if not isinstance(instance, bool):
            yield jsonschema.ValidationError(f"{instance} is not a boolean")
        truth_count = 1 if instance is True else 0
        composite_keys = (
            [
                value,
            ]
            if isinstance(value, str)
            else value
        )  # either a string or a list
        for composite_key in composite_keys:
            other_instance = self.get_parameter_value(composite_key)
            if other_instance is True:
                truth_count += 1
        if truth_count > 1:
            yield jsonschema.ValidationError(f"More than {value} entry is True")

    def _validate_equal_len(self, validator, value, instance, schema):
        r"""
        Check that two list instances have equal evaluation of len()

        Example: check than WedgeMinAngles and WedgeMaxAngles are list of same length

        Parameters
        ----------
        validator: ~jsonschema.IValidator
        value: str, list
            entry(ies) associated with this `instance`.
        instance: object
            value for the entry being validated. Should be a list or be casted into a list
        schema: dict
            schema related to `instance`

        Raises
        ------
        ~jsonschema.ValidationError
            when the the type of `instance` is not Bool or when more than one entry in `value` plus
            `instance` evaluates to :py:obj:`True`.
        """
        if instance is not None:
            if isinstance(
                instance, str
            ):  # instance is a string representation of a list
                instance = instance.replace("[", "").replace("]", "").split(",")
            target_len = len(instance)
            composite_keys = (
                [
                    value,
                ]
                if isinstance(value, str)
                else value
            )  # either a string or a list
            for composite_key in composite_keys:
                other_instance = self.get_parameter_value(composite_key)
                if instance is None:
                    yield jsonschema.ValidationError(
                        f"list(s) {composite_keys} have different length than instance"
                    )
                if isinstance(
                    other_instance, str
                ):  # instance is a string representation of a list
                    instance = instance.replace("[", "").replace("]", "").split(",")
                if len(other_instance) != target_len:
                    yield jsonschema.ValidationError(
                        f"list(s) {composite_keys} have different length than instance"
                    )

    def _validate_wedge_sources(self, validator, value, instance, schema):
        r"""
        Check that we can derive wedge angels from the given parameters.

        Two sources of wedge parameter are identified:
        1. custom WedgeMinAngles and WedgeMaxAngles.
        2. automatic wedges, requires autoWedgeQmin, autoWedgeQmax, autoWedgeQdelta, autoWedgeAzimuthalDelta,
        autoWedgePeakWidth, autoWedgeBackgroundWidth, and autoWedgeSignalToNoiseMin.

        Parameters
        ----------
        validator: ~jsonschema.IValidator
        value: list
            sets of wedge specifications. Each item in this list is a list itself, containing the parameter names
            specifying the wedge specifications.
        instance: object
            value for the entry being validated. Perform validation only if this instance evaluates to 'wedge'..
        schema: dict
            schema related to `instance`

        Raises
        ------
        ~jsonschema.ValidationError
            when every set of wedge specifications contains at least one empty parameter.
        """
        if instance == "wedge":
            source_set_valid_found = False  # True if we find one source set that allows us to specify the wedge angles
            for source_set in value:
                for composite_key in source_set:
                    other_instance = self.get_parameter_value(composite_key)
                    if other_instance is None:
                        break  # the source set is invalid because we're missing this one parameter
                else:
                    source_set_valid_found = True  # all instances in the source set are not empty. It's a valid set
            if source_set_valid_found is False:
                yield jsonschema.ValidationError(
                    f"We cannot define the wedge angles given the current"
                    f"values or parameters {value}"
                )

    def _validate_flux_file_tof(self, validator, value, instance, schema):
        r"""
        Check that the entry specifying the flux file is not-empty for the selected normalization.

        For example, if `instance` evaluates to 'Monitor', check parameter 'fluxMonitorRatioFile' is not
        empty.

        Parameters
        ----------
        validator: ~jsonschema.IValidator
        value: list
            items in this list are pairs of normalization type and parameter name containing the location of the
            associated flux file
        instance: object
            value for the entry being validated.
        schema: dict
            schema related to `instance`

        Raises
        ------
        ~jsonschema.ValidationError
            when every set of wedge specifications contains at least one empty parameter.
        """
        # assigns `None` for 'Time' normalization, since there's no flux file associated to this type of normalization
        composite_key = value.get(instance, None)
        if composite_key is not None:
            other_instance = self.get_parameter_value(
                composite_key
            )  # path to the flux file
            if other_instance is None:
                yield jsonschema.ValidationError(
                    f"No flux file was specified for {instance} normalization"
                )


def _instrument_json_generator(instrument=None, field="default"):
    r"""
    For each instrument schema, yield a resolved ~drtsans.redparms.DefaultJson instance.

    Parameters
    ----------
    instrument: str
        Name of the instrument. If :py:obj:`None` then generate for all instruments
    field: str
        entry in the parameter's schema that we want to use as default value for the parameter.

    Returns
    -------
    tuple
        A two item tuple. First item is instrument name and second item is the schema trimmed instance
    """
    schema_dir = os.path.join(configdir, "schema")
    instrument_names = (
        instrument_standard_names()
        if instrument is None
        else [instrument_standard_name(instrument)]
    )
    for name in instrument_names:
        schema_file = os.path.join(schema_dir, f"{name}.json")
        with open(schema_file, "r") as file_handle:
            schema_unresolved = json.load(file_handle)
            schema_resolved = resolver_common.dereference(schema_unresolved)
            yield name, DefaultJson(schema_resolved, field=field)


def default_reduction_parameters(instrument_name):
    r"""
    Get the dictionary of reduction parameters with default values from the schema

    Parameters
    ----------
    instrument_name: str

    Returns
    -------
    dict
    """
    _, generated_parameters = list(
        _instrument_json_generator(instrument=instrument_name)
    )[0]
    return generated_parameters.parameters


def pretty_print_schemae(save_dir):
    r"""
    Save a trimmed version of the instrument schemae for revision to file (BIOSANS.txt,...)

    Parameters
    ----------
    save_dir: str
        Absolute path where printed schemae are saved
    """
    for name, default_json in _instrument_json_generator():
        save_path = os.path.join(save_dir, f"{name}.txt")
        open(save_path, "w").write(str(default_json))


def generate_json_files(save_dir, timestamp=True, field="default"):
    r"""
    For each instrument schema, dump only the physical properties and their default
    values into a JSON formatted file.

    Parameters
    ----------
    save_dir: str
        Absolute path where printed schemae are saved
    timestamp: bool
        Include a 'timestamp' entry
    """
    for name, default_json in _instrument_json_generator(field=field):
        save_path = os.path.join(save_dir, f"{name}.json")
        default_json.dump(save_path)


def validate_reduction_parameters(parameters):
    r"""
    Validate reduction parameters against the instrument's schema.

    Parameters
    ----------
    parameters: dict, ~drtsans.redparms.ReductionParameters
        Reduction configuration

    Returns
    -------
    dict
      Validated reduction parameters
    """
    if isinstance(parameters, ReductionParameters):
        parameters = parameters.parameters
    instrument_name = instrument_standard_name(parameters["instrumentName"])
    schema = load_schema(instrument_name)
    parameters = ReductionParameters(parameters, schema)
    parameters.validate()
    return deepcopy(parameters.parameters)


def update_reduction_parameters(parameters_original, parameter_changes, validate=True):
    r"""
    Update the values of a reduction parameters dictionary with values from another dictionary. Handles nested
    dictionaries. Validate after update is done.

    Dictionary `parameters_original` is not modified, but a new copy is produced an updated with
    `parameter_changes`

    Parameters
    ----------
    parameters_original: dict
    parameter_changes: dict
    validate: bool
        Perform validation of the parameters
    Returns
    -------
    dict
    """
    parameters_updated = deepcopy(parameters_original)
    _update_reduction_parameters(parameters_updated, parameter_changes)
    if validate is False:
        return parameters_updated
    return validate_reduction_parameters(parameters_updated)


def _update_reduction_parameters(parameters_original, parameter_changes):
    r"""
    Update the values of a reduction parameters dictionary with values from another dictionary. Handles nested
    dictionaries. Update is performed in-place.

    Parameters
    ----------
    parameters_original: dict
    parameter_changes: dict

    Returns
    -------
    dict
    """
    for name, value in parameter_changes.items():
        if isinstance(value, dict):
            _update_reduction_parameters(parameters_original[name], value)
        else:
            parameters_original[name] = value


def reduction_parameters(
    parameters_particular=None, instrument_name=None, validate=True
):
    r"""
    Serve all necessary (and validated if so desired) parameters for a reduction session of
    a particular instrument.

    Parameters
    ----------
    parameters_particular: dict
        Non-default parameters, particular to the reduction session. If :py:obj:`None`, then the
        default parameters for the specified instrument are passed.
    instrument_name: str
        Mix the non-default parameters with the remaining default parameters appropriate for this instrument.
        If left as :py:obj:`None`,  the instrument name is looked under keyword 'instrumentName' in
        dictionary `parameters_particular`
    validate: bool
        Perform validation of the parameters
    Returns
    -------
    dict
    """
    if parameters_particular is None and instrument_name is None:
        raise RuntimeError(
            "Either `parameters_particular` or `instrument_name` must be specified"
        )
    if instrument_name is None:
        instrument_name = parameters_particular["instrumentName"]
    instrument_name = instrument_standard_name(
        instrument_name
    )  # e.g. change CG2 to GPSANS
    reduction_input = default_reduction_parameters(instrument_name)
    if parameters_particular is None:
        if validate is False:
            return reduction_input  # nothing else to do
        return validate_reduction_parameters(reduction_input)
    return update_reduction_parameters(
        reduction_input, parameters_particular, validate=validate
    )


def load_schema(instrument_name):
    r"""
    Load the schema appropriate to an instrument.

    Parameters
    ----------
    instrument_name: str
        One of the standard instrument names (BIOSANS, EQSANS, GPSANS)

    Returns
    -------
    dict
    """

    file_path = os.path.join(configdir, "schema", f"{instrument_name}.json")
    return json.load(open(file_path, "r"))
