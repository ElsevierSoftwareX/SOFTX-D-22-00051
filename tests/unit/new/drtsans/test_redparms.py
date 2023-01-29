import json
import jsonschema
import os
from pathlib import Path
import pytest
import shutil
import tempfile

from drtsans import configdir
from drtsans.instruments import instrument_standard_names, instrument_standard_name
from drtsans.redparms import (
    DefaultJson,
    Suggest,
    ReductionParameters,
    ReferenceResolver,
    generate_json_files,
    load_schema,
    reduction_parameters,
    resolver_common,
    update_reduction_parameters,
)


@pytest.fixture(scope="module")
def redparms_data():
    schema_common = json.loads(
        r"""
    {
      "$schema": "http://json-schema.org/draft-07/schema#",

      "definitions": {
        "runNumberTypes": {
          "anyOf": [{"type": "string", "minLength": 1, "pattern": "^[1-9][0-9]*$"},
                    {"type": "integer", "minimum": 1},
                    {"type": "array", "items": {"type": ["integer", "string"]}}],
          "preferredType": "int"
          },
        "transmissionValueTypes": {
          "anyOf": [{"type": "string", "pattern": "^$|^0?.[0-9]*$"},
                    {"type": "number", "exclusiveMinimum": 0, "maximum": 1}],
                    "preferredType": "float"
        },
        "safeStringPositiveFloat": {
          "anyOf": [{"type": "string", "pattern": "^$|^[0-9]*.[0-9]*$"},
                    {"type": "number", "exclusiveMinimum": 0}],
          "preferredType": "float"
        }
      },

      "instrumentName": {
        "type": "string",
        "description": "The name of the instrument. Valid values are BIOSANS, EQSANS, and GPSANS",
        "enum": ["BIOSANS", "EQSANS", "GPSANS"],
        "examples": ["BIOSANS", "EQSANS", "GPSANS"]
      },
      "iptsNumber": {
        "anyOf": [{"type": "string", "minLength": 1, "pattern": "^[1-9][0-9]*$"},
                  {"type": "integer", "minimum": 1}],
        "preferredType": "int",
        "description": "The IPTS number for the data files",
        "examples": ["24769"]
      },

      "sample": {
        "type": "object",
        "properties": {
          "runNumber": {
            "$ref": "common.json#/definitions/runNumberTypes",
            "description": "The run number(s) for the sample"
          },
          "transmission": {
            "type": "object",
            "properties": {
              "runNumber": {
                "$ref": "common.json#/definitions/runNumberTypes",
                "description": "The run number(s) for the transmission sample."
              },
              "value": {"$ref": "common.json#/definitions/transmissionValueTypes"}
            },
            "maxProperties": 2,
            "required": ["runNumber", "value"],
            "description": "The transmission for the sample"
          }
        },
        "maxProperties": 2,
        "required": ["transmission", "runNumber"]
      },

      "configuration": {
        "outputDir": {"type": "string", "description": "Output folder"},
        "useTimeSlice": {
          "type": "boolean",
          "useentry": "timeSliceInterval",
          "description": "Indicate whether the data should be processed as time slices"
        },
        "timeSliceInterval": {
          "$ref": "common.json#/definitions/safeStringPositiveFloat",
          "description": "Interval for time slicing"
        }
      }
    }"""
    )

    schema_common_dir = tempfile.mkdtemp(dir="/tmp")
    schema_common_file = os.path.join(schema_common_dir, "common.json")
    with open(schema_common_file, "w") as file_handle:
        json.dump(schema_common, file_handle)

    schema_instrument = json.loads(
        r"""
    {
      "$schema": "http://json-schema.org/draft-07/schema#",
      "type": "object",
      "properties": {
        "instrumentName": {"$ref": "common.json#/instrumentName", "default": "BIOSANS"},
        "iptsNumber": {"$ref": "common.json#/iptsNumber", "default": 42},
        "sample":     {"$ref": "common.json#/sample"},

        "configuration": {
          "type": "object",
          "properties": {
            "outputDir": {"$ref":  "common.json#/configuration/outputDir"},
            "timeSliceInterval": {"$ref":  "common.json#/configuration/timeSliceInterval"}
          },
          "maxProperties": 2,
          "required": ["outputDir", "timeSliceInterval"]
        }

      },
      "maxProperties": 4,
      "required": ["instrumentName", "iptsNumber", "sample", "configuration"]
    }"""
    )

    reduction_parameters = json.loads(
        r"""
    {"instrumentName": "BIOSANS",
     "iptsNumber": 42,
      "sample": {"runNumber": 24, "transmission": {"value": 0.95}},
      "configuration": {"outputDir": "/tmp", "timeSliceInterval": 20}
    }"""
    )

    yield {
        "schema_common": schema_common,
        "schema_common_file": schema_common_file,
        "schema_instrument": schema_instrument,
        "reduction_parameters": reduction_parameters,
    }
    shutil.rmtree(schema_common_dir)  # clean up before leaving module's scope


@pytest.mark.parametrize("instrument_name", instrument_standard_names())
def test_load_schema(instrument_name):
    r"""correct loading of the schema for each instrument"""
    schema = load_schema(instrument_name)
    assert (
        instrument_standard_name(schema["properties"]["instrumentName"]["default"])
        == instrument_name
    )


class TestReferenceResolver:
    def test_init(self, redparms_data):
        file_name = redparms_data["schema_common_file"]
        resolver = ReferenceResolver(file_name)
        assert resolver._resolver.base_uri == f"file://{os.path.dirname(file_name)}/"
        assert resolver._resolver.referrer == os.path.basename(file_name)

    def test_resolve_uri(self, redparms_data):
        resolver = ReferenceResolver(redparms_data["schema_common_file"])
        resolved = resolver._resolve_uri("common.json#/iptsNumber")
        compared = {
            "anyOf": [
                {"type": "string", "minLength": 1, "pattern": "^[1-9][0-9]*$"},
                {"type": "integer", "minimum": 1},
            ],
            "preferredType": "int",
            "description": "The IPTS number for the data files",
            "examples": ["24769"],
        }
        assert resolved == compared
        resolved = resolver._resolve_uri("common.json#/configuration/timeSliceInterval")
        compared = {
            "$ref": "common.json#/definitions/safeStringPositiveFloat",
            "description": "Interval for time slicing",
        }
        assert resolved == compared

    def test_derefence(self, redparms_data):
        resolver = ReferenceResolver(redparms_data["schema_common_file"])
        unresolved = {
            "background": {
                "iptsNumber": {"$ref": "common.json#/iptsNumber"},
                "default": 12345,
            }
        }
        resolved = resolver.dereference(unresolved)
        compared = {
            "background": {
                "iptsNumber": {
                    "anyOf": [
                        {"type": "string", "minLength": 1, "pattern": "^[1-9][0-9]*$"},
                        {"type": "integer", "minimum": 1},
                    ],
                    "preferredType": "int",
                    "description": "The IPTS number for the data files",
                    "examples": ["24769"],
                },
                "default": 12345,
            }
        }
        assert resolved == compared
        unresolved = {
            "configuration": {
                "outputDir": {"type": "string", "description": "Output folder"},
                "timeSliceInterval": {
                    "$ref": "common.json#/definitions/safeStringPositiveFloat",
                    "description": "Interval for time slicing",
                },
            }
        }
        resolved = resolver.dereference(unresolved)
        compared = {
            "configuration": {
                "outputDir": {"type": "string", "description": "Output folder"},
                "timeSliceInterval": {
                    "description": "Interval for time slicing",
                    "preferredType": "float",
                    "anyOf": [
                        {"type": "string", "pattern": "^$|^[0-9]*.[0-9]*$"},
                        {"type": "number", "exclusiveMinimum": 0},
                    ],
                },
            }
        }
        assert resolved == compared
        unresolved = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "configuration": {
                "type": "object",
                "properties": {
                    "outputDir": {"$ref": "common.json#/configuration/outputDir"}
                },
                "required": ["outputDir"],
            },
        }
        resolved = resolver.dereference(unresolved)
        compared = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "configuration": {
                "type": "object",
                "properties": {
                    "outputDir": {"type": "string", "description": "Output folder"}
                },
                "required": ["outputDir"],
            },
        }
        assert resolved == compared
        # reference common.json#/configuration/timeSliceInterval has inside another reference!
        unresolved = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "configuration": {
                "type": "object",
                "properties": {
                    "timeSliceInterval": {
                        "$ref": "common.json#/configuration/timeSliceInterval"
                    }
                },
            },
        }
        resolved = resolver.dereference(unresolved)
        compared = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "configuration": {
                "type": "object",
                "properties": {
                    "timeSliceInterval": {
                        "description": "Interval for time slicing",
                        "preferredType": "float",
                        "anyOf": [
                            {"type": "string", "pattern": "^$|^[0-9]*.[0-9]*$"},
                            {"type": "number", "exclusiveMinimum": 0},
                        ],
                    }
                },
            },
        }
        assert resolved == compared
        unresolved = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {"sample": {"$ref": "common.json#/sample"}},
        }
        resolved = resolver.dereference(unresolved)
        compared = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "sample": {
                    "type": "object",
                    "properties": {
                        "runNumber": {
                            "description": "The run number(s) for the sample",
                            "anyOf": [
                                {
                                    "type": "string",
                                    "minLength": 1,
                                    "pattern": "^[1-9][0-9]*$",
                                },
                                {"type": "integer", "minimum": 1},
                                {
                                    "type": "array",
                                    "items": {"type": ["integer", "string"]},
                                },
                            ],
                            "preferredType": "int",
                        },
                        "transmission": {
                            "type": "object",
                            "properties": {
                                "runNumber": {
                                    "description": "The run number(s) for the transmission sample.",
                                    "anyOf": [
                                        {
                                            "type": "string",
                                            "minLength": 1,
                                            "pattern": "^[1-9][0-9]*$",
                                        },
                                        {"type": "integer", "minimum": 1},
                                        {
                                            "type": "array",
                                            "items": {"type": ["integer", "string"]},
                                        },
                                    ],
                                    "preferredType": "int",
                                },
                                "value": {
                                    "anyOf": [
                                        {"type": "string", "pattern": "^$|^0?.[0-9]*$"},
                                        {
                                            "type": "number",
                                            "exclusiveMinimum": 0,
                                            "maximum": 1,
                                        },
                                    ],
                                    "preferredType": "float",
                                },
                            },
                            "maxProperties": 2,
                            "required": ["runNumber", "value"],
                            "description": "The transmission for the sample",
                        },
                    },
                    "maxProperties": 2,
                    "required": ["transmission", "runNumber"],
                }
            },
        }
        assert resolved == compared
        unresolved = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "sample": {"$ref": "common.json#/sample"},
                "configuration": {
                    "type": "object",
                    "properties": {
                        "timeSliceInterval": {
                            "$ref": "common.json#/configuration/timeSliceInterval"
                        }
                    },
                },
            },
            "maxProperties": 2,
            "required": ["sample", "configuration"],
        }
        resolved = resolver.dereference(unresolved)
        compared = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "sample": {
                    "type": "object",
                    "properties": {
                        "runNumber": {
                            "description": "The run number(s) for the sample",
                            "anyOf": [
                                {
                                    "type": "string",
                                    "minLength": 1,
                                    "pattern": "^[1-9][0-9]*$",
                                },
                                {"type": "integer", "minimum": 1},
                                {
                                    "type": "array",
                                    "items": {"type": ["integer", "string"]},
                                },
                            ],
                            "preferredType": "int",
                        },
                        "transmission": {
                            "type": "object",
                            "properties": {
                                "runNumber": {
                                    "description": "The run number(s) for the transmission sample.",
                                    "anyOf": [
                                        {
                                            "type": "string",
                                            "minLength": 1,
                                            "pattern": "^[1-9][0-9]*$",
                                        },
                                        {"type": "integer", "minimum": 1},
                                        {
                                            "type": "array",
                                            "items": {"type": ["integer", "string"]},
                                        },
                                    ],
                                    "preferredType": "int",
                                },
                                "value": {
                                    "anyOf": [
                                        {"type": "string", "pattern": "^$|^0?.[0-9]*$"},
                                        {
                                            "type": "number",
                                            "exclusiveMinimum": 0,
                                            "maximum": 1,
                                        },
                                    ],
                                    "preferredType": "float",
                                },
                            },
                            "maxProperties": 2,
                            "required": ["runNumber", "value"],
                            "description": "The transmission for the sample",
                        },
                    },
                    "maxProperties": 2,
                    "required": ["transmission", "runNumber"],
                },
                "configuration": {
                    "type": "object",
                    "properties": {
                        "timeSliceInterval": {
                            "description": "Interval for time slicing",
                            "anyOf": [
                                {"type": "string", "pattern": "^$|^[0-9]*.[0-9]*$"},
                                {"type": "number", "exclusiveMinimum": 0},
                            ],
                            "preferredType": "float",
                        }
                    },
                },
            },
            "maxProperties": 2,
            "required": ["sample", "configuration"],
        }
        assert resolved == compared

    # common.json can resolve itself, thus we include it in the list below
    @pytest.mark.parametrize("file_name", ["common"] + instrument_standard_names())
    def test_derefence_schemas(self, file_name):
        r"""Check all schemas can be resolved using common.json"""
        schema_dir = os.path.join(configdir, "schema")
        json_file = os.path.join(schema_dir, f"{file_name}.json")
        with open(json_file, "r") as file_handle:
            to_resolve = json.load(file_handle)
            resolver_common.dereference(to_resolve)


class TestSuggest:

    property_names = [
        "instrumentName",
        "iptsNumber",
        "sample",
        "thickness",
        "transmission",
        "runNumber",
        "runNumber",
    ]

    @pytest.mark.parametrize(
        "query, score",
        [("home", 0), ("Home", 1), ("hom", 1), ("ho_me", 1), ("Ho_E", 3)],
    )
    def test_levenshtein(self, query, score):
        assert Suggest.levenshtein(query, "home") == score

    def test_init(self):
        entries = Suggest(self.property_names)
        assert len(entries) == len(self.property_names) - 1

    @pytest.mark.parametrize(
        "query, top_match", [("runumber", "runNumber"), ("Sample", "sample")]
    )
    def test_top_match(self, query, top_match):
        entries = Suggest(self.property_names)
        assert entries.top_match(query) == top_match


@pytest.fixture(scope="module")
def default_json(redparms_data):
    resolver = ReferenceResolver(redparms_data["schema_common_file"])
    schema_resolved = resolver.dereference(redparms_data["schema_instrument"])
    return DefaultJson(schema_resolved)


class TestDefaultJson:
    def test_trim_schema(self, default_json):
        compared = {
            "instrumentName": "BIOSANS",
            "iptsNumber": 42,
            "sample": {
                "runNumber": None,
                "transmission": {"runNumber": None, "value": None},
            },
            "configuration": {"outputDir": None, "timeSliceInterval": None},
        }
        assert compared == default_json._json

    def test_str(self, default_json):
        compared = r"""#
# property-name (default value)
#
instrumentName = BIOSANS
iptsNumber = 42
sample:
    runNumber
    transmission:
        runNumber
        value
configuration:
    outputDir
    timeSliceInterval
"""
        assert compared == str(default_json)

    def test_dumps(self, default_json):
        compared = r"""{
  "instrumentName": "BIOSANS",
  "iptsNumber": 42,
  "sample": {
    "runNumber": null,
    "transmission": {
      "runNumber": null,
      "value": null
    }
  },
  "configuration": {
    "outputDir": null,
    "timeSliceInterval": null
  }
}"""
        assert default_json.dumps() == compared

    def test_dump(self, default_json):
        # open temporary file in 'read and write' mode
        _, json_file_path = tempfile.mkstemp(suffix=".json", dir="/tmp")
        default_json.dump(json_file_path)
        d = json.load(open(json_file_path, "r"))
        os.remove(json_file_path)
        compared = {
            "instrumentName": "BIOSANS",
            "iptsNumber": 42,
            "sample": {
                "runNumber": None,
                "transmission": {"runNumber": None, "value": None},
            },
            "configuration": {"outputDir": None, "timeSliceInterval": None},
        }
        assert d == compared

    def test_to_rest(self, default_json):
        d = default_json.to_rest()
        compared = r"""BIOSANS
=======

.. code-block:: python

   {
     "instrumentName": "BIOSANS",
     "iptsNumber": 42,
     "sample": {
       "runNumber": None,
       "transmission": {
         "runNumber": None,
         "value": None
       }
     },
     "configuration": {
       "outputDir": None,
       "timeSliceInterval": None
     }
   }

"""
        assert d == compared

    def test_property_names(self, default_json):
        compared = {
            "instrumentName",
            "iptsNumber",
            "sample",
            "configuration",
            "runNumber",
            "transmission",
            "runNumber",
            "value",
            "outputDir",
            "timeSliceInterval",
        }
        assert compared == default_json.property_names


class TestReductionParameters:
    def test_init(self, redparms_data):
        ReductionParameters(
            redparms_data["reduction_parameters"], redparms_data["schema_instrument"]
        )


class TestReductionParametersGPSANS:

    parameters_common = {
        "instrumentName": "GPSANS",
        "iptsNumber": 21981,
        "sample": {"runNumber": 9165, "thickness": 1.0},
        "outputFileName": "test_validator_datasource",
        "configuration": {"outputDir": "/tmp", "QbinType": "linear", "numQBins": 100},
    }
    parameters_all = reduction_parameters(parameters_common, validate=False)

    @pytest.mark.parametrize(
        "validator_name, parameter_changes",
        [
            ("dataSource", {"sample": {"runNumber": 666999666}}),
            ("evaluateCondition", {"configuration": {"numQBins": None}}),
        ],
    )
    def test_validators(self, validator_name, parameter_changes, reference_dir):
        parameter_changes["dataDirectories"] = str(Path(reference_dir.new.gpsans))
        with pytest.raises(jsonschema.ValidationError) as error_info:
            update_reduction_parameters(self.parameters_all, parameter_changes)
        assert validator_name in str(error_info.value)


def test_generate_json_files(tmpdir, cleanfile):
    directory = tmpdir.mkdir("generate_json_files")
    cleanfile(directory)
    generate_json_files(directory)


if __name__ == "__main__":
    pytest.main([__file__])
