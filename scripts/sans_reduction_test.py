import json
import sys
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
# from mantid.kernel import ConfigService
# ConfigService.setString('UpdateInstrumentDefinitions.OnStartup', '0')
from mantid.simpleapi import LoadEventNexus, Segfault  # noqa: E402


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        json_string = " ".join(sys.argv[1:])
        json_parameters = json.loads(json_string)
        filename = "{}_{}".format("EQSANS", json_parameters["runNumber"])
        w = LoadEventNexus(filename)
        Segfault(DryRun=not json_parameters["fail"])
    else:
        raise RuntimeError("reduction code requires a parameter json string")
