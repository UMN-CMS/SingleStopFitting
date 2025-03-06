import argparse
import json
from pathlib import Path

import uproot


def getOne(path):
    with uproot.open(path) as f:
        limit = f["limit"]
        sig = limit["limit"].array(library="np").tolist()[0]
    return {"significance" : sig}


def combineAll(files):
    ret = {}
    for f in files:
        f = Path(f)
        print(f)
        parts = f.parts
        signal_part = next(x for x in parts if "signal_" in x)
        one = getOne(f)
        _, coupling, mt, mx = signal_part.split("_")
        mt, mx = int(mt), int(mx)
        signal_metadata = dict(coupling=coupling, mass_stop=mt, mass_chargino=mx)
        ret[signal_part] = {"data": one, "meta": signal_metadata}
    return ret


def parseArguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-o", "--output")
    parser.add_argument("inputs", nargs="+")
    return parser.parse_args()


def main():
    args = parseArguments()
    res = combineAll(args.inputs)
    with open(args.output, "w") as f:
        f.write(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
