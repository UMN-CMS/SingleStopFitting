import argparse
from rich import print
import json
from pathlib import Path
from collections import defaultdict
import re

import uproot


def extractProperty(path, p):
    try:
        with uproot.open(path) as f:
            limit = f["limit"]
            sig = limit[p].array(library="np").tolist()
        return {p: sig}
    except Exception as e:
        print(e)
        return {p: None}


def groupFiles(files):
    ret = defaultdict(list)
    for f in files:
        f = Path(f)
        parts = f.parts
        signal_part = next(x for x in parts if "signal_" in x)
        _, coupling, mt, mx = signal_part.split("_")
        ret[(coupling, mt, mx)].append(f)
    return ret


def getBestFile(signal_files, pat):
    return {
        s: sorted(f, key=lambda x: not bool(re.search(pat, str(x))))[0]
        for s, f in signal_files.items()
    }


def parseArguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-o", "--output")
    parser.add_argument(
        "-f", "--favor", type=str, help="Factor signals matching pattern"
    )
    parser.add_argument("inputs", nargs="+")
    return parser.parse_args()


def main():
    args = parseArguments()
    file_groups = groupFiles(args.inputs)
    files = getBestFile(file_groups, args.favor)
    for s,p in files.items():
        ep = extractProperty(p,"r")
        print(p,ep)



if __name__ == "__main__":
    main()
