import argparse
import sys
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


def groupFiles(files, cut=None):
    ret = defaultdict(list)
    for f in files:
        f = Path(f)
        parts = f.parts
        if "uncomp" in str(f):
            algo = "uncomp"
        else:
            algo = "comp"

        signal_part = next(x for x in parts if "signal_" in x)
        _, coupling, mt, mx = signal_part.split("_")
        if int(mx) < 300:
            continue
        ret[(coupling, int(mt), int(mx))].append(
            {"signal": (coupling, int(mt), int(mx)), "algo": algo, "file": f}
        )
    return ret


def getBestFile(signal_files, algo=None, cut=None):
    ret = {}
    print(signal_files)
    for k, v in signal_files.items():
        if algo:
            ret[k] = next((x for x in v if x["algo"] == algo), v[0])
        elif cut:
            use = "comp" if k[2] / k[1] > cut else "uncomp"
            ret[k] = next(x for x in v if x["algo"] == use)
    return ret


def parseArguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-o", "--output", default="-")
    parser.add_argument(
        "-a", "--algo", type=str, help="Factor signals matching pattern"
    )
    parser.add_argument(
        "-c",
        "--cut",
        type=float,
    )
    parser.add_argument("-k", "--key", type=str)
    parser.add_argument("inputs", nargs="+")
    return parser.parse_args()


def main():
    args = parseArguments()
    file_groups = groupFiles(args.inputs)
    files = getBestFile(file_groups, algo=args.algo, cut=args.cut)
    res = [
        {"signal": s, "algo": p["algo"], "props": extractProperty(p["file"], args.key)}
        for s, p in files.items()
    ]
    if args.output == "-":
        json.dump(res, sys.stdout, indent=2)
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(exist_ok=True, parents=True)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
