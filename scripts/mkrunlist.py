from pathlib import Path
from rich import print
import argparse
from collections import namedtuple
import re

SignalParts = namedtuple("SignalParts", "coupling stop chargino")


def getSignalParts(path):
    s = next(x for x in path.parts if "signal_" in x)
    _, coupling, stop, charg = s.split("_")
    return SignalParts(coupling, int(stop), int(charg))


def getMassArea(m):
    mapping = reversed([(0, "low"), (1200, "med"), (1700, "high")])
    for x, y in mapping:
        if m > x:
            return y

def getCat(s, c):
    c = float(c)
    if c / s > 0.8:
        return ["comp"]
    elif c / s > 0.6:
        return ["comp", "uncomp"]
    else:
        return ["uncomp"]


def parseArgs():
    parser = argparse.ArgumentParser(description="Make the list")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("-o", "--output", type=str, help="Output file name")
    parser.add_argument("-b", "--background")
    parser.add_argument("-s", "--signal")

    return parser.parse_args()

def getFileNoCase(path,pattern):
    for x in path.glob("*"):
        if x.name.lower() == pattern.lower():
            return x
    return None
    

def main():
    args = parseArgs()
    inputs = [Path(x) for x in args.inputs]
    signal_mapping = {getSignalParts(x): x for x in inputs}
    for s, path in signal_mapping.items():
        cats = getCat(s.stop, s.chargino)
        area = getMassArea(s.stop)
        for c in cats:
            f = getFileNoCase(path, args.signal.format(c,c))
            cols = ["signal_" + "_".join(map(str,s)), c, str(f)]
            cols.append(args.background.format(c, area))
            print(" ".join(cols))


if __name__ == "__main__":
    main()
