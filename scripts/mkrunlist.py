from pathlib import Path
import itertools as it
import argparse
from collections import namedtuple
import re

SignalParts = namedtuple("SignalParts", "coupling stop chargino")


def getSignalParts(path):
    s = next(x for x in path.parts if "signal_" in x)
    _, coupling, stop, charg = s.split("_")
    return SignalParts(coupling, int(stop), int(charg))


def getMassArea(m):
    mapping = reversed([(0, "low"), (1250, "med"), (1750, "high")])
    for x, y in mapping:
        if m > x:
            return y


def getRegion(s, c, cat=None):
    ratio = c / s
    mapping = reversed([(0, "uncomp"), (0.7, "comp"), (0.89, "ucomp")])
    for x, y in mapping:
        if ratio > x:
            return [y]


def getCat(s, c):
    c = float(c)
    # return ["comp", "uncomp"]
    if c / s > 0.75:
        return ["comp"]
    # elif c / s > 0.6:
    #     return ["comp", "uncomp"]
    else:
        return ["uncomp"]


def parseArgs():
    parser = argparse.ArgumentParser(description="Make the list")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("-o", "--output", type=str, help="Output file name")
    parser.add_argument("-b", "--background")
    parser.add_argument("-s", "--signal")
    parser.add_argument("--subpath", type=str, default="{cat}")
    parser.add_argument("--background-toys", type=int, default=None)
    parser.add_argument("-i", "--injections", nargs="+", type=float, required=True)
    parser.add_argument("--spreads", nargs="+", type=float, required=True)

    return parser.parse_args()


def getFileNoCase(path, pattern):
    for x in path.rglob("*.pkl"):
        if x.name.lower() == pattern.lower():
            return x
    return None


def main():
    args = parseArgs()
    inputs = [Path(x) for x in args.inputs]
    signal_mapping = {getSignalParts(x): x for x in inputs}
    ret = []
    bt = args.background_toys or 1
    for s, path in signal_mapping.items():
        cats = getCat(s.stop, s.chargino)
        regions = getRegion(s.stop, s.chargino)
        area = getMassArea(s.stop)
        for c, r, t, inject, spread in it.product(
            cats, regions, range(bt), args.injections, args.spreads
        ):
            vals = dict(
                area=area,
                cat=c,
                region=r,
                **s._asdict(),
                toy=t,
                spread=spread,
                inject=inject,
            )

            f = getFileNoCase(
                path,
                args.signal.format(**vals),
            )
            cols = [
                "signal_" + "_".join(map(str, s)),
                f"Signal{s.coupling}",
                args.subpath.format(**vals).replace(".","p"),
                str(f),
            ]
            cols.append(args.background.format(**vals))
            cols += [str(inject), str(spread)]
            ret.append(" ".join(cols))
    print("\n".join(ret))


if __name__ == "__main__":
    main()
