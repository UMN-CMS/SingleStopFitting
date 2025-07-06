from pathlib import Path
import itertools as it
import argparse
from collections import namedtuple
import re

SignalParts = namedtuple("SignalParts", "coupling stop chargino")


def getSignalParts(path):
    s = next(x for x in path.parts if "signal_" in x)
    _, _, coupling, stop, charg = s.split("_")
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


SPECIAL_CATS = {}


EXCLUDE = [
    (800, 400),
    (800, 600),
    (800, 700),
    (900, 700),
    (1000, 100),
    (2000, 100),
    (1500, 100),
]


SPECIAL_SPREADS = {
    (1500.0, 1100.0): [1.5],
    (2000.0, 1500.0): [1.4],
    (1200.0, 900.0): [1.4],
}


def getCat(s, c):
    if (s, c) in SPECIAL_CATS:
        return SPECIAL_CATS[(s, c)]
    c = float(c)
    s = float(s)
    if c / s >= 0.75:
        return ["comp"]
    else:
        return ["uncomp"]


def getSpreads(s, c):
    if (s, c) in SPECIAL_SPREADS:
        return SPECIAL_SPREADS[(s, c)]
    if c / s < 0.74:
        return [1.75]
    else:
        return [1.4]


LUMI_SCALE_MAP = {
    "2016_preVFP": 19.65,
    "2016_postVFP": 16.98,
    "2017": 41.48,
    "2018": 59.83,
    "2022_preEE": 7.98,
    "2022_postEE": 26.67,
    "2023_preBPix": 17.65,
    "2023_postBPix": 9.451,
}


def parseArgs():
    parser = argparse.ArgumentParser(description="Make the list")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("-o", "--output", type=str, help="Output file name")
    parser.add_argument("-b", "--background")
    parser.add_argument("-s", "--signal")
    parser.add_argument("--subpath", type=str, default="{cat}")
    parser.add_argument("--years", type=str, nargs="+")
    parser.add_argument("--background-toys", type=int, default=None)
    parser.add_argument("-i", "--injections", nargs="+", type=float, required=True)
    parser.add_argument(
        "--spreads", nargs="+", type=float, required=False, default=None
    )
    parser.add_argument("--outdir", type=str)

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
        if (s.stop, s.chargino) in EXCLUDE:
            continue
        spreads = args.spreads or getSpreads(s.stop, s.chargino)
        cats = getCat(s.stop, s.chargino)
        regions = getRegion(s.stop, s.chargino)
        area = getMassArea(s.stop)
        for c, r, t, inject, spread, year in it.product(
            cats, regions, range(bt), args.injections, spreads, args.years
        ):
            vals = dict(
                signal_name=next(x for x in path.parts if "signal_" in x),
                area=area,
                cat=c,
                region=r,
                **s._asdict(),
                toy=t,
                spread=spread,
                inject=inject,
                year=year,
                lumi=LUMI_SCALE_MAP[year],
            )

            f = getFileNoCase(
                path,
                args.signal.format(**vals),
            )
            cols = [
                # "signal_" + "_".join(map(str, s)),
                vals["signal_name"],
                args.outdir.format(**vals).replace(".", "p"),
                f"Signal{s.coupling}",
                args.subpath.format(**vals).replace(".", "p"),
                str(f),
            ]
            cols.append(args.background.format(**vals))
            cols += [str(inject), str(spread)]
            cols.append(str(vals["lumi"]))
            ret.append(" ".join(cols))
    print("\n".join(ret))


if __name__ == "__main__":
    main()
