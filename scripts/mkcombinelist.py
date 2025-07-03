from pathlib import Path
import itertools as it
import argparse
from collections import namedtuple
import re

SignalParts = namedtuple("SignalParts", "coupling stop chargino")


def parseArgs():
    parser = argparse.ArgumentParser(description="Make the list")
    parser.add_argument("inputs")
    parser.add_argument("--base")
    return parser.parse_args()


def main():
    args = parseArgs()
    # inputs = [Path(x) for x in args.inputs]
    for path in Path(".").glob(args.inputs):
        print(path.relative_to(args.base))
        


if __name__ == "__main__":
    main()
