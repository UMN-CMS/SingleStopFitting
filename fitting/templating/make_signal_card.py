from .latex_env import renderTemplate
import itertools as it
from collections import defaultdict
from pathlib import Path
import json
import argparse
from rich import print
from fitting.core import SignalRunCollection


# def loadOneMeta(p):
#     p = Path(p)
#     parent = p.parent
#     with open(p, "r") as f:
#         metadata = json.load(f)
#     data = metadata
#     plots = {
#         x.stem: str(x)
#         for x in it.chain(parent.glob("*.pdf"), parent.parent.glob("*.pdf"))
#     }
#     plots["covar_center"] = next((y for x, y in plots.items() if "covariance_" in x), None )
#     data = {**metadata, "plots": plots}
#     return data


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-r", "--rate-injected", type=float)
    parser.add_argument("-o", "--output")
    return parser.parse_args()


def main():
    args = parseArgs()

    ret = defaultdict(list)
    with open(args.input, "r") as f:
        data = SignalRunCollection.model_validate_json(f.read())

    data = data.filter(signal_injected=args.rate_injected)

    r = renderTemplate("signal_card.tex", {"all_signals": data.model_dump()})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(r)


if __name__ == "__main__":
    main()
