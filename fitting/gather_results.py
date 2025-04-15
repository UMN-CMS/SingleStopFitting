import argparse
import sys

# from pydantic import BaseClass
import fitting
from rich import print
import json
from pathlib import Path
from collections import defaultdict
import re
import itertools as it
from collections import defaultdict, ChainMap
from pathlib import Path
import logging
import json
import uproot
from collections import namedtuple


SignalId = namedtuple("SignalId", "algo coupling mt mx")

logger = logging.getLogger(__name__)


def idFromMeta(data):
    return SignalId(data["algo"], data["coupling"], data["mt"], data["mx"])


def loadOneGPR(directory):
    logger.info(f"Loading gpr directory {directory}")
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    chi_path = directory / "chi2_info.json"
    post_pred_path = directory / "post_pred_data.json"

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    with open(chi_path, "r") as f:
        chi2_data = json.load(f)

    with open(post_pred_path, "r") as f:
        post_pred_data = json.load(f)

    data = {
        "signal_id": idFromMeta(metadata),
        "metadata": metadata,
        "chi2_info": chi2_data,
        "post_pred_info": post_pred_data,
    }

    plots = {
        x.stem: str(x)
        for x in it.chain(directory.glob("*.png"), directory.parent.glob("*.png"))
    }
    plots["covar_center"] = next(y for x, y in plots.items() if "covariance_" in x)
    data = {**data, "gpr_plots": plots}
    return data


def extractProperty(path, p, tree="limit"):
    try:
        with uproot.open(path) as f:
            limit = f[tree]
            sig = limit[p].array(library="np").tolist()
        return sig
    except Exception as e:
        print(e)
        return None


class ExtractGOF(object):
    name = "GOF"

    def __call__(self, directory):
        directory = Path(directory)
        obs = directory / "higgsCombine.gof_obs.GoodnessOfFit.mH120.root"
        toys = directory / "higgsCombine.gof_toys.GoodnessOfFit.mH120.123456.root"
        logger.info(f"Attempting to load GOF from {obs} and {toys}")
        if not (obs.exists() and toys.exists()):
            logger.info(f'Obs exists: {obs.exists()} : "{obs}"')
            logger.info(f'Toys exists: {toys.exists()} : "{toys}"')
            return None
        obs_val = extractProperty(obs, "limit")
        toys_vals = extractProperty(toys, "limit")
        return {"obs": obs_val, "toys": toys_vals}


class ExtractFit(object):
    def __init__(self, name, tag):
        self.name = name
        self.tag = tag

    def __call__(self, directory):
        directory = Path(directory)
        f = directory / f"higgsCombine.{self.tag}.MultiDimFit.mH120.root"
        logger.info(f"Attempting to load fit from {f}")
        if not f.exists():
            return None
        val = extractProperty(f, "r")
        if val:
            val = val[0]
        else:
            val = None
        return {"r": val}


combine_extractors = [
    ExtractGOF(),
    ExtractFit("fit", "fit"),
    ExtractFit("fit_asimov", "fitasimov"),
]


def loadOneCombine(directory):
    directory = Path(directory)
    logger.info(f"Loading combine directory {directory}")
    with open(directory / "metadata.json", "r") as f:
        metadata = json.load(f)
    all_data = {
        "signal_id": idFromMeta(metadata),
        "metadata": metadata,
    }
    data = {}
    for extractor in combine_extractors:
        data[extractor.name] = extractor(directory)

    all_data["data"] = data
    return all_data


def main(args):

    gathered = {}
    for d in [loadOneGPR(d) for d in args.gpr_dirs]:
        sid = d["signal_id"]
        if sid not in gathered:
            gathered[sid] = {"signal_info": sid._asdict(), "injections": []}
        this_inject = {
            "signal_injected": d["metadata"]["signal_injected"],
            "chi2_info": d["chi2_info"],
            "post_pred_info": d["post_pred_info"],
            "gpr_plots": d["gpr_plots"],
        }
        gathered[sid]["injections"].append(this_inject)

    if args.combine_dirs:
        for c in [loadOneCombine(d) for d in args.combine_dirs]:
            item = gathered[c["signal_id"]]
            i = c["metadata"]["signal_injected"]
            injected = next(x for x in item["injections"] if x["signal_injected"] == i)
            injected.update(c["data"])

    gathered = list(gathered.values())

    if args.output == "-":
        json.dump(gathered, sys.stdout, indent=2)
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(exist_ok=True, parents=True)
        with open(out_path, "w") as f:
            json.dump(gathered, f, indent=2)


def addGatherParser(parser):
    import argparse

    parser.add_argument("--combine-dirs", nargs="+")
    parser.add_argument("--gpr-dirs", required=True, nargs="+")
    parser.add_argument("-o", "--output", default="-")
    parser.set_defaults(func=main)

    return parser
