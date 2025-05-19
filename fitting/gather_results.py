import argparse
import sys

from pydantic import BaseModel
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


class InjectionResult(BaseModel):
    injection: float

    regression_plots: dict[str, str]


class SignalResult(BaseModel):
    signal_id: SignalId
    signal_metadata: dict


def idFromMeta(data):
    return SignalId(data["algo"], data["coupling"], data["mt"], data["mx"])


def loadOneGPR(directory):
    logger.debug(f"Loading gpr directory {directory}")
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    chi_path = directory / "chi2_info.json"
    post_pred_path = directory / "post_pred_data.json"


    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except OSError as e:
        logger.warn(f"Could not find {metadata_path}")
        return None

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
        for x in it.chain(directory.glob("*.pdf"), directory.parent.glob("*.pdf"))
    }
    plots["covar_center"] = next(
        (y for x, y in plots.items() if "covariance_" in x), None
    )
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
        logger.debug(f"Attempting to load GOF from {obs} and {toys}")
        if not (obs.exists() and toys.exists()):
            logger.debug(f'Obs exists: {obs.exists()} : "{obs}"')
            logger.debug(f'Toys exists: {toys.exists()} : "{toys}"')
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
        logger.debug(f"Attempting to load fit from {f}")
        if not f.exists():
            return None
        val = extractProperty(f, "r")
        if val:
            val = val[0]
        else:
            val = None
        return {"r": val}


class ExtractSig(object):
    def __init__(self, name, tag):
        self.name = name
        self.tag = tag

    def __call__(self, directory, for_inject=None):
        if for_inject is not None:
            if for_inject not in directory:
                return None
        directory = Path(directory)
        f = directory / f"higgsCombine.{self.tag}.Significance.mH120.root"
        logger.debug(f"Attempting to load fit from {f}")
        if not f.exists():
            logger.debug(f"Failed to load significance from {f}")
            return None
        val = extractProperty(f, "limit")
        if val:
            val = val[0]
        else:
            val = None
        return val


class ExtractSigInject(object):
    def __init__(self, name, tag, vals=(1, 4, 9, 16), for_inject=None):
        self.name = name
        self.tag = tag
        self.vals = vals
        self.for_inject = for_inject

    def __call__(self, directory):
        if self.for_inject is not None and self.for_inject not in str(directory):
            return None
        directory = Path(directory)
        ret = []
        for v in self.vals:
            t = self.tag + str(v)
            f = directory / f"higgsCombine.{t}.Significance.mH120.root"
            logger.debug(f"Attempting to load fit from {f}")
            if not f.exists():
                logger.debug(f"Failed to load significance from {f}")
                val= None
            else:
                val = extractProperty(f, "limit")
            logger.debug(f)
            logger.debug(val)
            if val:
                val = val[0]
            else:
                val = None
            ret.append((v, val))
        return ret

class ExtractRateInject(object):
    def __init__(self, name, tag, vals=(1, 4, 9, 16), for_inject=None):
        self.name = name
        self.tag = tag
        self.vals = vals
        self.for_inject = for_inject

    def __call__(self, directory):
        if self.for_inject is not None and self.for_inject not in str(directory):
            return None
        directory = Path(directory)
        ret = []
        for v in self.vals:
            t = self.tag + str(v)
            f = directory / f"higgsCombine.{t}.MultiDimFit.mH120.root"
            logger.debug(f"Attempting to load fit from {f}")
            if not f.exists():
                logger.debug(f"Failed to load fit from {f}")
                val= None
            else:
                val = extractProperty(f, "r")
            logger.debug(f)
            logger.debug(val)
            if val:
                val = val[0]
            else:
                val = None
            ret.append((v, val))
        return ret



class ExtractLimit(object):
    def __init__(self, name, tag):
        self.name = name
        self.tag = tag

    def __call__(self, directory):
        directory = Path(directory)
        f = directory / f"higgsCombine.{self.tag}.AsymptoticLimits.mH120.root"
        logger.debug(f"Attempting to load fit from {f}")
        if not f.exists():
            logger.debug(f"Failed to load limit from {f}")
            return None
        val = extractProperty(f, "limit")
        if val:
            k = ["5", "16", "50", "84", "95", "obs"]
            val = dict(zip(k, val))
        else:
            val = None
        return val


combine_extractors = [
    ExtractGOF(),
    ExtractFit("fit", "fit"),
    ExtractFit("fit_asimov", "fitasimov"),
    ExtractLimit("lim", "limit"),
    ExtractSig("sig", "sig"),
    ExtractSigInject("sig_inject", "sig", for_inject="0p0"),
    ExtractRateInject("fit_inject", "fit", for_inject="0p0"),
]


def loadOneCombine(directory):
    directory = Path(directory)
    logger.debug(f"Loading combine directory {directory}")
    with open(directory / "metadata.json", "r") as f:
        metadata = json.load(f)
    all_data = {
        "signal_id": idFromMeta(metadata),
        "metadata": metadata,
    }
    data = {}
    for extractor in combine_extractors:
        try:
            data[extractor.name] = extractor(directory)
        except Exception as e:
            data[extractor.name] = None

            

    all_data["data"] = data
    return all_data


def main(args):

    gathered = {}
    for d in [loadOneGPR(d) for d in args.gpr_dirs]:
        if not d:
            continue
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
            try:
                item = gathered[c["signal_id"]]
                i = c["metadata"]["signal_injected"]
                injected = next(x for x in item["injections"] if x["signal_injected"] == i)
                injected.update(c["data"])
            except Exception as e:
                logger.warn(f"Failed to gather for combine")
                pass
                

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
