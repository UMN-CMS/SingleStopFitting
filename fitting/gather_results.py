import argparse
from typing import Annotated, Any
import sys

from pydantic import BaseModel, Field, TypeAdapter
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
from fitting.core import Metadata, SignalPoint, SignalRun, SignalRunCollection


# SignalId = namedtuple("SignalId", "algo coupling mt mx")

logger = logging.getLogger(__name__)


def loadOneGPR(directory):
    logger.debug(f"Loading gpr directory {directory}")
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    chi_path = directory / "chi2_info.json"
    post_pred_path = directory / "post_pred_data.json"

    try:
        with open(metadata_path, "r") as f:
            metadata = Metadata.model_validate_json(f.read())
    except OSError as e:
        logger.warn(f"Could not find {metadata_path}")
        return None

    with open(chi_path, "r") as f:
        chi2_data = json.load(f)

    try:
        with open(post_pred_path, "r") as f:
            post_pred_data = json.load(f)
    except OSError as e:
        logger.warn(f"Error with file {post_pred_path}")
        post_pred_data = None

    data = SignalRun(
        metadata=metadata, chi2_info=chi2_data, post_pred_info=post_pred_data
    )

    plots = {
        x.stem: str(x)
        for x in it.chain(
            directory.glob("*.pdf"),
            directory.parent.glob("*.pdf"),
            directory.glob("*.png"),
            directory.parent.glob("*.png"),
        )
    }
    plots["covar_center"] = next(
        (y for x, y in plots.items() if "covariance_" in x), None
    )
    data.regression_plots = plots
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
    name = "gof"

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
        logger.debug(f"Extracted rate {val} from file {f}")
        if val:
            val = val[0]
        else:
            logger.warn(f"No rate found in file {f}")
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
                val = None
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
                logger.info(f"Failed to load fit from {f}")
                val = None
            else:
                val = extractProperty(f, "r")
            logger.debug(f)
            logger.debug(val)
            if val:
                val = val[0]
            else:
                logger.warn(f"No rate found")
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
            logger.info(f"Failed to load limit from {f}")
            return None
        val = extractProperty(f, "limit")
        if val:
            k = ["5", "16", "50", "84", "95", "obs"]
            val = dict(zip(k, val))
        else:
            val = None
        logger.debug(f"Extracted limit {val} from {f}")
        return val


combine_extractors = [
    # ExtractGOF(),
    ExtractFit("extracted_rate", "fit"),
    # ExtractFit("fit_asimov", "fitasimov"),
    ExtractLimit("limit", "lim"),
    ExtractSig("significance", "sig"),
    # ExtractSigInject("significance_for_inject", "sig", for_inject="0p0"),
    # ExtractRateInject("rate_for_inject", "fit", for_inject="0p0"),
]


def loadOneCombine(directory):
    directory = Path(directory)
    logger.debug(f"Loading combine directory {directory}")
    metadata_path = directory / "metadata.json"

    try:
        with open(metadata_path, "r") as f:
            metadata = Metadata.model_validate_json(f.read())
    except OSError as e:
        logger.warn(f"Could not find {metadata_path}")
        return None

    all_data = {
        "signal": metadata.signal_point,
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

    gathered = defaultdict(list)
    total_gpr = 0
    for d in (loadOneGPR(d) for d in Path(".").glob(args.gpr_dirs)):
        total_gpr += 1
        if d is not None:
            gathered[(d.signal_point, d.metadata.year)].append(d)
    total_combine = 0
    if args.combine_dirs:
        for c in (loadOneCombine(d) for d in Path(".").glob(args.combine_dirs)):
            total_combine += 1

            try:
                i = c["metadata"].fit_params.injected_signal
                t = c["metadata"].fit_region.background_toy
                s = c["metadata"].window.spread
                y = c["metadata"].year

                l = gathered[(c["signal"], y)]
                injected = next(
                    (
                        x
                        for x in l
                        if x.metadata.fit_params.injected_signal == i
                        and x.metadata.fit_region.background_toy == t
                        and x.metadata.window.spread == s
                        and x.metadata.year == y
                    ),
                    None,
                )
                if injected is None:
                    logger.warn(
                        f"Could not find associated gpr for combine point {c['metadata'].signal_point}"
                    )

                injected.inference_data.update(c["data"])

            except Exception as e:
                logger.warn(f"Failed to gather for combine")
                # raise

    gathered = [y for x in gathered.values() for y in x]
    # print(gathered)
    # gathered  = dict(gathered)

    # to_write = signal_run_list_adapter.dump_json(gathered, indent=2).decode("utf-8")
    # to_write = signal_run_list_adapter.dump_json(gathered, indent=2).decode("utf-8")
    coll = SignalRunCollection(gathered)
    to_write = coll.model_dump_json(indent=2)
    logger.info(f"Gathered {total_gpr} GPR dirs and {total_combine} combine dirs")
    # to_write = SignalRun.dump_json(gathered, indent=2).decode("utf-8")
    if args.output == "-":
        sys.stdout.write(to_write)
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(exist_ok=True, parents=True)
        with open(out_path, "w") as f:
            f.write(to_write)


def addGatherParser(parser):
    import argparse

    parser.add_argument("--combine-dirs", type=str, default=None)
    parser.add_argument("--gpr-dirs", required=True, type=str)
    parser.add_argument("-o", "--output", default="-")
    parser.set_defaults(func=main)

    return parser
