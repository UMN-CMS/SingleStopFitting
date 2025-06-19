import logging
import matplotlib.pyplot as plt
from fitting.plotting.plot_tools import plotRaw
from fitting.config import Config
import json
from pathlib import Path
import gpytorch
from fitting.plotting.plot_tools import plotRaw

import argparse

import pyro
import pyro.distributions as pyrod
import pyro.infer as pyroi
import numpy as np
import code
import readline
import rlcompleter

#

import torch
import uproot
from fitting.regression import loadModel, getModelingData, getPosteriorProcess
from fitting.utils import getScaledEigenvecs

from .datacard import Channel, DataCard, Process, Systematic

torch.set_default_dtype(torch.float64)


logger = logging.getLogger(__name__)


def decomposedModel(variations):
    cv = pyro.sample(
        "normals",
        pyrod.Normal(torch.zeros(variations.shape[0]), torch.ones(variations.size(0))),
    )
    ret = torch.sum(variations * torch.unsqueeze(cv, dim=-1), axis=0)
    return pyro.deterministic("obs", ret)


def tensorToHist(array):
    a = array.numpy()
    hist = (a, np.arange(0, a.shape[0] + 1))
    return hist


def saveVariation(X, Y, E, name, save_dir):
    import matplotlib.pyplot as plt
    import mplhep

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots()
    plotRaw(ax, E, X, Y)
    # mplhep.hist2dplot(ax=ax)
    fig.savefig(save_dir / f"{name}.png")
    plt.close(fig)


def createHists(
    obs,
    pred,
    signal_data,
    root_file,
    sig_percent=0.0,
    save_n_variations=None,
    save_dir=None,
):
    cov_mat = pred.covariance_matrix
    mean = torch.clip(pred.mean, min=0, max=None)
    vals, vecs = getScaledEigenvecs(cov_mat)
    root_file["bkg_estimate"] = tensorToHist(mean)
    root_file["signal"] = tensorToHist(signal_data.Y)
    root_file["data_obs"] = tensorToHist(obs.Y)
    if sig_percent is not None:
        wanted = vals >= vals[0] * sig_percent
    else:
        wanted = torch.full_like(vals, False, dtype=bool)
        print(f"Not including systematics")
    nz = int(torch.count_nonzero(wanted))
    print(f"There are {nz} egeinvariations at least {sig_percent} of the max ")
    good_vals, good_vecs = vals[wanted], vecs[wanted]
    all_vars = []
    for i, (va, ve) in enumerate(zip(good_vals, good_vecs)):
        v = torch.sqrt(va)
        # v = va

        var = v * ve

        all_vars.append(var)
        # raw_h_up = torch.clip(mean + var, min=0, max=None)
        # raw_h_down = torch.clip(mean - var, min=0, max=None)
        raw_h_up = mean + var
        raw_h_down = mean - var
        h_up = tensorToHist(raw_h_up)
        h_down = tensorToHist(raw_h_down)

        if save_n_variations is not None and i < save_n_variations:
            logger.info(f"Saving variation {i}")
            saveVariation(
                signal_data.X,
                var,
                signal_data.E,
                f"CMS_GPRMVN_Eigenvar_Rank_{i}",
                save_dir=save_dir,
            )
            saveVariation(
                signal_data.X,
                raw_h_up,
                signal_data.E,
                f"EVAR_{i}_UP",
                save_dir=save_dir,
            )
            saveVariation(
                signal_data.X,
                raw_h_down,
                signal_data.E,
                f"EVAR_{i}_DOWN",
                save_dir=save_dir,
            )

        root_file[f"bkg_estimate_CMS_GPRMVN_Eigenvar_Rank_{i}Up"] = h_up
        root_file[f"bkg_estimate_CMS_GPRMVN_Eigenvar_Rank_{i}Down"] = h_down

    all_vars = torch.stack(all_vars)
    predictive = pyroi.Predictive(decomposedModel, num_samples=800)
    pred = predictive(all_vars)
    v = torch.var(pred["obs"], axis=0)
    saveVariation(signal_data.X, v, signal_data.E, f"Recomposed", save_dir=save_dir)
    return nz


def createDatacard(
    obs, pred, signal_data, output_dir, signal_meta=None, syst_threshold=0.05
):
    print(f"Generating combine datacard in {output_dir}")
    signal_meta = signal_meta or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    root_path = output_dir / "histograms.root"
    root_file = uproot.recreate(root_path)

    nz = createHists(
        obs,
        pred,
        signal_data,
        root_file,
        syst_threshold,
        save_n_variations=2,
        save_dir=output_dir / "evars",
    )

    card = DataCard()

    bkg = Process("BackgroundEstimate", False)
    sig = Process("Signal", True)
    b1 = Channel("SignalRegion")
    card.addChannel(b1)
    card.addProcess(sig)
    card.addProcess(bkg)

    card.setProcessRate(sig, b1, -1)
    card.setProcessRate(bkg, b1, -1)
    card.addShape(
        bkg,
        b1,
        "histograms.root",
        "bkg_estimate",
        "bkg_estimate_$SYSTEMATIC",
    )
    card.addShape(sig, b1, "histograms.root", "signal", "")

    card.addObservation(b1, "histograms.root", "data_obs", int(torch.sum(obs.Y)))

    for i in range(0, nz):
        s = Systematic(f"EVAR_{i}", "shape")
        card.addSystematic(s)
        card.setProcessSystematic(bkg, s, b1, 1)

    out_card = output_dir / "datacard.txt"
    logger.info(f"Saving to {out_card}")
    with open(out_card, "w") as f:
        f.write(card.dumps())



def main(args):
    for data_path in args.inputs:
        p = Path(data_path)
        parent = p.parent
        signal_name = next(x for x in p.parts if "signal_" in x)
        print(signal_name)
        relative = parent.relative_to(Path("."))
        if args.base:
            relative = relative.relative_to(Path(args.base))
        signal_data_path = parent.parent / "signal_data.pth"
        sig_data = torch.load(
            signal_data_path, weights_only=False
        )  # , weights_only=True)
        bkg_data = torch.load(p, weights_only=False)
        model = loadModel(bkg_data)
        obs, mask = getModelingData(bkg_data)
        blinded = obs.getMasked(mask)
        pred = getPosteriorProcess(model, obs, bkg_data.transform)
        signal_data = sig_data["signal_data"]

        # import code
        # import readline
        # import rlcompleter
        #
        # vars = globals()
        # vars.update(locals())
        # readline.set_completer(rlcompleter.Completer(vars).complete)
        # readline.parse_and_bind("tab: complete")
        # code.InteractiveConsole(vars).interact()

        save_dir = args.output / relative
        save_dir.mkdir(exist_ok=True, parents=True)

        def saveFunc(name, obj):
            import json

            if isinstance(obj, dict):
                with open(save_dir / f"{name}.json", "w") as f:
                    json.dump(obj, f)
            else:
                ext = Config.IMAGE_TYPE
                name = name.replace("(", "").replace(")", "").replace(".", "p")
                obj.savefig((save_dir / name).with_suffix(f".{ext}"))
                plt.close(obj)

        if args.blind_only:
            pred = gpytorch.distributions.MultivariateNormal(
                pred.mean[mask],
                pred.covariance_matrix[..., mask][mask, ...],
            )
            obs = blinded
            signal_data = signal_data.getMasked(mask)

        _, coupling, mt, mx = signal_name.split("_")
        mt, mx = int(mt), int(mx)
        # signal_metadata = dict(
        #     name=signal_name,
        #     coupling=coupling,
        #     mass_stop=mt,
        #     mass_chargino=mx,
        #     rate=bkg_data.metadata.fit_params.injected_signal,
        # )
        fig, ax = plt.subplots()
        plotRaw(ax, signal_data.E, signal_data.X, pred.variance)
        saveFunc("combine_post_variance", fig)

        createDatacard(
            obs,
            pred,
            signal_data,
            args.output / relative,
            # signal_meta=signal_metadata,
            syst_threshold=args.syst_threshold,
        )


def addDatacardGenerateParser(parser):
    parser.add_argument("--output")
    parser.add_argument("--base")
    parser.add_argument("--syst-threshold", default=0.0, type=float)
    parser.add_argument(
        "--blind-only", default=True, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("inputs", nargs="+")
    parser.set_defaults(func=main)


if __name__ == "__main__":
    main()
