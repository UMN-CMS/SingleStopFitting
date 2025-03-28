import logging
import json
from pathlib import Path

import argparse
import numpy as np

import torch
import uproot
from fitting.regression import loadModel, getModelingData, getPosteriorProcess
from fitting.utils import getScaledEigenvecs

from .datacard import Channel, DataCard, Process, Systematic

torch.set_default_dtype(torch.float64)


logger = logging.getLogger(__name__)


def tensorToHist(array):
    a = array.numpy()
    hist = (a, np.arange(0, a.shape[0] + 1))
    return hist


def createHists(obs, pred, signal_data, root_file, sig_percent=0.0):
    cov_mat = pred.covariance_matrix
    mean = pred.mean
    vals, vecs = getScaledEigenvecs(cov_mat)
    root_file["bkg_estimate"] = tensorToHist(mean)
    root_file["signal"] = tensorToHist(signal_data.Y)
    root_file["data_obs"] = tensorToHist(obs.Y)
    wanted = torch.sqrt(vals) >= torch.sqrt(vals[0]) * sig_percent
    nz = int(torch.count_nonzero(wanted))
    print(f"There are {nz} egeinvariations at least {sig_percent} of the max ")
    good_vals, good_vecs = vals[wanted], vecs[wanted]
    print(torch.max(vals))
    for i, (va, ve) in enumerate(zip(good_vals, good_vecs)):
        v = torch.sqrt(va)
        # v = va
        # print(f"Magnitude is {torch.abs(v).max()}")
        h_up = tensorToHist(mean + v * ve)
        root_file[f"bkg_estimate_EVAR_{i}Up"] = h_up
        h_down = tensorToHist(mean - v * ve)
        root_file[f"bkg_estimate_EVAR_{i}Down"] = h_down
    return nz


def createDatacard(obs, pred, signal_data, output_dir, signal_meta=None):
    print(f"Generating combine datacard in {output_dir}")
    signal_meta = signal_meta or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    root_path = output_dir / "histograms.root"
    root_file = uproot.recreate(root_path)

    nz = createHists(obs, pred, signal_data, root_file, 0.01)

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

    card.addObservation(
        b1, "histograms.root", "data_obs", int(torch.sum(obs.Y))
    )

    for i in range(0, nz):
        s = Systematic(f"EVAR_{i}", "shape")
        card.addSystematic(s)
        card.setProcessSystematic(bkg, s, b1, 1)

    with open(output_dir / "datacard.txt", "w") as f:
        f.write(card.dumps())

    metadata = {"signal_metadata" : signal_meta }

    with open(output_dir / "metadata.json", 'w') as f:
        f.write(json.dumps(metadata))

def main(args):
    for data_path in args.inputs:
        p = Path(data_path)
        parent = p.parent
        signal_name = next(x for x in p.parts if "signal_" in x)
        print(signal_name)
        relative = parent.relative_to(Path("."))
        if args.base:
            relative=relative.relative_to(Path(args.base))
        signal_data_path = parent.parent / "signal_data.pth"
        sig_data = torch.load(signal_data_path)  # , weights_only=True)
        bkg_data = torch.load(p)
        model = loadModel(bkg_data)
        obs, mask = getModelingData(bkg_data)
        blinded = obs.getMasked(mask)
        pred = getPosteriorProcess(model, obs, bkg_data.transform)
        _, coupling ,mt, mx = signal_name.split("_")
        mt,mx = int(mt), int(mx)
        signal_metadata = dict(name=signal_name, coupling=coupling, mass_stop = mt, mass_chargino=mx, rate=bkg_data.metadata["signal_injected"])
        (args.output / relative).mkdir(exist_ok=True, parents=True)
        createDatacard(obs, pred, sig_data["signal_data"], args.output / relative, signal_meta=signal_metadata) 



def addDatacardGenerateParser(parser):
    parser.add_argument('--output')
    parser.add_argument('--base')
    parser.add_argument('inputs',nargs="+" )
    parser.set_defaults(func=main)



    

if __name__ == "__main__":
    main()
