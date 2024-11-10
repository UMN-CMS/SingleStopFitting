from pathlib import Path

import numpy as np

import torch
import uproot
from fitting.utils import getScaledEigenvecs
import logging


logger = logging.getLogger(__name__)

from fitting.estimate import RegressionModel, SignalData
from .datacard import Channel, DataCard, Process, Systematic



def tensorToHist(array):
    a = array.numpy()
    hist = (a, np.arange(0, a.shape[0] + 1))
    return hist


def createHists(regression_data, signal_data, root_file, sig_percent = 0.0):
    bm = regression_data.domain_mask
    cov_mat = regression_data.posterior_dist.covariance_matrix
    mean = regression_data.posterior_dist.mean
    vals,vecs = getScaledEigenvecs(cov_mat)

    root_file["bkg_estimate"] = tensorToHist(mean)
    root_file["signal"] = tensorToHist(signal_data.Y[bm])

    root_file["data_obs"] = tensorToHist(regression_data.test_data.Y + 0 * signal_data.Y[bm])

    wanted = vals > vals[0] * sig_percent
    nz = int(torch.count_nonzero(wanted))
    print(f"There are {nz} egeinvariations at least {sig_percent} of the max ")
    good_vals, good_vecs = vals[wanted], vecs[wanted]
    print(vals)
    print(torch.max(vals))
    for i, (va,ve) in enumerate(zip(good_vals,good_vecs)):
        #print(f"Magnitude is {torch.abs(v).max()}")
        h_up = tensorToHist(mean + va * ve)
        root_file[f"bkg_estimate_EVAR_{i}Up"] = h_up
        h_down = tensorToHist(mean - va * ve)
        root_file[f"bkg_estimate_EVAR_{i}Down"] = h_down
    return nz


def createDatacard(regression_data, signal_data, output_dir, num_bkg_systs=None):
    print(f"Generating combine datacard in {output_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    root_path = output_dir / "histograms.root"
    root_file = uproot.recreate(root_path)

    if num_bkg_systs is None:
        num_bkg_systs = regression_data.test_data.Y.shape[0]
    nz = createHists(regression_data, signal_data, root_file, 0.05)

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
        b1, "histograms.root", "data_obs", int(torch.sum(regression_data.test_data.Y))
    )


    for i in range(0, nz):
        s = Systematic(f"EVAR_{i}", "shape")
        card.addSystematic(s)
        card.setProcessSystematic(bkg, s, b1, 1)

    with open(output_dir / "datacard.txt", "w") as f:
        f.write(card.dumps())


def main():
    print("Starting")
    import sys

    path = Path(sys.argv[1]).absolute()
    spath = Path(sys.argv[2]).absolute()
    d = torch.load(path)
    s = torch.load(spath)
    pd = d.posterior_dist
    sd = s.signal_data

    createDatacard(d, sd, "combineoutput/testout1", 1000) #int(pd.mean.size(0)))

if __name__ == "__main__":
    main()
