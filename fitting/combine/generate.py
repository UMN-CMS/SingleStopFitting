import logging
import sys
from pathlib import Path

import numpy as np

import fitting.models
import fitting.transformations as transformations
import gpytorch
import torch
import uproot
from fitting.regression import DataValues, makeRegressionData
from fitting.utils import getScaledEigenvecs, modelToPredMVN, chi2Bins

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
    wanted = vals > vals[0] * sig_percent
    nz = int(torch.count_nonzero(wanted))
    print(f"There are {nz} egeinvariations at least {sig_percent} of the max ")
    good_vals, good_vecs = vals[wanted], vecs[wanted]
    print(vals)
    print(torch.max(vals))
    for i, (va, ve) in enumerate(zip(good_vals, good_vecs)):
        # print(f"Magnitude is {torch.abs(v).max()}")
        h_up = tensorToHist(mean + va * ve)
        root_file[f"bkg_estimate_EVAR_{i}Up"] = h_up
        h_down = tensorToHist(mean - va * ve)
        root_file[f"bkg_estimate_EVAR_{i}Down"] = h_down
    return nz


def createDatacard(obs, pred, signal_data, output_dir):
    print(f"Generating combine datacard in {output_dir}")
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


def getPrediction(bkg_data):
    hist = bkg_data["input_data"]
    raw_regression_data, *_ = makeRegressionData(hist)
    bm = bkg_data["blind_mask"]
    dm = bkg_data["domain_mask"]

    all_data = raw_regression_data.getMasked(dm)
    blinded_data = all_data.getMasked(~bm)

    transform = transformations.getNormalizationTransform(blinded_data)

    normalized_blinded_data = transform.transform(blinded_data)
    normalized_all_data = transform.transform(all_data)

    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=normalized_blinded_data.V,
        learn_additional_noise=False,
        noise_constraint=gpytorch.constraints.GreaterThan(1e-10),
    )
    model = fitting.models.NonStatParametric(
        normalized_blinded_data.X, normalized_blinded_data.Y, likelihood
    )
    model.load_state_dict(bkg_data["model_dict"])
    model.eval()
    likelihood.eval()

    pred_dist = modelToPredMVN(
        model,
        likelihood,
        normalized_all_data,
        slope=transform.transform_y.slope,
        intercept=transform.transform_y.intercept,
    )
    pred_data = DataValues(all_data.X, pred_dist.mean, pred_dist.variance, all_data.E)
    good_bin_mask = all_data.Y > 50
    global_chi2_bins = chi2Bins(pred_data.Y, all_data.Y, all_data.V, good_bin_mask)
    blinded_chi2_bins = chi2Bins(pred_data.Y, all_data.Y, all_data.V, bm)

    return all_data, pred_dist


def main():
    base_dir = Path(sys.argv[1]).absolute()
    out_dir = Path(sys.argv[2])
    signal_dirs = list(base_dir.glob("signal*"))

    for signal_dir in signal_dirs:
        signal_name = signal_dir.relative_to(base_dir).parts[0]
        print(signal_name)
        signal_data_path = signal_dir / "signal_data.pth"
        est_path = signal_dir / "inject_r_0p0"
        est_data_path = est_path / "train_model.pth"
        sig_data = torch.load(signal_data_path)  # , weights_only=True)
        bkg_data = torch.load(est_data_path)
        obs, pred = getPrediction(bkg_data)
        createDatacard(obs, pred, sig_data["signal_data"], out_dir / signal_name / "inject_r_0p0") 


if __name__ == "__main__":
    main()
