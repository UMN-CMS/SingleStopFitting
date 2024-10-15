import itertools as it
import json
import logging
import pickle as pkl
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import fitting.utils as fit_utils
import gpytorch
import hist
import linear_operator
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import uhi
from gpytorch.kernels import ScaleKernel as SK
from matplotlib.patches import Polygon
from rich import print

from . import models, regression, transformations, windowing
from .plot_tools import createSlices, getPolyFromSquares, makeSquares, simpleGrid
from .utils import chi2Bins

torch.set_default_dtype(torch.float64)


logger = logging.getLogger(__name__)


# class LargeFeatureExtractor(torch.nn.Sequential):
#     def __init__(self):
#         super(LargeFeatureExtractor, self).__init__()
#         self.add_module("linear1", torch.nn.Linear(2, 32))
#         self.add_module("relu1", torch.nn.ReLU())
#         self.add_module("linear2", torch.nn.Linear(32, 16))
#         self.add_module("relu2", torch.nn.ReLU())
#         self.add_module("linear3", torch.nn.Linear(16, 8))
#         self.add_module("relu3", torch.nn.ReLU())
#         self.add_module("linear4", torch.nn.Linear(8, 2))


# ft = LargeFeatureExtractor()


# class GPRegressionModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = gpytorch.kernels.InducingPointKernel(
#             gpytorch.kernels.ScaleKernel(
#                 gpytorch.kernels.RBFKernel(ard_num_dims=2)
#             ),
#             likelihood=likelihood,
#             inducing_points=train_x[::2].clone(),
#         )

#         self.feature_extractor = ft
#         self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

#     def forward(self, x):
#         projected_x = self.feature_extractor(x)
#         projected_x = self.scale_to_bounds(projected_x)
#         mean_x = self.mean_module(projected_x)
#         covar_x = self.covar_module(projected_x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


@dataclass
class RegressionModel:
    input_data: hist.Hist
    window: Any
    domain_mask: torch.Tensor

    train_data: torch.Tensor
    test_data: torch.Tensor

    trained_model: gpytorch.models.ExactGP

    raw_posterior_dist: gpytorch.distributions.MultivariateNormal
    posterior_dist: gpytorch.distributions.MultivariateNormal


@dataclass
class SignalData:
    signal_data: torch.Tensor
    domain_mask: torch.Tensor = None
    signal_hist: hist.Hist = None
    signal_name: str = None


def saveDiagnosticPlots(plots, save_dir):
    for name, (fig, ax) in plots.items():
        o = (save_dir / name).with_suffix(".pdf")
        fig.savefig(o)


def makeSigBkgPlot(train_data, test_data, signal_data, window):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout="tight")
    mask = regression.getBlindedMask(
        test_data.X, test_data.Y, test_data.Y, test_data.V, window
    )
    squares = makeSquares(test_data.X[mask], test_data.E)
    points = getPolyFromSquares(squares)
    # plotting.drawAs2DHist(ax[0][1], plotting.PlotObject.fromHist(sig_hist))

    simpleGrid(ax[0], train_data.E, train_data.X, train_data.Y)
    simpleGrid(ax[1], signal_data.E, signal_data.X, signal_data.Y)
    poly = mpl.patches.Polygon(points, edgecolor="red", fill=False)
    ax[0].add_patch(poly)
    poly = mpl.patches.Polygon(points, edgecolor="red", fill=False)
    ax[1].add_patch(poly)
    return {"sig_bkg": (fig, ax)}


def makeDiagnosticPlots(pred, raw_test, raw_train, mask=None, inducing_points=None):
    ret = {}
    if mask is not None:
        squares = makeSquares(raw_test.X[mask], raw_test.E)
        points = getPolyFromSquares(squares)

    def addWindow(ax):
        if mask is None:
            return
        else:
            poly = Polygon(points, edgecolor="green", linewidth=3, fill=False)
            ax.add_patch(poly)

    def addInducing(ax):
        if inducing_points is not None:
            ax.scatter(inducing_points[:, 0], inducing_points[:, 1], c="red", s=1)

    pred_mean = pred.Y
    pred_variances = pred.V

    all_x2 = (pred_mean - raw_test.Y) ** 2 / raw_test.V
    x2 = torch.sum(all_x2)

    fig, ax = plt.subplots(layout="tight")
    simpleGrid(ax, raw_test.E, raw_train.X, raw_train.Y)
    ax.set_title("Masked Inputs (Training)")
    # plotting.addTitles2D(ax, plotting.PlotObject.fromHist(raw_hist))
    addWindow(ax)
    ret["training_points"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    f = simpleGrid(ax, raw_test.E, raw_test.X, pred_mean)
    ax.set_title("GPR Mean Prediction")
    # plotting.addTitles2D(ax, plotting.PlotObject.fromHist(raw_hist))
    addWindow(ax)
    ret["gpr_mean"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    f = simpleGrid(ax, raw_test.E, raw_test.X, pred_mean)
    ax.set_title("GPR Mean Prediction")
    # plotting.addTitles2D(ax, plotting.PlotObject.fromHist(raw_hist))
    addInducing(ax)
    ret["inducing_gpr_mean"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    simpleGrid(ax, raw_test.E, raw_test.X, raw_test.Y)
    ax.set_title("Observed Outputs")
    addWindow(ax)
    ret["observed_outputs"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    f = simpleGrid(ax, raw_test.E, raw_test.X, raw_test.V)
    ax.set_title("Observed Variances")
    # plotting.addTitles2D(ax, plotting.PlotObject.fromHist(raw_hist))
    addWindow(ax)
    ret["observed_variances"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    f = simpleGrid(ax, raw_test.E, raw_test.X, pred.V)
    ax.set_title("Pred Variances")
    # plotting.addTitles2D(ax, plotting.PlotObject.fromHist(raw_hist))
    addWindow(ax)
    ret["predicted_variances"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    f = simpleGrid(ax, raw_test.E, raw_test.X, torch.sqrt(pred.V) / raw_test.Y)
    f.set_clim(0, 1)
    ax.set_title("Relative Uncertainty (std/val)")
    # plotting.addTitles2D(ax, plotting.PlotObject.fromHist(raw_hist))
    addWindow(ax)
    ret["relative_uncertainty"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    f = simpleGrid(
        ax,
        raw_test.E,
        raw_test.X,
        (raw_test.Y - pred_mean) / torch.sqrt(pred.V),
        cmap="coolwarm",
    )
    f.set_clim(-2.5, 2.5)
    ax.set_title("Pulls")
    # plotting.addTitles2D(ax, plotting.PlotObject.fromHist(raw_hist))
    ax.cax.set_ylabel(r"$\frac{N_{obs}-N_{pred}}{\sigma_{p}}$")
    addWindow(ax)
    ret["pulls_pred"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    f = simpleGrid(
        ax,
        raw_test.E,
        raw_test.X,
        (raw_test.Y - pred_mean) / torch.sqrt(raw_test.V),
        cmap="coolwarm",
    )
    f.set_clim(-2.5, 2.5)
    ax.set_title("Pulls")
    # plotting.addTitles2D(ax, plotting.PlotObject.fromHist(raw_hist))
    ax.cax.set_ylabel(r"$\frac{N_{obs}-N_{pred}}{\sigma_{o}}$")
    addWindow(ax)
    ret["pulls_obs"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")

    all_pulls = (raw_test.Y - pred_mean) / torch.sqrt(raw_test.V)

    p = all_pulls[torch.abs(all_pulls) < np.inf]
    p = all_pulls
    ax.hist(p, bins=np.linspace(-5.0, 5.0, 21), density=True)
    X = torch.linspace(-5, 5, 100)
    g = torch.distributions.Normal(0, 1)
    Y = torch.exp(g.log_prob(X))
    ax.plot(X, Y)

    ax.set_xlabel(r"$\frac{N_{obs}-N_{pred}}{\sigma_{p}}$")
    ax.set_ylabel("Count")
    ret["global_pulls_hist"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    p = all_pulls[mask]
    ax.hist(p, bins=np.linspace(-5.0, 5.0, 21), density=True)
    X = torch.linspace(-5, 5, 100)
    g = torch.distributions.Normal(0, 1)
    Y = torch.exp(g.log_prob(X))
    ax.plot(X, Y)

    ax.set_xlabel(r"$\frac{N_{obs}-N_{pred}}{\sigma_{p}}$")
    ax.set_ylabel("Count")
    ret["window_pulls_hist"] = (fig, ax)

    return ret


def makeSlicePlots(pred, test_data, hist, window, dim, save_dir):
    pred_mean, _ = fit_utils.pointsToGrid(test_data.X, pred.Y, test_data.E)
    pred_var, _ = fit_utils.pointsToGrid(test_data.X, pred.V, test_data.E)
    obs_vals, _ = fit_utils.pointsToGrid(test_data.X, test_data.Y, test_data.E)
    obs_vars, filled = fit_utils.pointsToGrid(test_data.X, test_data.V, test_data.E)

    (save_dir / "slices" / f"along_{dim}").mkdir(parents=True, exist_ok=True)

    for val, f, ax in createSlices(
        pred_mean.hist,
        pred_var.hist,
        obs_vals.hist,
        obs_vars.hist,
        test_data.E,
        filled,
        observed_title="CRData",
        mask_function=window,
        just_window=window is not None,
        slice_dim=dim,
    ):
        # plotting.addTitles1D(
        #     ax, plotting.PlotObject.fromHist(hist[{hist.axes[dim].name: sum}])
        # )
        o = (
            save_dir
            / "slices"
            / f"along_{dim}"
            / (f"slice_{round(float(val),3)}".replace(".", "p") + ".pdf")
        )
        f.savefig(o)
        plt.close(f)


def bumpCut(X, Y):
    m = Y > (1 - 200 / X)
    return m


def makePredModel(model, likelihood, data, slope=None, intercept=None):
    pred_dist = regression.getPrediction(model, likelihood, data)
    with gpytorch.settings.cholesky_max_tries(30):
        psd_pred_dist = fit_utils.fixMVN(pred_dist)

    if slope is not None and intercept is not None:
        pred_dist = fit_utils.affineTransformMVN(psd_pred_dist, slope, intercept)
    else:
        pred_dist = type(pred_dist)(
            psd_pred_dist.mean,
            psd_pred_dist.covariance_matrix.to_dense(),
        )

    return pred_dist


def makePredData(model, likelihood, data, base_data, slope=None, intercept=None):
    pred_dist = makePredModel(model, likelihood, data, slope=slope, intercept=intercept)

    return regression.DataValues(
        base_data.X, pred_dist.mean, pred_dist.variance, base_data.E
    )


def doCompleteRegression(
    inhist,
    window_func,
    kernel=None,
    model_maker=None,
    save_dir="plots",
):

    min_counts = 50
    train_data = regression.makeRegressionData(
        inhist, window_func, domain_mask_function=bumpCut, exclude_less=min_counts
    )
    test_data, domain_mask = regression.makeRegressionData(
        inhist,
        None,
        get_mask=True,
        domain_mask_function=bumpCut,
        exclude_less=min_counts,
    )
    train_transform = transformations.getNormalizationTransform(train_data)
    normalized_train_data = train_transform.transform(train_data)
    normalized_test_data = train_transform.transform(test_data)

    use_cuda = True
    if torch.cuda.is_available() and use_cuda:
        train = normalized_train_data.toGpu()
        norm_test = normalized_test_data.toGpu()
        print("USING CUDA")
    else:
        train = normalized_train_data
        norm_test = normalized_test_data

    lr = 0.01
    ok = False
    while not ok:
        model, likelihood = regression.createModel(
            train, kernel=kernel, model_maker=model_maker, learn_noise=False
        )

        # likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        #     noise=train.V,
        #     learn_additional_noise=False,
        # )
        # model = GPRegressionModel(train.X, train.Y, likelihood)
        print(model)


        if torch.cuda.is_available() and use_cuda:
            model = model.cuda()
            likelihood = likelihood.cuda()

        if hasattr(model.covar_module, "initialize_from_data"):
            model.covar_module.initialize_from_data(train.X, train.Y)

        def validate(model):
            X = normalized_test_data.X.cuda()
            Y = normalized_test_data.Y.cuda()
            V = normalized_test_data.V.cuda()
            mask = regression.getBlindedMask(test_data.X, None, None, None, window_func)
            mask = mask.cuda()
            output = model(X)
            bpred_mean = output.mean
            chi2 = chi2Bins(Y, bpred_mean, V, mask)
            #chi2 = chi2Bins(Y, bpred_mean, output.variance, mask)
            return chi2

        try:
            model, likelihood, evidence = regression.optimizeHyperparams(
                model,
                likelihood,
                train,
                bar=False,
                iterations=1600,
                lr=lr,
                get_evidence=True,
                chi2mask=train_data.Y > min_counts,
                val=validate,
            )
            if torch.cuda.is_available() and use_cuda:
                model = model.cpu()
                likelihood = likelihood.cpu()

            model.eval()
            likelihood.eval()
            logger.info("Done with loop")
            print("Done with loop")

            slope = train_transform.transform_y.slope
            intercept = train_transform.transform_y.intercept

            raw_pred_dist = makePredModel(model, likelihood, normalized_train_data)
            pred_dist = makePredModel(
                model, likelihood, normalized_train_data, slope, intercept
            )

            train_pred_data = makePredData(
                model, likelihood, normalized_train_data, train_data, slope, intercept
            )
            test_pred_data = makePredData(
                model, likelihood, normalized_test_data, test_data, slope, intercept
            )

            normalized_train_pred_data = makePredData(
                model, likelihood, normalized_train_data, normalized_train_data
            )
            normalized_test_pred_data = makePredData(
                model, likelihood, normalized_test_data, normalized_test_data
            )

            good_bin_mask = test_data.Y > min_counts
            global_chi2_bins = chi2Bins(
                test_pred_data.Y, test_data.Y, test_data.V, good_bin_mask
            )
            global_normed_chi2_bins = chi2Bins(
                normalized_test_pred_data.Y,
                normalized_test_data.Y,
                normalized_test_data.V,
                good_bin_mask,
            )
            good_bin_mask = train_data.Y > min_counts
            blinded_chi2_bins = chi2Bins(
                train_pred_data.Y, train_data.Y, train_data.V, good_bin_mask
            )
            blinded_normed_chi2_bins = chi2Bins(
                normalized_train_pred_data.Y,
                train.Y.cpu(),
                train.V.cpu(),
                good_bin_mask,
            ).item()
            print(normalized_train_data.V)
            print(normalized_train_pred_data.V)
            print(f"Global Chi2/bins = {global_chi2_bins}")
            print(f"Unblinded Chi2/bins = {blinded_chi2_bins}")
            print(f"Normed Global Chi2/bins = {global_normed_chi2_bins}")
            print(f"Normed UnBlinded Chi2/bins = {blinded_normed_chi2_bins}")

            if global_chi2_bins < 1.5:
                ok = True
            else:
                pass
                # logger.warning("Bad global Chi2, retrying")
            ok = True

        except (
            linear_operator.utils.errors.NanError,
            linear_operator.utils.errors.NotPSDError,
        ) as e:
            lr = lr + random.random() / 1000
            logger.warning(f"CHOLESKY FAILED: retrying with lr={round(lr,3)}")
            logger.warning(e)

    data = {
        "evidence": float(evidence),
        "global_chi2/bins": float(global_chi2_bins),
        "model_string": str(model),
    }

    if window_func:
        mask = regression.getBlindedMask(
            test_pred_data.X, test_pred_data.Y, test_data.Y, test_data.V, window_func
        )
        bpred_mean = test_pred_data.Y
        obs_mean = test_data.Y
        obs_var = test_data.V
        chi2 = chi2Bins(obs_mean, bpred_mean, obs_var, mask)
        avg_chi = torch.sum(
            (obs_mean - bpred_mean) / torch.sqrt(obs_var)
        ) / obs_mean.size(0)
        abs_avg_chi = torch.sum(
            torch.abs(obs_mean - bpred_mean) / torch.sqrt(obs_var)
        ) / obs_mean.size(0)
        data.update(
            {
                "chi2_blinded": float(chi2),
                "avg_chi_blinded": float(avg_chi),
                "abs_avg_chi_blinded": float(abs_avg_chi),
            }
        )
        print(f"Blinded Chi^2/bins = {float(chi2)}")
        print(f"Blinded Chi/bins = {float(avg_chi)}")
        print(f"Blinded Abs(Chi)/bins = {float(abs_avg_chi)}")
    else:
        mask = None

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    if True:
        diagnostic_plots = makeDiagnosticPlots(
            test_pred_data, test_data, train_data, mask
        )
        saveDiagnosticPlots(diagnostic_plots, save_dir)
        # makeSlicePlots(pred_data, test_data, inhist, window_func, 0, save_dir)
        # makeSlicePlots(pred_data, test_data, inhist, window_func, 1, save_dir)

    model_dict = model.state_dict()

    save_data = RegressionModel(
        input_data=inhist,
        window=window_func,
        domain_mask=domain_mask,
        train_data=train_data,
        test_data=test_data,
        trained_model=model,
        raw_posterior_dist=raw_pred_dist,
        posterior_dist=pred_dist,
    )
    torch.save(save_data, save_dir / "train_model.pth")
    torch.save(model_dict, save_dir / "model_dict.pth")
    torch.save(pred_dist, save_dir / "posterior_latent.pth")
    with open(save_dir / "info.json", "w") as f:
        json.dump(data, f)
    return save_data


def createWindowForSignal(signal_data, axes=(200, 0.06)):
    max_idx = torch.argmax(signal_data.Y)
    max_x = signal_data.X[max_idx].round(decimals=2)
    return windowing.EllipseWindow(max_x.tolist(), list(axes))


def doRegressionForSignal(
    signal_name, signal_hist, bkg_hist, kernel, base_dir, kernel_name=""
):
    logging.info(f"Signal is: {signal_name}")

    inducing_ratio = 2

    def mm(train_x, train_y, likelihood, kernel, **kwargs):
        # return models.ExactAnyKernelModel(train_x, train_y, likelihood, kernel=kernel)
        return models.InducingPointModel(
            train_x, train_y, likelihood, kernel, inducing=train_x[::inducing_ratio]
        )
        # return GPRegressionModel  # (train_x, train_y, likelihood)

    if signal_hist:
        signal_regression_data = regression.makeRegressionData(signal_hist)
        window = createWindowForSignal(signal_regression_data)
        sd = SignalData(signal_regression_data, None, signal_hist, signal_name)
    else:
        window = None

    print(f"Now processing area {window.toString() if window else 'NoWindow'}")

    path = base_dir / (window.toString() if window else "NoWindow")
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)

    if signal_hist:
        torch.save(sd, path / "signal_data.pth")

    # dirdata = futil.DirectoryData(path)
    # dir_data = {"inducing_ratio": inducing_ratio, "signal": signal_name or "None"}
    # if window:
    #     dir_data["window"] = window.toDict()
    # dirdata.setGlobal(dir_data)

    d = doCompleteRegression(
        bkg_hist, window, save_dir=path, model_maker=mm, kernel=kernel
    )

    # if signal_hist:
    #     r = makeSigBkgPlot(
    #         d.train_data, d.test_data, signal_regression_data, window
    #     )
    #     saveDiagnosticPlots(r, dirdata.directory)


def doEstimationForSignals(signals, bkg_hist, kernel, base_dir, kernel_name=""):
    base_dir = Path(base_dir)

    i = 0
    for signal_name, signal_hist in signals:
        i = i + 1
        doRegressionForSignal(
            signal_name,
            signal_hist,
            bkg_hist,
            kernel,
            base_dir,
            kernel_name=kernel_name,
        )

        plt.close("all")


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module("linear1", torch.nn.Linear(2, 64))
        self.add_module("relu1", torch.nn.ReLU())
        self.add_module("linear2", torch.nn.Linear(64, 32))
        self.add_module("relu2", torch.nn.ReLU())
        self.add_module("linear4", torch.nn.Linear(32, 2))


def func(x):
    return x - torch.tensor([0.4, 0.4], device=x.device)


def main():

    with open(
        "regression_results/2018_Signal312_nn_uncomp_0p67_m14_vs_mChiUncompRatio.pkl",
        "rb",
    ) as f:
        signal312 = pkl.load(f)

    with open(
        "regression_results/2018_Control_nn_uncomp_0p67_m14_vs_mChiUncompRatio.pkl",
        "rb",
    ) as f:
        control = pkl.load(f)

    bkg_hist = control["Data2018", "Control"]["hist_collection"][
        "histogram"
    ]  # [hist.loc(1000) :, hist.loc(0.4) :]
    signal_hist_names = [
        "signal_312_1500_600",
        "signal_312_1500_900",
        "signal_312_2000_900",
    ]
    signals_to_scan = [
        (sn, signal312[sn, "Signal312"]["hist_collection"]["histogram"]["central", ...])
        for sn in signal_hist_names
    ]

    rbf = SK(gpytorch.kernels.RBFKernel(ard_num_dims=2))

    grbf = SK(models.GeneralRBF(ard_num_dims=2))
    # gsmk = models.GeneralSpectralMixture(ard_num_dims=2, num_mixtures=4)

    smk = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=12, ard_num_dims=2)

    kernels = {
        #"rbf": rbf,
        #"smk": smk,
        # "grq": grq,
        # "grbf": grbf ,
        # "grbf3": grbf + grbf + grbf,
        # "nngrf" : nngrbf,
        # "nnsmk_tiny": nnsmk_tiny,
        # "nnrbf_deep": nnrbf_deep,
        # "nnrbf_huge": nnrbf_huge,
        # "nnrbf_tiny": nnrbf_tiny,
        # "nnrbf_32_16_8": nnrbf32_16_8,
        # "nnrbf_32_32_8": SK(models.NNRBFKernel(odim=2, layer_sizes=(32, 32, 8))),
        # "nnrbf_500_500_50": SK(models.NNRBFKernel(odim=2, layer_sizes=(500, 500, 50))),
        # "nnrbf": SK(models.NNRBFKernel(odim=2,nn=LargeFeatureExtractor())),
        # "nnrbf_64_16_8": SK(models.NNRBFKernel(odim=2, layer_sizes=(64, 16, 8))),
        #"nnrbf_32_16": SK(models.NNRBFKernel(odim=2, layer_sizes=(32, 8))),
        "nnrbf_16_16_8_8": SK(
            models.NNRBFKernel(odim=2, layer_sizes=(16, 16, 8, 8))
        ),
        #"nnrbf_10_5": SK(models.NNRBFKernel(odim=2, layer_sizes=(10, 5)))
        #"nnrq_10_5": SK(models.NNRQKernel(odim=2, layer_sizes=(10, 5)))
        #k"nnm_10_5": SK(models.NNMaternKernel(odim=2, layer_sizes=(10, 5)))
        # "nnrbf_4": SK(models.NNRBFKernel(odim=2, layer_sizes=(4,))),
        # "testf": SK(models.FunctionRBF(func, ard_num_dims=2)),
    }

    p = Path("allscans/control/")
    for n, k in kernels.items():
        print(n)
        doEstimationForSignals(signals_to_scan, bkg_hist, k, p / n, kernel_name=n)


if __name__ == "__main__":
    main()
