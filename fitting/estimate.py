import argparse
from .core import FitParams, SignalPoint, FitRegion, Metadata
from fitting.config import Config
from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer, ConfigDict
import copy
from typing import Any
import json
import logging
import gpytorch
import pickle as pkl
from pathlib import Path


from .diagnostics import plotDiagnostics

from .regression import DataValues

import hist
import matplotlib as mpl
from .utils import chi2Bins, computePosterior, dataToHist
import matplotlib.pyplot as plt
import mplhep
import torch

from . import models, regression
from .blinder import (
    GaussianWindow2D,
    MinYCut,
    StaticWindow,
    MaxXCut,
    WindowAnd,
    MinXCut,
)
from .plotting.plots import windowPlots2D

torch.set_default_dtype(torch.float64)


logger = logging.getLogger(__name__)


def saveDiagnosticPlots(plots, save_dir):
    for name, (fig, ax) in plots.items():
        o = (save_dir / name).with_suffix(".pdf")
        fig.savefig(o)
        plt.close(fig)


def validate(model, train, test, window_mask, train_transform):
    import torch

    model.eval()
    with (
        torch.no_grad(),
        gpytorch.settings.fast_computations(
            covar_root_decomposition=False, log_prob=False, solves=False
        ),
        gpytorch.settings.fast_pred_samples(state=False),
        gpytorch.settings.fast_pred_var(state=False),
        gpytorch.settings.lazily_evaluate_kernels(state=True),
        gpytorch.settings.max_cholesky_size(1200),
        # gpytorch.settings.max_eager_kernel_size(1200),
        gpytorch.settings.linalg_dtypes(
            default=torch.float64, symeig=torch.float64, cholesky=torch.float64
        ),
    ):
        post = model(test.X)

        extra_noise = None
        if model.likelihood.second_noise_covar is not None:
            extra_noise = model.likelihood.second_noise

        full_post = computePosterior(
            model, model.likelihood, test, extra_noise=extra_noise
        )

        post_reg = post.mean

        with gpytorch.settings.min_fixed_noise(double_value=1e-8):
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=test.V[~window_mask] + model.likelihood.second_noise
            )
            pred_unblind = likelihood(model(test.X[~window_mask]))
        with gpytorch.settings.min_fixed_noise(double_value=1e-8):
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=test.V[window_mask] + model.likelihood.second_noise
            )
            pred_blind = likelihood(model(test.X[window_mask]))

    real_y = test.Y
    real_v = test.V
    if hasattr(model.likelihood, "second_noise"):
        logger.info(f"Adding extra noise {model.likelihood.second_noise}")
        pred_v = full_post.variance

    # post_reg = train_transform.transform_y.iTransformData(model(test.X).mean)
    # real = train_transform.transform_y.iTransformData(test.Y)
    chi2_blind_post_raw = chi2Bins(post_reg, real_y, real_v, mask=window_mask)
    chi2_post_raw = chi2Bins(post_reg, real_y, real_v, mask=~window_mask)
    logger.info(
        f"Validate Statistical Chi2 (seen={chi2_post_raw:0.3f}) (blind={chi2_blind_post_raw:0.3f})"
    )

    chi2_blind_pred_raw = chi2Bins(post_reg, real_y, pred_v, mask=window_mask)
    chi2_pred_raw = chi2Bins(post_reg, real_y, pred_v, mask=~window_mask)
    logger.info(
        f"Validate Predictive Chi2 (seen={chi2_pred_raw:0.3f}) (blind={chi2_blind_pred_raw:0.3f})"
    )
    unblind_nlpd = gpytorch.metrics.negative_log_predictive_density(
        pred_unblind, test.Y[~window_mask]
    )
    blind_nlpd = gpytorch.metrics.negative_log_predictive_density(
        pred_blind, test.Y[window_mask]
    )
    logger.info(f"NLPD (seen={unblind_nlpd:0.3f}) (blind={blind_nlpd:0.3f})")

    model.train()
    return (
        chi2_post_raw.detach(),
        chi2_blind_post_raw.detach(),
    )


def regress(
    data,
    base_save_dir,
    window=None,
    iterations=100,
    min_counts=0,
    use_cuda=False,
    window_spread=1.0,
    learning_rate=0.02,
):

    base_save_dir = Path(base_save_dir)
    base_save_dir.mkdir(exist_ok=True, parents=True)

    # model = models.NonStatParametric2D
    model = models.MyNNRBFModel2D
    # model = models.MyVariational2DModel

    trained_model = regression.doCompleteRegression(
        data,
        model,
        MinYCut(min_y=min_counts),
        # WindowAnd(
        #     window_1=WindowAnd(
        #         window_1=MinYCut(min_y=min_counts),
        #         window_2=MaxXCut(max_x=torch.tensor([100000.0, 0.95])),
        #     ),
        #     window_2=MinXCut(max_x=torch.tensor([-20.0, 0.7])),
        # ),
        window,
        iterations=iterations,
        use_cuda=use_cuda,
        learn_noise=True,
        lr=learning_rate,
        validate_function=validate,
    )
    return trained_model


def estimateSingle2DWithWindow(
    signal_name,
    signal_hist,
    bkg_hist,
    window,
    base_dir,
    blinding_signal=True,
    use_cuda=False,
    iterations=100,
    signal_injections=None,
    learning_rate=0.02,
    rebin_signal=1,
    use_other_model=None,
    use_other_kernel=None,
    extra_metadata=None,
    no_contamination=False,
):
    extra_metadata = extra_metadata or {}
    signal_injections = signal_injections or [0.0, 1.0, 4.0, 16.0]
    sig_dir = base_dir  # / signal_name
    sig_dir.mkdir(exist_ok=True, parents=True)
    signal_regression_data = DataValues.fromHistogram(signal_hist)

    def saveFunc(name, fig):
        ext = Config.IMAGE_TYPE
        fig.savefig((sig_dir / name).with_suffix(f".{ext}"))
        plt.close(fig)

    if no_contamination:
        window_mask = window(signal_regression_data.X)
        logger.warn(f"Removing signal bins outside blinded window")
        signal_regression_data.Y[~window_mask] = 0
        h = signal_regression_data.toHist()
        signal_hist.view(flow=False).value = h.values()
        signal_hist.view(flow=False).variance = h.variances()

    sd = dict(
        signal_data=signal_regression_data,
        signal_hist=signal_hist,
        signal_name=signal_name,
    )

    windowPlots2D(signal_regression_data, window, saveFunc)

    torch.save(sd, sig_dir / "signal_data.pth")
    _, coupling, stop, chi = signal_name.split("_")

    fr = FitRegion(
        stop_bounds=torch.aminmax(signal_regression_data.X[:, 0]),
        ratio_bounds=torch.aminmax(signal_regression_data.X[:, 1]),
    )

    signal_point = SignalPoint(
        coupling=coupling,
        mt=stop,
        mx=chi,
    )

    for r in signal_injections:
        fp = FitParams(
            iterations=iterations, learning_rate=learning_rate, injected_signal=r
        )

        metadata = Metadata(
            signal_point=signal_point,
            fit_region=fr,
            fit_params=fp,
            window=window,
            other_data=extra_metadata,
        )

        save_dir = sig_dir / f"inject_r_{str(round(r,3)).replace('.','p')}"
        meta = {"signal_name": signal_name}
        save_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Injecting background with signals strength {round(r,3)}")
        to_estimate = bkg_hist + r * signal_hist
        if use_other_model:
            trained_model = regression.updateModelNewData(
                use_other_model,
                to_estimate,
                MinYCut(min_y=0),
                window,
            )

        else:
            trained_model = regress(
                to_estimate,
                base_dir,
                min_counts=-1,
                window=window,
                use_cuda=use_cuda,
                iterations=iterations,
                learning_rate=learning_rate,
            )
        trained_model.metadata = metadata
        with open(save_dir / "metadata.json", "w") as f:
            f.write(metadata.model_dump_json())

        torch.save(trained_model, save_dir / "bkg_estimation_result.pth")
        plotDiagnostics(save_dir, trained_model)


def estimateSingle2D(
    background_path,
    signal_path,
    signal_name,
    signal_selection,
    background_name,
    base_dir,
    window_spread=1.0,
    blinding_signal=True,
    inject_other_signals=None,
    rebin_background=1,
    rebin_signal=1,
    scale_background=None,
    min_base_variance=None,
    extra_metadata=None,
    use_fit_as_signal=False,
    static_window_path=None,
    poisson_rescale=None,
    scale_signal_to_lumi=None,
    **kwargs,
):
    base_dir = Path(base_dir)
    extra_metadata = extra_metadata or {}

    with open(signal_path, "rb") as f:
        signal_file = pkl.load(f)
    with open(background_path, "rb") as f:
        background = pkl.load(f)
    bkg_hist = background

    logger.info(bkg_hist)

    if scale_background is not None:
        bkg_hist = bkg_hist.copy(deep=True)
        bkg_hist = bkg_hist * scale_background
        bkg_hist.view(flow=True).variance = bkg_hist.view(flow=True).value

    logger.info(bkg_hist)
    logger.info(f"Post scale background is ")
    if poisson_rescale:
        import numpy as np

        logger.info(f"Rescaling by {poisson_rescale} and preserving poisson statistics")

        bkg_hist = bkg_hist.copy(deep=True)
        print(bkg_hist)
        v = copy.deepcopy(bkg_hist.view(flow=False).value)
        new = v * poisson_rescale
        bkg_hist[...] = np.stack([new, new], axis=-1)
        print(bkg_hist)

    if min_base_variance:
        import numpy as np

        bkg_hist = bkg_hist.copy(deep=True)
        v = bkg_hist.view(flow=False).variance
        bkg_hist.view(flow=False).variance = np.clip(v, a_min=5, a_max=None)

    bkg_hist = bkg_hist[hist.rebin(rebin_background), hist.rebin(rebin_background)]
    bkg_bin_size = np.mean(np.diff(bkg_hist.axes[0].centers))
    logger.info(f"Background bin size is {bkg_bin_size}")
    a1, a2 = bkg_hist.axes
    a1_min, a1_max = a1.edges.min() + 0.000001, a1.edges.max()
    a2_min, a2_max = a2.edges.min() + 0.000001, a2.edges.max()
    signal_hist = signal_file[signal_name, signal_selection]["hist"]

    signal_params = signal_file[signal_name, signal_selection]["params"]
    if scale_signal_to_lumi is not None:
        current_lumi = signal_params["dataset"]["era"]["lumi"]
        lumi_scale = scale_signal_to_lumi / current_lumi
        signal_hist = lumi_scale * signal_hist
        logger.warn(f"Scaling signal histogram by {lumi_scale}")

    extra_metadata = extra_metadata or {}
    extra_metadata["signal_params"] = signal_params

    sig_bin_size = np.mean(np.diff(signal_hist.axes[0].centers))
    logger.info(f"Signal bin size is {sig_bin_size}")
    ratio = bkg_bin_size / sig_bin_size
    rebin_signal = round(ratio)
    logger.info(f"Rebinning signal by {rebin_signal} based on ratio {ratio:0.2f}")
    signal_hist = signal_hist[
        hist.loc(a1_min) : hist.loc(a1_max), hist.loc(a2_min) : hist.loc(a2_max)
    ]

    signal_hist = signal_hist[hist.rebin(rebin_signal), hist.rebin(rebin_signal)]

    logger.info(bkg_hist)
    logger.info(signal_hist)

    sig_dir = base_dir  # / signal_name
    sig_dir.mkdir(exist_ok=True, parents=True)
    signal_regression_data = DataValues.fromHistogram(signal_hist)
    if blinding_signal:
        import scipy

        try:
            if static_window_path:
                window = StaticWindow.fromFile(static_window_path)
            else:
                window = GaussianWindow2D.fromData(
                    signal_regression_data, spread=window_spread
                )
            d = window.model_dump()
            print(d)
            print(GaussianWindow2D(**d))
            extra_metadata["window_spread"] = window_spread
        except (scipy.optimize.OptimizeWarning, RuntimeError) as e:
            raise e
            window = None
    else:
        logger.warn(f"Could not find a window for signal {signal_name}")
        window = None

    if use_fit_as_signal:
        logger.warn(f"Remaking signal from 2D Gaussian fit")
        signal_hist_total = signal_hist.sum().value
        fit_vals = window.vals(signal_regression_data.X)
        scale = signal_hist_total / fit_vals.sum()
        logger.warn(
            f"Raw signal hist has norm : {signal_hist_total:0.2f}, fit has norm {fit_vals.sum():0.2f}. Rescaling by {scale:0.2f}"
        )
        signal_regression_data.Y = fit_vals * scale
        signal_regression_data.V = fit_vals * scale
        h = signal_regression_data.toHist()
        signal_hist.view(flow=False).value = h.values()
        signal_hist.view(flow=False).variance = h.variances()

    sd = dict(
        signal_data=signal_regression_data,
        signal_hist=signal_hist,
        signal_name=signal_name,
    )
    if not inject_other_signals:
        estimateSingle2DWithWindow(
            signal_name,
            signal_hist,
            bkg_hist,
            window,
            base_dir,
            rebin_signal=rebin_signal,
            extra_metadata=extra_metadata,
            **kwargs,
        )
    else:
        for other_signal in inject_other_signals:
            path, name = other_signal.split(":")
            with open(path, "rb") as f:
                signal_file = pkl.load(f)

            signal_hist = signal_file[name, signal_selection]["hist"]
            sig_bin_size = np.mean(np.diff(signal_hist.axes[0].centers))
            logger.info(f"Signal bin size is {sig_bin_size}")
            ratio = bkg_bin_size / sig_bin_size
            rebin_signal = round(ratio)
            logger.info(
                f"Rebinning signal by {rebin_signal} based on ratio {ratio:0.2f}"
            )
            signal_hist = signal_hist[
                hist.loc(a1_min) : hist.loc(a1_max), hist.loc(a2_min) : hist.loc(a2_max)
            ]

            signal_hist = signal_hist[
                hist.rebin(rebin_signal), hist.rebin(rebin_signal)
            ]
            with (
                gpytorch.settings.fast_computations(
                    covar_root_decomposition=False, log_prob=False, solves=False
                ),
                gpytorch.settings.fast_pred_samples(state=False),
                gpytorch.settings.fast_pred_var(state=False),
                gpytorch.settings.lazily_evaluate_kernels(state=False),
            ):
                estimateSingle2DWithWindow(
                    signal_name,
                    signal_hist,
                    bkg_hist,
                    window,
                    Path(base_dir) / f"inject_{name}",
                    rebin_signal=rebin_signal,
                    **kwargs,
                )


def argEq(x):
    left, right = x.split("=")
    return {left: right}


def main(args):
    import gpytorch

    # gpytorch.settings.lazily_evaluate_kernels(state=False)
    # gpytorch.settings.cholesky_max_tries(1000000)
    # gpytorch.settings.lazily_evaluate_kernels(state=True)
    torch.set_default_dtype(torch.float64)

    mpl.use("Agg")
    mplhep.style.use("CMS")
    if not args.blind_signal:
        logger.warn(f"Not blinding signal window")

    other_model = None
    if args.use_other_model:
        other_model_data = torch.load(args.use_other_model, weights_only=False)
        other_model = regression.loadModel(other_model_data)

    extra_metadata = {
        k: v for d in map(argEq, args.metadata or []) for k, v in d.items()
    }

    logger.info(extra_metadata)
    estimateSingle2D(
        background_path=args.background,
        signal_path=args.signal,
        signal_name=args.name,
        signal_selection=args.region,
        background_name=None,
        base_dir=args.outdir,
        use_cuda=args.cuda,
        window_spread=args.spread,
        learning_rate=args.learning_rate,
        rebin_signal=args.rebin_signal,
        iterations=args.iterations,
        rebin_background=args.rebin_background,
        blinding_signal=args.blind_signal,
        scale_background=args.scale_background,
        signal_injections=args.injected,
        use_fit_as_signal=args.use_fit_as_signal,
        min_base_variance=5,
        use_other_model=other_model,
        inject_other_signals=args.inject_other_signals,
        extra_metadata=extra_metadata,
        no_contamination=args.no_contamination,
        poisson_rescale=args.poisson_rescale,
        static_window_path=args.static_window_path,
        scale_signal_to_lumi=args.scale_signal_to_lumi,
    )


def addToParser(parser):
    parser.add_argument(
        "-o", "--outdir", type=str, help="Output directory for estimate", required=True
    )
    parser.add_argument(
        "-b",
        "--background",
        type=str,
        help="Input path for the background",
        required=True,
    )
    parser.add_argument(
        "-s", "--signal", type=str, help="Input path for the signal", required=True
    )
    parser.add_argument(
        "-n", "--name", type=str, help="Name for the signal", required=True
    )
    parser.add_argument("--scale-background", type=float, default=1.0)
    parser.add_argument("--rebin-signal", default=1, type=int, help="Rebinning")
    parser.add_argument("--rebin-background", default=1, type=int, help="Rebinning")
    parser.add_argument("-r", "--region", type=str, help="Region", required=True)
    parser.add_argument("-l", "--learning-rate", type=float, default=0.02)
    parser.add_argument("--poisson-rescale", type=float, default=None)
    parser.add_argument("--injected", type=float, nargs="*", default=[0.0])
    parser.add_argument("-i", "--iterations", type=int, default=100)
    parser.add_argument(
        "--cuda",
        help="Use cuda",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--scale-signal-to-lumi",
        help="Scale Signal",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--use-fit-as-signal",
        action="store_true",
        help="Use the 2D gaussian as the signal shape",
        default=False,
    )
    parser.add_argument(
        "--no-contamination",
        action="store_true",
        help="Remove any signal outside of blinding window",
        default=False,
    )
    parser.add_argument("--static-window-path", type=str, default=None)
    parser.add_argument("--spread", type=float, default=1.5)
    parser.add_argument("--inject-other-signals", default=None, type=str, nargs="*")
    parser.add_argument("--metadata", default=None, type=str, nargs="*")
    parser.add_argument("--use-other-model", type=str)
    parser.add_argument(
        "--blind-signal", default=True, action=argparse.BooleanOptionalAction
    )

    parser.set_defaults(func=main)
    return parser
