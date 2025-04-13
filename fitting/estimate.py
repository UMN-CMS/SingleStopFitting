import argparse
import json
import logging
import pickle as pkl
from pathlib import Path


from .diagnostics import plotDiagnostics

from .regression import DataValues

import hist
import matplotlib as mpl
from .utils import chi2Bins
import matplotlib.pyplot as plt
import mplhep
import torch

from . import models, regression
from .blinder import GaussianWindow2D, MinYCut
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
    post_reg = model(test.X).mean
    real_y = test.Y
    real_v = test.V

    # post_reg = train_transform.transform_y.iTransformData(model(test.X).mean)
    # real = train_transform.transform_y.iTransformData(test.Y)
    chi2_blind_post_raw = chi2Bins(post_reg, real_y, real_v, mask=window_mask)
    chi2_post_raw = chi2Bins(post_reg, real_y, real_v, mask=~window_mask)
    logger.info(
        f"Validate Chi2 (seen={chi2_post_raw:0.3f}) (blind={chi2_blind_post_raw:0.3f})"
    )

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
):
    extra_metadata = extra_metadata or {}
    signal_injections = signal_injections or [0.0, 1.0, 4.0, 16.0]
    sig_dir = base_dir  # / signal_name
    sig_dir.mkdir(exist_ok=True, parents=True)
    signal_regression_data = DataValues.fromHistogram(signal_hist)
    sd = dict(
        signal_data=signal_regression_data,
        signal_hist=signal_hist,
        signal_name=signal_name,
    )

    print(bkg_hist)
    print(signal_hist)

    def saveFunc(name, fig):
        ext = "png"
        fig.savefig((sig_dir / name).with_suffix(f".{ext}"))
        plt.close(fig)

    windowPlots2D(signal_regression_data, window, saveFunc)
    torch.save(sd, sig_dir / "signal_data.pth")
    _, coupling, stop, chi = signal_name.split("_")
    model_data = {
        "coupling": coupling,
        "mt": int(stop),
        "mx": int(chi),
        "x_bounds": [
            float(signal_regression_data.X[:, 0].min()),
            float(signal_regression_data.X[:, 0].max()),
        ],
        "y_bounds": [
            float(signal_regression_data.X[:, 1].min()),
            float(signal_regression_data.X[:, 1].max()),
        ],
    }

    model_data.update(extra_metadata)

    for r in signal_injections:
        print(window)
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
                # min_counts=20,
                window=window,
                use_cuda=True,
                iterations=iterations,
                learning_rate=learning_rate,
            )
        this_injection_data = {**model_data, "signal_injected": r}
        trained_model.metadata.update(this_injection_data)
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(this_injection_data, f, indent=2)

        # trained_model.metadata.update(add_metadata)
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
    **kwargs,
):
    base_dir = Path(base_dir)

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
            window = GaussianWindow2D.fromData(
                signal_regression_data, spread=window_spread
            )
        except (scipy.optimize.OptimizeWarning, RuntimeError) as e:
            raise e
            window = None
    else:
        logger.warn(f"Could not find a window for signal {signal_name}")
        window = None
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
        min_base_variance=5,
        use_other_model=other_model,
        inject_other_signals=args.inject_other_signals,
        extra_metadata=extra_metadata,
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
    parser.add_argument("--injected", type=float, nargs="*", default=[0.0])
    parser.add_argument("-i", "--iterations", type=int, default=100)
    parser.add_argument("--cuda", action="store_true", help="Use cuda", default=False)
    parser.add_argument("--spread", type=float, default=1.5)
    parser.add_argument("--inject-other-signals", default=None, type=str, nargs="*")
    parser.add_argument("--metadata", default=None, type=str, nargs="*")
    parser.add_argument("--use-other-model", type=str)
    parser.add_argument(
        "--blind-signal", default=True, action=argparse.BooleanOptionalAction
    )

    parser.set_defaults(func=main)
    return parser
