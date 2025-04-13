import linear_operator
import torch
import gpytorch
from rich import print
from uhi.numpy_plottable import NumPyPlottableHistogram
import numpy as np
import logging
import warnings
import torch

from linear_operator import settings
from linear_operator.utils.errors import NanError, NotPSDError
from linear_operator.utils.warnings import NumericalWarning
import code
import readline
import rlcompleter


logger = logging.getLogger(__name__)


def pointsToGrid(points_x, points_y, edges, set_unfilled=None):
    if len(edges) > 1:
        filled = torch.histogramdd(
            points_x, bins=edges, weight=torch.full_like(points_y, True)
        )
        ret = torch.histogramdd(points_x, bins=edges, weight=points_y)
        return ret, filled.hist.bool()
    else:
        filled = torch.histogram(
            points_x, bins=edges[0], weight=torch.full_like(points_y, True)
        )
        ret = torch.histogram(points_x, bins=edges[0], weight=points_y)
        return ret, filled.hist.bool()


def dataToHist(X, Y, E, V=None):
    Z, filled = pointsToGrid(X, Y, E)
    Z = np.where(filled.numpy(), Z.hist.numpy(), np.nan)
    if V is not None:
        V, filled = pointsToGrid(X, V, E)
        V = np.where(filled.numpy(), V.hist.numpy(), np.nan)

    return NumPyPlottableHistogram(Z, *tuple(x.numpy() for x in E), variances=V)


def chi2Bins(obs, exp, var, mask=None, min_var=0, power=2):
    if mask is not None:
        obs, exp, var = obs[mask], exp[mask], var[mask]
    m = var > min_var
    obs, exp, var = obs[..., m], exp[..., m], var[..., m]

    return torch.sum((obs - exp).pow(power) / var, dim=-1) / exp.size(-1)


def fixMVN(mvn):
    with gpytorch.settings.cholesky_max_tries(10000):
        X = linear_operator.utils.cholesky.psd_safe_cholesky(mvn.covariance_matrix)
    fixed_cv = X @ X.T
    mu = mvn.mean
    fixed = type(mvn)(mu, fixed_cv)
    # assert(torch.allclose(fixed_cv, mvn.covariance_matrix))
    return fixed


def fixCov(mat):
    X = linear_operator.utils.cholesky.psd_safe_cholesky(mat)
    fixed_cv = X @ X.T
    return fixed_cv


def affineTransformMVN(mvn, slope, intercept):
    cm = mvn.covariance_matrix
    mu = mvn.mean
    new_mu = slope * mu + intercept
    d = slope * torch.eye(cm.shape[0], device=slope.device)
    new_cm = d @ cm @ d.T
    # new_cm = fixCov(new_cm)
    ret = type(mvn)(new_mu, new_cm)
    return ret


def getScaledEigenvecs(cov_mat, top=None):
    # linear_operator.utils.cholesky.psd_safe_cholesky(cov_mat)
    vals, vecs = torch.linalg.eigh(cov_mat)
    vals = vals.real
    vecs = vecs.real

    X = vecs @ torch.diag(torch.sqrt(vals))
    logger.info(vals)
    assert torch.allclose(X @ X.T, cov_mat)

    vals = torch.flip(vals, (0,))
    vecs = torch.flip(vecs, (0,))

    if top is not None:
        eva = vals[:top]
        eve = vecs[:top]
    else:
        eva = vals
        eve = vecs

    return eva, eve


def fixCovar(mat):
    vals, vecs = torch.linalg.eigh(mat)
    vals = vals.clamp_min(0.0)
    covar = vecs @ torch.diag(vals) @ vecs.T
    psd_covar_C = linear_operator.utils.cholesky.psd_safe_cholesky(covar)
    covar = psd_covar_C @ psd_covar_C.T
    return covar


def computePosterior(
    model, likelihood, data, slope=None, intercept=None, extra_noise=None
):
    # model.eval()
    with (
        torch.no_grad(),
        gpytorch.settings.fast_computations(
            covar_root_decomposition=False, log_prob=False, solves=False
        ),
        gpytorch.settings.fast_pred_samples(state=False),
        gpytorch.settings.fast_pred_var(state=False),
        gpytorch.settings.lazily_evaluate_kernels(state=False),
    ):
        pred_dist = model(data.X)

    pred_dist = fixMVN(pred_dist)
    # logger.info(f"Initial Transfomed variances are {pred_dist.variance * slope**2}")

    # vars = globals()
    # vars.update(locals())
    # vars = {"histogram": h}
    # readline.set_completer(rlcompleter.Completer(vars).complete)
    # readline.parse_and_bind("tab: complete")
    # code.InteractiveConsole(vars).interact()
    # logger.info(pred_dist.covariance_matrix)
    # logger.info(pred_dist.covariance_matrix.dtype)

    # logger.info(f"Initial mean is {pred_dist.mean * slope}")
    if extra_noise is not None:
        en = extra_noise 
        new_cov = torch.diag(torch.ones(pred_dist.variance.size(0)) * en).detach()
        logger.info(f"Adding extra noise to covariance {torch.sqrt(en*slope**2)}")
        pred_dist = gpytorch.distributions.MultivariateNormal(
            pred_dist.mean, pred_dist.covariance_matrix + new_cov
        )

    if slope is not None and intercept is not None:
        pred_dist = affineTransformMVN(pred_dist, slope, intercept)

    return pred_dist
