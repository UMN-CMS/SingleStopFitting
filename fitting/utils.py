import linear_operator
import torch
import gpytorch
from uhi.numpy_plottable import NumPyPlottableHistogram
import numpy as np


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
    X = linear_operator.utils.cholesky.psd_safe_cholesky(mvn.covariance_matrix)
    fixed_cv = X @ X.T
    mu = mvn.mean
    fixed = type(mvn)(mu, fixed_cv)
    # assert(torch.allclose(fixed_cv, mvn.covariance_matrix))
    return fixed


def affineTransformMVN(mvn, slope, intercept):
    cm = mvn.covariance_matrix
    mu = mvn.mean
    new_mu = slope * mu + intercept
    d = slope * torch.eye(cm.shape[0], device=slope.device)
    new_cm = d @ cm @ d.T
    ret = type(mvn)(new_mu, new_cm)
    return ret


def getScaledEigenvecs(cov_mat, top=None):
    linear_operator.utils.cholesky.psd_safe_cholesky(cov_mat)
    vals, vecs = torch.linalg.eigh(cov_mat)
    vals = vals.real
    vecs = vecs.real

    X = vecs @ torch.diag(torch.sqrt(vals))
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


def computePosterior(model, likelihood, data, slope=None, intercept=None):
    with torch.no_grad():
        pred_dist = model(data.X)
    with gpytorch.settings.cholesky_max_tries(30):
        psd_pred_dist = fixMVN(pred_dist)
    if slope is not None and intercept is not None:
        pred_dist = affineTransformMVN(psd_pred_dist, slope, intercept)
    else:
        pred_dist = type(pred_dist)(
            psd_pred_dist.mean,
            psd_pred_dist.covariance_matrix.to_dense(),
        )

    return pred_dist
