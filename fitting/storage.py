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

def getPrediction(bkg_data, model_class):
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
    model = model_class(
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
