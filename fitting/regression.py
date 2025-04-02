from dataclasses import dataclass

import linear_operator
import copy
import fitting.transformations as transformations
import gpytorch
import hist
import torch
from rich import print
import logging

from .utils import dataToHist
from dataclasses import dataclass
from typing import Any


import gpytorch
import hist
import torch

from . import transformations
from .utils import dataToHist, computePosterior, chi2Bins

logger = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    model_class: Any
    model_state: dict

    input_data: hist.Hist

    domain_mask: torch.Tensor
    blind_mask: torch.Tensor

    transform: torch.Tensor
    metadata: dict

    learned_noise: bool = False


def getModelingData(trained_model, other_data=None):
    hist = trained_model.input_data
    raw_regression_data = DataValues.fromHistogram(trained_model.input_data)
    dm = trained_model.domain_mask
    if other_data is None:
        all_data = raw_regression_data.getMasked(dm)
    else:
        all_data = other_data

    if trained_model.blind_mask is not None:
        bm = trained_model.blind_mask
    else:
        bm = torch.zeros_like(all_data.Y, dtype=bool)

    return all_data, bm


def loadModel(trained_model, other_data=None):
    model_class = trained_model.model_class
    model_state = trained_model.model_state

    all_data, bm = getModelingData(trained_model, other_data)
    blinded_data = all_data.getMasked(~bm)

    transform = trained_model.transform
    normalized_blinded_data = transform.transform(blinded_data)
    normalized_all_data = transform.transform(all_data)

    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=normalized_blinded_data.V,
        learn_additional_noise=trained_model.learned_noise,
        noise_constraint=gpytorch.constraints.GreaterThan(1e-10),
    )
    model = model_class(
        normalized_blinded_data.X, normalized_blinded_data.Y, likelihood
    )
    model.load_state_dict(model_state)

    model.eval()
    return model


def getPosteriorProcess(model, data, transform):
    normalized_data = transform.transform(data)
    pred_dist = computePosterior(
        model,
        model.likelihood,
        normalized_data,
        slope=transform.transform_y.slope,
        intercept=transform.transform_y.intercept,
    )
    return pred_dist


@dataclass
class DataValues:
    X: torch.Tensor
    Y: torch.Tensor
    V: torch.Tensor
    E: torch.Tensor

    def getMasked(self, mask):
        return DataValues(self.X[mask], self.Y[mask], self.V[mask], self.E)

    def __getitem__(self, m):
        return self.getMasked(m)

    def toGpu(self):
        return DataValues(
            self.X.cuda(), self.Y.cuda(), self.V.cuda(), tuple(x.cuda() for x in self.E)
        )

    def fromGpu(self):
        return DataValues(
            self.X.cpu(), self.Y.cpu(), self.V.cpu(), tuple(x.cpu() for x in self.E)
        )

    @property
    def dim(self):
        return len(self.E)

    def toHist(self):
        return dataToHist(self.X, self.Y, self.E, self.V)

    @staticmethod
    def fromHistogram(histogram):
        edges = tuple(torch.from_numpy(a.edges) for a in histogram.axes)
        centers = tuple(torch.diff(e) / 2 + e[:-1] for e in edges)
        bin_values = torch.from_numpy(histogram.values())
        bin_vars = torch.from_numpy(histogram.variances())
        bin_values = bin_values.T
        bin_vars = bin_vars.T
        centers_grid = torch.meshgrid(*centers, indexing="xy")
        centers_grid = torch.stack(centers_grid, axis=-1)

        flat_centers = torch.flatten(centers_grid, end_dim=1)
        flat_bin_values = torch.flatten(bin_values)
        flat_bin_vars = torch.flatten(bin_vars)

        ret = DataValues(
            flat_centers,
            flat_bin_values,
            flat_bin_vars,
            edges,
        )
        return ret


def optimizeHyperparams(
    model, likelihood, train_data, iterations=200, lr=0.01, validate_function=None
):
    model.train()
    likelihood.train()
    logger.info(
        f"Optimizing {sum(torch.numel(x) for x in model.parameters())} hyperparameters on {len(train_data.Y)} data points"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    evidence = None

    with linear_operator.settings.max_cg_iterations(10000):
        for i in range(iterations):
            optimizer.zero_grad()
            output = model(train_data.X)
            loss = -mll(output, train_data.Y)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            # slr = scheduler.get_last_lr()[0]
            if (i % (iterations // 20) == 0) or i == iterations - 1:
                logger.info(f"Iter {i} (lr={lr:0.4f}): Loss={round(loss.item(),4)}")
                if validate_function is not None:
                    validate_function(model)
            torch.cuda.empty_cache()

    return model, likelihood, loss


def optimizeHyperparamsVar(
    model, likelihood, train_data, iterations=200, lr=0.01, validate_function=None
):
    from torch.utils.data import TensorDataset, DataLoader

    model.train()
    likelihood.train()
    logger.info(
        f"Optimizing {sum(torch.numel(x) for x in model.parameters())} hyperparameters on {len(train_data.Y)} data points"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, train_data.Y.size(0))

    evidence = None
    train_dataset = TensorDataset(train_data.X, train_data.Y)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    for i in range(iterations):
        # Within each iteration, we will go over each minibatch of data
        minibatch_iter = train_loader
        for x_batch, y_batch in minibatch_iter:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
        if (i % (iterations // 20) == 0) or i == iterations - 1:
            logger.info(f"Iter {i} (lr={lr:0.4f}): Loss={round(loss.item(),4)}")
            if validate_function is not None:
                validate_function(model)

    return model, likelihood, loss


def updateModelNewData(
    model, histogram, domain_blinder, window_blinder=None, learn_noise=False
):
    model = copy.deepcopy(model)
    all_data = DataValues.fromHistogram(histogram)
    domain_mask = domain_blinder(all_data.X, all_data.Y)
    test_data = all_data[domain_mask]
    if window_blinder is not None:
        window_mask = window_blinder(test_data.X)
    else:
        window_mask = ~torch.ones_like(test_data.Y, dtype=bool)
    train_data = test_data[~window_mask]
    train_transform = transformations.getNormalizationTransform(train_data)
    normalized_train_data = train_transform.transform(train_data)
    normalized_test_data = train_transform.transform(test_data)

    train = normalized_train_data
    norm_test = normalized_test_data

    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=train.V,
        learn_additional_noise=learn_noise,
        noise_constraint=gpytorch.constraints.GreaterThan(1e-20),
    )
    model.set_train_data(train.X, train.Y, strict=False)
    model.likelhood = likelihood

    model.eval()
    likelihood.eval()

    trained_model = TrainedModel(
        model_class=type(model),
        model_state=model.state_dict(),
        input_data=histogram,
        domain_mask=domain_mask,
        blind_mask=window_mask,
        transform=train_transform,
        metadata={},
        learned_noise=learn_noise,
    )
    return trained_model


def doCompleteRegression(
    histogram,
    model_class,
    domain_blinder,
    window_blinder,
    use_cuda=True,
    iterations=300,
    lr=0.001,
    learn_noise=False,
    validate_function=None,
):

    all_data = DataValues.fromHistogram(histogram)
    domain_mask = domain_blinder(all_data.X, all_data.Y)

    test_data = all_data[domain_mask]
    if window_blinder is not None:
        window_mask = window_blinder(test_data.X)
    else:
        window_mask = ~torch.ones_like(test_data.Y, dtype=bool)

    train_data = test_data[~window_mask][::1]
    # print(torch.count_nonzero(domain_mask))
    # print(torch.count_nonzero(window_mask))
    # print(test_data.X.shape)
    # print(train_data.X.shape)

    train_transform = transformations.getNormalizationTransform(train_data)
    normalized_train_data = train_transform.transform(train_data)
    normalized_test_data = train_transform.transform(test_data)

    if torch.cuda.is_available() and use_cuda:
        logger.info("USING CUDA")
        train = normalized_train_data.toGpu()
        norm_test = normalized_test_data.toGpu()
    else:
        train = normalized_train_data
        norm_test = normalized_test_data

    # logger.info(train.Y)
    # logger.info(train.V)
    # logger.info(f"Learn additional noise is: {learn_noise}")
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=train.V,
        learn_additional_noise=learn_noise,
        noise_constraint=gpytorch.constraints.GreaterThan(1e-10),
    )

    # likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model = model_class(train.X, train.Y, likelihood)
    logger.info(f"Using model:")
    logger.info(model)

    if torch.cuda.is_available() and use_cuda:
        model = model.cuda()
        likelihood = likelihood.cuda()

    if validate_function is not None:

        def vf(model):
            validate_function(model, train, norm_test, window_mask)

    else:
        vf = None

    if isinstance(model, gpytorch.models.ApproximateGP):
        model, likelihood, evidence = optimizeHyperparamsVar(
            model, likelihood, train, iterations=iterations, lr=lr, validate_function=vf
        )
    else:
        model, likelihood, evidence = optimizeHyperparams(
            model, likelihood, train, iterations=iterations, lr=lr, validate_function=vf
        )

    if torch.cuda.is_available() and use_cuda:
        model = model.cpu()
        likelihood = likelihood.cpu()

    model.eval()
    likelihood.eval()

    # normalized_chi2 = model
    # post_reg = model(normalized_test_data.X).mean
    # post_blind = post_reg[~window_mask]
    # tr = torch.exp(post_reg)
    # chi2_blind_post_raw = chi2Bins(tr, test_data.Y, test_data.V, mask=window_mask)
    # logger.info(f"Chi2Blind raw: {chi2_blind_post_raw:0.3f}")

    trained_model = TrainedModel(
        model_class=model_class,
        model_state=model.state_dict(),
        input_data=histogram,
        domain_mask=domain_mask,
        blind_mask=window_mask,
        transform=train_transform,
        metadata={},
        learned_noise=learn_noise,
    )
    return trained_model
