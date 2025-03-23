from pathlib import Path

import hist
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC, NUTS
from fitting.regression import loadModel, getModelingData, getPosteriorProcess

torch.set_default_dtype(torch.float64)


def statModel(bkg_pdf, signal_dist, observed=None):
    # r = pyro.sample("rate", dist.Uniform(-20, 20))
    r = pyro.param("rate", lambda: 1+torch.randn(()).to(signal_dist.device))
    r = r.to(signal_dist.device)
    print(r)
    background = torch.clamp(pyro.sample("bkg_estimate", bkg_pdf), 1)
    obs_hist = (r * signal_dist) + background
    print((obs_hist - background).abs().sum())
    with pyro.plate("bins", signal_dist.shape[0]):
        return pyro.sample("observed", dist.Poisson(obs_hist), obs=observed)


def runMCMC(model, *args, **kwargs):
    nuts_kernel = NUTS(
        model,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        jit_compile=True,
    )
    mcmc = MCMC(nuts_kernel, num_samples=800, warmup_steps=400, num_chains=1)
    mcmc.run(*args, **kwargs)
    return mcmc


def runSVI(model, *args, **kwargs):
    guide = pyro.infer.autoguide.AutoDelta(model)
    adam = pyro.optim.Adam({"lr": 0.05})
    elbo = pyro.infer.Trace_ELBO()
    svi = pyro.infer.SVI(model, guide, adam, elbo)
    losses = []
    for step in range(200):
        loss = svi.step(*args, **kwargs)
        losses.append(loss)
        if step % 1 == 0:
            print("Elbo loss: {}".format(loss))
    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name).data.cpu().numpy())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load(
        "condor_results_2025_03_20/signal_312_1500_600/uncomp/inject_r_0p0/bkg_estimation_result.pth"
    )
    model = loadModel(d)
    model = model.cuda()
    all_data, bm = getModelingData(d)
    all_data = all_data.toGpu()
    m = all_data.Y >= 1
    all_data = all_data.getMasked(m)
    transform = d.transform.toCuda()
    print(transform)
    print(model)
    pred_dist = getPosteriorProcess(model, all_data, transform)
    s = torch.load(
        "condor_results_2025_03_20/signal_312_1500_600/uncomp/signal_data.pth"
    )
    print(s)
    # s[~bm] = 0
    s = s["signal_data"].toGpu()
    runSVI(statModel, pred_dist, s.Y[m], observed=pred_dist.mean.round())



if __name__ == "__main__":
    main()
