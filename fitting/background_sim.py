import pickle as pkl
from pathlib import Path
import torch
from . import regression
from .utils import dataToHist


def makeSimulatedBackground(inhist, model_class, outdir, use_cuda=True, num=10):
    trained_model = regression.doCompleteRegression(
        inhist,
        None,
        model_class,
        use_cuda=use_cuda,
        min_counts=0,
    )
    model, transform, all_data, pred_dist = regression.getPrediction(trained_model)
    poiss = torch.distributions.Poisson(torch.clamp(pred_dist.mean, min=0))

    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    torch.save(trained_model, outdir / "simulated_trained_model.pth")
    for i in range(num):
        sampled = poiss.sample()
        vals = dataToHist(all_data.X, sampled, all_data.E, sampled)
        new_hist = inhist.copy(deep=True)
        new_hist.view(flow=True).value = 0
        new_hist.view(flow=True).variance = 0
        new_hist.view(flow=False).value = vals.values()
        new_hist.view(flow=False).variance = vals.variances()
        with open(outdir / f"background_{i}.pkl", "wb") as f:
            pkl.dump(new_hist, f)
