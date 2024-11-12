import arviz as az
import fitting.models
import pyro
import pyro.distributions as pyrod
import pyro.infer as pyroi
import torch
from fitting.storage import getPrediction


def statModel(bkg_mvn, observed=None):
    background = pyro.sample("background", bkg_mvn)
    with pyro.plate("bins", bkg_mvn.mean.shape[0]):
        return pyro.sample(
            "observed", pyrod.Poisson(torch.clamp(background, 0)), obs=observed
        )

def main():
    bkg_data = torch.load(
        "allscans/control_reduced/signal_312_1500_600/inject_r_0p0/train_model.pth"
    )
    data, pred = getPrediction(bkg_data, fitting.models.NonStatParametric2D)
    print(pred)
    predictive = pyroi.Predictive(
        statModel,
        num_samples=800,
    )

    samples = predictive(pred)
    x = samples["background"].squeeze()
    print(x.shape)
    print(x)

    variance = pred.variance
    # v = torch.sqrt(variance[500])
    o = torch.sqrt(data.V[1000])
    s = torch.std(x[:,1000])

    print(o)
    print(s)
    
    #inf_data = az.convert_to_inference_data(samples,group="posterior_predictive")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main()
