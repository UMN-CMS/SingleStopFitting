from pathlib import Path

import argparse

import fitting.models
import torch
from fitting.regression import DataValues
from fitting.storage import getPrediction


def parseArguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('inputs',nargs="+" )
    return parser.parse_args()


def sliceData(data, dim, val):
    E = data.E[int(not(dim))]

    X=X[:,dim]
    mask = torch.close(X)
    X,Y,V = X[mask],Y[mask],V[mask]
    s = torcg.argsort(X)
    X,Y,V=X[s],Y[s],V[s]
    return DataValues(X,Y,V,E)


def main():
    args = parseArguments()
    for data_path in args.inputs:
        p = Path(data_path)
        parent = p.parent
        signal_name = next(x for x in p.parts if "signal_" in x)
        relative = parent.relative_to(Path("."))
        bkg_data = torch.load(p)
        obs, pred = getPrediction(bkg_data, model_class=fitting.models.NonStatParametric2D)
        pred_data = DataValues(obs.X, pred.mean, pred.variance, obs.E)
        r=  sliceData(pred_data, 1, obs.X[10][0])
        print(r)


if __name__ == "__main__":
    main()

    
