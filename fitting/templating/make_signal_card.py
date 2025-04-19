from .latex_env import renderTemplate
import itertools as it
from collections import defaultdict
from pathlib import Path
import json
from rich import print


def loadOneMeta(p):
    p = Path(p)
    parent = p.parent
    with open(p, "r") as f:
        metadata = json.load(f)
    data = metadata
    plots = {
        x.stem: str(x)
        for x in it.chain(parent.glob("*.pdf"), parent.parent.glob("*.pdf"))
    }
    plots["covar_center"] = next((y for x, y in plots.items() if "covariance_" in x), None )
    data = {**metadata, "plots": plots}
    return data



def main():
    d = "condor_results_2025_04_17_asimov/"
    ret = defaultdict(list)
    for p in Path(d).rglob("metadata.json"):
        data = loadOneMeta(p)
        signal = (data["algo"], data["coupling"], data["mt"], data["mx"])
        ret[signal].append(data)

    for x in ret.values():
        x.sort(key=lambda x: x["signal_injected"])

    to_use = [
        next(z for z in x if z["signal_injected"] == 0.0)
        for _, x in sorted(ret.items(), key=lambda x: x[0])
    ]
    print(to_use)
    r = renderTemplate("signal_card.tex", {"test": "Hello", "all_signals": to_use})
    with open("generated/signal_cards.tex", "w") as f:
        f.write(r)

    # print(ret)

    # with open("condor_results_2025_04_14/signal_312_1500_1300/uncomp/inject_r_1p0/metadata.json",'r') as f:
    #     data = json.load(f)
    # t = renderTemplate("signal_card.tex",data)
    # print(t)


if __name__ == "__main__":
    main()
