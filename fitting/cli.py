import argparse
import fitting.estimate
import fitting.combine.plot_sensitivity
import fitting.diagnostics
import fitting.background_sim
from rich import print
import fitting.predictive
import fitting.gather_results
from fitting.combine.generate import addDatacardGenerateParser
from .logging import setupLogging


def jsonToNamespace(path, defaults):
    from types import SimpleNamespace
    import json

    with open(path, "r") as f:
        data = json.load(f)
        data = {**defaults, **data}
        data = SimpleNamespace(**data)
    return data


def parseAndProcess():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--log",
        dest="log_level",
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    subparsers = parser.add_subparsers()
    funcs = {}

    def addToCli(f, name):
        ret = subparsers.add_parser(name)
        f(ret)
        funcs[name] = (
            ret.get_default("func"),
            {
                x.dest: x.default
                for x in ret._actions
                if isinstance(
                    x, (argparse._StoreAction, argparse.BooleanOptionalAction)
                )
            },
        )
        return ret

    x = addToCli(fitting.estimate.addToParser, "estimate")
    addToCli(fitting.diagnostics.addDiagnosticsToParser, "plots")
    addToCli(fitting.diagnostics.addCovarsToParser, "covars")
    addToCli(fitting.diagnostics.addEigensToParser, "eigens")
    addToCli(fitting.background_sim.addSimParser, "bkg-smooth")
    addToCli(fitting.predictive.addPValueParser, "model-checks")
    addToCli(fitting.gather_results.addGatherParser, "gather")
    addToCli(fitting.combine.plot_sensitivity.addPlotSensitivityParser, "plot-sens")
    addToCli(addDatacardGenerateParser, "make-datacard")

    config_parser = subparsers.add_parser("run-config")
    config_parser.add_argument("input")

    def runConfig(args):
        from types import SimpleNamespace
        import json

        with open(args.input, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]
        for d in data:
            command, defaults = funcs[d["command"]]
            all_data = {**defaults}
            all_data.update(d)
            ns = SimpleNamespace(**all_data)
            command(ns)

    config_parser.set_defaults(func=runConfig)

    args = parser.parse_args()
    setupLogging(args.log_level)

    # import code
    # import readline
    # import rlcompleter
    #
    # vars = globals()
    # vars.update(locals())
    # readline.set_completer(rlcompleter.Completer(vars).complete)
    # readline.parse_and_bind("tab: complete")
    # code.InteractiveConsole(vars).interact()

    args.func(args)


if __name__ == "__main__":

    parseAndProcess()
