import argparse
import fitting.estimate
import fitting.diagnostics
import fitting.background_sim
from fitting.combine.generate import addDatacardGenerateParser


def parseAndProcess():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    fitting.estimate.addToParser(subparsers.add_parser("estimate"))
    fitting.diagnostics.addDiagnosticsToParser(subparsers.add_parser("plots"))
    fitting.diagnostics.addCovarsToParser(subparsers.add_parser("covars"))
    fitting.diagnostics.addEigensToParser(subparsers.add_parser("eigens"))
    fitting.background_sim.addSimParser(subparsers.add_parser("bkg-smooth"))
    addDatacardGenerateParser(subparsers.add_parser("make-datacard"))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":

    parseAndProcess()
