import argparse
import fitting.estimate
import fitting.diagnostics
import fitting.background_sim
import fitting.predictive
import fitting.gather_results
from fitting.combine.generate import addDatacardGenerateParser
from .logging import setupLogging


def parseAndProcess():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--log",
        dest="log_level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    subparsers = parser.add_subparsers()
    fitting.estimate.addToParser(subparsers.add_parser("estimate"))
    fitting.diagnostics.addDiagnosticsToParser(subparsers.add_parser("plots"))
    fitting.diagnostics.addCovarsToParser(subparsers.add_parser("covars"))
    fitting.diagnostics.addEigensToParser(subparsers.add_parser("eigens"))
    fitting.background_sim.addSimParser(subparsers.add_parser("bkg-smooth"))
    fitting.predictive.addPValueParser(subparsers.add_parser("model-checks"))
    fitting.gather_results.addGatherParser(subparsers.add_parser("gather"))
    addDatacardGenerateParser(subparsers.add_parser("make-datacard"))

    args = parser.parse_args()

    setupLogging(args.log_level)

    args.func(args)


if __name__ == "__main__":

    parseAndProcess()
