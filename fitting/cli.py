import argparse
import fitting.estimate
import fitting.diagnostics

def parseAndProcess():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    estim = subparsers.add_parser("estimate")
    fitting.estimate.addToParser(estim)

    plots = subparsers.add_parser("plots")
    fitting.diagnostics.addToParser(plots)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    parseAndProcess()
    
