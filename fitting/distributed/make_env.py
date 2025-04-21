try:
    import htcondor
    import classad

    HAS_CONDOR = True
except ImportError as e:
    HAS_CONDOR = False



