import mplhep


def addCMS(ax, text=""):
    if text is not None:
        text = "\n" + text
    mplhep.cms.label(
        data=True,
        label="Preliminary" + text,
        year=2018,
        lumi=59.8,
        # energy=13,
        ax=ax,
        loc=0,
    )
