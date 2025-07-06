import mplhep


def addCMS(ax, text="", loc=0, year="2018"):
    if text is not None:
        text = "\n" + text
    mplhep.cms.label(
        data=True,
        label="Preliminary" + text,
        year=year,
        # lumi=59.8,
        # energy=13,
        ax=ax,
        loc=loc,
    )
