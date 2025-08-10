import mplhep

LUMI_SCALE_MAP = {
    "2016_preVFP": 19.65,
    "2016_postVFP": 16.98,
    "2017": 41.48,
    "2018": 59.83,
    "2022_preEE": 7.98,
    "2022_postEE": 26.67,
    "2023_preBPix": 17.65,
    "2023_postBPix": 9.451,
}


def addCMS(ax, text="", loc=1, year="2018", coupling="312"):
    if text is not None:
        text = "\n" + text

    energy = 13 if year.startswith("201") else 13.6
    lumi = LUMI_SCALE_MAP[year]
    lumi_text =f"{lumi} fb$^{{-1}}$ ({energy} TeV)"
    info_text = year + ", " + lumi_text
    mplhep.cms.lumitext(text=info_text, ax=ax)
    a, b, c = mplhep.cms.text(text=f"Preliminary", ax=ax, loc=1)
