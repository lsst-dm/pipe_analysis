import numpy as np


def addMagnitudes(cat, fluxColName="PsFlux"):
    """Add magnitude columns.

    Parameters
    ----------
    cat : `pandas.core.frame.DataFrame`
    fluxColName

    Returns
    -------
    cat : `pandas.core.frame.DataFrame`
    """

    for band in ["g", "r", "i", "z", "y"]:
        magColName = band + fluxColName[:-4] + "Mag"
        mags = -2.5*np.log10(cat[band + fluxColName].values/3631.0e9)
        cat[magColName] = mags

    return cat


def addColors(cat, magColName="PsMag"):
    """Add colors
    """

    bands = ["g", "r", "i", "z", "y"]
    for (i, band) in enumerate(bands[:-1]):
        colorColName = band + bands[i + 1] + magColName
        color = cat[band + magColName].values - cat[bands[i + 1] + magColName].values
        cat[colorColName] = color

    return cat


def stellarLocusFits(cat, magColName="PsMag"):

    # Coefficents from Ivezic 2004 adapted to HSC
    # The perpendicular fits
    wPerp = {"name": "wPerp", "range": "griBlue", "HSC-G": -0.274, "HSC-R": 0.803,
             "HSC-I": -0.259, "HSC-Z": 0.0, "const": 0.041}
    xPerp = {"name": "xPerp", "range": "griRed", "HSC-G": -0.680, "HSC-R": 0.731,
             "HSC-I": -0.051, "HSC-Z": 0.0, "const": 0.792}
    yPerp = {"name": "yPerp", "range": "rizRed", "HSC-G": 0.0, "HSC-R": -0.227,
             "HSC-I": 0.793, "HSC-Z": -0.566, "const": -0.017}

    # The parallel fits
    # Currently using the SDSS values (IS THIS STILL TRUE?)
    wPara = {"name": "wPara", "range": "griBlue", "HSC-G": 0.888, "HSC-R": -0.427,
             "HSC-I": -0.461, "HSC-Z": 0.0, "const": -0.478}
    xPara = {"name": "xPara", "range": "griRed", "HSC-G": 0.075, "HSC-R": 0.922,
             "HSC-I": -0.997, "HSC-Z": 0.0, "const": -1.442}
    yPara = {"name": "yPara", "range": "rizRed", "HSC-G": 0.0, "HSC-R": 0.928,
             "HSC-I": -0.557, "HSC-Z": -0.372, "const": -1.332}

    # Straight line fits for the perpendicular ranges
    # The following fits were derived in the process of calibrating
    # the above coeffs (all three RC2 tracts gave ~ the same fits).
    # May remove later if deemed no longer useful.
    wFit = {"name": "wFit", "range": "griBlue", "HSC-G": 0.52, "HSC-R": -0.52,
            "HSC-I": 0.0, "HSC-Z": 0.0, "const": -0.08}
    xFit = {"name": "xFit", "range": "griRed", "HSC-G": 13.35, "HSC-R": -13.35,
            "HSC-I": 0.0, "HSC-Z": 0.0, "const": -15.54}
    yFit = {"name": "yFit", "range": "rizRed", "HSC-G": 0.0, "HSC-R": 0.40,
            "HSC-I": -0.40, "HSC-Z": 0.0, "const": 0.03}

    # Use the equations from Ivezic 2004 to calculate these values for each
    # source
    for coeffs in [wPerp, xPerp, yPerp, wPara, xPara, yPara, wFit, xFit, yFit]:
        vals = (cat["g" + magColName].values*coeffs["HSC-G"]
                + cat["r" + magColName].values*coeffs["HSC-R"]
                + cat["i" + magColName].values*coeffs["HSC-I"]
                + cat["z" + magColName].values*coeffs["HSC-Z"] + coeffs["const"])
        cat[coeffs["name"]] = vals

    return cat


def addGalacticExtinction(cat):
    """Adds galactic extinction for each source
    """
    pass


def addUseForQAFlag(cat):
    """Add a flag to say if the source should be used for QA purposes
    """
    for col in cat.columns:
        if "lag" in col:
            print(col)

    use = ((np.isfinite(cat["gPsMag"])) & (np.isfinite(cat["rPsMag"]))
           & (np.isfinite(cat["iPsMag"])) & (np.isfinite(cat["zPsMag"]))
           & (np.isfinite(cat["zPsMag"])) & (np.isfinite(cat["gCModelMag"]))
           & (np.isfinite(cat["rCModelMag"])) & (np.isfinite(cat["iCModelMag"]))
           & (np.isfinite(cat["zCModelMag"])) & (np.isfinite(cat["yCModelMag"])))

    for band in ["g", "r", "i", "z", "y"]:
        print(cat[band + "Centroid_flag_notAtMaximum"])
        print(cat[band + "Shape_flag"].values[0])
        print(cat[band + "PsfFlux_flag"].values[0], type(cat[band + "PsfFlux_flag"].values[0]))
        usePerBand = ((cat[band + "Centroid_flag_notAtMaximum"] == 0) & (cat[band + "Shape_flag"] == 0)
                      & (cat[band + "PsfFlux_flag"] == 0)
                      & (cat[band + "PixelFlags_saturatedCenter"] == 0)
                      & (cat[band + "Extendedness_flag"] == 0))
        use = use & usePerBand
    cat["useForQAFlag"] = use
    return cat


def addSNColumn(cat):
    """Add a S/N column for each band
    """
    for band in ["g", "r", "i", "z", "y"]:
        SN = cat[band + "PsFlux"].values / cat[band + "PsFluxErr"].values
        cat[band + "SnPsFlux"] = SN

    for col in cat.columns:
        if "xx" in col:
            print(col)
    return cat


def addUseForStatsColumn(cat):
    """Add a column to indicate if a source should be used in
    calculating statistics.
    """
    # Can potentially get rid of this as S/N is needed 
    # as its own column for other plots
    useForStats = np.zeros(len(cat))
    SN = cat["iPsFlux"].values / cat["iPsFluxErr"].values

    lowSn = np.where((SN > 500))[0]
    useForStats[lowSn] = 2

    highSn = np.where((SN > 2700))[0]
    useForStats[highSn] = 1

    cat["useForStats"] = useForStats
    return cat


def addDeconvMoments(cat):
    """Add moments to the catalog
    """
    for band in ["g", "r", "i", "z", "y"]:
        shape = cat[band + "ShapeRound_xx"].values + cat[band + "ShapeRound_yy"].values
        psfShape = cat[band + "IxxPsf"].values + cat[band + "IyyPsf"].values
        cat[band + "DeconvMoments"] = shape
        cat[band + "PsDeconvMoments"] = psfShape

    return cat
