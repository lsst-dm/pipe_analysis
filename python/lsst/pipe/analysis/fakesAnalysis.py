# This file is part of pipe analysis
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Make QA plots for data with fake sources inserted
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib import colors, gridspec, patches
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import UnivariateSpline
import astropy.coordinates as coord
from astropy import units as u
from astropy.stats import mad_std as sigmaMAD

__all__ = ["addDegreePositions", "matchCatalogs", "addNearestNeighbor", "getPlotInfo",
           "fakesAreaDepth", "fakesPositionCompare", "fakesMagnitudeBlendedness",
           "fakesMagnitudeNearestNeighbor", "fakesMagnitudeCompare"]


def addDegreePositions(catalog, raCol, decCol):
    """Add RA and Dec columns in degrees, requires input columns in radians.

    Parameters
    ----------
    catalog : `pandas.core.frame.DataFrame`
        The catalog to add posistions in degrees to.
    raCol : `str`
        The column name for the RA column to be converted.
    decCol : `str`
        The column name for the Declination column to be converted.

    Returns
    -------
    catalog : `pandas.core.frame.DataFrame`
        The original catalog but with a column for the posistion in degrees added.
    """

    assert np.max(catalog[raCol].values) <= np.pi*2.0, "Input not in radians"
    assert np.max(catalog[decCol].values) <= np.pi*2.0, "Input not in radians"
    assert np.min(catalog[raCol].values) >= 0.0, "Input not in radians"
    assert np.min(catalog[decCol].values) >= 0.0, "Input not in radians"

    catalog[raCol + "_deg"] = np.rad2deg(catalog[raCol].values)
    catalog[decCol + "_deg"] = np.rad2deg(catalog[decCol].values)

    return catalog


def matchCatalogs(catalog1, raCol1, decCol1, catalog2, raCol2, decCol2, units=u.degree,
                  matchRadius=coord.Angle(0.1, unit=u.arcsecond)):
    """Match two dataframes together by RA and Dec.

    Parameters
    ----------
    catalog1 : `pandas.core.frame.DataFrame`
        The catalog to be matched to catalog2.
    catalog2 : `pandas.core.frame.DataFrame`
        The catalog that catalo1 is matched to.
    raCol1 : `string`
        The column name for the RA column in catalog1.
    decCol1 : `string`
        The column name for the Dec. column in catalog1.
    raCol2 : `string`
        The column name for the RA column in catalog2.
    decCol2 : `string`
        The column name for the Dec. column in catalog2.
    matchRadius : `astropy.coordinates.angles.Angle`
        Radius within which to match the nearest source, can be in any units supported by astropy.
        Default is coord.Angle(0.1, unit=u.arcsecond), 0.1 arseconds.
    units : `astropy.units.core.Unit`
        Units of the RA and Dec. given, defaults to degrees, can be anything supported by astropy.

    Returns
    -------
    catalog1Matched : `pandas.core.frame.DataFrame`
        Catalog with only the rows that had a match in catalog2.
    catalog2Matched : `pandas.core.frame.DataFrame`
        Catalog with only the rows that matched catalog1.

    Notes
    -----
    Returns two shortened catalogs, with the matched rows in the same order and objects without a match
    removed. Matches the first catalog to the second, multiple objects from the first catalog can match
    the same object in the second. Uses astropy's match_coordinates_sky and their units framework.
    """

    skyCoords1 = coord.SkyCoord(catalog1[raCol1], catalog1[decCol1], unit=units)
    skyCoords2 = coord.SkyCoord(catalog2[raCol2], catalog2[decCol2], unit=units)
    inds, dists, _ = coord.match_coordinates_sky(skyCoords1, skyCoords2)

    ids = (dists < matchRadius)

    matchedInds = inds[ids]
    catalog1["matched"] = np.zeros(len(catalog1))
    catalog1["matched"][ids] = 1
    catalog1Matched = catalog1[ids]
    catalog2Matched = catalog2.iloc[matchedInds].copy()

    return catalog1Matched, catalog2Matched, catalog1


def addNearestNeighbor(catalog, raCol, decCol, units=u.degree):
    """Add the distance to the nearest neighbor in the catalog.

    Parameters
    ----------
    catalog : `pandas.core.frame.DataFrame`
        Catalog to add the distance to the nearest neighbor to.
    raCol : `string`
        Column name for the RA column to be used.
    decCol : `string`
        Column name for the Declination column to be used.
    units : `astropy.units.core.Unit`
        Units of the RA and Dec. given, defaults to degrees, can be anything supported by astropy.

    Returns
    -------
    catalog : `pandas.core.frame.DataFrame`
        Catalog with a column 'nearestNeighbor' containing the distance to the neareast neighbor.

    Notes
    -----
    Distance added is in degrees.
    """

    skyCoords = coord.SkyCoord(catalog[raCol], catalog[decCol], unit=units)
    inds, dists, _ = coord.match_coordinates_sky(skyCoords, skyCoords, nthneighbor=2)

    catalog["nearestNeighbor"] = dists.degree

    return catalog


def getPlotInfo(repoInfo):
    """Parse the repoInfo into a dict of useful info for plots.

    Parameters
    ----------
    repoInfo : `lsst.pipe.base.struct.Struct`

    Returns
    -------
    plotInfoDict : `dict`
    """

    camera = repoInfo.camera.getName()
    dataId = repoInfo.dataId
    filterName = dataId["filter"]
    visit = str(dataId["visit"])
    tract = str(dataId["tract"])
    dataset = repoInfo.dataset
    rerun = list(repoInfo.butler.storage.repositoryCfgs)[0]
    plotInfoDict = {"camera": camera, "filter": filterName, "visit": visit, "tract": tract,
                    "dataset": dataset, "rerun": rerun}

    return plotInfoDict


def addProvenanceInfo(fig, plotInfoDict):
    """Add some useful provenance information to the plot

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`
        The figure that the information should be added to
    plotInfoDict : `dict`
        A dict containing useful information to add to the plot.
    """

    plt.text(0.85, 0.98, "Camera: " + plotInfoDict["camera"], fontsize=8, alpha=0.8,
             transform=fig.transFigure)
    plt.text(0.85, 0.96, "Filter: " + plotInfoDict["filter"], fontsize=8, alpha=0.8,
             transform=fig.transFigure)
    plt.text(0.85, 0.94, "Visit: " + plotInfoDict["visit"], fontsize=8, alpha=0.8, transform=fig.transFigure)
    plt.text(0.85, 0.92, "Tract: " + plotInfoDict["tract"], fontsize=8, alpha=0.8, transform=fig.transFigure)

    plt.text(0.02, 0.98, "rerun: " + plotInfoDict["rerun"], fontsize=8, alpha=0.8, transform=fig.transFigure)

    if "jointcal" in plotInfoDict["dataset"]:
        plt.text(0.02, 0.02, "JointCal Used? Yes", fontsize=8, alpha=0.8, transform=fig.transFigure)
    else:
        plt.text(0.02, 0.02, "JointCal Used? No", fontsize=8, alpha=0.8, transform=fig.transFigure)

    return fig


def fakesAreaDepth(inputFakesMatched, processedFakesMatched, plotInfoDict, areaDict, repoInfo,
                   measColType="base_PsfFlux_", distNeighbor=2.0 / 3600.0):
    """Plot the area vs depth for the given catalog

    Parameters
    ----------
    inputFakesMatched : `pandas.core.frame.DataFrame`
        The catalog used to add the fakes originally, matched to the processed catalog.
    processedFakesMatched : `pandas.core.frame.DataFrame`
        The catalog produced by the stack from the images with fakes in.
    plotInfoDict : `dict`
        A dict containing useful information to add to the plot, passed to addProvenanceInfo.
    areaDict : `dict`
        A dict containing the area of each ccd.
    repoInfo : `lsst.pipe.base.struct.Struct`
        A struct that contains information about the input data.
    measColType : `string`
        default : 'base_CircularApertureFlux_25_0_'
        Which type of flux/magnitude column to use for calculating the depth.

    Returns
    -------
    medMag10 : `float`
        The median of the magnitudes at flux/flux error of 10 from all the ccds.
    medMag100 : `float`
        The median of the magnitudes at flux/flux error of 100 from all the ccds.

    Notes
    -----
    Returns the median magnitude from all the CCDs at a signal to noise of 10 and a signal to noise of 100.
    measColType needs to have an associated magnitude column already computed. plotInfoDict should also
    contain outputName which is the output path and filename for the plot to be written to.
    """

    ccds = list(set(processedFakesMatched["ccdId"].values))

    mag10s = np.array([np.nan]*len(ccds))
    mag100s = np.array([np.nan]*len(ccds))
    areas = []

    for (i, ccd) in enumerate(ccds):
        onCcd = ((processedFakesMatched["ccdId"].values == ccd) &
                 (np.isfinite(processedFakesMatched[measColType + "mag"].values)) &
                 (processedFakesMatched["nearestNeighbor"].values >= distNeighbor) &
                 (inputFakesMatched["sourceType"].values == "star"))

        mags = processedFakesMatched[measColType + "mag"].values[onCcd]
        fluxes = processedFakesMatched[measColType + "instFlux"].values[onCcd]
        fluxErrs = processedFakesMatched[measColType + "instFluxErr"].values[onCcd]
        areas.append(areaDict[ccd])

        magsSNRatio = list(zip(fluxes/fluxErrs, mags))
        magsSNRatio.sort(key=lambda t: t[0])
        [fluxDivFluxErrsSorted, magsSorted] = [list(unzippedTuple) for unzippedTuple in zip(*magsSNRatio)]

        if len(fluxDivFluxErrsSorted) > 1:
            # Use a spline to interpolate the data and use it to give an estimate of the
            # magnitude limit at signal to noise of 10 and 100
            if len(list(set(fluxDivFluxErrsSorted))) < len(fluxDivFluxErrsSorted):
                # The spline will complain if there is a duplicate in the list.
                # Find the indices of the duplicated point and remove it
                D = defaultdict(list)
                for (j, item) in enumerate(fluxDivFluxErrsSorted):
                    D[item].append(j)
                repeatedInds = [value for key, value in D.items() if len(value) > 1]
                for inds in repeatedInds:
                    del fluxDivFluxErrsSorted[inds[1]]
                    del magsSorted[inds[1]]
            interpSpline = UnivariateSpline(fluxDivFluxErrsSorted, magsSorted, s=0)

            mag10s[i] = interpSpline(10.0)
            mag100s[i] = interpSpline(100.0)

    sigmas = ["10", "100"]
    areas = np.array(areas)/(3600.0**2)
    for (i, depths) in enumerate([mag10s, mag100s]):
        bins = np.linspace(min(depths), max(depths), 101)
        areasOut = np.zeros(100)
        n = 0
        magLimHalfArea = None
        magLim25 = None
        magLim75 = None
        while n < len(bins)-1:
            ids = np.where((depths >= bins[n]) & (depths < bins[n + 1]))[0]
            areasOut[n] += np.sum(areas[ids])
            if np.sum(areasOut) > np.sum(areas) / 2.0 and magLimHalfArea is None:
                magLimHalfArea = (bins[n] + bins[n + 1]) / 2.0
            if np.sum(areasOut) > np.sum(areas) * 0.25 and magLim25 is None:
                magLim25 = (bins[n] + bins[n + 1]) / 2.0
            if np.sum(areasOut) > np.sum(areas) * 0.75 and magLim75 is None:
                magLim75 = (bins[n] + bins[n + 1]) / 2.0
            n += 1

        cumAreas = np.zeros(100)
        n = 0
        for area in areasOut[::-1]:
            cumAreas[n] = area + cumAreas[n-1]
            n += 1

        plt.plot(bins[::-1][1:], cumAreas)
        plt.xlabel("Magnitude Limit (" + sigmas[i] + " sigma)", fontsize=15)
        plt.ylabel("Area / deg^2", fontsize=15)
        plt.title('Total Area to Given Depth \n (Recovered fake stars with no match within 2")')
        labelHA = "Mag. Lim. for 50% of the area: {:0.2f}".format(magLimHalfArea)
        plt.axvline(magLimHalfArea, label=labelHA, color="k", ls=":")
        label25 = "Mag. Lim. for 25% of the area: {:0.2f}".format(magLim25)
        plt.axvline(magLim25, label=label25, color="k", ls="--")
        label75 = "Mag. Lim. for 75% of the area: {:0.2f}".format(magLim75)
        plt.axvline(magLim75, label=label75, color="k", ls="--")
        plt.legend(loc="best")

        fig = plt.gcf()
        fig = addProvenanceInfo(fig, plotInfoDict)

        repoInfo.dataId["description"] = "areaDepth" + sigmas[i]
        repoInfo.butler.put(fig, "plotFakes", repoInfo.dataId)
        plt.close()

    medMag10 = np.median(mag10s)
    medMag100 = np.median(mag100s)

    return medMag10, medMag100


def fakesPositionCompare(inputFakesMatched, processedFakesMatched, plotInfoDict, repoInfo,
                         raFakesCol="raJ2000_deg", decFakesCol="decJ2000_deg", raCatCol="coord_ra_deg",
                         decCatCol="coord_dec_deg", magCol="base_PsfFlux_mag"):
    """Make a plot showing the RA and Dec offsets from the input positions.

    Parameters
    ----------
    inputFakesMatched : `pandas.core.frame.DataFrame`
        The catalog used to add the fakes originally, matched to the processed catalog.
    processedFakesMatched : `pandas.core.frame.DataFrame`
        The catalog produced by the stack from the images with fakes in.
    plotInfoDict : `dict`
        A dict containing useful information to add to the plot, passed to addProvenanceInfo.
    repoInfo : `lsst.pipe.base.struct.Struct`
        A struct that contains information about the input data.
    raFakesCol : `string`
        default : 'raJ2000_deg'
        The RA column to use from the fakes catalog.
    decFakesCol : `string`
        default : 'decJ2000_deg'
        The Dec. column to use from the fakes catalog.
    raCatCol : `string`
        default : 'coord_ra_deg'
        The RA column to use from the catalog.
    decCatCol : `string`
        default : 'coord_dec_deg'
        The Dec. column to use from the catalog.
    magCol : `string`
        default : 'base_CircularApertureFlux_25_0_mag'
        The magnitude column to use from the catalog.

    Returns
    -------
    dRAMed : `float`
        The median RA difference.
    dRASigmaMAD : `float`
        The sigma MAD from the RA difference.
    dDecMed : `float`
        The median Dec. difference.
    dDecSigmaMAD : `float`
        The sigma MAD from the Dec. difference.

    Notes
    -----
    The two input catalogs need to be pre matched and in the same order so that the entry for an object is
    in the same row in each catalog. The delta RA and Dec is given in milli arcseconds. The plot is made
    using only objects that were stars in the input catalog of fakes. `plotInfoDict` needs to be a dict
    containing camera, filter, visit, tract and dataset (jointcal or not). Returns the median and sigma MAD
    for the RA and Dec offsets. plotInfoDict should also contain the magnitude limit that the plot should go
    to (magLim) and the path and filename that the figure should be written to (outputName).
    """

    pointsToUse = (processedFakesMatched[magCol].values < plotInfoDict["magLim"])

    processedFakesMatched = processedFakesMatched[pointsToUse]
    inputFakesMatched = inputFakesMatched[pointsToUse]

    stars = (inputFakesMatched["sourceType"] == "star")

    dRA = (inputFakesMatched[raFakesCol].values - processedFakesMatched[raCatCol].values)[stars]
    dRA *= (3600 * 1000 * np.cos(np.deg2rad(inputFakesMatched[decFakesCol].values[stars])))
    dDec = (inputFakesMatched[decFakesCol].values - processedFakesMatched[decCatCol].values)[stars]
    dDec *= (3600 * 1000)

    dRAMax = np.max(np.fabs(dRA))
    dDecMax = np.max(np.fabs(dDec))

    nBins = int(len(inputFakesMatched) / 100)
    gs = gridspec.GridSpec(4, 4)

    # Make a 2D histogram of the position offsets
    axHist = plt.subplot(gs[1:, :-1])
    axHist.axvline(0.0, ls=":", color="gray", alpha=0.5, zorder=0)
    axHist.axhline(0.0, ls=":", color="gray", alpha=0.5, zorder=0)
    _, _, _, im = axHist.hist2d(dRA, dDec, bins=nBins, cmap="winter", norm=colors.LogNorm(), zorder=10)
    axHist.set_xlim(-1 * dRAMax, dRAMax)
    axHist.set_ylim(-1 * dDecMax, dDecMax)
    axHist.tick_params(top=True, right=True)

    # Make a 1D histogram of the RA offsets
    axRA = plt.subplot(gs[0, :-1], sharex=axHist)
    axRA.hist(dRA, bins=nBins, log=True)
    axRA.axes.get_xaxis().set_visible(False)
    axRA.set_ylabel("Number")

    # Make a 1D histogram of the Dec. offsets
    axDec = plt.subplot(gs[1:, -1], sharey=axHist)
    axDec.hist(dDec, orientation="horizontal", bins=nBins, log=True)
    axDec.axes.get_yaxis().set_visible(False)
    axDec.set_xlabel("Number")

    # Add a color bar for the 2D histogram
    divider = make_axes_locatable(axDec)
    cax = divider.append_axes("right", size="8%", pad=0.00)
    plt.colorbar(im, cax=cax, orientation="vertical")

    # Add some statistics to the axis in the top right corner
    axStats = plt.subplot(gs[0, -1])
    axStats.axes.get_xaxis().set_visible(False)
    axStats.axes.get_yaxis().set_visible(False)

    dRAMed = np.median(dRA)
    dDecMed = np.median(dDec)
    dRASigmaMAD = sigmaMAD(dRA)
    dDecSigmaMAD = sigmaMAD(dDec)

    infoMedRA = r"Med. $\delta$RA %0.3f" % dRAMed
    infoMadRA = r"$\sigma_{MAD}$ $\delta$RA %0.3f" % dRASigmaMAD
    infoMedDec = r"Med. $\delta$Dec %0.3f" % dDecMed
    infoMadDec = r"$\sigma_{MAD}$ $\delta$Dec %0.3f" % dDecSigmaMAD

    axStats.text(0.05, 0.8, infoMedRA, fontsize=10)
    axStats.text(0.05, 0.6, infoMadRA, fontsize=10)
    axStats.text(0.05, 0.4, infoMedDec, fontsize=10)
    axStats.text(0.05, 0.2, infoMadDec, fontsize=10)
    axStats.text(0.05, 0.05, "(mas)", fontsize=8)

    axHist.set_xlabel(r"$\delta$RA / mas")
    axHist.set_ylabel(r"$\delta$Dec / mas")

    axRA.set_title("Position Offsets for Input Fakes - Recovered Fakes")
    numInfo = "Num. Sources: %d \n Mag. Limit: %0.2f" % (len(dRA), plotInfoDict["magLim"])
    bbox = dict(facecolor="white", edgecolor="k", alpha=0.8)
    plt.text(-2.9, 0.7, numInfo, fontsize=8, bbox=bbox)

    plt.subplots_adjust(top=0.9, hspace=0.0, wspace=0.0, right=0.93, left=0.12, bottom=0.1)

    fig = plt.gcf()
    fig = addProvenanceInfo(fig, plotInfoDict)

    repoInfo.dataId["description"] = "positionCompare"
    repoInfo.butler.put(fig, "plotFakes", repoInfo.dataId)
    plt.close()

    return dRAMed, dRASigmaMAD, dDecMed, dDecSigmaMAD


def plotWithOneHist(xs, ys, maskForStats, xLabel, yLabel, title, plotInfoDict):
    """Makes a generic plot with a scatter plot/2D histogram in the dense areas and a collapsed histogram of
       the y axis.

    Parameters
    ----------
    xs : `numpy.ndarray`
         The array to be plotted on the x axis.
    ys : `numpy.ndarray`
         The array to be plotted on the y axis.
    maskForStats : `numpy.ndarray`
         An array of the ids of the objects to be used in calculating the printed stats.
    xLabel : `string`
         The text to go on the xLabel of the plot.
    yLabel : `string`
         The text to go on the yLabel of the plot.
    title : `string`
         The text to be displayed as the plot title.
    plotInfoDict : `dict`
        A dict containing useful information to add to the plot, passed to addProvenanceInfo.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
    """

    medYs = np.median(ys)
    sigmaMadYs = sigmaMAD(ys)

    # Make a colormap that looks good
    r, g, b = colors.colorConverter.to_rgb("C0")
    r1, g1, b1 = colors.colorConverter.to_rgb("midnightblue")
    colorDict = {"blue": ((0.0, b, b), (1.0, b1, b1)), "red": ((0.0, r, r), (1.0, r1, r1)),
                 "green": ((0.0, g, g), (1.0, g1, g1))}
    colorDict["alpha"] = ((0.0, 0.0, 0.0), (0.05, 0.3, 0.3), (0.5, 0.8, 0.8), (1.0, 1.0, 1.0))
    newBlues = colors.LinearSegmentedColormap("newBlues", colorDict)

    fig = plt.figure()
    gs = gridspec.GridSpec(1, 4)
    ax = fig.add_subplot(gs[:, :-1])
    fiveSigmaHigh = medYs + 5.0 * sigmaMadYs
    fiveSigmaLow = medYs - 5.0 * sigmaMadYs
    binSize = (fiveSigmaHigh - fiveSigmaLow) / 101.0
    yEdges = np.arange(fiveSigmaLow, fiveSigmaHigh, binSize)
    [xs1, xs25, xs50, xs75, xs95, xs97] = np.percentile(xs, [1, 25, 50, 75, 95, 97])
    xScale = (np.max(xs)-np.min(xs))/20.0
    xEdges = np.arange(xs1 - xScale, xs95, (xs95 - (xs1 - xScale))/40.0)

    counts, xBins, yBins = np.histogram2d(xs, ys, bins=(xEdges, yEdges))
    countsYs = np.sum(counts, axis=1)

    ids = np.where((countsYs > 50.0))[0]
    xEdgesPlot = xEdges[1:][ids]

    # Create the codes needed to turn the sigmaMad lines into a path to speed up checking which points are
    # inside the area
    codes = np.ones(len(xEdgesPlot) * 2) * Path.LINETO
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY

    meds = np.zeros(len(xEdgesPlot))
    sigmaMadVerts = np.zeros((len(xEdgesPlot) * 2, 2))
    sigmaMads = np.zeros(len(xEdgesPlot))

    for (i, xEdge) in enumerate(xEdgesPlot):
        ids = np.where((xs < xEdge) & (xs > xEdges[i]) & (np.isfinite(ys)))[0]
        med = np.median(ys[ids])
        sigmaMad = sigmaMAD(ys[ids])
        meds[i] = med
        sigmaMads[i] = sigmaMad
        sigmaMadVerts[i, :] = [xEdge, med + sigmaMad]
        sigmaMadVerts[-(i + 1), :] = [xEdge, med - sigmaMad]

    medDiff = np.median(ys[maskForStats])
    medLine, = ax.plot(xEdgesPlot, meds, "k", label="Median: {:0.2f}".format(medDiff))

    # Make path to check which points lie within one sigma mad
    sigmaMadPath = Path(sigmaMadVerts, codes)

    # Add lines for the median +/- 0.5 * sigmaMad
    halfSigmaMadLine, = ax.plot(xEdgesPlot, meds + 0.5 * sigmaMads, "k", alpha=0.8,
                                label=r"$\frac{1}{2}$ $\sigma_{MAD}$")
    ax.plot(xEdgesPlot, meds - 0.5 * sigmaMads, "k", alpha=0.8)

    # Add lines for the median +/- sigmaMad
    statsSigmaMad = sigmaMAD(ys[maskForStats])
    sigmaMadLine, = ax.plot(xEdgesPlot, sigmaMadVerts[:len(xEdgesPlot), 1], "k", alpha=0.6,
                            label=r"$\sigma_{MAD}$: " + "{:0.2f}".format(statsSigmaMad))
    ax.plot(xEdgesPlot[::-1], sigmaMadVerts[len(xEdgesPlot):, 1], "k", alpha=0.6)

    # Add lines for the median +/- 2 * sigmaMad
    twoSigmaMadLine, = ax.plot(xEdgesPlot, meds + 2.0 * sigmaMads, "k", alpha=0.4, label=r"2 $\sigma_{MAD}$")
    ax.plot(xEdgesPlot, meds - 2.0 * sigmaMads, "k", alpha=0.4)

    # Check which points are outside 1 sigmaMad of the median and plot these as points, histogram the rest
    inside = sigmaMadPath.contains_points(np.array([xs, ys]).T)
    _, _, _, histIm = ax.hist2d(xs[inside], ys[inside], bins=(xEdgesPlot, yEdges), cmap=newBlues, zorder=-2,
                                cmin=1)
    notStatsPoints, = ax.plot(xs[~inside & ~maskForStats], ys[~inside & ~maskForStats], "x", ms=3, alpha=0.3,
                              mfc="C0", zorder=-1, label="Not used in Stats")
    statsPoints, = ax.plot(xs[maskForStats & ~inside], ys[maskForStats & ~inside], ".", ms=3, alpha=0.3,
                           mfc="C0", zorder=-1, mec="C0", label="Used in stats")

    # Divide the data into sections and plot historgrams for them
    sectionColors = ["C0", "royalblue", "midnightblue", "darkslateblue"]
    sectionBounds = [xs25, xs50, xs75]
    sectionSelections = [np.where((xs < xs25))[0], np.where((xs < xs50) & (xs >= xs25))[0],
                         np.where((xs < xs75) & (xs >= xs50))[0], np.where((xs >= xs75))[0]]

    for (i, sectionColor) in enumerate(sectionColors):
        if i != len(sectionColors) - 1:
            quartileLine = ax.axvline(sectionBounds[i], color="k", ls="--", zorder=-10, label="Quartiles")
            ax.arrow(sectionBounds[i], medYs + 2.5 * sigmaMadYs, -1.0 * xScale / 4, 0, head_width=2.0,
                     head_length=0.25, color=sectionColor, fc=sectionColor)
        if i != 0:
            ax.arrow(sectionBounds[i - 1], medYs + 2.5 * sigmaMadYs, xScale / 4, 0, head_width=2.0,
                     head_length=0.25, color=sectionColor, fc=sectionColor)

    # Add a 1d histogram showing the offsets in magnitude difference
    axHist = plt.subplot(gs[:, -1], sharey=ax)
    axHist.axhline(0.0, color="grey", ls="--", zorder=-11)
    axHist.hist(ys, orientation="horizontal", bins=yBins, color="C0", alpha=0.5)
    axHist.axes.get_yaxis().set_visible(False)
    axHist.set_xlabel("Number")
    for (i, section) in enumerate(sectionSelections):
        axHist.hist(ys[section], histtype="step", orientation="horizontal", bins=yBins,
                    color=sectionColors[i])

    # Add a color bar for the 2d histogram
    divider = make_axes_locatable(axHist)
    cax = divider.append_axes("right", size="8%", pad=0.00)
    plt.colorbar(histIm, cax=cax, orientation="vertical", label="Number of Points Per Bin")

    ax.set_xlabel(xLabel, fontsize=10)
    ax.set_ylabel(yLabel, fontsize=10)
    ax.axhline(0.0, color="grey", ls="--", zorder=-11)

    # Make the plot limits median + 3 sigmaMad from all the points (not just those < the mag lim)
    ax.set_ylim(medYs - 3.0 * sigmaMadYs, medYs + 3 * sigmaMadYs)
    ax.set_xlim(xs1 - xScale, xs97)

    # Add legends, needs to be split up as otherwise too large
    ax.set_title(title)
    sigmaLines = [medLine, sigmaMadLine, halfSigmaMadLine, twoSigmaMadLine]
    infoLines = [quartileLine]

    legendQuartiles = ax.legend(handles=infoLines, loc="upper left", fontsize=10, framealpha=0.9,
                                borderpad=0.2)
    ax.add_artist(legendQuartiles)
    legendSigmaLines = ax.legend(handles=sigmaLines, loc="lower left", ncol=2, fontsize=10, framealpha=0.9,
                                 borderpad=0.2)
    plt.draw()
    plt.subplots_adjust(wspace=0.0)

    # Make the legends line up nicely
    legendBBox = legendSigmaLines.get_window_extent()
    yLegFigure = legendBBox.transformed(plt.gcf().transFigure.inverted()).ymin
    fig.legend(handles=[notStatsPoints, statsPoints], fontsize=8, borderaxespad=0, loc="lower left",
               bbox_to_anchor=(0.66, yLegFigure), bbox_transform=fig.transFigure, framealpha=0.9,
               markerscale=2)
    # Add the infomation about the data origins
    fig = plt.gcf()
    fig = addProvenanceInfo(fig, plotInfoDict)

    return fig


def fakesMagnitudeCompare(inputFakesMatched, processedFakesMatched, plotInfoDict, repoInfo,
                          magCol="base_PsfFlux_mag"):
    """Make a plot showing the comparison between the input and extracted magnitudes.

    Parameters
    ----------
    inputFakesMatched : `pandas.core.frame.DataFrame`
        The catalog used to add the fakes originally, matched to the processed catalog.
    processedFakesMatched : `pandas.core.frame.DataFrame`
        The catalog produced by the stack from the images with fakes in.
    plotInfoDict : `dict`
        A dict containing useful information to add to the plot.
    repoInfo : `lsst.pipe.base.struct.Struct`
        A struct that contains information about the input data.
    magCol : `string`
        default : 'base_PsfFlux_mag'
        The magnitude column to use from the catalog.

    Returns
    -------
    medStars : `float`
        Median magnitude difference for all stars

    Notes
    -----
    The two input catalogs need to be pre matched and in the same order so that the entry for an object is
    in the same row in each catalog. The magnitude difference is given in milli mags. The plot is made
    using only objects that were stars in the input catalog of fakes. `plotInfoDict` needs to be a dict
    containing camera, filter, visit, tract and dataset (jointcal or not). plotInfoDict should also contain
    the magnitude limit that the plot should go to (magLim).
    """
    band = plotInfoDict["filter"][-1].lower()

    stars = (inputFakesMatched["sourceType"] == "star")

    fakeMagStars = inputFakesMatched[band[-1].lower() + "magVar"].values[stars]
    catMagStars = processedFakesMatched[magCol].values[stars]
    finiteStars = np.where(np.isfinite(catMagStars))[0]
    ys = (catMagStars[finiteStars] - fakeMagStars[finiteStars])*1000
    xs = fakeMagStars[finiteStars]
    maskForStats = (xs < plotInfoDict["magLim"])
    xLabel = "Input Magnitude / milli mags"
    yLabel = "Output - Input Magnitude / milli mags"
    title = "Magnitude Difference For Fake Stars"

    fig = plotWithOneHist(xs, ys, maskForStats, xLabel, yLabel, title, plotInfoDict)

    # Don't have good mags for galaxies at this point.
    # To Do: coadd version of plot with cmodel mags.

    repoInfo.dataId["description"] = "magnitudeCompare"
    repoInfo.butler.put(fig, "plotFakes", repoInfo.dataId)

    plt.close()


def fakesMagnitudeNearestNeighbor(inputFakesMatched, processedFakesMatched, plotInfoDict, repoInfo,
                                  magCol="base_PsfFlux_mag"):
    """Make a plot showing the comparison between the input and extracted magnitudes against the distance
       to the neareast neighbor.

    Parameters
    ----------
    inputFakesMatched : `pandas.core.frame.DataFrame`
        The catalog used to add the fakes originally, matched to the processed catalog.
    processedFakesMatched : `pandas.core.frame.DataFrame`
        The catalog produced by the stack from the images with fakes in.
    plotInfoDict : `dict`
        A dict containing useful information to add to the plot.
    repoInfo : `lsst.pipe.base.struct.Struct`
        A struct that contains information about the input data.
    magCol : `string`
        default : 'base_PsfFlux_mag'
        The magnitude column to use from the catalog.

    Notes
    -----
    The two input catalogs need to be pre matched and in the same order so that the entry for an object is
    in the same row in each catalog. The magnitude difference is given in milli mags. The plot is made
    using only objects that were stars in the input catalog of fakes. `plotInfoDict` needs to be a dict
    containing camera, filter, visit, tract and dataset (jointcal or not). plotInfoDict should also contain
    the magnitude limit that the plot should go to (magLim).
    """
    band = plotInfoDict["filter"][-1].lower()

    stars = (inputFakesMatched["sourceType"] == "star")

    fakeMagStars = inputFakesMatched[band[-1].lower() + "magVar"].values[stars]
    catMagStars = processedFakesMatched[magCol].values[stars]
    nearestNeighborDistance = processedFakesMatched["nearestNeighbor"].values[stars] * 3600.0
    finiteValues = np.where((np.isfinite(catMagStars)) & (np.isfinite(nearestNeighborDistance)))[0]
    ys = (catMagStars[finiteValues] - fakeMagStars[finiteValues]) * 1000
    xs = nearestNeighborDistance[finiteValues]
    mags = fakeMagStars[finiteValues]
    maskForStats = (mags < plotInfoDict["magLim"])

    # Don't have good mags for galaxies at this point.
    # To Do: coadd version of plot with cmodel mags.

    xLabel = "Distance to Nearest Neighbor / arcsecs"
    yLabel = "Output - Input Magnitude / milli mags"
    title = "Magnitude Difference For Fake Stars \nAgainst Distance to Nearest Neighbor"

    fig = plotWithOneHist(xs, ys, maskForStats, xLabel, yLabel, title, plotInfoDict)

    repoInfo.dataId["description"] = "magnitudeNearestNeighbor"
    repoInfo.butler.put(fig, "plotFakes", repoInfo.dataId)

    plt.close()


def fakesMagnitudeBlendedness(inputFakesMatched, processedFakesMatched, plotInfoDict, repoInfo,
                              magCol="base_PsfFlux_mag"):
    """Make a plot showing the comparison between the input and extracted magnitudes against blendedness.

    Parameters
    ----------
    inputFakesMatched : `pandas.core.frame.DataFrame`
        The catalog used to add the fakes originally, matched to the processed catalog.
    processedFakesMatched : `pandas.core.frame.DataFrame`
        The catalog produced by the stack from the images with fakes in.
    plotInfoDict : `dict`
        A dict containing useful information to add to the plot, passed to addProvenanceInfo.
    repoInfo : `lsst.pipe.base.struct.Struct`
        A struct that contains information about the input data.
    magCol : `string`
        default : 'base_PsfFlux_mag'
        The magnitude column to use from the catalog.

    Notes
    -----
    The two input catalogs need to be pre matched and in the same order so that the entry for an object is
    in the same row in each catalog. The magnitude difference is given in milli mags. The plot is made
    using only objects that were stars in the input catalog of fakes. `plotInfoDict` needs to be a dict
    containing camera, filter, visit, tract and dataset (jointcal or not). plotInfoDict should also contain
    the magnitude limit that the plot should go to (magLim).
    """
    band = plotInfoDict["filter"][-1].lower()

    stars = (inputFakesMatched["sourceType"] == "star")

    fakeMagStars = inputFakesMatched[band[-1].lower() + "magVar"].values[stars]
    catMagStars = processedFakesMatched[magCol].values[stars]
    blendedness = np.log10(processedFakesMatched["base_Blendedness_abs"].values[stars])
    finiteValues = np.where((np.isfinite(catMagStars)) & (np.isfinite(blendedness)))[0]
    ys = (catMagStars[finiteValues] - fakeMagStars[finiteValues]) * 1000
    xs = blendedness[finiteValues]
    mags = fakeMagStars[finiteValues]
    maskForStats = (mags < plotInfoDict["magLim"])

    xLabel = "log10(Blendedness)"
    yLabel = "Output - Input Magnitude / milli mags"
    title = "Magnitude Difference For Fake Stars \nAgainst Blendedness"

    fig = plotWithOneHist(xs, ys, maskForStats, xLabel, yLabel, title, plotInfoDict)

    repoInfo.dataId["description"] = "magnitudeBlendedness"
    repoInfo.butler.put(fig, "plotFakes", repoInfo.dataId)

    plt.close()


def fakesCompletenessPlot(inputFakes, inputFakesMatched, processedFakesMatched, plotInfoDict, areaDict,
                          repoInfo, raFakesCol="raJ2000_deg", decFakesCol="decJ2000_deg",
                          raCatCol="coord_ra_deg", decCatCol="coord_dec_deg", distNeighbor=2.0 / 3600.0):
    """Makes three plots, one showing a two dimensional histogram of the fraction of fakes recovered,
    one a 1D histogram showing the area against the depth and one showing the fraction of input fakes
    recovered by magnitude.

    Parameters
    ----------
    inputFakes : `pandas.core.frame.DataFrame`
        The catalog used to add the fakes originally.
    inputFakesMatched : `pandas.core.frame.DataFrame`
        The catalog used to add the fakes originally, matched to the processed catalog.
    processedFakesMatched : `pandas.core.frame.DataFrame`
        The catalog produced by the stack from the images with fakes in.
    plotInfoDict : `dict`
        A dict containing useful information to add to the plot, passed to addProvenanceInfo.
     areaDict : `dict`
        A dict containing the area of each ccd.
    repoInfo : `lsst.pipe.base.struct.Struct`
        A struct that contains information about the input data.
    raFakesCol : `string`
        default : 'raJ2000_deg'
        The RA column to use from the fakes catalog.
    decFakesCol : `string`
        default : 'decJ2000_deg'
        The Dec. column to use from the fakes catalog.
    raCatCol : `string`
        default : 'coord_ra_deg'
        The RA column to use from the catalog.
    decCatCol : `string`
        default : 'coord_dec_deg'
        The Dec. column to use from the catalog.

    Notes
    -----
    Makes 3 plots to study the completeness of the data.
    The first is a 2D histogram of the fraction of fakes recovered.
    The second is a cumulative plot of the area which has fainter sources recovered than a given magnitude.
    The third is a histogram showing the fraction recovered in each magnitude bin with the number input and
    recovered overplotted.
    """

    band = plotInfoDict["filter"][-1].lower()

    # Make a color map that looks nice
    r, g, b = colors.colorConverter.to_rgb("C0")
    r1, g1, b1 = colors.colorConverter.to_rgb("midnightblue")
    colorDict = {"blue": ((0.0, b, b), (1.0, b1, b1)), "red": ((0.0, r, r), (1.0, r1, r1)),
                 "green": ((0.0, g, g), (1.0, g1, g1))}
    colorDict["alpha"] = ((0.0, 0.2, 0.2), (0.05, 0.3, 0.3), (0.5, 0.8, 0.8), (1.0, 1.0, 1.0))
    newBlues = colors.LinearSegmentedColormap("newBlues", colorDict)

    # Find the fake stars"
    stars = np.where((inputFakes["sourceType"] == "star"))[0]
    starsMatched = np.where((inputFakesMatched["sourceType"] == "star"))[0]

    # Find the min/max ra/dec and use them to find an approximate scale for the plot limits.
    minRa = np.min(inputFakesMatched[raFakesCol].values[starsMatched])
    maxRa = np.max(inputFakesMatched[raFakesCol].values[starsMatched])
    minDec = np.min(inputFakesMatched[decFakesCol].values[starsMatched])
    maxDec = np.max(inputFakesMatched[decFakesCol].values[starsMatched])

    scaleRa = np.fabs((maxRa - minRa) / 20.0)
    scaleDec = np.fabs((maxDec - minDec) / 20.0)

    rasFakes = inputFakes[raFakesCol].values[stars]
    decsFakes = inputFakes[decFakesCol].values[stars]

    rasFakesMatched = inputFakesMatched[raFakesCol].values[starsMatched]
    decsFakesMatched = inputFakesMatched[decFakesCol].values[starsMatched]

    fig, ax = plt.subplots()

    # Run through the ccds, plot them and then histogram the data on them
    for ccd in list(set(processedFakesMatched["ccdId"].values)):
        # Find the ccd corners and sizes
        corners = areaDict["corners_" + str(ccd)]
        xy = (corners[0].getRa().asDegrees(), corners[0].getDec().asDegrees())
        width = corners[1].getRa().asDegrees() - corners[0].getRa().asDegrees()
        height = corners[1].getDec().asDegrees() - corners[0].getDec().asDegrees()

        # Some of the ccds are rotated and some have xy0 as being on the right with negative width/height
        # this upsets the binning so find the min and max to calculate positive bin widths from.
        minX = np.min([xy[0], corners[1].getRa().asDegrees()])
        maxX = np.max([xy[0], corners[1].getRa().asDegrees()])
        minY = np.min([xy[1], corners[1].getDec().asDegrees()])
        maxY = np.max([xy[1], corners[1].getDec().asDegrees()])
        if np.fabs(width) > np.fabs(height):
            binWidth = (maxX - minX) / 10
        else:
            binWidth = (maxY - minY) / 10
        xEdges = np.arange(minX, maxX + binWidth, binWidth)
        yEdges = np.arange(minY, maxY + binWidth, binWidth)

        # Plot the ccd outlines
        ccdPatch = patches.Rectangle(xy, width, height, fill=False, edgecolor="k", alpha=0.7)
        ax.add_patch(ccdPatch)

        # Check which points are on the ccd, the ccd patch is in different coordinates to the data, hence
        # the transform being required
        trans = ccdPatch.get_patch_transform()
        histPoints = ccdPatch.get_path().contains_points(list(zip(rasFakes, decsFakes)), transform=trans)
        pointsFakesMatched = list(zip(rasFakesMatched, decsFakesMatched))
        histPointsMatched = ccdPatch.get_path().contains_points(pointsFakesMatched, transform=trans)

        # Histogram them both and then take the ratio
        hFakes, _, _ = np.histogram2d(rasFakes[histPoints], decsFakes[histPoints], bins=(xEdges, yEdges))
        hFakesMatched, _, _ = np.histogram2d(rasFakesMatched[histPointsMatched],
                                             decsFakesMatched[histPointsMatched], bins=(xEdges, yEdges))
        X, Y = np.meshgrid(xEdges, yEdges)
        # Plot 1 - the ratio to give the fraction missed as it looks better
        fracIm = plt.pcolormesh(X, Y, 1.0 - hFakesMatched.T/hFakes.T, cmap=newBlues, vmin=0.0, vmax=1.0)

    # Add a color bar
    colorBar = plt.gcf().colorbar(fracIm, ax=plt.gca())
    colorBar.set_label("Fraction Missed")

    plt.xlabel("R. A. / Degrees")
    plt.ylabel("Declination / Degrees")
    plt.title('Fraction of Sources Missed \n (Fake stars)')

    # Add useful information to the plot
    fig = plt.gcf()
    fig = addProvenanceInfo(fig, plotInfoDict)

    plt.xlim(minRa - scaleRa, maxRa + scaleRa)
    plt.ylim(minDec - scaleDec, maxDec + scaleDec)
    plt.subplots_adjust(right=0.99)

    # Save the graph
    repoInfo.dataId["description"] = "completenessHist2D"
    repoInfo.butler.put(fig, "plotFakes", repoInfo.dataId)
    plt.close()

    # Now condense this information into a histogram

    depths = []
    ccds = list(set(processedFakesMatched["ccdId"].values))
    areas = np.zeros(len(ccds))

    # Find the faintest recovered isolated star on each ccd
    for (i, ccd) in enumerate(ccds):
        onCcd = ((processedFakesMatched["ccdId"].values == ccd) &
                 (processedFakesMatched["nearestNeighbor"].values >= distNeighbor) &
                 (inputFakesMatched["sourceType"].values == "star"))

        depths.append(np.max(inputFakesMatched[band + "magVar"].values[onCcd]))
        areas[i] = areaDict[ccd] / (3600.0**2)

    # Make a cumulative histogram of the area vs magnitude of the faintest source for each ccd
    bins = np.linspace(min(depths), max(depths), 101)
    areasOut = np.zeros(100)
    n = 0
    magLimHalfArea = None
    magLim25 = None
    magLim75 = None
    while n < len(bins)-1:
        ids = np.where((depths >= bins[n]) & (depths < bins[n + 1]))[0]
        areasOut[n] += np.sum(areas[ids])
        if np.sum(areasOut) > np.sum(areas) / 2.0 and magLimHalfArea is None:
            magLimHalfArea = (bins[n] + bins[n + 1]) / 2.0
        if np.sum(areasOut) > np.sum(areas) * 0.25 and magLim25 is None:
            magLim25 = (bins[n] + bins[n + 1]) / 2.0
        if np.sum(areasOut) > np.sum(areas) * 0.75 and magLim75 is None:
            magLim75 = (bins[n] + bins[n + 1]) / 2.0
        n += 1

    cumAreas = np.zeros(100)
    n = 0
    for area in areasOut[::-1]:
        cumAreas[n] = area + cumAreas[n-1]
        n += 1

    plt.plot(bins[::-1][1:], cumAreas)
    plt.xlabel("Faintest Source Recovered / mags", fontsize=15)
    plt.ylabel("Area / deg^2", fontsize=15)
    plt.title('Faintest Source Recovered \n (Fake stars with no match within 2")')
    labelHA = "Mag. Lim. for 50% of the area: {:0.2f}".format(magLimHalfArea)
    plt.axvline(magLimHalfArea, label=labelHA, color="k", ls=":")
    label25 = "Mag. Lim. for 25% of the area: {:0.2f}".format(magLim25)
    plt.axvline(magLim25, label=label25, color="k", ls="--")
    label75 = "Mag. Lim. for 75% of the area: {:0.2f}".format(magLim75)
    plt.axvline(magLim75, label=label75, color="k", ls="--")
    plt.legend(loc="best")

    # Add useful information to the plot
    fig = plt.gcf()
    fig = addProvenanceInfo(fig, plotInfoDict)

    # Save the graph
    repoInfo.dataId["description"] = "completenessAreaDepth"
    repoInfo.butler.put(fig, "plotFakes", repoInfo.dataId)
    plt.close()

    # Make plot showing the fraction recovered in magnitude bins
    fig, axLeft = plt.subplots()
    axRight = axLeft.twinx()
    axLeft.tick_params(axis="y", labelcolor="C0")
    axLeft.set_ylabel("Fraction Recovered", color="C0")
    axLeft.set_xlabel("Magnitude / mags")
    axRight.set_ylabel("Number of Sources")
    plt.title("Fraction of Sources Recovered at Each Magnitude")
    overlap = np.isfinite(inputFakes["onCcd"])
    nInput, bins, _ = axRight.hist(inputFakes[band + "magVar"][overlap], bins=100, log=True, histtype="step",
                                   label="Input Fakes", color="black")
    nOutput, _, _ = axRight.hist(inputFakesMatched[band + "magVar"], bins=bins, log=True, histtype="step",
                                 label="Recovered Fakes", color="grey")

    # Find bin where the fraction recovered first falls below 0.5
    lessThanHalf = np.where((nOutput/nInput < 0.5))[0]
    mag50 = np.min(bins[lessThanHalf])
    xlims = plt.gca().get_xlim()
    axLeft.plot([xlims[0], mag50], [0.5, 0.5], ls=":", color="grey")
    plt.xlim(xlims)
    axLeft.plot([mag50, mag50], [0, 0.5], ls=":", color="grey")
    axRight.legend(loc="upper left", ncol=2)
    axLeft.axhline(1.0, color="grey", ls="--")
    axLeft.bar(bins[:-1], nOutput/nInput, width=np.diff(bins), align="edge", color="C0", alpha=0.5, zorder=10)
    bboxDict = dict(boxstyle="round", facecolor="white", alpha=0.75)
    info50 = "Magnitude at 50% recovered: {:0.2f}".format(mag50)
    axLeft.text(0.3, 0.15, info50, transform=fig.transFigure, bbox=bboxDict, zorder=11)

    # Add useful information to the plot
    fig = plt.gcf()
    addProvenanceInfo(fig, plotInfoDict)

    # Save the graph
    repoInfo.dataId["description"] = "completenessHist"
    repoInfo.butler.put(fig, "plotFakes", repoInfo.dataId)
    plt.close()
