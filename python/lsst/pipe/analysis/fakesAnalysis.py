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
from scipy import stats

import lsst.pipe.base as pipeBase

from .utils import addMetricMeasurement

__all__ = ["addDegreePositions", "matchCatalogs", "addNearestNeighbor", "calcFakesAreaDepth",
           "plotFakesAreaDepth", "fakesPositionCompare", "fakesMagnitudeBlendedness",
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
        The catalog that catalog1 is matched to.
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
        default is coord.Angle(0.1, unit=u.arcsecond), 0.1 arseconds.
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
    the same object in the second. Uses astropy's match_coordinates_sky and their units framework. Adds a new
    column to the catalogs, matchDistance, that contains the distance to the matched source in degrees.
    """

    skyCoords1 = coord.SkyCoord(catalog1[raCol1], catalog1[decCol1], unit=units)
    skyCoords2 = coord.SkyCoord(catalog2[raCol2], catalog2[decCol2], unit=units)
    inds, dists, _ = coord.match_coordinates_sky(skyCoords1, skyCoords2)

    ids = (dists < matchRadius)

    matchedInds = inds[ids]
    matchDists = dists[ids]

    catalog1["matched"] = np.zeros(len(catalog1))
    catalog1["matched"].iloc[ids] = 1

    catalog1Matched = catalog1[ids].copy()
    catalog2Matched = catalog2.iloc[matchedInds].copy()

    catalog1Matched["matchDistance"] = matchDists.degree
    catalog2Matched["matchDistance"] = matchDists.degree

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


def addProvenanceInfo(fig, plotInfoDict):
    """Add some useful provenance information to the plot

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`
        The figure that the information should be added to
    plotInfoDict : `dict`
        A dictionary of information about the data being plotted.
            ``"cameraName"``
                The name of camera used to take the data
            ``"filter"``
                The filter used for this data
            ``"visit"``
                The visit of the data; only included if the data is from a single epoch dataset
            ``"tract"``
                The tract that the data comes from
            ``"photoCalibDataset"``
                The dataset used for the calibration, for example; jointcal
            ``"rerun"``
                The rerun the data is stored in
            ``"skyWcsDataset"``
                The sky Wcs dataset used
            ``"patch"``
                The patch that the data is from; only included if the data is from a coadd dataset
    """
    plt.text(0.85, 0.98, "Camera: " + plotInfoDict["cameraName"], fontsize=8, alpha=0.8,
             transform=fig.transFigure)
    plt.text(0.85, 0.96, "Filter: " + plotInfoDict["filter"], fontsize=8, alpha=0.8,
             transform=fig.transFigure)
    plt.text(0.85, 0.94, "Visit: " + plotInfoDict["visit"], fontsize=8, alpha=0.8, transform=fig.transFigure)
    plt.text(0.85, 0.92, "Tract: " + plotInfoDict["tract"], fontsize=8, alpha=0.8, transform=fig.transFigure)

    plt.text(0.02, 0.98, "rerun: " + plotInfoDict["rerun"], fontsize=8, alpha=0.8, transform=fig.transFigure)

    if "jointcal" in plotInfoDict["photoCalibDataset"]:
        plt.text(0.02, 0.02, "JointCal Used? Yes", fontsize=8, alpha=0.8, transform=fig.transFigure)
    else:
        plt.text(0.02, 0.02, "JointCal Used? No", fontsize=8, alpha=0.8, transform=fig.transFigure)

    return fig


def calcFakesAreaDepth(inputFakesMatched, processedFakesMatched, areaDict, measColType="base_PsfFlux_",
                       distNeighbor=2.0 / 3600.0, numSigmas=10):
    """Calculate the area vs depth for the given catalog
    Parameters
    ----------
    inputFakesMatched : `pandas.core.frame.DataFrame`
        The catalog used to add the fakes originally, matched to the processed catalog.
    processedFakesMatched : `pandas.core.frame.DataFrame`
        The catalog produced by the stack from the images with fakes in.
    areaDict : `dict`
        A dict containing the area of each ccd.
        Examples of keys, there is one of these for every ccd number specified when the code is called.
            ``"1"``
                The area of the the ccd not covered by the `bad` mask, in arcseconds
            ``"corners_1"``
                The corners of the ccd, `list` of `lsst.geom.SpherePoint`s, in degrees.
    measColType : `string`
        default : 'base_CircularApertureFlux_25_0_'
        Which type of flux/magnitude column to use for calculating the depth.
    distNeighbor : `float`
        The smallest distance to the nearest other source allowed to use the object for
        calculations.
    numSigmas : `float`
        default : '10'
        How many sigmas to calculate the median depth for.

    Returns
    -------
    depthsToPlot : `np.ndarray`
        The depth bins used to calculate the area to.
    cumAreas : `np.ndarray`
        The area to each depth
    magLimHalfArea : `float`
        The magnitude limit which half the data is deeper than.
    magLim25 : `float`
        The magnitude limit which 25% of the data is deeper than.
    magLim75 : `float`
        The magnitude limit which 75% of the data is deeper than.
    medMagsToLimit : `float`
        The median of the magnitudes at numSigmas for each ccd.

    Notes
    -----
    measColType needs to have an associated magnitude column already computed.
    """

    ccds = list(set(processedFakesMatched["ccdId"].values))

    magsToLimit = np.array([np.nan]*len(ccds))
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

            magsToLimit[i] = interpSpline(numSigmas)

    areas = np.array(areas)/(3600.0**2)
    bins = np.linspace(min(magsToLimit), max(magsToLimit), 101)
    areasOut = np.zeros(100)
    n = 0
    magLimHalfArea = None
    magLim25 = None
    magLim75 = None
    while n < len(bins)-1:
        ids = np.where((magsToLimit >= bins[n]) & (magsToLimit < bins[n + 1]))[0]
        areasOut[n] += np.sum(areas[ids])
        if np.sum(areasOut) > np.sum(areas)/2.0 and magLimHalfArea is None:
            magLimHalfArea = (bins[n] + bins[n + 1])/2.0
        if np.sum(areasOut) > np.sum(areas)*0.25 and magLim25 is None:
            magLim25 = (bins[n] + bins[n + 1])/2.0
        if np.sum(areasOut) > np.sum(areas)*0.75 and magLim75 is None:
            magLim75 = (bins[n] + bins[n + 1])/2.0
        n += 1

    cumAreas = np.cumsum(areasOut[::-1])

    medMagsToLimit = np.median(magsToLimit)
    depthsToPlot = bins[::-1][1:]

    return pipeBase.Struct(depthsToPlot=depthsToPlot, cumAreas=cumAreas, magLimHalfArea=magLimHalfArea,
                           magLim25=magLim25, magLim75=magLim75, medMagsToLimit=medMagsToLimit)


def plotFakesAreaDepth(inputFakesMatched, processedFakesMatched, plotInfoDict, areaDict,
                       measColType="base_PsfFlux_", distNeighbor=2.0/3600.0, numSigmas=10):
    """Plot the area vs depth for the given catalog

    Parameters
    ----------
    inputFakesMatched : `pandas.core.frame.DataFrame`
        The catalog used to add the fakes originally, matched to the processed catalog.
    processedFakesMatched : `pandas.core.frame.DataFrame`
        The catalog produced by the stack from the images with fakes in.
    plotInfoDict : `dict`
        A dictionary of information about the data being plotted.
            ``"camera"``
                The camera used to take the data
            ``"filter"``
                The filter used for this data
            ``"visit"``
                The visit of the data; only included if the data is from a single epoch dataset
            ``"tract"``
                The tract that the data comes from
            ``"photoCalibDataset"``
                The dataset used for the calibration, for example; jointcal
            ``"rerun"``
                The rerun the data is stored in
            ``"skyWcsDataset"``
                The sky Wcs dataset used
            ``"patch"``
                The patch that the data is from; only included if the data is from a coadd dataset
    areaDict : `dict`
        A dict containing the area of each ccd.
        Examples of keys, there is one of these for every ccd number specified when the code is called.
            ``"1"``
                The area of the the ccd not covered by the `bad` mask, in arcseconds
            ``"corners_1"``
                The corners of the ccd, `list` of `lsst.geom.SpherePoint`s, in degrees.
    measColType : `string`
        default : 'base_CircularApertureFlux_25_0_'
        Which type of flux/magnitude column to use for calculating the depth.
    distNeighbor : `float`
        The smallest distance to the nearest other source allowed to use the object for
        calculations.
    numSigmas : `float`
        default : '10'
        How many sigmas to calculate the median depth for.

    Yields
    -------
    `lsst.pipe.base.Struct`
        A struct containing the figure and its associated properties.
        ``fig``
            The figure to be saved (`matplotlib.figure.Figure`)
        ``stats``
            The statistics calculated for the plot (`dict`), the dict contains:
                MedDepth : `float`
                    The median of the magnitudes at flux/flux error of ``numSigmas`` from all the ccds.
        ``description``
            The name the plot is saved under (`str`), for this plot ``areaDepth<numGigmas>``
        ``style``
            The style of the plot being made (`str`), set to ``fakes``.

    Notes
    -----
    measColType needs to have an associated magnitude column already computed.
    """
    yield
    areaDepthStruct = calcFakesAreaDepth(inputFakesMatched, processedFakesMatched, areaDict,
                                         measColType="base_PsfFlux_", distNeighbor=2.0 / 3600.0,
                                         numSigmas=numSigmas)

    plt.plot(areaDepthStruct.depthsToPlot, areaDepthStruct.cumAreas)
    plt.xlabel("Magnitude Limit ({:0.0f} sigma)".format(numSigmas), fontsize=15)
    plt.ylabel("Area (deg^2)", fontsize=15)
    plt.title('Total Area to Given Depth \n (Recovered fake stars with no match within 2")')
    labelHA = "Mag. Lim. for 50% of the area: {:0.2f}".format(areaDepthStruct.magLimHalfArea)
    plt.axvline(areaDepthStruct.magLimHalfArea, label=labelHA, color="k", ls=":")
    label25 = "Mag. Lim. for 25% of the area: {:0.2f}".format(areaDepthStruct.magLim25)
    plt.axvline(areaDepthStruct.magLim25, label=label25, color="k", ls="--")
    label75 = "Mag. Lim. for 75% of the area: {:0.2f}".format(areaDepthStruct.magLim75)
    plt.axvline(areaDepthStruct.magLim75, label=label75, color="k", ls="--")
    plt.legend(loc="best")

    fig = plt.gcf()
    fig = addProvenanceInfo(fig, plotInfoDict)

    description = "areaDepth{:0.0f}".format(numSigmas)
    stats = {"MedDepth{:0.0f}".format(numSigmas): areaDepthStruct.medMagsToLimit}

    yield pipeBase.Struct(fig=fig, stats=stats, description=description, style="fakes")


def plotWithTwoHists(xs, ys, xName, yName, xLabel, yLabel, title, plotInfoDict, statsUnit, yLims=False,
                     xLims=False):
    """Makes a generic plot with a 2D histogram and collapsed histograms of each axis.

    Parameters
    ----------
    xs : `numpy.ndarray`
         The array to be plotted on the x axis.
    ys : `numpy.ndarray`
         The array to be plotted on the y axis.
    xName : `string`
         The name to be used in the text for the x axis statistics.
    yName : `string`
         The name to be used in the text for the y axis statistics.
    xLabel : `string`
         The text to go on the xLabel of the plot.
    yLabel : `string`
         The text to go on the yLabel of the plot.
    title : `string`
         The text to be displayed as the plot title.
    plotInfoDict : `dict`
        A dictionary of information about the data being plotted.
            ``"cameraName"``
                The name of the camera used to take the data
            ``"filter"``
                The filter used for this data
            ``"visit"``
                The visit of the data; only included if the data is from a single epoch dataset
            ``"tract"``
                The tract that the data comes from
            ``"photoCalibDataset"``
                The dataset used for the calibration, for example; jointcal
            ``"rerun"``
                The rerun the data is stored in
            ``"skyWcsDataset"``
                The sky Wcs dataset used
            ``"patch"``
                The patch that the data is from; only included if the data is from a coadd dataset
    statsUnit : `string`
        The text used to describe the units of the statistics calculated.
    yLims : `Bool` or `tuple`
        The y axis limits to use for the plot, default is False and they are calculated from the data.
        If being given a tuple of (yMin, yMax).
    xLims : `Bool` or `tuple`
        The x axis limits to use for the plot, default is False and they are calculated from the data.
        If being given a tuple of (xMin, xMax).

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The resulting figure.
    xsMed : `float`
        The median of the values plotted on the x axis.
    xsSigmaMAD : `float`
        The sigma MAD from the x axis values.
    ysMed : `float`
        The median of the values plotted on the y axis.
    ysSigmaMAD : `float`
        The sigma MAD from the y axis values.

    Notes
    -----
    The `plotInfoDict` needs to be a dict containing camera, filter, visit, tract and
    dataset (jointcal or not), it is used to add information to the plot. Returns the median and sigma MAD
    for the x and y values.
    """

    xsMed = np.median(xs)
    ysMed = np.median(ys)
    xsSigmaMAD = sigmaMAD(xs)
    ysSigmaMAD = sigmaMAD(ys)

    nBins = 20.0*np.log10(len(xs))
    if xLims:
        xMin = xLims[0]
        xMax = xLims[1]
    else:
        xMin = xsMed - 3.0*xsSigmaMAD
        xMax = xsMed + 3.0*xsSigmaMAD
    if yLims:
        yMin = yLims[0]
        yMax = yLims[1]
    else:
        yMin = ysMed - 3.0*ysSigmaMAD
        yMax = ysMed + 3.0*ysSigmaMAD
    xEdges = np.arange(xMin, xMax, (xMax - xMin)/nBins)
    yEdges = np.arange(yMin, yMax, (yMax - yMin)/nBins)
    bins = (xEdges, yEdges)

    gs = gridspec.GridSpec(4, 4)

    # Make a 2D histogram of the position offsets
    axHist = plt.subplot(gs[1:, :-1])
    axHist.axvline(0.0, ls=":", color="gray", alpha=0.5, zorder=0)
    axHist.axhline(0.0, ls=":", color="gray", alpha=0.5, zorder=0)
    _, _, _, im = axHist.hist2d(xs, ys, bins=bins, cmap="winter", norm=colors.LogNorm(), zorder=10)
    axHist.set_xlim(xMin, xMax)
    axHist.set_ylim(yMin, yMax)
    axHist.tick_params(top=True, right=True)

    # Make a 1D histogram of the offsets for the xs
    axXs = plt.subplot(gs[0, :-1], sharex=axHist)
    axXs.hist(xs, bins=xEdges, log=True)
    axXs.axes.get_xaxis().set_visible(False)
    axXs.set_ylabel("Number")

    # Make a 1D histogram of the offsets for the ys
    axYs = plt.subplot(gs[1:, -1], sharey=axHist)
    axYs.hist(ys, orientation="horizontal", bins=yEdges, log=True)
    axYs.axes.get_yaxis().set_visible(False)
    axYs.set_xlabel("Number")

    # Add a color bar for the 2D histogram
    divider = make_axes_locatable(axYs)
    cax = divider.append_axes("right", size="8%", pad=0.00)
    plt.colorbar(im, cax=cax, orientation="vertical", label="Points Per Bin")

    # Add some statistics to the axis in the top right corner
    axStats = plt.subplot(gs[0, -1])
    axStats.axes.get_xaxis().set_visible(False)
    axStats.axes.get_yaxis().set_visible(False)

    infoMedXs = r"Med. $\delta$%s %0.3f" % (xName, xsMed)
    infoMadXs = r"$\sigma_{MAD}$ $\delta$%s %0.3f" % (xName, xsSigmaMAD)
    infoMedYs = r"Med. $\delta$%s %0.3f" % (yName, ysMed)
    infoMadYs = r"$\sigma_{MAD}$ $\delta$%s %0.3f" % (yName, ysSigmaMAD)

    axStats.text(0.05, 0.8, infoMedXs, fontsize=10)
    axStats.text(0.05, 0.6, infoMadXs, fontsize=10)
    axStats.text(0.05, 0.4, infoMedYs, fontsize=10)
    axStats.text(0.05, 0.2, infoMadYs, fontsize=10)
    axStats.text(0.05, 0.05, statsUnit, fontsize=8)

    axHist.set_xlabel(xLabel)
    axHist.set_ylabel(yLabel)

    axXs.set_title(title)
    numInfo = "Num. Sources: %d \n Mag. Limit: %0.2f" % (len(xs), plotInfoDict["magLim"])
    bbox = dict(facecolor="white", edgecolor="k", alpha=0.8)
    plt.text(-2.9, 0.7, numInfo, fontsize=8, bbox=bbox)

    plt.subplots_adjust(top=0.9, hspace=0.0, wspace=0.0, right=0.90, left=0.12, bottom=0.1)

    fig = plt.gcf()
    fig = addProvenanceInfo(fig, plotInfoDict)

    return fig, xsMed, xsSigmaMAD, ysMed, ysSigmaMAD


def plotWithOneHist(xs, ys, maskForStats, xLabel, yLabel, title, plotInfoDict, binThresh=50.0, yLims=False,
                    xLims=False):
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
        A dictionary of information about the data being plotted.
            ``"cameraName"``
                The name of the camera used to take the data
            ``"filter"``
                The filter used for this data
            ``"visit"``
                The visit of the data; only included if the data is from a single epoch dataset
            ``"tract"``
                The tract that the data comes from
            ``"photoCalibDataset"``
                The dataset used for the calibration, for example; jointcal
            ``"rerun"``
                The rerun the data is stored in
            ``"skyWcsDataset"``
                The sky Wcs dataset used
            ``"patch"``
                The patch that the data is from; only included if the data is from a coadd dataset
    binThresh : `float`
        The number of points there needs to be (in a vertical bin) to trigger binning the data rather
        than plotting points.
    yLims : `Bool` or `tuple`
        The y axis limits to use for the plot, default is False and they are calculated from the data.
        If being given a tuple of (yMin, yMax).
    xLims : `Bool` or `tuple`
        The x axis limits to use for the plot, default is False and they are calculated from the data.
        If being given a tuple of (xMin, xMax).

    Returns
    -------
    fig : `matplotlib.figure.Figure`
    """

    medYs = np.median(ys[maskForStats])
    sigmaMadYs = sigmaMAD(ys[maskForStats])

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
    fiveSigmaHigh = medYs + 5.0*sigmaMadYs
    fiveSigmaLow = medYs - 5.0*sigmaMadYs
    binSize = (fiveSigmaHigh - fiveSigmaLow)/101.0
    yEdges = np.arange(fiveSigmaLow, fiveSigmaHigh, binSize)
    [xs1, xs25, xs50, xs75, xs95, xs97] = np.percentile(xs, [1, 25, 50, 75, 95, 97])
    xScale = (xs97 - xs1)/20.0  # This is ~5% of the data range
    # 40 was used as the number of bins because it looked good, might need to be changed in the future
    xEdges = np.arange(xs1 - xScale, xs95, (xs95 - (xs1 - xScale))/40.0)

    counts, xBins, yBins = np.histogram2d(xs, ys, bins=(xEdges, yEdges))
    countsYs = np.sum(counts, axis=1)

    ids = np.where((countsYs > binThresh))[0]
    xEdgesPlot = xEdges[ids][1:]
    xEdges = xEdges[ids]

    # Create the codes needed to turn the sigmaMad lines into a path to speed up checking which points are
    # inside the area
    codes = np.ones(len(xEdgesPlot)*2)*Path.LINETO
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY

    meds = np.zeros(len(xEdgesPlot))
    sigmaMadVerts = np.zeros((len(xEdgesPlot)*2, 2))
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
    halfSigmaMadLine, = ax.plot(xEdgesPlot, meds + 0.5*sigmaMads, "k", alpha=0.8,
                                label=r"$\frac{1}{2}$ $\sigma_{MAD}$")
    ax.plot(xEdgesPlot, meds - 0.5 * sigmaMads, "k", alpha=0.8)

    # Add lines for the median +/- sigmaMad
    statsSigmaMad = sigmaMAD(ys[maskForStats])
    sigmaMadLine, = ax.plot(xEdgesPlot, sigmaMadVerts[:len(xEdgesPlot), 1], "k", alpha=0.6,
                            label=r"$\sigma_{MAD}$: " + "{:0.2f}".format(statsSigmaMad))
    ax.plot(xEdgesPlot[::-1], sigmaMadVerts[len(xEdgesPlot):, 1], "k", alpha=0.6)

    # Add lines for the median +/- 2 * sigmaMad
    twoSigmaMadLine, = ax.plot(xEdgesPlot, meds + 2.0*sigmaMads, "k", alpha=0.4, label=r"2 $\sigma_{MAD}$")
    ax.plot(xEdgesPlot, meds - 2.0*sigmaMads, "k", alpha=0.4)

    # Check which points are outside 1 sigmaMad of the median and plot these as points, histogram the rest
    inside = sigmaMadPath.contains_points(np.array([xs, ys]).T)
    _, _, _, histIm = ax.hist2d(xs[inside], ys[inside], bins=(xEdgesPlot, yEdges), cmap=newBlues, zorder=-2,
                                cmin=1)
    notStatsPoints, = ax.plot(xs[~inside & ~maskForStats], ys[~inside & ~maskForStats], "x", ms=3, alpha=0.3,
                              mfc="C0", zorder=-1, label="Not used in Stats")
    statsPoints, = ax.plot(xs[maskForStats & ~inside], ys[maskForStats & ~inside], ".", ms=3, alpha=0.3,
                           mfc="C0", zorder=-1, mec="C0", label="Used in stats")

    # Divide the data into sections and plot historgrams for them
    sectionColors = ["mediumpurple", "blueviolet", "darkviolet", "purple"]
    sectionBounds = [xs25, xs50, xs75]
    sectionSelections = [np.where((xs < xs25))[0], np.where((xs < xs50) & (xs >= xs25))[0],
                         np.where((xs < xs75) & (xs >= xs50))[0], np.where((xs >= xs75))[0]]

    if np.median(meds[:10]) < 0:
        yArrowPos = medYs - 2.5*sigmaMadYs
    else:
        yArrowPos = medYs + 2.5*sigmaMadYs

    yScale = (6.0*sigmaMadYs)/40.0
    headLength = 11*xScale/20
    for (i, sectionColor) in enumerate(sectionColors):
        if i != len(sectionColors) - 1:
            quartileLine = ax.axvline(sectionBounds[i], color="k", ls="--", zorder=-10, label="Quartiles")
            ax.arrow(sectionBounds[i], yArrowPos, -1.0*xScale/4, 0, head_width=yScale,
                     head_length=headLength, color=sectionColor, fc=sectionColor)
        if i != 0:
            ax.arrow(sectionBounds[i - 1], yArrowPos, xScale/4, 0, head_width=yScale,
                     head_length=headLength, color=sectionColor, fc=sectionColor)

    # Add a 1d histogram showing the offsets in the ys axis
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
    if yLims:
        ax.set_ylim(yLims[0], yLims[1])
    else:
        if np.fabs(np.max(meds)) > medYs + 3*sigmaMadYs:
            ax.set_ylim(medYs - 4.0*sigmaMadYs, medYs + 4.0*sigmaMadYs)
        else:
            ax.set_ylim(medYs - 3.0*sigmaMadYs, medYs + 3*sigmaMadYs)
    if xLims:
        ax.set_xlim(xLims[0], xLims[1])
    else:
        ax.set_xlim(xs1 - xScale, xs97)

    # Add legends, needs to be split up as otherwise too large
    # Use the median value to pick which side they should go on
    ax.set_title(title, fontsize=12)
    sigmaLines = [medLine, sigmaMadLine, halfSigmaMadLine, twoSigmaMadLine]
    infoLines = [quartileLine]

    if np.median(meds[:10]) < 0:
        locQuartiles = "lower left"
        locSigmaLines = "upper left"
    else:
        locQuartiles = "upper left"
        locSigmaLines = "lower left"

    legendQuartiles = ax.legend(handles=infoLines, loc=locQuartiles, fontsize=10, framealpha=0.9,
                                borderpad=0.2)
    ax.add_artist(legendQuartiles)
    legendSigmaLines = ax.legend(handles=sigmaLines, loc=locSigmaLines, ncol=2, fontsize=10, framealpha=0.9,
                                 borderpad=0.2)
    plt.draw()
    plt.subplots_adjust(wspace=0.0)

    # Make the legends line up nicely
    if np.median(meds[:10]) < 0:
        lowerLeftLegend = legendQuartiles
    else:
        lowerLeftLegend = legendSigmaLines
    legendBBox = lowerLeftLegend.get_window_extent()
    yLegFigure = legendBBox.transformed(plt.gcf().transFigure.inverted()).ymin
    fig.legend(handles=[notStatsPoints, statsPoints], fontsize=8, borderaxespad=0, loc="lower left",
               bbox_to_anchor=(0.66, yLegFigure), bbox_transform=fig.transFigure, framealpha=0.9,
               markerscale=2)
    # Add the infomation about the data origins
    fig = plt.gcf()
    fig = addProvenanceInfo(fig, plotInfoDict)

    return fig, medDiff, statsSigmaMad


def fakesPositionCompare(inputFakesMatched, processedFakesMatched, plotInfoDict,
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
        A dictionary of information about the data being plotted.
            ``"cameraName"``
                The name of the camera used to take the data
            ``"filter"``
                The filter used for this data
            ``"visit"``
                The visit of the data; only included if the data is from a single epoch dataset
            ``"tract"``
                The tract that the data comes from
            ``"photoCalibDataset"``
                The dataset used for the calibration, for example; jointcal
            ``"rerun"``
                The rerun the data is stored in
            ``"skyWcsDataset"``
                The sky Wcs dataset used
            ``"patch"``
                The patch that the data is from; only included if the data is from a coadd dataset
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

    Yields
    -------
    `lsst.pipe.base.Struct`
        A struct containing the figure and its associated properties.
        ``fig``
            The figure to be saved (`matplotlib.figure.Figure`)
        ``stats``
           The statistics calculated for the plot (`dict`), the dict contains:
                dRAMed : `float`
                    The median RA difference.
                dRASigmaMAD : `float`
                    The sigma MAD from the RA difference.
                dDecMed : `float`
                    The median Dec. difference.
                dDecSigmaMAD : `float`
                    The sigma MAD from the Dec. difference.

        ``description``
            The name the plot is saved under (`str`), for this plot ``positionCompare``.
        ``style``
            The style of the plot being made (`str`), set to ``fakes``.

    Notes
    -----
    The two input catalogs need to be pre matched and in the same order so that the entry for an object is
    in the same row in each catalog. The delta RA and Dec is given in milli arcseconds. The plot is made
    using only objects that were stars in the input catalog of fakes. `plotInfoDict` needs to be a dict
    containing camera, filter, visit, tract and dataset (jointcal or not). Returns the median and sigma MAD
    for the RA and Dec offsets. plotInfoDict should also contain the magnitude limit that the plot should go
    to (magLim) and the path and filename that the figure should be written to (outputName).
    """
    yield
    pointsToUse = (processedFakesMatched[magCol].values < plotInfoDict["magLim"])

    processedFakesMatched = processedFakesMatched[pointsToUse]
    inputFakesMatched = inputFakesMatched[pointsToUse]

    stars = (inputFakesMatched["sourceType"] == "star")

    dRA = (inputFakesMatched[raFakesCol].values - processedFakesMatched[raCatCol].values)[stars]
    dRA *= (3600*1000*np.cos(np.deg2rad(inputFakesMatched[decFakesCol].values[stars])))
    dDec = (inputFakesMatched[decFakesCol].values - processedFakesMatched[decCatCol].values)[stars]
    dDec *= (3600*1000)

    xLabel = r"$\delta$RA (mas)"
    yLabel = r"$\delta$Dec (mas)"
    title = "Position Offsets for Input Fakes - Recovered Fakes"
    xName = "RA"
    yName = "Dec"
    statsUnit = "(mas)"

    fig, dRAMed, dRASigmaMAD, dDecMed, dDecSigmaMAD = plotWithTwoHists(dRA, dDec, xName, yName, xLabel,
                                                                       yLabel, title, plotInfoDict, statsUnit)

    description = "positionCompare"
    stats = {"dRAMed": dRAMed, "dRASigmaMAD": dRASigmaMAD, "dDecMed": dDecMed, "dDecSigmaMAD": dDecSigmaMAD}

    yield pipeBase.Struct(fig=fig, stats=stats, description=description, style="fakes")


def focalPlaneBinnedValues(ras, decs, zs, title, colorBarLabel, areaDict, plotInfoDict, statistic="median",
                           plotLims=False):
    """Make a plot of values across the focal plane

    Parameters
    ----------
    ras : `numpy.ndarray`
        The R.As of the points to plot, assumed to be in degrees.
    decs : `numpy.ndarray`
        The declinations of the points to plot, assumed to be in degrees.
    zs : `numpy.ndarray`
        The values to be plotted over the focal plane.
    title : `str`
        The text to be displayed as the plot title.
    colorBarLabel : `str`
        The text to be displayed as the color bar label.
    areaDict : `dict`
        A dict containing the area of each ccd.
        Examples of keys, there is one of these for every ccd number specified when the code is called.
            ``"1"``
                The area of the the ccd not covered by the `bad` mask, in arcseconds
            ``"corners_1"``
                The corners of the ccd, `list` of `lsst.geom.SpherePoint`s, in degrees.
    plotInfoDict : `dict`
        A dictionary of information about the data being plotted.
            ``"cameraName"``
                The name of the camera used to take the data
            ``"filter"``
                The filter used for this data
            ``"visit"``
                The visit of the data; only included if the data is from a single epoch dataset
            ``"tract"``
                The tract that the data comes from
            ``"photoCalibDataset"``
                The dataset used for the calibration, for example; jointcal
            ``"rerun"``
                The rerun the data is stored in
            ``"skyWcsDataset"``
                The sky Wcs dataset used
            ``"patch"``
                The patch that the data is from; only included if the data is from a coadd dataset
    statistic : `str` or `function`
        default : 'median'
        Either a string that can be used by scipy.stats.binned_statistic_2d to determine
        the function (see the docs for scipy.stats.binned_statistic_2d) or a user defined
        function that returns the value to be displayed in each bin.
    plotLims : `bool` or `tuple`
        default : 'False'
        If the plot should have specified plot limits rather than calculating them from the data
        this should be a tuple of (vmin, vmax).

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The resulting figure.
    """
    # Make a color map that looks nice
    r, g, b = colors.colorConverter.to_rgb("C0")
    r1, g1, b1 = colors.colorConverter.to_rgb("midnightblue")
    colorDict = {"blue": ((0.0, b, b), (1.0, b1, b1)), "red": ((0.0, r, r), (1.0, r1, r1)),
                 "green": ((0.0, g, g), (1.0, g1, g1))}
    colorDict["alpha"] = ((0.0, 0.2, 0.2), (0.05, 0.3, 0.3), (0.5, 0.8, 0.8), (1.0, 1.0, 1.0))
    newBlues = colors.LinearSegmentedColormap("newBlues", colorDict)

    fig, ax = plt.subplots()

    ccds = []
    for key in areaDict.keys():
        if type(key) is int:
            ccds.append(key)

    # Calculate a min and max value for the plot to be between
    if plotLims:
        (vmin, vmax) = plotLims
    else:
        zsAllMed = np.median(zs)
        zsAllMad = sigmaMAD(zs)
        vmin = zsAllMed - 2.0*zsAllMad
        vmax = zsAllMed + 2.0*zsAllMad

    raPlotMin = None
    raPlotMax = None
    decPlotMin = None
    decPlotMax = None

    # Run through the ccds, plot them and then histogram the data on them
    for ccd in ccds:
        # Find the ccd corners and sizes
        corners = areaDict["corners_" + str(ccd)]
        xy = (corners[0].getRa().asDegrees(), corners[0].getDec().asDegrees())
        width = corners[2].getRa().asDegrees() - corners[0].getRa().asDegrees()
        height = corners[2].getDec().asDegrees() - corners[0].getDec().asDegrees()

        # Some of the ccds are rotated and some have xy0 as being on the right with negative width/height
        # this upsets the binning so find the min and max to calculate positive bin widths from.
        minX = np.min([xy[0], corners[2].getRa().asDegrees()])
        maxX = np.max([xy[0], corners[2].getRa().asDegrees()])
        minY = np.min([xy[1], corners[2].getDec().asDegrees()])
        maxY = np.max([xy[1], corners[2].getDec().asDegrees()])

        if np.fabs(width) > np.fabs(height):
            binWidth = (maxX - minX)/10
        else:
            binWidth = (maxY - minY)/10

        # Find the max and the min of the ccds - this informs the plot limits
        if raPlotMax is None or maxX > raPlotMax:
            raPlotMax = maxX
        if raPlotMin is None or minX < raPlotMin:
            raPlotMin = minX
        if decPlotMax is None or maxY > decPlotMax:
            decPlotMax = maxY
        if decPlotMin is None or minY < decPlotMin:
            decPlotMin = minY

        xEdges = np.arange(minX, maxX + binWidth, binWidth)
        yEdges = np.arange(minY, maxY + binWidth, binWidth)

        # Plot the ccd outlines
        ccdPatch = patches.Rectangle(xy, width, height, fill=False, edgecolor="k", alpha=0.7)
        ax.add_patch(ccdPatch)

        # Check which points are on the ccd, the ccd patch is in different coordinates to the data, hence
        # the transform being required
        trans = ccdPatch.get_patch_transform()
        pointsOnCcd = ccdPatch.get_path().contains_points(list(zip(ras, decs)), transform=trans)

        # Calculate the statistic on the values in each bin then plot it
        zsVals, _, _, _ = stats.binned_statistic_2d(ras[pointsOnCcd], decs[pointsOnCcd], zs[pointsOnCcd],
                                                    statistic=statistic, bins=(xEdges, yEdges))
        X, Y = np.meshgrid(xEdges, yEdges)
        fracIm = plt.pcolormesh(X, Y, zsVals.T, vmin=vmin, vmax=vmax, cmap=newBlues)
        plt.draw()

    # Add a color bar
    colorBar = plt.gcf().colorbar(fracIm, ax=plt.gca())
    colorBar.set_label(colorBarLabel)

    plt.xlabel("R. A. (Degrees)")
    plt.ylabel("Dec. (Degrees)")
    plt.title(title)

    # Add useful information to the plot
    fig = plt.gcf()
    fig = addProvenanceInfo(fig, plotInfoDict)

    scaleRa = np.fabs((raPlotMax - raPlotMin)/20.0)
    scaleDec = np.fabs((decPlotMax - decPlotMin)/20.0)

    plt.xlim(raPlotMin - scaleRa, raPlotMax + scaleRa)
    plt.ylim(decPlotMin - scaleDec, decPlotMax + scaleDec)
    plt.subplots_adjust(right=0.99)

    return plt.gcf()


def fakesMagnitudeCompare(inputFakesMatched, processedFakesMatched, plotInfoDict, magCol="base_PsfFlux_mag",
                          verifyJob=None):
    """Make a plot showing the comparison between the input and extracted magnitudes.

    Parameters
    ----------
    inputFakesMatched : `pandas.core.frame.DataFrame`
        The catalog used to add the fakes originally, matched to the processed catalog.
    processedFakesMatched : `pandas.core.frame.DataFrame`
        The catalog produced by the stack from the images with fakes in.
    plotInfoDict : `dict`
        A dictionary of information about the data being plotted.
            ``"cameraName"``
                The name of the camera used to take the data
            ``"filter"``
                The filter used for this data
            ``"visit"``
                The visit of the data; only included if the data is from a single epoch dataset
            ``"tract"``
                The tract that the data comes from
            ``"photoCalibDataset"``
                The dataset used for the calibration, for example; jointcal
            ``"rerun"``
                The rerun the data is stored in
            ``"skyWcsDataset"``
                The sky Wcs dataset used
            ``"patch"``
                The patch that the data is from; only included if the data is from a coadd dataset
    magCol : `string`
        default : 'base_PsfFlux_mag'
        The magnitude column to use from the catalog.

    Yields
    -------
    `lsst.pipe.base.Struct`
        A struct containing the figure and its associated properties.
        ``fig``
            The figure to be saved (`matplotlib.figure.Figure`)
        ``stats``
           The statistics calculated for the plot (`dict`), the dict contains:
                dRAMed : `float`
                    The median RA difference.
                dRASigmaMAD : `float`
                    The sigma MAD from the RA difference.
                dDecMed : `float`
                    The median Dec. difference.
                dDecSigmaMAD : `float`
                    The sigma MAD from the Dec. difference.

        ``description``
            The name the plot is saved under (`str`), for this plot ``magnitudeCompare_<mag col used>``
        ``style``
            The style of the plot being made (`str`), set to ``fakes``.

    Notes
    -----
    The two input catalogs need to be pre matched and in the same order so that the entry for an object is
    in the same row in each catalog. The magnitude difference is given in milli mags. The plot is made
    using only objects that were stars in the input catalog of fakes. `plotInfoDict` needs to be a dict
    containing camera, filter, visit, tract and dataset (jointcal or not). plotInfoDict should also contain
    the magnitude limit that the plot should go to (magLim). Adds two metrics to the fakesAnalysis metrics
    file, fake_stars_magDiff and fake_stars_magDiffSigmaMad which can be used to track the evolution of the
    median difference (and the sigma MAD of the distribution) between the input and extracted magnitudes for
    fake stars brighter than the given magnitude limit.
    """
    yield
    band = plotInfoDict["filter"][-1].lower()

    stars = (inputFakesMatched["sourceType"] == "star")

    fakeMagStars = inputFakesMatched[band[-1].lower() + "magVar"].values[stars]
    catMagStars = processedFakesMatched[magCol].values[stars]
    finiteStars = np.where(np.isfinite(catMagStars))[0]
    ys = (catMagStars[finiteStars] - fakeMagStars[finiteStars])*1000
    xs = fakeMagStars[finiteStars]
    maskForStats = (xs < plotInfoDict["magLim"])
    xLabel = "Input Magnitude (mmag)"
    yLabel = "Output - Input Magnitude (mmag)"
    title = "Magnitude Difference For Fake Stars \n (" + magCol + ")"

    fig, med, sigmaMad = plotWithOneHist(xs, ys, maskForStats, xLabel, yLabel, title, plotInfoDict)
    if verifyJob:
        magName = magCol.replace("Flux", "")
        if "Aperture" in magCol:
            magName = magName.replace("Aperture", "Aper")
        if "Circular" in magCol:
            magName = magName.replace("Circular", "Circ")
        addMetricMeasurement(verifyJob, "pipe_analysis.fake_stars_magDiff_" + magName, med*u.mmag)
        addMetricMeasurement(verifyJob, "pipe_analysis.fake_stars_magDiffSigMad_" + magName, sigmaMad*u.mmag)
    # Don't have good mags for galaxies at this point.
    # To Do: coadd version of plot with cmodel mags.

    description = "magnitudeCompare_" + magCol
    stats = None

    yield pipeBase.Struct(fig=fig, stats=stats, description=description, style="fakes")


def fakesMagnitudeNearestNeighbor(inputFakesMatched, processedFakesMatched, plotInfoDict,
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
        A dictionary of information about the data being plotted.
            ``"cameraName"``
                The name of the camera used to take the data
            ``"filter"``
                The filter used for this data
            ``"visit"``
                The visit of the data; only included if the data is from a single epoch dataset
            ``"tract"``
                The tract that the data comes from
            ``"photoCalibDataset"``
                The dataset used for the calibration, for example; jointcal
            ``"rerun"``
                The rerun the data is stored in
            ``"skyWcsDataset"``
                The sky Wcs dataset used
            ``"patch"``
                The patch that the data is from; only included if the data is from a coadd dataset
    magCol : `string`
        default : 'base_PsfFlux_mag'
        The magnitude column to use from the catalog.

    Yields
    ------
    `lsst.pipe.base.Struct`
        A struct containing the figure and its associated properties.
        ``fig``
            The figure to be saved (`matplotlib.figure.Figure`)
        ``stats``
           The statistics calculated for the plot, here no stats are calculated so it is None
        ``description``
            The name the plot is saved under (`str`), for this plot ``magnitudeNearestNeighbour``
        ``style``
            The style of the plot being made (`str`), set to ``fakes``.

    Notes
    -----
    The two input catalogs need to be pre matched and in the same order so that the entry for an object is
    in the same row in each catalog. The magnitude difference is given in milli mags. The plot is made
    using only objects that were stars in the input catalog of fakes. `plotInfoDict` needs to be a dict
    containing camera, filter, visit, tract and dataset (jointcal or not). plotInfoDict should also contain
    the magnitude limit that the plot should go to (magLim).
    """
    yield
    band = plotInfoDict["filter"][-1].lower()

    stars = (inputFakesMatched["sourceType"] == "star")

    fakeMagStars = inputFakesMatched[band[-1].lower() + "magVar"].values[stars]
    catMagStars = processedFakesMatched[magCol].values[stars]
    nearestNeighborDistance = processedFakesMatched["nearestNeighbor"].values[stars]*3600.0
    finiteValues = np.where((np.isfinite(catMagStars)) & (np.isfinite(nearestNeighborDistance)))[0]
    ys = (catMagStars[finiteValues] - fakeMagStars[finiteValues])*1000
    xs = nearestNeighborDistance[finiteValues]
    mags = fakeMagStars[finiteValues]
    maskForStats = (mags < plotInfoDict["magLim"])

    # Don't have good mags for galaxies at this point.
    # To Do: coadd version of plot with cmodel mags.

    xLabel = "Distance to Nearest Neighbor (arcsec)"
    yLabel = "Output - Input Magnitude (mmag)"
    title = "Magnitude Difference For Fake Stars Against \nDistance to Nearest Neighbor (" + magCol + ")"

    fig, _, _ = plotWithOneHist(xs, ys, maskForStats, xLabel, yLabel, title, plotInfoDict)

    description = "magnitudeNearestNeighbor"
    stats = None

    yield pipeBase.Struct(fig=fig, stats=stats, description=description, style="fakes")


def fakesMagnitudeBlendedness(inputFakesMatched, processedFakesMatched, plotInfoDict,
                              magCol="base_PsfFlux_mag"):
    """Make a plot showing the comparison between the input and extracted magnitudes against blendedness.

    Parameters
    ----------
    inputFakesMatched : `pandas.core.frame.DataFrame`
        The catalog used to add the fakes originally, matched to the processed catalog.
    processedFakesMatched : `pandas.core.frame.DataFrame`
        The catalog produced by the stack from the images with fakes in.
    plotInfoDict : `dict`
        A dictionary of information about the data being plotted.
            ``"cameraName"``
                The name of the camera used to take the data
            ``"filter"``
                The filter used for this data
            ``"visit"``
                The visit of the data; only included if the data is from a single epoch dataset
            ``"tract"``
                The tract that the data comes from
            ``"photoCalibDataset"``
                The dataset used for the calibration, for example; jointcal
            ``"rerun"``
                The rerun the data is stored in
            ``"skyWcsDataset"``
                The sky Wcs dataset used
            ``"patch"``
                The patch that the data is from; only included if the data is from a coadd dataset
    magCol : `string`
        default : 'base_PsfFlux_mag'
        The magnitude column to use from the catalog.

    Yields
    ------
    `lsst.pipe.base.Struct`
        A struct containing the figure and its associated properties.
        ``fig``
            The figure to be saved (`matplotlib.figure.Figure`)
        ``stats``
           The statistics calculated for the plot, here no stats are calculated so it is None
        ``description``
            The name the plot is saved under (`str`), for this plot ``magnitudeBlendedness``
        ``style``
            The style of the plot being made (`str`), set to ``fakes``.


    Notes
    -----
    The two input catalogs need to be pre matched and in the same order so that the entry for an object is
    in the same row in each catalog. The magnitude difference is given in milli mags. The plot is made
    using only objects that were stars in the input catalog of fakes. `plotInfoDict` needs to be a dict
    containing camera, filter, visit, tract and dataset (jointcal or not). plotInfoDict should also contain
    the magnitude limit that the plot should go to (magLim).
    """
    yield
    band = plotInfoDict["filter"][-1].lower()

    stars = (inputFakesMatched["sourceType"] == "star")

    fakeMagStars = inputFakesMatched[band[-1].lower() + "magVar"].values[stars]
    catMagStars = processedFakesMatched[magCol].values[stars]
    blendedness = np.log10(processedFakesMatched["base_Blendedness_abs"].values[stars])
    finiteValues = np.where((np.isfinite(catMagStars)) & (np.isfinite(blendedness)))[0]
    ys = (catMagStars[finiteValues] - fakeMagStars[finiteValues])*1000
    xs = blendedness[finiteValues]
    mags = fakeMagStars[finiteValues]
    maskForStats = (mags < plotInfoDict["magLim"])

    xLabel = "log10(Blendedness)"
    yLabel = "Output - Input Magnitude (mmag)"
    title = "Magnitude Difference For Fake Stars \nAgainst Blendedness"

    fig, _, _ = plotWithOneHist(xs, ys, maskForStats, xLabel, yLabel, title, plotInfoDict)

    description = "magnitudeBlendedness"
    stats = None

    yield pipeBase.Struct(fig=fig, stats=stats, description=description, style="fakes")


def fakesCompletenessPlot(inputFakes, inputFakesMatched, processedFakesMatched, plotInfoDict, areaDict,
                          raFakesCol="raJ2000_deg", decFakesCol="decJ2000_deg",
                          raCatCol="coord_ra_deg", decCatCol="coord_dec_deg", distNeighbor=2.0/3600.0):
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
        A dictionary of information about the data being plotted.
            ``"cameraName"``
                The name of the camera used to take the data
            ``"filter"``
                The filter used for this data
            ``"visit"``
                The visit of the data; only included if the data is from a single epoch dataset
            ``"tract"``
                The tract that the data comes from
            ``"photoCalibDataset"``
                The dataset used for the calibration, for example; jointcal
            ``"rerun"``
                The rerun the data is stored in
            ``"skyWcsDataset"``
                The sky Wcs dataset used
            ``"patch"``
                The patch that the data is from; only included if the data is from a coadd dataset
    areaDict : `dict`
        A dict containing the area of each ccd.
        Examples of keys, there is one of these for every ccd number specified when the code is called.
            ``"1"``
                The area of the the ccd not covered by the `bad` mask, in arcseconds
            ``"corners_1"``
                The corners of the ccd, `list` of `lsst.geom.SpherePoint`s, in degrees.
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

    Yields
    ------
    `lsst.pipe.base.Struct`
        A struct containing the figure and its associated properties.
        ``fig``
            The figure to be saved (`matplotlib.figure.Figure`)
        ``stats``
           The statistics calculated for the plot, here no stats are calculated so it is None
        ``description``
            The name the plot is saved under (`str`), for this plot ``completenessHist2D``
        ``style``
            The style of the plot being made (`str`), set to ``fakes``.

    `lsst.pipe.base.Struct`
        A struct containing the figure and its associated properties.
        ``fig``
            The figure to be saved (`matplotlib.figure.Figure`)
        ``stats``
           The statistics calculated for the plot, here no stats are calculated so it is None
        ``description``
            The name the plot is saved under (`str`), for this plot ``completenessAreaDepth``
        ``style``
            The style of the plot being made (`str`), set to ``fakes``.

    `lsst.pipe.base.Struct`
        A struct containing the figure and its associated properties.
        ``fig``
            The figure to be saved (`matplotlib.figure.Figure`)
        ``stats``
           The statistics calculated for the plot, here no stats are calculated so it is None
        ``description``
            The name the plot is saved under (`str`), for this plot ``completenessHist``
        ``style``
            The style of the plot being made (`str`), set to ``fakes``.

    Notes
    -----
    Makes 3 plots to study the completeness of the data.
    The first is a 2D histogram of the fraction of fakes recovered.
    The second is a cumulative plot of the area which has fainter sources recovered than a given magnitude.
    The third is a histogram showing the fraction recovered in each magnitude bin with the number input and
    recovered overplotted.
    """
    yield

    band = plotInfoDict["filter"][-1].lower()

    # Find the fake stars
    stars = np.where((inputFakes["sourceType"] == "star"))[0]

    def histFunction(zs):
        zsOut = 1.0 - (np.sum(zs)/len(zs)) if any(zs) else np.nan
        return zsOut

    zs = inputFakes["matched"].values[stars]
    ras = inputFakes[raFakesCol].values[stars]
    decs = inputFakes[decFakesCol].values[stars]
    title = "Fraction of Sources Missed \n (Fake Stars)"
    colorBarLabel = "Fraction Missed"

    fig = focalPlaneBinnedValues(ras, decs, zs, title, colorBarLabel, areaDict, plotInfoDict,
                                 statistic=histFunction, plotLims=(0, 1))
    description = "completenessHist2D"
    stats = None
    yield pipeBase.Struct(fig=fig, stats=stats, description=description, style="fakes")
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
        areas[i] = areaDict[ccd]/(3600.0**2)

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
        if np.sum(areasOut) > np.sum(areas)/2.0 and magLimHalfArea is None:
            magLimHalfArea = (bins[n] + bins[n + 1])/2.0
        if np.sum(areasOut) > np.sum(areas)*0.25 and magLim25 is None:
            magLim25 = (bins[n] + bins[n + 1])/2.0
        if np.sum(areasOut) > np.sum(areas)*0.75 and magLim75 is None:
            magLim75 = (bins[n] + bins[n + 1])/2.0
        n += 1

    cumAreas = np.zeros(100)
    n = 0
    for area in areasOut[::-1]:
        cumAreas[n] = area + cumAreas[n-1]
        n += 1

    plt.plot(bins[::-1][1:], cumAreas)
    plt.xlabel("Faintest Source Recovered (mag)", fontsize=15)
    plt.ylabel("Area (deg^2)", fontsize=15)
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

    description = "completenessAreaDepth"
    stats = None
    yield pipeBase.Struct(fig=fig, stats=stats, description=description, style="fakes")
    plt.close()

    # Make plot showing the fraction recovered in magnitude bins
    fig, axLeft = plt.subplots()
    axRight = axLeft.twinx()
    axLeft.tick_params(axis="y", labelcolor="C0")
    axLeft.set_ylabel("Fraction Recovered", color="C0")
    axLeft.set_xlabel("Input Magnitude (mag)")
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

    description = "completenessHist"
    stats = None
    yield pipeBase.Struct(fig=fig, stats=stats, description=description, style="fakes")


def fakesMagnitudePositionError(inputFakesMatched, processedFakesMatched, plotInfoDict, areaDict,
                                raFakesCol="raJ2000_deg", decFakesCol="decJ2000_deg", raCatCol="coord_ra_deg",
                                decCatCol="coord_dec_deg", magCol="base_PsfFlux_mag"):
    """Make plots showing the comparison between the input minus the extracted magnitudes and the
       difference in the input and extracted positions, one showing the position difference locations
       in R.A. and Dec., one showing position differences against output - input magnitude difference
       and one showing position errors against input magnitude.

    Parameters
    ----------
    inputFakesMatched : `pandas.core.frame.DataFrame`
        The catalog used to add the fakes originally matched to the processed catalog.
    processedFakesMatched : `pandas.core.frame.DataFrame`
        The catalog produced by the stack from the images with fakes in.
    plotInfoDict : `dict`
        A dictionary of information about the data being plotted.
            ``"cameraName"``
                The name of the camera used to take the data
            ``"filter"``
                The filter used for this data
            ``"visit"``
                The visit of the data; only included if the data is from a single epoch dataset
            ``"tract"``
                The tract that the data comes from
            ``"photoCalibDataset"``
                The dataset used for the calibration, for example; jointcal
            ``"rerun"``
                The rerun the data is stored in
            ``"skyWcsDataset"``
                The sky Wcs dataset used
            ``"patch"``
                The patch that the data is from; only included if the data is from a coadd dataset
    areaDict : `dict`
        A dict containing the area of each ccd.
        Examples of keys, there is one of these for every ccd number specified when the code is called.
            ``"1"``
                The area of the the ccd not covered by the `bad` mask, in arcseconds
            ``"corners_1"``
                The corners of the ccd, `list` of `lsst.geom.SpherePoint`s, in degrees.
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
        default : 'base_PsfFlux_mag'
        The magnitude column to use from the catalog.

    Yields
    ------
    `lsst.pipe.base.Struct`
        A struct containing the figure and its associated properties.
        ``fig``
            The figure to be saved (`matplotlib.figure.Figure`)
        ``stats``
           The statistics calculated for the plot, here no stats are calculated so it is None
        ``description``
            The name the plot is saved under (`str`), for this plot ``magnitudePosErrs``
        ``style``
            The style of the plot being made (`str`), set to ``fakes``.

   `lsst.pipe.base.Struct`
        A struct containing the figure and its associated properties.
        ``fig``
            The figure to be saved (`matplotlib.figure.Figure`)
        ``stats``
           The statistics calculated for the plot, here no stats are calculated so it is None
        ``description``
            The name the plot is saved under (`str`), for this plot ``magDiffPosErrs``
        ``style``
            The style of the plot being made (`str`), set to ``fakes``.

   `lsst.pipe.base.Struct`
        A struct containing the figure and its associated properties.
        ``fig``
            The figure to be saved (`matplotlib.figure.Figure`)
        ``stats``
           The statistics calculated for the plot, here no stats are calculated so it is None
        ``description``
            The name the plot is saved under (`str`), for this plot ``PosErrsCcd``
        ``style``
            The style of the plot being made (`str`), set to ``fakes``.

    Notes
    -----
    The two input catalogs need to be pre matched and in the same order so that the entry for an object is
    in the same row in each catalog. The magnitude difference is given in milli mags. The plot is made
    using only objects that were stars in the input catalog of fakes. `plotInfoDict` needs to be a dict
    containing camera, filter, visit, tract and dataset (jointcal or not). plotInfoDict should also contain
    the magnitude limit that the plot should go to (magLim).
    """
    yield
    band = plotInfoDict["filter"][-1].lower()

    stars = (inputFakesMatched["sourceType"] == "star")

    fakeMagStars = inputFakesMatched[band[-1].lower() + "magVar"].values[stars]
    catMagStars = processedFakesMatched[magCol].values[stars]

    dRA = (inputFakesMatched[raFakesCol].values - processedFakesMatched[raCatCol].values)[stars]
    dRA *= (3600*1000*np.cos(np.deg2rad(inputFakesMatched[decFakesCol].values[stars])))
    dDec = (inputFakesMatched[decFakesCol].values - processedFakesMatched[decCatCol].values)[stars]
    dDec *= (3600*1000)

    positionErrors = np.log10(np.sqrt(dDec**2.0 + dRA**2.0))
    finiteValues = np.where((np.isfinite(fakeMagStars)) & (np.isfinite(positionErrors)) &
                            (np.isfinite(catMagStars)))[0]
    xs = fakeMagStars[finiteValues]
    ys = positionErrors[finiteValues]
    maskForStats = (xs < plotInfoDict["magLim"])
    yLabel = r"log10($\sqrt{\delta RA^2 + \delta Dec^2}$ (mas))"
    xLabel = "Input Magnitude (mag)"
    title = "Position Differences Against Magnitude for Fake Stars\n(" + magCol + ")"

    fig, _, _ = plotWithOneHist(xs, ys, maskForStats, xLabel, yLabel, title, plotInfoDict)

    description = "magnitudePosErrs"
    stats = None
    yield pipeBase.Struct(fig=fig, stats=stats, description=description, style="fakes")

    # Add magnitude differences against position errors
    xs = positionErrors[finiteValues]
    ys = (catMagStars[finiteValues] - fakeMagStars[finiteValues])*1000
    title = "Position Differences Against Output - Input Magnitudes\n for Fake Stars (" + magCol + ")"
    xLabel = r"log10($\sqrt{\delta RA^2 + \delta Dec^2}$ (mas))"
    yLabel = "Output - Input Magnitude (mmag)"
    fig, _, _ = plotWithOneHist(xs, ys, maskForStats, xLabel, yLabel, title, plotInfoDict)

    description = "magDiffsPosErrs"
    stats = None
    yield pipeBase.Struct(fig=fig, stats=stats, description=description, style="fakes")

    # Add binned position errors across focal plane
    ras = inputFakesMatched[raFakesCol].values[stars]
    decs = inputFakesMatched[decFakesCol].values[stars]
    zs = np.log10(np.sqrt(dDec**2.0 + dRA**2.0))
    title = "Average Position Differences Across the CCDs"
    colorBarLabel = r"log10($\sqrt{\delta RA^2 + \delta Dec^2}$ (mas))"
    fig = focalPlaneBinnedValues(ras, decs, zs, title, colorBarLabel, areaDict, plotInfoDict,
                                 statistic="median")

    description = "PosErrsCcd"
    stats = None
    yield pipeBase.Struct(fig=fig, stats=stats, description=description, style="fakes")
