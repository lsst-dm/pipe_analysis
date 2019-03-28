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
from matplotlib import colors, gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import UnivariateSpline

from lsst.qa.explorer.match import match_lists

__all__ = ["addDegreePositions", "matchCatalogs", "addNearestNeighbor", "sigmaMAD", "getPlotInfo",
           "areaDepth", "positionCompare"]


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

    catalog[raCol + "_deg"] = np.rad2deg(catalog[raCol].values)
    catalog[decCol + "_deg"] = np.rad2deg(catalog[decCol].values)

    return catalog


def matchCatalogs(catalog1, raCol1, decCol1, catalog2, raCol2, decCol2, matchRadius=0.1/3600.0):
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
    matchRadius : `float`
        Radius within which to match the nearest source, in degrees, defaults to 0.1 arcseconds (0.1/3600.0).

    Returns
    -------
    catalog1Matched : `pandas.core.frame.DataFrame`
        Catalog with only the rows that had a match in catalog2.
    catalog2Matched : `pandas.core.frame.DataFrame`
        Catalog with only the rows that matched catalog1.

    Notes
    -----
    matchRadius is in degrees, the catalogs need to be in degrees rather than radians. Returns two shortened
    catalogs, with the matched rows in the same order and objects without a match removed. Matches the first
    catalog to the second, multiple objects from the first catalog can match the same object in the second.
    """

    dists, inds = match_lists(catalog1[raCol1], catalog1[decCol1], catalog2[raCol2], catalog2[decCol2],
                              matchRadius)

    ids = (inds != len(catalog2))
    matchedInds = inds[ids]
    catalog1Matched = catalog1[ids]
    catalog2Matched = catalog2.iloc[matchedInds].copy()

    return catalog1Matched, catalog2Matched


def addNearestNeighbor(catalog, raCol, decCol):
    """Add the distance to the nearest neighbour in the catalog.

    Parameters
    ----------
    catalog : `pandas.core.frame.DataFrame`
        Catalog to add the distance to the nearest neighbour to.
    raCol : `string`
        Column name for the RA column to be used.
    decCol : `string`
        Column name for the Declination column to be used.

    Returns
    -------
    catalog : `pandas.core.frame.DataFrame`
        Catalog with a column 'nearestNeighbor' containing the distance to the neareast neighbour.

    Notes
    -----
    Distance added is in degrees.
    """

    dists, inds = match_lists(catalog[raCol], catalog[decCol], catalog[raCol], catalog[decCol],
                              20.0 / 3600, numNei=2)

    distNN = np.array([d[1] for d in dists])
    catalog["nearestNeighbor"] = distNN

    return catalog


def sigmaMAD(xs, med):
    """Calculate the sigma given by the median absolute deviation.

    Parameters
    ----------
    xs : `np.array`
        Values to calculate the median absolute deviation of.
    med : `float`
        Median of the values in xs.

    Returns
    -------
    sigmaMad : `float`
        The sigma derived from the median absolute deviation.
    """

    sigmaMad = 1.4826 * np.median(np.fabs(xs - med))

    return sigmaMad


def getPlotInfo(repoInfo):
    """Parse the repoInfo into a dict of useful info for plots.

    Parameters
    ----------
    repoInfo : `lsst.pipe.base.struct.Struct`

    Returns
    -------
    info : `dict`
    """

    camera = repoInfo.camera.getName()
    dataId = repoInfo.dataId
    filterName = dataId["filter"]
    visit = str(dataId["visit"])
    tract = str(dataId["tract"])
    dataset = repoInfo.dataset
    rerun = list(repoInfo.butler.storage.repositoryCfgs)[0]
    info = {"camera": camera, "filter": filterName, "visit": visit, "tract": tract, "dataset": dataset,
            "rerun": rerun}

    return info


def areaDepth(inputFakesMatched, processedFakesMatched, info, areaDict, repoInfo,
              measColType="base_PsfFlux_"):

    """Plot the area vs depth for the given catalog

    Parameters
    ----------
    inputFakesMatched : `pandas.core.frame.DataFrame`
        The catalog used to add the fakes originally matched to the processed catalog.
    processedFakesMatched : `pandas.core.frame.DataFrame`
        The catalog produced by the stack from the images with fakes in.
    info : `dict`
        A dict containing useful information to add to the plot.
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
    measColType needs to have an associated magnitude column already computed. Info should also contain
    outputName which is the output path and filename for the plot to be written to.
    """

    ccds = list(set(processedFakesMatched["ccdId"].values))

    mag10s = np.array([np.nan]*len(ccds))
    mag100s = np.array([np.nan]*len(ccds))
    areas = []

    for (i, ccd) in enumerate(ccds):
        onCcd = ((processedFakesMatched["ccdId"].values == ccd) &
                 (np.isfinite(processedFakesMatched[measColType + "mag"].values)) &
                 (processedFakesMatched["nearestNeighbor"].values > 2.0 / 3600.0) &
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

        ax = plt.gca()
        plt.text(0.96, 1.02, "Camera: " + info["camera"], fontsize=8, alpha=0.8, transform=ax.transAxes)
        plt.text(0.96, 1.05, "Filter: " + info["filter"], fontsize=8, alpha=0.8, transform=ax.transAxes)
        plt.text(0.96, 1.08, "Visit: " + info["visit"], fontsize=8, alpha=0.8, transform=ax.transAxes)
        plt.text(0.96, 1.11, "Tract: " + info["tract"], fontsize=8, alpha=0.8, transform=ax.transAxes)

        if "jointcal" in info["dataset"]:
            plt.text(-0.1, -0.1, "JointCal Used? Yes", fontsize=8, alpha=0.8, transform=ax.transAxes)
        else:
            plt.text(-0.1, -0.1, "JointCal Used? No", fontsize=8, alpha=0.8, transform=ax.transAxes)

        plt.text(-0.1, 1.12, "Rerun: " + info["rerun"], fontsize=8, alpha=0.8, transform=ax.transAxes)

        fig = plt.gcf()
        repoInfo.dataId["description"] = "areaDepth" + sigmas[i]
        repoInfo.butler.put(fig, "plotFakes", repoInfo.dataId)
        plt.close()

    medMag10 = np.median(mag10s)
    medMag100 = np.median(mag100s)

    return medMag10, medMag100


def positionCompare(inputFakesMatched, processedFakesMatched, info, repoInfo, raFakesCol="raJ2000_deg",
                    decFakesCol="decJ2000_deg", raCatCol="coord_ra_deg", decCatCol="coord_dec_deg",
                    magCol="base_PsfFlux_mag"):

    """Make a plot showing the RA and Dec offsets from the input positions.

    Parameters
    ----------
    inputFakesMatched : `pandas.core.frame.DataFrame`
        The catalog used to add the fakes originally matched to the processed catalog.
    processedFakesMatched : `pandas.core.frame.DataFrame`
        The catalog produced by the stack from the images with fakes in.
    info : `dict`
        A dict containing useful information to add to the plot.
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
    using only objects that were stars in the input catalog of fakes. `info` needs to be a dict containing
    camera, filter, visit, tract and dataset (jointcal or not). Returns the median and sigma MAD for the RA
    and Dec offsets. Info should also contain the magnitude limit that the plot should go to (magLim) and the
    path and filename that the figure should be written to (outputName).
    """

    pointsToUse = (processedFakesMatched[magCol].values < info["magLim"])
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
    dRASigmaMAD = sigmaMAD(dRA, dRAMed)
    dDecSigmaMAD = sigmaMAD(dDec, dDecMed)

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
    numInfo = "Num. Sources: %d \n Mag. Limit: %0.2f" % (len(dRA), info["magLim"])
    bbox = dict(facecolor="white", edgecolor="k", alpha=0.8)
    plt.text(-2.9, 0.7, numInfo, fontsize=8, bbox=bbox)

    plt.text(0.5, 1.04, "Camera: " + info["camera"], fontsize=8, alpha=0.8)
    plt.text(0.5, 1.16, "Filter: " + info["filter"], fontsize=8, alpha=0.8)
    plt.text(0.5, 1.28, "Visit: " + info["visit"], fontsize=8, alpha=0.8)
    plt.text(0.5, 1.40, "Tract: " + info["tract"], fontsize=8, alpha=0.8)

    plt.text(-3.55, 1.40, "rerun: " + info["rerun"], fontsize=8, alpha=0.8)

    if "jointcal" in info["dataset"]:
        plt.text(-3.55, -3.45, "JointCal Used? Yes", fontsize=8, alpha=0.8)
    else:
        plt.text(-3.55, -3.45, "JointCal Used? No", fontsize=8, alpha=0.8)

    plt.subplots_adjust(top=0.9, hspace=0.0, wspace=0.0, right=0.93, left=0.12, bottom=0.1)

    fig = plt.gcf()
    repoInfo.dataId["description"] = "positionCompare"
    repoInfo.butler.put(fig, "plotFakes", repoInfo.dataId)
    plt.close()

    return dRAMed, dRASigmaMAD, dDecMed, dDecSigmaMAD
