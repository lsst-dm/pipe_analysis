from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import numpy as np

import lsst.afw.cameraGeom as cameraGeom
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
from lsst.pipe.base import Struct

from .utils import checkHscStack, findCcdKey

try:
    from lsst.meas.mosaic.updateExposure import applyMosaicResultsExposure
except ImportError:
    applyMosaicResultsExposure = None

__all__ = ["AllLabeller", "StarGalaxyLabeller", "OverlapsStarGalaxyLabeller", "MatchesStarGalaxyLabeller",
           "CosmosLabeller", "plotText", "annotateAxes", "labelVisit", "labelCamera",
           "filterStrFromFilename", "plotCameraOutline", "plotTractOutline", "plotPatchOutline",
           "plotCcdOutline", "rotatePixelCoords", "bboxToXyCoordLists", "getRaDecMinMaxPatchList",
           "percent", "setPtSize", "getQuiver", "makeAlphaCmap", "buildTractImage"]


class AllLabeller(object):
    labels = {"all": 0}
    plot = ["all"]

    def __call__(self, catalog):
        return np.zeros(len(catalog))


class StarGalaxyLabeller(object):
    labels = {"star": 0, "galaxy": 1}
    plot = ["star", "galaxy"]
    _column = "base_ClassificationExtendedness_value"

    def __call__(self, catalog):
        return np.where(catalog[self._column] < 0.5, 0, 1)


class OverlapsStarGalaxyLabeller(StarGalaxyLabeller):
    labels = {"star": 0, "galaxy": 1, "split": 2}
    plot = ["star", "galaxy", "split"]

    def __init__(self, first="first_", second="second_"):
        self._first = first
        self._second = second

    def __call__(self, catalog):
        first = np.where(catalog[self._first + self._column] < 0.5, 0, 1)
        second = np.where(catalog[self._second + self._column] < 0.5, 0, 1)
        return np.where(first == second, first, 2)


class MatchesStarGalaxyLabeller(StarGalaxyLabeller):
    _column = "src_base_ClassificationExtendedness_value"


class CosmosLabeller(StarGalaxyLabeller):
    """Do star/galaxy classification using Alexie Leauthaud's Cosmos catalog"""
    def __init__(self, filename, radius):
        original = afwTable.BaseCatalog.readFits(filename)
        good = (original["CLEAN"] == 1) & (original["MU.CLASS"] == 2)
        num = good.sum()
        cosmos = afwTable.SimpleCatalog(afwTable.SimpleTable.makeMinimalSchema())
        cosmos.reserve(num)
        for ii in range(num):
            cosmos.addNew()
        cosmos["id"][:] = original["NUMBER"][good]
        cosmos["coord_ra"][:] = original["ALPHA.J2000"][good]*(1.0*afwGeom.degrees).asRadians()
        cosmos["coord_dec"][:] = original["DELTA.J2000"][good]*(1.0*afwGeom.degrees).asRadians()
        self.cosmos = cosmos
        self.radius = radius

    def __call__(self, catalog):
        # A kdTree would be better, but this is all we have right now
        matches = afwTable.matchRaDec(self.cosmos, catalog, self.radius)
        good = set(mm.second.getId() for mm in matches)
        return np.array([0 if ii in good else 1 for ii in catalog["id"]])


def plotText(textStr, plt, axis, xLoc, yLoc, prefix="", fontSize=9, color="k", coordSys="axes", **kwargs):
    """Label the plot with the string provided at a given location

    Parameters
    ----------
    textStr : `str`
       String of text to plot.
    plt : `matplotlib.pyplot`
       Instance of matplotlib pyplot to plot ``textStr``.
    axis : `matplotlib.axes._axes.Axes`
       Particular matplotlib axes of ``plt`` on which to plot ``testStr``.
    xLoc, yLoc : `float`
       x and y coordinates, in corrdinate system set by ``coordSys``, at which to plot the ``textStr``.
       The string will be centered both horizontally and vertically at this position.
    prefix : `str`, optional
       Optional prefix to add to ``textStr``.
    fontSize : `int` or `str`, optional
       Size of font for plotting of ``textStr``.  May be either an absolute font size in points, or a
       size string, relative to the default font size.  Default is 9 points.
    color : `str`, optional
       Color to plot ``textStr``.  Can be any matplotlib color str.  Default is k (for black).
    coordSys : `str`, optional
       Coordinate system for ``xLoc``, ``yLoc``.  Choices and matplotlib mappings are:
       axes => axis.transAxes [the default]
       data => axis.transData
       figure => axis.transFigure
    **kwargs
       Arbitrary keyword arguments.  These can include any of those accecpted
       by matplotlib's matplotlib.pyplot.text function (i.e. are properties of
       the matplotlib.text.Text class).  Of particular interest here include:

       - ``rotation`` : Angle in degrees to rotate ``textStr`` for plotting
                        or one of strings "vertical" or "horizontal".  The
                        matplotlib default is 0 degrees (`int` or `str`).
       - ``alpha`` : The matplotlib blending value, between 0 (transparent)
                     and 1 (opaque).  The matplotlib default is 1 (`float`).

    Raises
    ------
    `ValueError`
       If unrecognized ``coordSys`` is requested (i.e. something other than axes, data, or figure)
    """
    if coordSys == "axes":
        transform = axis.transAxes
    elif coordSys == "data":
        transform = axis.transData
    elif coordSys == "figure":
        transform = axis.transFigure
    else:
        raise ValueError("Unrecognized coordSys: {}.  Must be one of axes, data, figure".format(coordSys))
    fontSize = int(fontSize - min(3, len(textStr)/10))
    plt.text(xLoc, yLoc, prefix + textStr, ha="center", va="center", fontsize=fontSize, transform=transform,
             color=color, **kwargs)


def annotateAxes(filename, plt, axes, stats, dataSet, magThreshold, x0=0.03, y0=0.96, yOff=0.05,
                 fontSize=8, ha="left", va="top", color="blue", isHist=False, hscRun=None, matchRadius=None,
                 writeMinMax=None, unitScale=1.0):
    xOffFact = 0.67*len(" N = {0.num:d} (of {0.total:d})".format(stats[dataSet]))
    axes.annotate(dataSet+r" N = {0.num:d} (of {0.total:d})".format(stats[dataSet]),
                  xy=(x0, y0), xycoords="axes fraction", ha=ha, va=va, fontsize=fontSize, color=color)
    axes.annotate(r" [mag<{0:.1f}]".format(magThreshold), xy=(x0*xOffFact, y0), xycoords="axes fraction",
                  ha=ha, va=va, fontsize=fontSize, color="k", alpha=0.55)
    meanStr = "{0.mean:.4f}".format(stats[dataSet])
    stdevStr = "{0.stdev:.4f}".format(stats[dataSet])
    statsUnitStr = None
    if unitScale == 1000.0:
        meanStr = "{0.mean:.2f}".format(stats[dataSet])
        stdevStr = "{0.stdev:.2f}".format(stats[dataSet])
        statsUnitStr = " (milli)"
        if any(ss in filename for ss in ["_ra", "_dec", "distance"]):
            statsUnitStr = " (mas)"
        if any(ss in filename for ss in ["Flux", "_photometry", "matches_mag"]):
            statsUnitStr = " (mmag)"
    lenStr = 0.12 + 0.017*(max(len(meanStr), len(stdevStr)))

    axes.annotate("mean = ", xy=(x0 + 0.12, y0 - yOff),
                  xycoords="axes fraction", ha="right", va=va, fontsize=fontSize, color="k")
    axes.annotate(meanStr, xy=(x0 + lenStr, y0 - yOff),
                  xycoords="axes fraction", ha="right", va=va, fontsize=fontSize, color="k")
    if statsUnitStr is not None:
        axes.annotate(statsUnitStr, xy=(x0 + lenStr + 0.006, y0 - yOff),
                      xycoords="axes fraction", ha="left", va=va, fontsize=fontSize, color="k")
    axes.annotate("stdev = ", xy=(x0 + 0.12, y0 - 2*yOff),
                  xycoords="axes fraction", ha="right", va=va, fontsize=fontSize, color="k")
    axes.annotate(stdevStr, xy=(x0 + lenStr, y0 - 2*yOff),
                  xycoords="axes fraction", ha="right", va=va, fontsize=fontSize, color="k")
    yOffMult = 3
    if writeMinMax is not None:
        axes.annotate("Min, Max (all stars) = ({0:.2f}, {1:.2f})\"".format(), xy=(x0, y0 - yOffMult*yOff),
                      xycoords="axes fraction", ha=ha, va=va, fontsize=fontSize)
        yOffMult += 1
    if matchRadius is not None:
        axes.annotate("Match radius = {0:.2f}\"".format(matchRadius), xy=(x0, y0 - yOffMult*yOff),
                      xycoords="axes fraction", ha=ha, va=va, fontsize=fontSize)
        yOffMult += 1
    if hscRun is not None:
        axes.annotate("HSC stack run: {0:s}".format(hscRun), xy=(x0, y0 - yOffMult*yOff),
                      xycoords="axes fraction", ha=ha, va=va, fontsize=fontSize, color="#800080")
    if isHist:
        l1 = axes.axvline(stats[dataSet].median, linestyle="dotted", color="0.7")
        l2 = axes.axvline(stats[dataSet].median+stats[dataSet].clip, linestyle="dashdot", color="0.7")
        axes.axvline(stats[dataSet].median-stats[dataSet].clip, linestyle="dashdot", color="0.7")
    else:
        l1 = axes.axhline(stats[dataSet].median, linestyle="dotted", color="0.7", label="median")
        l2 = axes.axhline(stats[dataSet].median+stats[dataSet].clip, linestyle="dashdot", color="0.7",
                          label="clip")
        axes.axhline(stats[dataSet].median-stats[dataSet].clip, linestyle="dashdot", color="0.7")
    return l1, l2


def labelVisit(filename, plt, axis, xLoc, yLoc, color="k", fontSize=9):
    labelStr = None
    if filename.find("tract-") >= 0:
        labelStr = "tract: "
        i1 = filename.find("tract-") + len("tract-")
        i2 = filename.find("/", i1)
        labelStr += filename[i1:i2]
    if filename.find("visit-") >= 0:
        labelStr += " visit: "
        i1 = filename.find("visit-") + len("visit-")
        i2 = filename.find("/", i1)
        labelStr += filename[i1:i2]
    if labelStr is not None:
        plt.text(xLoc, yLoc, labelStr, ha="center", va="center", fontsize=fontSize,
                 transform=axis.transAxes, color=color)


def labelCamera(camera, plt, axis, xLoc, yLoc, color="k", fontSize=10):
    labelStr = "camera: " + str(camera.getName())
    plt.text(xLoc, yLoc, labelStr, ha="center", va="center", fontsize=fontSize,
             transform=axis.transAxes, color=color)


def filterStrFromFilename(filename):
    """!Determine filter string from filename
    """
    filterStr = None
    f1 = filename.find("plots/") + len("plots/")
    f2 = filename.find("/", f1)
    filterStr = filename[f1:f2]

    return filterStr


def plotCameraOutline(plt, axes, camera, ccdList, color="k", fontSize=6):
    axes.tick_params(which="both", direction="in", labelleft=False, labelbottom=False)
    axes.locator_params(nbins=6)
    axes.ticklabel_format(useOffset=False)
    camRadius = max(camera.getFpBBox().getWidth(), camera.getFpBBox().getHeight())/2
    camRadius = np.round(camRadius, -2)
    camLimits = np.round(1.25*camRadius, -2)
    intCcdList = [int(ccd) for ccd in ccdList]
    for ccd in camera:
        if ccd.getId() in intCcdList:
            ccdCorners = ccd.getCorners(cameraGeom.FOCAL_PLANE)
            plt.gca().add_patch(patches.Rectangle(ccdCorners[0], *list(ccdCorners[2] - ccdCorners[0]),
                                                  fill=True, facecolor="y", edgecolor="k", ls="solid"))
    axes.set_xlim(-camLimits, camLimits)
    axes.set_ylim(-camLimits, camLimits)
    axes.add_patch(patches.Circle((0, 0), radius=camRadius, color="k", alpha=0.2))
    if camera.getName() == "HSC":
        for x, y, t in ([-1, 0, "N"], [0, 1, "W"], [1, 0, "S"], [0, -1, "E"]):
            axes.text(1.085*camRadius*x, 1.085*camRadius*y, t, ha="center", va="center",
                      fontsize=fontSize - 1)
    axes.text(-0.82*camRadius, 0.95*camRadius, "%s" % camera.getName(), ha="center", fontsize=fontSize,
              color=color)


def plotTractOutline(axes, tractInfo, patchList, fontSize=5, maxDegBeyondPatch=1.5):
    """Plot the the outline of the tract and patches highlighting those with data

    As some skyMap settings can define tracts with a large number of patches, this can
    become very crowded.  So, if only a subset of patches are included, find the outer
    boudary of all patches in patchList and only plot to maxDegBeyondPatch degrees
    beyond those boundaries (in all four directions).

    Parameters
    ----------
    tractInfo : `lsst.skymap.tractInfo.ExplicitTractInfo`
       Tract information object for extracting tract RA and DEC limits.
    patchList : `list` of `str`
       List of patch IDs with data to be plotted.  These will be color shaded in the outline plot.
    fontSize : `int`
       Font size for plot labels.
    maxDegBeyondPatch : `float`
       Maximum number of degrees to plot beyond the border defined by all patches with data to be plotted.
    """
    buff = 0.02
    axes.tick_params(which="both", direction="in", labelsize=fontSize)
    axes.locator_params(nbins=6)
    axes.ticklabel_format(useOffset=False)

    tractRa, tractDec = bboxToXyCoordLists(tractInfo.getBBox(), wcs=tractInfo.getWcs())
    patchBoundary = getRaDecMinMaxPatchList(patchList, tractInfo, pad=maxDegBeyondPatch)

    xMin = min(max(tractRa), patchBoundary.raMax) + buff
    xMax = max(min(tractRa), patchBoundary.raMin) - buff
    yMin = max(min(tractDec), patchBoundary.decMin) - buff
    yMax = min(max(tractDec), patchBoundary.decMax) + buff
    xlim = xMin, xMax
    ylim = yMin, yMax
    axes.fill(tractRa, tractDec, fill=True, edgecolor='k', lw=1, linestyle='solid',
              color="black", alpha=0.2)
    for ip, patch in enumerate(tractInfo):
        patchIndexStr = str(patch.getIndex()[0]) + "," + str(patch.getIndex()[1])
        color = "k"
        alpha = 0.05
        if patchIndexStr in patchList:
            color = ("c", "g", "r", "b", "m")[ip%5]
            alpha = 0.5
        ra, dec = bboxToXyCoordLists(patch.getOuterBBox(), wcs=tractInfo.getWcs())
        deltaRa = abs(max(ra) - min(ra))
        deltaDec = abs(max(dec) - min(dec))
        pBuff = 0.5*max(deltaRa, deltaDec)
        centerRa = min(ra) + 0.5*deltaRa
        centerDec = min(dec) + 0.5*deltaDec
        if (centerRa < xMin + pBuff and centerRa > xMax - pBuff and
                centerDec > yMin - pBuff and centerDec < yMax + pBuff):
            axes.fill(ra, dec, fill=True, color=color, lw=1, linestyle="solid", alpha=alpha)
            if patchIndexStr in patchList or (centerRa < xMin - 0.2*pBuff and
                                              centerRa > xMax + 0.2*pBuff and
                                              centerDec > yMin + 0.2*pBuff and
                                              centerDec < yMax - 0.2*pBuff):
                axes.text(percent(ra), percent(dec, 0.5), str(patchIndexStr),
                          fontsize=fontSize - 1, horizontalalignment="center", verticalalignment="center")
    axes.text(percent((xMin, xMax), 1.06), percent((yMin, yMax), -0.08), "RA",
              fontsize=fontSize, horizontalalignment="center", verticalalignment="center", color="green")
    axes.text(percent((xMin, xMax), 1.15), percent((yMin, yMax), 0.01), "Dec",
              fontsize=fontSize, horizontalalignment="center", verticalalignment="center",
              rotation="vertical", color="green")
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)


def plotCcdOutline(axes, butler, dataId, ccdList, zpLabel=None, fontSize=8):
    """!Plot outlines of CCDs in ccdList
    """
    dataIdCopy = dataId.copy()
    if "raftName" in dataId:  # Pop these (if present) so that the ccd is looked up by just the detector field
        dataIdCopy.pop("raftName", None)
        dataIdCopy.pop("detectorName", None)
    for ccd in ccdList:
        ccdKey = findCcdKey(dataId)
        ccdLabelStr = str(ccd)
        if "raft" in dataId:
            if len(ccd) != 4:
                if len(ccd) > 4:
                    errorStr = "Only raft/sensor combos with x,y coords 0 through 9 have been accommodated"
                else:
                    errorStr = "Only 2 by 2 = 4 integer raft/sensor combo names have been accommodated"
                RuntimeError(errorStr)
            raft = ccd[0] + "," + ccd[1]
            dataIdCopy["raft"] = raft
            ccd = ccd[-2] + "," + ccd[-1]
            ccdLabelStr = "R" + str(raft) + "S" + str(ccd)
        dataIdCopy[ccdKey] = ccd
        calexp = butler.get("calexp", dataIdCopy)
        dataRef = butler.dataRef("raw", dataId=dataIdCopy)
        # Check metadata to see if stack used was HSC
        metadata = butler.get("calexp_md", dataIdCopy)
        hscRun = checkHscStack(metadata)
        if zpLabel is not None:
            if zpLabel == "MEAS_MOSAIC" or "MEAS_MOSAIC_1" in zpLabel:
                applyMosaicResultsExposure(dataRef, calexp=calexp)

        wcs = calexp.getWcs()
        w = calexp.getWidth()
        h = calexp.getHeight()
        if zpLabel is not None:
            if hscRun and (zpLabel == "MEAS_MOSAIC" or "MEAS_MOSAIC_1" in zpLabel):
                nQuarter = calexp.getDetector().getOrientation().getNQuarter()
                if nQuarter%2 != 0:
                    w = calexp.getHeight()
                    h = calexp.getWidth()

        ras = list()
        decs = list()
        for x, y in zip([0, w, w, 0, 0], [0, 0, h, h, 0]):
            xy = afwGeom.Point2D(x, y)
            ra = np.rad2deg(np.float64(wcs.pixelToSky(xy)[0]))
            dec = np.rad2deg(np.float64(wcs.pixelToSky(xy)[1]))
            ras.append(ra)
            decs.append(dec)
        axes.plot(ras, decs, "k-", linewidth=1)
        xy = afwGeom.Point2D(w/2, h/2)
        centerX = np.rad2deg(np.float64(wcs.pixelToSky(xy)[0]))
        centerY = np.rad2deg(np.float64(wcs.pixelToSky(xy)[1]))
        axes.text(centerX, centerY, "%s" % str(ccdLabelStr), ha="center", va="center", fontsize=fontSize)


def plotPatchOutline(axes, tractInfo, patchList, plotUnits="deg", idFontSize=None):
    """!Plot outlines of patches in patchList
    """
    validWcsUnits = ["deg", "rad"]
    idFontSize = max(5, 9 - int(0.4*len(patchList))) if not idFontSize else idFontSize
    for ip, patch in enumerate(tractInfo):
        if str(patch.getIndex()[0])+","+str(patch.getIndex()[1]) in patchList:
            if len(patchList) < 9:
                if plotUnits in validWcsUnits:
                    xCoord, yCoord = bboxToXyCoordLists(patch.getOuterBBox(), wcs=tractInfo.getWcs(),
                                                        wcsUnits=plotUnits)
                else:
                    xCoord, yCoord = bboxToXyCoordLists(patch.getOuterBBox(), wcs=None)
                xCoords = xCoord + (xCoord[0], )
                yCoords = yCoord + (yCoord[0], )
                axes.plot(xCoords, yCoords, color="black", lw=0.5, linestyle="solid")
            if plotUnits in validWcsUnits:
                xCoord, yCoord = bboxToXyCoordLists(patch.getInnerBBox(), tractInfo.getWcs(),
                                                    wcsUnits=plotUnits)
            else:
                xCoord, yCoord = bboxToXyCoordLists(patch.getInnerBBox(), wcs=None)
            xCoords = xCoord + (xCoord[0], )
            yCoords = yCoord + (yCoord[0], )
            axes.plot(xCoords, yCoords, color="black", lw=0.8, linestyle="dashed")
            axes.text(percent(xCoords), percent(yCoords, 0.5), str(patch.getIndex()),
                      fontsize=idFontSize, horizontalalignment="center", verticalalignment="center")


def rotatePixelCoords(sources, width, height, nQuarter):
    """Rotate catalog (x, y) pixel coordinates such that LLC of detector in FP is (0, 0)
    """
    xKey = sources.schema.find("slot_Centroid_x").key
    yKey = sources.schema.find("slot_Centroid_y").key
    for s in sources:
        x0 = s[xKey]
        y0 = s[yKey]
        if nQuarter == 1:
            s.set(xKey, height - y0 - 1.0)
            s.set(yKey, x0)
        if nQuarter == 2:
            s.set(xKey, width - x0 - 1.0)
            s.set(yKey, height - y0 - 1.0)
        if nQuarter == 3:
            s.set(xKey, y0)
            s.set(yKey, width - x0 - 1.0)
    return sources


def bboxToXyCoordLists(bbox, wcs=None, wcsUnits="deg"):
    """Get the corners of a BBox and convert them to x and y coord lists.

    Parameters
    ----------
    bbox : `lsst.geom.Box2I`
       The bounding box under consideration.
    wcs : `lsst.afw.geom.SkyWcs`, optional
       If provided, the coordinate lists returned will be Ra and Dec in
       `wcsUnits`.  Ignored if ``wcs`` is `None`.  Default is "deg".
    wcsUnits : `str`, optional
       Coordinate units to be returned if a wcs is provided (ignored
       otherwise).  Can be either "deg" or "rad".  Default is "deg".

    Raises
    ------
    `RuntimeError`
       If ``wcsUnits`` is neither "deg" nor "rad".

    Returns
    -------
    xCoords, yCoords : `list` of `float`
       The lists associated with the x and y coordinates in appropriate uints.
    """
    validWcsUnits = ["deg", "rad"]
    corners = []
    for corner in bbox.getCorners():
        p = afwGeom.Point2D(corner.getX(), corner.getY())
        if wcs:
            if wcsUnits not in validWcsUnits:
                raise RuntimeError("wcsUnits must be one of {:}".format(validWcsUnits))
            coord = wcs.pixelToSky(p)
            if wcsUnits == "deg":
                corners.append([coord.getRa().asDegrees(), coord.getDec().asDegrees()])
            elif wcsUnits == "rad":
                corners.append([coord.getRa().asRadians(), coord.getDec().asRadians()])
        else:
            coord = p
            corners.append([coord.getX(), coord.getY()])
    xCoords, yCorrds = zip(*corners)
    return xCoords, yCorrds


def getRaDecMinMaxPatchList(patchList, tractInfo, pad=0.0, nDecimals=4, raMin=360.0, raMax=0.0,
                            decMin=90.0, decMax=-90.0):
    """Find the max and min RA and DEC (deg) boundaries encompased in the patchList

    Parameters
    ----------
    patchList : `list` of `str`
       List of patch IDs.
    tractInfo : `lsst.skymap.tractInfo.ExplicitTractInfo`
       Tract information associated with the patches in patchList
    pad : `float`
       Pad the boundary by pad degrees
    nDecimals : `int`
       Round coordinates to this number of decimal places
    raMin, raMax : `float`
       Initiate minimum[maximum] RA determination at raMin[raMax] (deg)
    decMin, decMax : `float`
       Initiate minimum[maximum] DEC determination at decMin[decMax] (deg)

    Returns
    -------
    `lsst.pipe.base.Struct`
       Contains the ra and dec min and max values for the patchList provided
    """
    for ip, patch in enumerate(tractInfo):
        if str(patch.getIndex()[0])+","+str(patch.getIndex()[1]) in patchList:
            raPatch, decPatch = bboxToXyCoordLists(patch.getOuterBBox(), wcs=tractInfo.getWcs())
            raMin = min(np.round(min(raPatch) - pad, nDecimals), raMin)
            raMax = max(np.round(max(raPatch) + pad, nDecimals), raMax)
            decMin = min(np.round(min(decPatch) - pad, nDecimals), decMin)
            decMax = max(np.round(max(decPatch) + pad, nDecimals), decMax)
    return Struct(
        raMin=raMin,
        raMax=raMax,
        decMin=decMin,
        decMax=decMax,
    )


def percent(values, p=0.5):
    """Return a value a faction of the way between the min and max values in a list."""
    m = min(values)
    interval = max(values) - m
    return m + p*interval


def setPtSize(num, ptSize=12):
    """Set the point size according to the size of the catalog"""
    if num > 10:
        ptSize = min(12, max(4, int(20/np.log10(num))))
    return ptSize


def getQuiver(x, y, e1, e2, ax, color=None, scale=3, width=0.005, label=''):
    """Return the quiver object for the given input parameters"""
    theta = [np.math.atan2(a, b)/2.0 for a, b in zip(e1, e2)]
    e = np.sqrt(e1**2 + e2**2)
    c1 = e*np.cos(theta)
    c2 = e*np.sin(theta)
    if color is None:
        color = e
    q = ax.quiver(x, y, c1, c2, color=color, angles='uv', scale=scale, units='width', pivot='middle',
                  width=width, headwidth=0.0, headlength=0.0, headaxislength=0.0, label=label)
    return q


def makeAlphaCmap(cmap=plt.cm.viridis, alpha=1.0):
    """Given a matplotlib colormap, return it but with given alpha transparency

    Parameters
    ----------
    cmap : `matplotlib.colors.ListedColormap`, optional
       The matplotlib colormap to make transparent with level ``alpha``.
       Default color map is `plt.cm.viridis`.
    alpha : `float`, optional
       The matplotlib blending value, between 0 (transparent) and 1 (opaque)
       (1.0 by default).

    Returns
    -------
    alphaCmap : `matplotlib.colors.ListedColormap`
       The matplotlib colormap ``cmap`` but with transparency level ``alpha``.
    """
    alphaCmap = cmap(np.arange(cmap.N))
    alphaCmap[:, -1] = alpha
    alphaCmap = ListedColormap(alphaCmap)
    return alphaCmap


def buildTractImage(butler, dataId, tractInfo, patchList=None, coaddName="deep"):
    """Build up an image of an entire tract or list of patches

    Parameters
    ----------
    butler : `lsst.daf.persistence.Butler`
    dataId : `lsst.daf.persistence.DataId`
       An instance of `lsst.daf.persistence.DataId` from which to extract the
       filter name.
    tractInfo : `lsst.skymap.tractInfo.ExplicitTractInfo`
       Tract information object.
    patchList : `list` of `str`, optional
       A list of the patches to include.  If `None`, the full list of patches
       in ``tractInfo`` will be included.
    coaddName : `str`, optional
       The base name of the coadd (e.g. "deep" or "goodSeeing").
       Default is "deep".

    Raises
    ------
    `RuntimeError`
       If ``nPatches`` is zero, i.e. no data was found.

    Returns
    -------
    image : `lsst.afw.image.ImageF`
       The full tract or patches in ``patchList`` image.
    """
    tractBbox = tractInfo.getBBox()
    nPatches = 0
    if not patchList:
        patchList = []
        nPatchX, nPatchY = tractInfo.getNumPatches()
        for iPatchX in range(nPatchX):
            for iPatchY in range(nPatchY):
                patchList.append("%d,%d" % (iPatchX, iPatchY))
    tractArray = np.full((tractBbox.getMaxY() + 1, tractBbox.getMaxX() + 1), np.nan, dtype="float32")
    for patch in patchList:
        expDataId = {"filter": dataId["filter"], "tract": tractInfo.getId(), "patch": patch}
        try:
            exp = butler.get(coaddName + "Coadd_calexp", expDataId, immediate=True)
            bbox = butler.get(coaddName + "Coadd_calexp_bbox", expDataId, immediate=True)
            tractArray[bbox.getMinY():bbox.getMaxY() + 1,
                       bbox.getMinX():bbox.getMaxX() + 1] = exp.maskedImage.image.array
            nPatches += 1
        except Exception:
            continue
    if nPatches == 0:
        raise RuntimeError("No data found for tract {:}".format(tractInfo.getId()))
    tractArray = np.flipud(tractArray)
    image = afwImage.ImageF(afwGeom.ExtentI(tractBbox.getMaxX() + 1, tractBbox.getMaxY() + 1))
    image.array[:] = tractArray
    return image
