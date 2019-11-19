from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import numpy as np

import lsst.afw.cameraGeom as cameraGeom
import lsst.afw.cameraGeom.utils as cgUtils
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.daf.persistence as dafPersist
import lsst.geom as geom
from lsst.display.matplotlib.matplotlib import AsinhNormalize
from lsst.pipe.base import Struct

from .utils import checkHscStack, findCcdKey, popIdAndCcdKeys

try:
    from lsst.meas.mosaic.updateExposure import applyMosaicResultsExposure
except ImportError:
    applyMosaicResultsExposure = None

__all__ = ["AllLabeller", "StarGalaxyLabeller", "OverlapsStarGalaxyLabeller", "MatchesStarGalaxyLabeller",
           "CosmosLabeller", "plotText", "annotateAxes", "labelVisit", "labelCamera",
           "filterStrFromFilename", "plotCameraOutline", "plotTractOutline", "plotPatchOutline",
           "plotCcdOutline", "rotatePixelCoords", "rotatePoint", "bboxToXyCoordLists", "getMinMaxPatchList",
           "getMinMaxCcdList", "computeEqualAspectLimits", "percent", "setPtSize", "getQuiver",
           "makeAlphaCmap", "buildTractImage", "buildVisitImage", "getArrayFromImage", "makeDiffImages",
           "makeAsinhNormFromArray", "determineUberCalLabel", "plotDirectionArrows", "plotSkyLimitLabels"]


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

    def __call__(self, catalog1, catalog2=None):
        catalog2 = catalog2 if catalog2 else catalog1
        first = np.where(catalog1[self._first + self._column] < 0.5, 0, 1)
        second = np.where(catalog2[self._second + self._column] < 0.5, 0, 1)
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
        cosmos["coord_ra"][:] = original["ALPHA.J2000"][good]*(1.0*geom.degrees).asRadians()
        cosmos["coord_dec"][:] = original["DELTA.J2000"][good]*(1.0*geom.degrees).asRadians()
        self.cosmos = cosmos
        self.radius = radius

    def __call__(self, catalog):
        # A kdTree would be better, but this is all we have right now
        matches = afwTable.matchRaDec(self.cosmos, catalog, self.radius)
        good = set(mm.second.getId() for mm in matches)
        return np.array([0 if ii in good else 1 for ii in catalog["id"]])


def plotText(textStr, plt, axis, xLoc, yLoc, prefix="", fontSize=None, color="k", coordSys="axes", **kwargs):
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
    fontSize : `int` or `str` or `None`, optional
       Size of font for plotting of ``textStr``.  May be either an absolute font size in points, or a
       size string, relative to the default font size.  Default is `None`, in which case an automatic
       scaling based on the length of ``textStr`` will be used.
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
    if not fontSize:
        fontSize = int(9 - min(3, len(textStr)/10))
    plt.text(xLoc, yLoc, prefix + textStr, ha="center", va="center", fontsize=fontSize, transform=transform,
             color=color, **kwargs)


def annotateAxes(filename, plt, axes, statsConf, dataSet, magThresholdConf, signalToNoiseStrConf=None,
                 statsHigh=None, magThresholdHigh=None, signalToNoiseHighStr=None,
                 x0=0.03, y0=0.96, yOff=0.05, fontSize=8, ha="left", va="top", color="blue",
                 isHist=False, hscRun=None, matchRadius=None, matchRadiusUnitStr="\"",
                 unitScale=1.0, doPrintMedian=False):
    """Label the plot with the statistical computation results

    Parameters
    ----------
    filename : `str`
       String representing the full path of the plot output filename.  Used
       here to select for/against certain annotations for certain styles of
       plots.
    plt : `matplotlib.pyplot`
       Instance of the matplotlib plot to be annotated.
    axes : `matplotlib.axes._axes.Axes`
       Particular matplotlib axes of ``plt`` on which to plot the annotations.
    statsConf : `lsst.pipe.analysis.utils.Stats`
       `lsst.pipe.analysis.utils.Stats` object that contains the results from
       the "configured" statistical computation results for the threshold type
       and values set in ``analysis.config.suseSignalToNoiseThreshold`` and:
       - ``analysis.config.signalToNoiseThreshold`` if the former is `True` or
       - ``analysis.config.magThreshold`` if it is `False`.
    dataset : `str`
       Name of the catalog dataset to for which annotations are being added.
       Valid strings are "star", "galaxy", "all", and "split".
    magThresholdConf : `float`
       The "configured" value for the magnitude threshold (i.e. the value set
       in ``analysis.config.magThreshold`` if the threshold was set based on
       magnitude or the "effective" magnitude threshold if the cut was based
       on S/N).
    signalToNoiseStrConf : `str` or `None`, optional
       A string representing the type of threshold used in culling the data to
       the subset of the quantity that was used in the statistics computation
       of ``statsConf``: "S/N" and "mag" indicate a threshold based on
       signal-to-noise or magnitude, respectively.  Default is `None`.
    statsHigh : `lsst.pipe.analysis.utils.Stats`, optional
       `lsst.pipe.analysis.utils.Stats` object that contains the results from
       the "high" statistical computation results whose value is set in
       ``analysis.config.signalToNoiseHighThreshold``.  Default is `None`.
    magThresholdHigh : `float`, optional
       The "effective" magnitude threshold based on the "high" S/N cut.
       Default is `None`.
    signalToNoiseHighStr : `str`
       A string representing the threshold used in culling of the dataset to
       the subset of the quantity that was used in the statistics computation
       of ``statsHigh``.  Default is `None`.
    x0, y0 : `float`, optional
       Axis coordinates controlling placement of annotations on the plot.
       Defaults are ``x0``=0.03 and ``y0``=0.96.
    yOff : `float`, optional
       Offset by which to separate annotations along the y-axis.
       Default is 0.05.
    fontSize : `int`, optional
       Font size for plot labels.  Default is 8.
    ha, va : `str`, optional
       Horizontal and vertical allignments for text labels.  Can be any valid
       matplotlib allignment string.  Defaults are ``ha``="left", ``va``="top".
    color : `str`, optional
       Color for annotations.  Can be any matplotlib color str.
       Default is "blue".
    isHist : `bool`, optional
       Boolean indicating if this is a histogram style plot (for slightly
       different annotation settings).  Default is `False`.
    hscRun : `str` or `None`, optional
       String representing "HSCPIPE_VERSION" fits header if the data were
       processed with the (now obsolete, but old reruns still exist)
       "HSC stack".  Default is `None`.
    matchRadius : `float` or `None`, optional
       Maximum search radius for source matching between catalogs.
       Default is `None`.
    matchRadiusUnitStr : `str`, optional
       String representing the units of the match radius (e.g. "arcsec",
       "pixel").  Default is "\"" (i.e. arcsec).
    unitScale : `float`, optional
       Number indicating any scaling of the units (e.g 1000.0 means units
       are in "milli" of the base unit).  Default is 1.0.
    doPrintMedian : `bool`, optional
       Boolean to indicate if the median (in addition to the mean) should
       be printed on the plot.  Default is `False`.

    Returns
    -------
    l1, l2 : `matplotlib.lines.Line2D`
       Output of the axes.axvline commands for the median and clipped
       values (used for plot legends).
    """
    xThresh = axes.get_xlim()[0] + 0.58*(axes.get_xlim()[1] - axes.get_xlim()[0])
    for stats, magThreshold, signalToNoiseStr, y00 in [[statsConf, magThresholdConf, signalToNoiseStrConf,
                                                        y0],
                                                       [statsHigh, magThresholdHigh, signalToNoiseHighStr,
                                                        0.18]]:
        axes.annotate(dataSet + r" N = {0.num:d} (of {0.total:d})".format(stats[dataSet]),
                      xy=(x0, y00), xycoords="axes fraction", ha=ha, va=va, fontsize=fontSize, color=color)
        if signalToNoiseStr:
            axes.annotate(signalToNoiseStr, xy=(xThresh, y00), xycoords=("data", "axes fraction"),
                          ha="right", va=va, fontsize=fontSize, color="k", alpha=0.8)
            axes.annotate(r" [mag$\lesssim${0:.1f}]".format(magThreshold), xy=(xThresh, y00 - yOff),
                          xycoords=("data", "axes fraction"),
                          ha="right", va=va, fontsize=fontSize, color="k", alpha=0.8)
        else:
            axes.annotate(r" [mag$\leqslant${0:.1f}]".format(magThreshold), xy=(xThresh, y00),
                          xycoords=("data", "axes fraction"),
                          ha="right", va=va, fontsize=fontSize, color="k", alpha=0.8)
        meanStr = "{0.mean:.4f}".format(stats[dataSet])
        medianStr = "{0.median:.4f}".format(stats[dataSet])
        stdevStr = "{0.stdev:.4f}".format(stats[dataSet])
        statsUnitStr = None
        if unitScale == 1000.0:
            meanStr = "{0.mean:.2f}".format(stats[dataSet])
            medianStr = "{0.median:.2f}".format(stats[dataSet])
            stdevStr = "{0.stdev:.2f}".format(stats[dataSet])
            statsUnitStr = " (milli)"
            if any(ss in filename for ss in ["_ra", "_dec", "distance"]):
                statsUnitStr = " (mas)"
            if any(ss in filename for ss in ["Flux", "_photometry", "_mag"]):
                statsUnitStr = " (mmag)"
        lenStr = 0.12 + 0.017*(max(len(meanStr), len(stdevStr)))
        strKwargs = dict(xycoords="axes fraction", va=va, fontsize=fontSize, color="k")
        yOffMult = 1
        axes.annotate("mean = ", xy=(x0 + 0.12, y00 - yOffMult*yOff), ha="right", **strKwargs)
        axes.annotate(meanStr, xy=(x0 + lenStr, y00 - yOffMult*yOff), ha="right", **strKwargs)
        if statsUnitStr is not None:
            axes.annotate(statsUnitStr, xy=(x0 + lenStr + 0.006, y00 - yOffMult*yOff), ha="left", **strKwargs)
        yOffMult += 1
        axes.annotate("stdev = ", xy=(x0 + 0.12, y00 - yOffMult*yOff), ha="right", **strKwargs)
        axes.annotate(stdevStr, xy=(x0 + lenStr, y00 - yOffMult*yOff), ha="right", **strKwargs)
        if doPrintMedian:
            yOffMult += 1
            axes.annotate("med = ", xy=(x0 + 0.12, y00 - yOffMult*yOff), ha="right", **strKwargs)
            axes.annotate(medianStr, xy=(x0 + lenStr, y00 - yOffMult*yOff), ha="right", **strKwargs)

    if matchRadius is not None:
        yOffMult += 1
        axes.annotate("Match radius = {0:.2f}{1:s}".format(matchRadius, matchRadiusUnitStr),
                      xy=(x0, y0 - yOffMult*yOff), ha=ha, **strKwargs)
    if hscRun is not None:
        yOffMult += 1
        axes.annotate("HSC stack run: {0:s}".format(hscRun), xy=(x0, y0 - yOffMult*yOff),
                      xycoords="axes fraction", ha=ha, va=va, fontsize=fontSize, color="#800080")
    if isHist:
        l1 = axes.axvline(stats[dataSet].median, linestyle="dotted", color="0.7")
        l2 = axes.axvline(stats[dataSet].median + stats[dataSet].clip, linestyle="dashdot", color="0.7")
        axes.axvline(stats[dataSet].median - stats[dataSet].clip, linestyle="dashdot", color="0.7")
    else:
        l1 = axes.axhline(stats[dataSet].median, linestyle="dotted", color="0.7", label="median")
        l2 = axes.axhline(stats[dataSet].median + stats[dataSet].clip, linestyle="dashdot", color="0.7",
                          label="clip")
        axes.axhline(stats[dataSet].median - stats[dataSet].clip, linestyle="dashdot", color="0.7")
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
    camRadius = np.round(camRadius, -1)
    camLimits = np.round(1.25*camRadius, -1)
    intCcdList = [int(ccd) for ccd in ccdList]
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors.pop(colors.index('#7f7f7f'))  # get rid of the gray one as is doesn't contrast well with white
    colors.append("gold")
    hasRotatedCcds = False
    for ccd in camera:
        if ccd.getOrientation().getNQuarter() != 0:
            hasRotatedCcds = True
            break
    for ic, ccd in enumerate(camera):
        ccdCorners = ccd.getCorners(cameraGeom.FOCAL_PLANE)
        if ccd.getType() == cameraGeom.DetectorType.SCIENCE:
            plt.gca().add_patch(patches.Rectangle(ccdCorners[0], *list(ccdCorners[2] - ccdCorners[0]),
                                                  facecolor="none", edgecolor="k", ls="solid", lw=0.5,
                                                  alpha=0.5))
        if ccd.getId() in intCcdList:
            if hasRotatedCcds:
                nQuarter = ccd.getOrientation().getNQuarter()
                fillColor = colors[nQuarter%len(colors)]
            elif ccd.getName()[0] == "R":
                try:
                    fillColor = colors[(int(ccd.getName()[1]) + int(ccd.getName()[2]))%len(colors)]
                except Exception:
                    fillColor = colors[ic%len(colors)]
            else:
                fillColor = colors[ic%len(colors)]
            ccdCorners = ccd.getCorners(cameraGeom.FOCAL_PLANE)
            plt.gca().add_patch(patches.Rectangle(ccdCorners[0], *list(ccdCorners[2] - ccdCorners[0]),
                                                  fill=True, facecolor=fillColor, edgecolor="k",
                                                  ls="solid", lw=1.0, alpha=0.7))
    axes.set_xlim(-camLimits, camLimits)
    axes.set_ylim(-camLimits, camLimits)
    if camera.getName() == "HSC":
        for x, y, t in ([-1, 0, "N"], [0, 1, "W"], [1, 0, "S"], [0, -1, "E"]):
            axes.text(1.085*camRadius*x, 1.085*camRadius*y, t, ha="center", va="center",
                      fontsize=fontSize - 1)
            axes.text(-0.82*camRadius, 1.04*camRadius, "%s" % camera.getName(), ha="center",
                      fontsize=fontSize, color=color)
    else:
        axes.text(0.0, 1.04*camRadius, "%s" % camera.getName(), ha="center", fontsize=fontSize,
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
    patchBoundary = getMinMaxPatchList(patchList, tractInfo)
    if maxDegBeyondPatch > 0:
        patchBoundary.raMin -= maxDegBeyondPatch
        patchBoundary.raMax += maxDegBeyondPatch
        patchBoundary.decMin -= maxDegBeyondPatch
        patchBoundary.decMax += maxDegBeyondPatch
    xMin = min(max(tractRa), patchBoundary.raMax) + buff
    xMax = max(min(tractRa), patchBoundary.raMin) - buff
    yMin = max(min(tractDec), patchBoundary.decMin) - buff
    yMax = min(max(tractDec), patchBoundary.decMax) + buff
    xlim = xMin, xMax
    ylim = yMin, yMax
    axes.fill(tractRa, tractDec, fill=False, edgecolor='k', lw=0.5, linestyle='solid', color="k", alpha=0.3)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors.pop(colors.index('#7f7f7f'))  # get rid of the gray one as that's our no-data colour
    colors.append("gold")
    for ip, patch in enumerate(tractInfo):
        patchIndexStr = str(patch.getIndex()[0]) + "," + str(patch.getIndex()[1])
        color = "k"
        alpha = 0.05
        (color, alpha) = (colors[ip%len(colors)], 0.5) if patchIndexStr in patchList else (color, alpha)
        ra, dec = bboxToXyCoordLists(patch.getOuterBBox(), wcs=tractInfo.getWcs())
        deltaRa = abs(max(ra) - min(ra))
        deltaDec = abs(max(dec) - min(dec))
        pBuff = 0.5*max(deltaRa, deltaDec)
        centerRa = min(ra) + 0.5*deltaRa
        centerDec = min(dec) + 0.5*deltaDec
        if (centerRa < xMin + pBuff and centerRa > xMax - pBuff and
                centerDec > yMin - pBuff and centerDec < yMax + pBuff):
            axes.fill(ra, dec, fill=True, color=color, lw=0.5, linestyle="solid", alpha=alpha)
            if patchIndexStr in patchList or (centerRa < xMin - 0.2*pBuff and
                                              centerRa > xMax + 0.2*pBuff and
                                              centerDec > yMin + 0.2*pBuff and
                                              centerDec < yMax - 0.2*pBuff):
                axes.text(percent(ra), percent(dec, 0.5), str(patchIndexStr),
                          fontsize=fontSize - 1, horizontalalignment="center", verticalalignment="center")
    axes.text(percent((xMin, xMax), 1.065), percent((yMin, yMax), -0.08), "RA",
              fontsize=fontSize, horizontalalignment="center", verticalalignment="center", color="green")
    axes.text(percent((xMin, xMax), 1.15), percent((yMin, yMax), -0.02), "Dec",
              fontsize=fontSize, horizontalalignment="center", verticalalignment="center",
              rotation="vertical", color="green")
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)


def plotCcdOutline(axes, butler, dataId, camera, ccdList, tractInfo=None, zpLabel=None, fontSize=8,
                   wcsKeysList=None, raDecProj=True):
    """!Plot outlines of CCDs in ccdList
    """
    wcsKeysList = (wcsKeysList if wcsKeysList is not None else
                   ["rawWcs", "initialWcs", "calexpWcs", "uberWcs"])
    wcsDict = dict()
    legendAddedDict = dict()
    for key in wcsKeysList:
        legendAddedDict[key] = False
    lineStyleDict = {"initialWcs": (0, (3, 5, 1, 5)), "rawWcs": ":", "calexpWcs": "--", "uberWcs": "-"}
    lineColorDict = {"initialWcs": "yellow", "rawWcs": "hotpink", "calexpWcs": "lightgrey",
                     "uberWcs": "black"}

    # Pop some ccd keys (if present) from a copy of the dataId so that the ccd
    # is looked up by just the detector field.
    dataIdCopy = dataId.copy()
    dataIdCopy = popIdAndCcdKeys(dataIdCopy)
    ccdKey = findCcdKey(dataId)
    for ccd in ccdList:
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
        detector = calexp.getDetector()
        # Check metadata to see if stack used was HSC
        metadata = butler.get("calexp_md", dataIdCopy)
        hscRun = checkHscStack(metadata)
        # Get WCSs at various stages: raw, calexp, ubercal (if present)
        try:
            raw = butler.get("raw", dataId=dataIdCopy)
            rawWcs = raw.getWcs()
        except dafPersist.butlerExceptions.NoResults:
            print("NOTE: could not locate raw dataset, not plotting rawWcs")
            rawWcs = None
        wcsDict["rawWcs"] = rawWcs
        if rawWcs:
            try:
                from lsst.obs.base import createInitialSkyWcs
                initialWcs = createInitialSkyWcs(raw.getInfo().getVisitInfo(), raw.getDetector())
            except Exception:
                initialWcs = None
        else:
            initialWcs = None

        wcsDict["initialWcs"] = initialWcs
        calexpWcs = calexp.getWcs()
        wcsDict["calexpWcs"] = calexpWcs
        if zpLabel and (zpLabel == "MEAS_MOSAIC" or "MEAS_MOSAIC_1" in zpLabel):
            applyMosaicResultsExposure(dataRef, calexp=calexp)
            uberWcs = calexp.getWcs()
        elif zpLabel and ("JOINTCAL" in zpLabel or "MMphotoCalib" in zpLabel or "JOINTCAL_1" in zpLabel):
            uberWcs = dataRef.get("jointcal_wcs")
        else:
            uberWcs = None
        wcsDict["uberWcs"] = uberWcs
        w = calexp.getWidth()
        h = calexp.getHeight()
        if zpLabel is not None:
            if hscRun and (zpLabel == "MEAS_MOSAIC" or "MEAS_MOSAIC_1" in zpLabel):
                nQuarter = calexp.getDetector().getOrientation().getNQuarter()
                if nQuarter%2 != 0:
                    w = calexp.getHeight()
                    h = calexp.getWidth()

        lineWidth = 0.5
        if not raDecProj:
            xs = list()
            ys = list()
            ccdCorners = detector.getCorners(cameraGeom.FOCAL_PLANE)
            ccdCenter = detector.getCenter(cameraGeom.FOCAL_PLANE)
            pixelSizeX = detector.getPixelSize()[0]
            pixelSizeY = detector.getPixelSize()[1]
            for corner in ccdCorners:
                xs.append(corner[0]/pixelSizeX)
                ys.append(corner[1]/pixelSizeY)
            xs.append(xs[0])  # to close the box for plotting
            ys.append(ys[0])
            centerX = ccdCenter[0]/pixelSizeX
            centerY = ccdCenter[1]/pixelSizeY
            axes.plot(xs, ys, color="black", linewidth=1)
            axes.text(centerX, centerY, "%s" % str(ccdLabelStr), ha="center", va="center",
                      fontsize=fontSize, color="black")
        else:
            for wcsKey in wcsKeysList:
                wcs = wcsDict[wcsKey]
                if wcs:
                    ras = list()
                    decs = list()
                    coords = list()
                    for x, y in zip([0, w, w, 0, 0], [0, 0, h, h, 0]):
                        xy = geom.Point2D(x, y)
                        ra = np.rad2deg(np.float64(wcs.pixelToSky(xy)[0]))
                        dec = np.rad2deg(np.float64(wcs.pixelToSky(xy)[1]))
                        ras.append(ra)
                        decs.append(dec)
                        coords.append(geom.SpherePoint(ra, dec, geom.degrees))
                    xy = geom.Point2D(w/2, h/2)
                    centerX = np.rad2deg(np.float64(wcs.pixelToSky(xy)[0]))
                    centerY = np.rad2deg(np.float64(wcs.pixelToSky(xy)[1]))
                    inTract = False
                    if tractInfo is not None:
                        for coord in coords:
                            if tractInfo.contains(coord):
                                inTract = True
                                break
                    if not tractInfo or inTract:
                        if not legendAddedDict[wcsKey]:
                            axes.plot(ras, decs, color=lineColorDict[wcsKey],
                                      linestyle=lineStyleDict[wcsKey], linewidth=lineWidth, label=wcsKey)
                            legendAddedDict[wcsKey] = True
                        else:
                            axes.plot(ras, decs, color=lineColorDict[wcsKey],
                                      linestyle=lineStyleDict[wcsKey], linewidth=lineWidth)

                        lineWidth += 1/sum(x is not False for x in wcsDict.values())

                        if (wcsKey == "calexpWcs" and not wcsDict["uberWcs"]) or wcsKey == "uberWcs":
                            axes.text(centerX, centerY, "%s" % str(ccdLabelStr), ha="center", va="center",
                                      fontsize=fontSize, color="black")


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
                xCoords = xCoord + [xCoord[0], ]
                yCoords = yCoord + [yCoord[0], ]
                axes.plot(xCoords, yCoords, color="black", lw=0.5, linestyle="solid")
            if plotUnits in validWcsUnits:
                xCoord, yCoord = bboxToXyCoordLists(patch.getInnerBBox(), wcs=tractInfo.getWcs(),
                                                    wcsUnits=plotUnits)
            else:
                xCoord, yCoord = bboxToXyCoordLists(patch.getInnerBBox(), wcs=None)
            xCoords = xCoord + [xCoord[0], ]
            yCoords = yCoord + [yCoord[0], ]
            axes.plot(xCoords, yCoords, color="black", lw=0.6, linestyle=(0, (5, 4)))
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


def rotatePoint(x0, y0, xToRotate, yToRotate, rotationAngle):
    """Rotate point clockwise by the given angle about the given origin

    Parameters
    ----------
    x0, y0 : `float`
       The x & y coordinates of the origin about which to perform the rotation.
    xToRotate, yToRotate : `float`
       The x & y coordinates of the point to be rotated about the origin.
    rotationAngle : `float`
       The angle in radians by which to rotate point in a clockwise direction.
       (``xToRotate``, ``yToRotate``) about origin (``x0``, ``y0``).

    Returns
    -------
    xRoated, yRotated : `float`
       The rotated x and y coordinates.
    """
    xRotated = (np.cos(-rotationAngle)*(xToRotate - x0) -
                np.sin(-rotationAngle)*(yToRotate - y0) + x0)
    yRotated = (np.sin(-rotationAngle)*(xToRotate - x0) +
                np.cos(-rotationAngle)*(yToRotate - y0) + y0)
    return xRotated, yRotated


def bboxToXyCoordLists(bbox, detector=None, wcs=None, wcsUnits="deg"):
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
    if detector is not None:
        if wcs is not None:
            raise RuntimeError("wcs must be None if detector is set (i.e. want coord in focal plane pixels)")
        bboxCorners = detector.getCorners(cameraGeom.FOCAL_PLANE)
    else:
        bboxCorners = bbox.getCorners()
    for corner in bboxCorners:
        p = geom.Point2D(corner.getX(), corner.getY())
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
    return list(xCoords), list(yCorrds)


def getMinMaxPatchList(patchList, tractInfo, nDecimals=None, raMin=360.0, raMax=0.0, decMin=90.0,
                       decMax=-90.0, xMin=1e15, xMax=-1e15, yMin=1e15, yMax=-1e15):
    """Find the min/max boundaries encompased in the patchList

    Results are provided in RA/Dec (deg) and tract x/y (pixels)

    Parameters
    ----------
    patchList : `list` of `str`
       List of patch IDs.
    tractInfo : `lsst.skymap.tractInfo.ExplicitTractInfo`
       Tract information associated with the patches in ``patchList``.
    nDecimals : `int`, optional
       Round coordinates to this number of decimal places.
    raMin, raMax : `float`, optional
       Initiate minimum[maximum] RA determination in degrees at
       ``raMin``[``raMax``].
    decMin, decMax : `float`, optional
       Initiate minimum[maximum] Dec determination in degrees at
       ``decMin``[``decMax``].
    xMin, xMax : `float`, optional
       Initiate minimum[maximum] x determination in pixels at
       ``xMin``[``xMax``].
    yMin, yMax : `float`, optional
       Initiate minimum[maximum] y determination in pixels at
       ``yMin``[``yMax]``.

    Returns
    -------
    `lsst.pipe.base.Struct`
       Contains the min and max values for the patch list provided in
       RA/Dec (degrees) and tract x/y (pixels).
    """
    for patch in tractInfo:
        if str(patch.getIndex()[0]) + "," + str(patch.getIndex()[1]) in patchList:
            xPatch, yPatch = bboxToXyCoordLists(patch.getOuterBBox(), wcs=None)
            xMin, xMax = min(min(xPatch), xMin), max(max(xPatch), xMax)
            yMin, yMax = min(min(yPatch), yMin), max(max(yPatch), yMax)
            raPatch, decPatch = bboxToXyCoordLists(patch.getOuterBBox(), wcs=tractInfo.getWcs())
            raMin, raMax = min(min(raPatch), raMin), max(max(raPatch), raMax)
            decMin, decMax = min(min(decPatch), decMin), max(max(decPatch), decMax)
        if nDecimals:
            raMin = np.round(raMin, nDecimals)
            raMax = np.round(raMax, nDecimals)
            decMin = np.round(decMin, nDecimals)
            decMax = np.round(decMax, nDecimals)

    return Struct(
        xMin=xMin,
        xMax=xMax,
        yMin=yMin,
        yMax=yMax,
        raMin=raMin,
        raMax=raMax,
        decMin=decMin,
        decMax=decMax,
    )


def getMinMaxCcdList(ccdList, dataId, butler, fpUnits="pixels", nDecimals=None, zpLabel=None,
                     raMin=360.0, raMax=0.0, decMin=90.0, decMax=-90.0, xMin=1e15, xMax=-1e15,
                     yMin=1e15, yMax=-1e15):
    """Find the min/max boundaries encompased in the ccdList

    Results are provided in RA/Dec (deg) and tract x/y (pixels)

    Parameters
    ----------
    ccdList : `list` of `str`
       List of CCD IDs.
    dataId : `lsst.daf.persistence.DataId`
       An instance of `lsst.daf.persistence.DataId`.  A copy will be made
       and the "detector/ccd" key will be used to update the copy to the
       current ccd when looping through ``ccdList``.
    butler : `lsst.daf.persistence.Butler`
    nDecimals : `int`, optional
       Round coordinates to this number of decimal places.
    raMin, raMax : `float`, optional
       Initiate minimum[maximum] RA determination in degrees at
       ``raMin``[``raMax``].
    decMin, decMax : `float`, optional
       Initiate minimum[maximum] Dec determination in degrees at
       ``decMin``[``decMax``].
    xMin, xMax : `float`, optional
       Initiate minimum[maximum] x determination in focal plane pixels at
       ``xMin``[``xMax``].
    yMin, yMax : `float`, optional
       Initiate minimum[maximum] y determination in focal plane pixels at
       ``yMin``[``yMax``].

    Returns
    -------
    `lsst.pipe.base.Struct`
       Contains the min and max values for the ccd list provided in
       RA/Dec (degrees) and focal plane x/y (pixels).
    """
    dataIdCopy = dataId.copy()
    dataIdCopy = popIdAndCcdKeys(dataIdCopy)
    ccdKey = findCcdKey(dataId)
    for ccd in ccdList:
        dataIdCopy[ccdKey] = ccd
        calexp = butler.get("calexp", dataIdCopy)
        detector = calexp.getDetector()
        dataRef = butler.dataRef("raw", dataId=dataIdCopy)
        if zpLabel and (zpLabel == "MEAS_MOSAIC" or "MEAS_MOSAIC_1" in zpLabel):
            applyMosaicResultsExposure(dataRef, calexp=calexp)
            wcs = calexp.getWcs()
        elif zpLabel and ("JOINTCAL" in zpLabel or "MMphotoCalib" in zpLabel or "JOINTCAL_1" in zpLabel):
            wcs = dataRef.get("jointcal_wcs")
        else:
            wcs = calexp.getWcs()

        xCcd, yCcd = bboxToXyCoordLists(calexp.getBBox(), detector=detector, wcs=None)
        if "pix" in fpUnits:
            for i in enumerate(xCcd):
                xCcd[i[0]] = xCcd[i[0]]//detector.getPixelSize()[0]
                yCcd[i[0]] = yCcd[i[0]]//detector.getPixelSize()[1]
        xMin, xMax = min(min(xCcd), xMin), max(max(xCcd), xMax)
        yMin, yMax = min(min(yCcd), yMin), max(max(yCcd), yMax)
        raCcd, decCcd = bboxToXyCoordLists(calexp.getBBox(), wcs=wcs)
        raMin, raMax = min(min(raCcd), raMin), max(max(raCcd), raMax)
        decMin, decMax = min(min(decCcd), decMin), max(max(decCcd), decMax)
        if nDecimals:
            raMin = np.round(raMin, nDecimals)
            raMax = np.round(raMax, nDecimals)
            decMin = np.round(decMin, nDecimals)
            decMax = np.round(decMax, nDecimals)

    return Struct(
        xMin=xMin,
        xMax=xMax,
        yMin=yMin,
        yMax=yMax,
        raMin=raMin,
        raMax=raMax,
        decMin=decMin,
        decMax=decMax,
    )


def computeEqualAspectLimits(xMin, xMax, yMin, yMax, percentPad=0.0):
    """Compute equal aspect plotting limits given xMin, xMax, yMin, yMax values

    Also allow for a padding of a fractional percent of the maximum deltaX/deltaY
    range to to be applied to the limits.

    Parameters
    ----------
    xMin, xMax : `float`
       Minimum and maximum \"y\" coordinate.
    yMin, yMax : `float`
       Minimum and maximum \"y\" coordinate.
    percentPad : `float`, optional
       A percentage amount by which to pad the axis limits on all four sides
       (5.0% by default).

    Returns
    -------
    `lsst.pipe.base.Struct`
       Contains the min and max values that provide equal aspect limits
       for the two axes based on the maximum range of the coordinate limits
       provided by ``xMin``, ``xMax``, ``yMin``, ``yMax``.  These values are
       padded by the percentage of the plot range provided by ``percentPad``.
    """
    deltaX, deltaY = xMax - xMin, yMax - yMin
    deltaPix = (1.0 + percentPad/100.0)*max(deltaX, deltaY)
    xPlotMin = (xMin + deltaX/2.0) - deltaPix/2.0
    xPlotMax = (xMin + deltaX/2.0) + deltaPix/2.0
    yPlotMin = (yMin + deltaY/2.0) - deltaPix/2.0
    yPlotMax = (yMin + deltaY/2.0) + deltaPix/2.0
    return Struct(
        xPlotMin=xPlotMin,
        xPlotMax=xPlotMax,
        yPlotMin=yPlotMin,
        yPlotMax=yPlotMax,
    )


def percent(values, p=0.5):
    """Return a value a faction of the way between the min and max values in a list."""
    m = min(values)
    interval = max(values) - m
    return m + p*interval


def setPtSize(num, ptSize=12):
    """Set the point size according to the size of the catalog"""
    if num > 10:
        ptSize = min(12, max(3, int(20/np.log10(num))))
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
    image = afwImage.ImageF(geom.ExtentI(tractBbox.getMaxX() + 1, tractBbox.getMaxY() + 1))
    image.array[:] = tractArray
    return image


def buildVisitImage(butler, dataId, camera, ccdList=None, dataType="calexp", dimensionLimit=4000,
                    doApplyExternalPhotoCalib=False, photoCalibType="jointcal", asMaskedImage=False):
    """Build up an image of an entire visit or list of ccds

    Parameters
    ----------
    butler : `lsst.daf.persistence.Butler`
       The butler associated with the dataset.
    dataId : `lsst.daf.persistence.DataId`
       An instance of `lsst.daf.persistence.DataId` from which to extract the
       filter name.
    camera : `lsst.afw.cameraGeom.Camera`
       The camera associated with the observation from which to build the
       visit image.
    ccdList : `list` of `str`, optional
       A list of the CCDs to include.  If `None`, the full list of CCDs
       in ``tractInfo`` will be included.
    dataType : `str`, optional
       The base name of the dataset (e.g. "raw" or "calexp").
       Default is "calexp".
    dimensionLimit : `int`, optional
       The mamximum dimension on a side in pixels.  The image will be binned
       by a factor of ceil(``dimensionLimit``/``dimensionMax``), where
       ``dimensionMax`` is the maximum dimention in Focal Plane pixels
       spanned by the ccds in ``ccdList``.  The maximum this value can
       be (per `lsst.afw.image.Image` limits) is 2**15 = 32768
    doApplyExternalPhotoCalib : `bool`, optional
       Whether to apply an external photometric calibration to the image.
       Default is `False`.
    photoCalibType : `str`, optional
       The butler-recognized datatype of the uber-calibration to apply.
       Currently, the only options are "jointcal", "fgcmcal", and
       "fgcmcal_tract" (but, to date, the latter two are still undergoing
       integration testing and only for obs_subaru).  Default is "jointcal".
    asMaskedImage : `bool`, optional
       Set to `True` if the desired return image type is
       `lsst.afw.image.MaskedImageF`.  Default is `False`.

    Returns
    -------
    image : `lsst.afw.image.ImageF` or `lsst.afw.image.MaskedImageF`
       Image of the full visit or just the CCDs in ``ccdList``.
    """
    if not ccdList:
        ccdList = []
        for ccd in camera:
            if ccd.getType() == cameraGeom.DetectorType.SCIENCE:
                ccdList.append(ccd.getId())

    visitLimits = getMinMaxCcdList(ccdList, dataId, butler, fpUnits="pixels")
    pixMin = geom.Point2I(int(visitLimits.xMin), int(visitLimits.yMin))
    pixMax = geom.Point2I(int(visitLimits.xMax), int(visitLimits.yMax))
    dimensionMax = max(pixMax[0] - pixMin[0], pixMax[1] - pixMin[1])
    binSize = int(-(dimensionMax//-dimensionLimit)) if dimensionMax > dimensionLimit else 1
    # DECam has a mismatch between the "detector" and "ccd/calexp" bboxes (both the rows and cols of
    # the latter are smaller by 2 pixels).  Need to bin by at least 10 to "avoid" this problem
    # in the cgUtils functions
    binSize = max(binSize, 10) if camera.getName() == "DECam" else binSize
    if any(isinstance(ccd, str) for ccd in ccdList):
        ccdList = list(map(int, ccdList))
    if doApplyExternalPhotoCalib:
        callback = (lambda im, ccd, imageSource:
                    cgUtils.applyExternalPhotoCalibCallback(im, ccd, imageSource,
                                                            photoCalibType=photoCalibType))
    else:
        callback = cgUtils.rawCallback if dataType == "raw" else None
    imageSource = cgUtils.ButlerImage(butler, dataType, callback=callback, verbose=True,
                                      visit=dataId["visit"], tract=dataId["tract"])
    imageFactory = afwImage.MaskedImageF if asMaskedImage else afwImage.ImageF
    visitImage = cgUtils.makeImageFromCamera(camera, detectorNameList=ccdList, background=0.0, bufferSize=0,
                                             imageSource=imageSource, imageFactory=imageFactory,
                                             binSize=binSize, asMaskedImage=asMaskedImage)

    visitArray = visitImage.image.array if hasattr(visitImage, "image") else visitImage.array
    visitArray = np.flipud(visitArray)
    if hasattr(visitImage, "image"):
        visitImage.image.array[:] = visitArray
    else:
        visitImage.array[:] = visitArray
    visitImage.setXY0(geom.Point2I(pixMin[0], pixMin[1]))
    return visitImage, binSize


def getArrayFromImage(image):
    """Extract the image array from any afwImage-like type

    Parameters
    ----------
    image : `lsst.afw.image.Image` or `lsst.afw.image.MaskedImage` or `lsst.afw.image.Exposure`
       Image from which to extract the `numpy.ndarray`.

    Returns
    -------
    imageArray : `numpy.ndarray` of `float`
       Array of ``image``.  If ``image`` is of type `numpy.ndarray`, just returns
       ``image`` itself.
    """
    if isinstance(image, np.ndarray):
        return image
    if hasattr(image, "image"):
        imageArray = image.image.array
    elif hasattr(image, "maskedImage"):
        imageArray = image.maskedImage.image.array
    else:
        imageArray = image.array
    return imageArray


def makeDiffImages(image1, image2):
    """Helper function to make direct and percentage difference images

    Parameters
    ----------
    image1, image2 : `lsst.afw.image.Image` or `lsst.afw.image.MaskedImage` or `lsst.afw.image.Exposure`
       Images from which to create the difference images.

    Raises
    ------
    `RuntimeError`
       If the bounding boxes of ``image1`` and ``image2`` do not match.

    Returns
    -------
    differenceImages : `lsst.pipe.base.Struct`
       The direct and percent difference images of ``image1`` and ``image2``.

       ``diffImage``
          The direct difference image (``image1 - ``image2``) (`lsst.afw.image.imageF`).
       ``percentDiffImage``
          The percent difference image (`lsst.afw.image.imageF`).
    """
    if not image1.getBBox() == image2.getBBox():
        raise RuntimeError("Bounding boxes of two input images must match (image1 bbox = {0:} "
                           "vs. image2 bbox {1:}".format(image1.getBBox(), image2.getBBox()))
    diffImage = afwImage.ImageF(image1.getBBox())
    imageArray1 = getArrayFromImage(image1)
    imageArray2 = getArrayFromImage(image2)
    diffImage.array[:] = imageArray1 - imageArray2
    percentDiffImage = afwImage.ImageF(image1.getBBox())
    percentDiffImage.array[:] = 200*(imageArray1 - imageArray2)/(imageArray1 + imageArray2)

    return Struct(
        diffImage=diffImage,
        percentDiffImage=percentDiffImage,
    )


def makeAsinhNormFromArray(imageArray, nMinDev=2.5, nMaxDev=8.0, Q=8.0):
    """Create an Asinh normalization for a given image array

    The minimum and maximum values used for the normalization are set by
    the (user definable) number of "standard deviations" (the factor 1.4826
    assumes normally distributed data) below and above the mean absolute
    deviation (imMad) of the image array, respectively.

    Parameters
    ----------
    imageArray : `numpy.ndarray`
       Image array from which to calculate the Asinh normalization.
    nMinDev : `float`, optional
       Number of deviations below 1.4826*imMad of the image to set the scaling
       minimum value.  Default is 2.5.
    nMinDev : `float`, optional
       Number of deviations above 1.4826*imMad of the image to set the scaling
       maximum value.  Default is 8.0.
    Q : `float`, optional
       The asinh softening parameter for asinh stretch.  Use Q=0 for linear stretch,
       increase Q to make brighter features visible.  Default is 8.0.

    Returns
    -------
    asinhNormalization : `lsst.pipe.base.Struct`
       The asinh nomalization with attrubutes:

       ``norm``
          The asinh normalization
          (`lsst.display.matplotlib.matplotlib.AsinhNormalize`).
       ``imMin``
          The minimum value used in the data range for the normalization
          (`float`).
       ``imMax``
          The maximum value used in the data range for the normalization
          (`float`).
    """
    imMed = np.nanmedian(imageArray)
    imMad = np.nanmedian(abs(imageArray - imMed))
    imMin = imMed - nMinDev*1.4826*imMad
    imMax = imMed + nMaxDev*1.4826*imMad
    imMin = -1e-3 if imMin == 0.0 else imMin
    imMax = 1e-3 if imMax == 0.0 else imMax
    norm = AsinhNormalize(minimum=imMin, dataRange=imMax - imMin, Q=Q)

    return Struct(
        norm=norm,
        imMin=imMin,
        imMax=imMax,
        imMed=imMed,
        imMad=imMad,
    )


def determineUberCalLabel(repoInfo, patch, coaddName="deep"):
    """Determine uber-calibration (meas_mosaic/jointcal) applied to make coadd.

    Parameters
    ----------
    repoInfo : `lsst.pipe.base.struct.Struct`
       A struct containing elements with repo information needed to create
       appropriate dataIds to look for the uber-calibration datasets.
    patch : `str`
       An existing patch to use in the coaddDataId.
    coaddName : `str`, optional
       The base name of the coadd (e.g. "deep" or "goodSeeing").
       Default is "deep".

    Returns
    -------
    uberCalLabel : `str`
       The label to be used for the uberCal used.
    """
    # Find a visit/ccd input so that you can check for meas_mosaic input (i.e. to set uberCalLabel)
    coaddDataId = {"tract": repoInfo.tractInfo.getId(), "patch": patch, "filter": repoInfo.filterName}
    coadd = repoInfo.butler.get(coaddName + "Coadd_calexp", coaddDataId, immediate=True)
    coaddInputs = coadd.getInfo().getCoaddInputs()
    try:
        visitDataId = {"visit": coaddInputs.ccds[0]["visit"], "ccd": coaddInputs.ccds[0]["ccd"],
                       "filter": repoInfo.filterName, "tract": repoInfo.tractInfo.getId()}
        repoInfo.butler.datasetExists("jointcal_photoCalib", dataId=visitDataId)
    except Exception:  # The above will throw if ccd is not a valid dataId key, try detector instead
        visitDataId = {"visit": coaddInputs.ccds[0]["visit"], "detector": coaddInputs.ccds[0]["ccd"],
                       "filter": repoInfo.filterName, "tract": repoInfo.tractInfo.getId()}
        repoInfo.butler.datasetExists("jointcal_photoCalib", dataId=visitDataId)

    if repoInfo.butler.datasetExists("fcr_md", dataId=visitDataId):
        uberCalLabel = "MEAS_MOSAIC"
    elif (not repoInfo.butler.datasetExists("fcr_md", dataId=visitDataId) and
          repoInfo.butler.datasetExists("jointcal_photoCalib", dataId=visitDataId)):
        uberCalLabel = "JOINTCAL"
    else:
        uberCalLabel = "None"

    return uberCalLabel


def plotDirectionArrows(axes, butler, dataId, camera, xPlotMin, xPlotMax, yPlotMin, yPlotMax):
    """Plot direction arrows according to the boresight rotation angle
    This assumes (the LSST defined) FP projected as:
          +x: E->W (-ve RA), +y: S->N (+ve Dec)
    at a boresight rotation angle of 0.

    Parameters
    ----------
    axes : `matplotlib.axes._axes.Axes`
       Particular matplotlib axes of ``plt`` on which to plot the annotations.
    butler : `lsst.daf.persistence.Butler`
       The butler associated with the dataset.
    dataId : `lsst.daf.persistence.DataId`
       An instance of `lsst.daf.persistence.DataId` from which to extract the
       filter name.
    camera : `lsst.afw.cameraGeom.Camera`
       The camera associated with the observation from which to build the
       visit image.
    xPlotMin, xPlotMax, yPlotMin, yPlotMax : `float`
       Plotting limits for the \"x\" and \"y\" coordinates.
    """
    calexp = butler.get("calexp", dataId)
    boresightRotAng = calexp.getInfo().getVisitInfo().getBoresightRotAngle().asRadians()
    dPix = int(0.08*max((xPlotMax - xPlotMin), (yPlotMax - yPlotMin)))
    xy0 = (xPlotMin + 1.5*dPix, yPlotMax - 1.6*dPix)
    # The following just moves the arrows to be as close to the upper left
    # corner as possible based on the boresight rotation angel.
    xy0 = (xy0[0] - abs(np.sin(boresightRotAng)*np.cos(boresightRotAng/3.0))*dPix,
           xy0[1] + abs(np.sin(boresightRotAng/6.0)*np.cos(np.pi/2.0 - boresightRotAng))*dPix)
    xyNorth = (xy0[0], xy0[1] + dPix)
    xyEast = (xy0[0] - dPix, xy0[1])
    rotatedNorth = (rotatePoint(xy0[0], xy0[1], xyNorth[0], xyNorth[1], boresightRotAng))
    rotatedEast = (rotatePoint(xy0[0], xy0[1], xyEast[0], xyEast[1], boresightRotAng))
    deltaNx, deltaNy = rotatedNorth[0] - xy0[0], rotatedNorth[1] - xy0[1]
    deltaEx, deltaEy = rotatedEast[0] - xy0[0], rotatedEast[1] - xy0[1]
    deltaFrac = 0.06
    axes.plot(xy0[0], xy0[1], markersize=3, marker="o", color="black")
    arrowKwargs = dict(xycoords="data", textcoords="data", ha="center", va="center",
                       arrowprops=dict(arrowstyle="<|-", facecolor="springgreen"))
    axes.annotate("", xy=xy0, xytext=rotatedNorth, **arrowKwargs)
    # DECam has flipX = True, but can't access this info until RFC-605 is implemented
    # on DM-20746
    if camera.getName() == "DECam":
        axes.annotate("S", xy=(rotatedNorth[0] + deltaFrac*deltaNx,
                               rotatedNorth[1] + deltaFrac*deltaNy), fontsize=7, **arrowKwargs)
    else:
        axes.annotate("N", xy=(rotatedNorth[0] + deltaFrac*deltaNx,
                               rotatedNorth[1] + deltaFrac*deltaNy), fontsize=7, **arrowKwargs)
    axes.annotate("", xy=xy0, xytext=rotatedEast, **arrowKwargs)
    axes.annotate("E", xy=(rotatedEast[0] + deltaFrac*deltaEx,
                           rotatedEast[1] + deltaFrac*deltaEy), fontsize=7, **arrowKwargs)
    return axes


def plotSkyLimitLabels(axes, raDecProj, plt, butler=None, dataId=None, camera=None, ccdList=None,
                       tractInfo=None, xPlotMin=None, xPlotMax=None, yPlotMin=None, yPlotMax=None,
                       raDecMin=None, raDecMax=None, filterLabelStr=""):
    """Plot limits in pixel and/or RA/Dec units

    Parameters
    ----------
    axes : `matplotlib.axes._axes.Axes`
       Particular matplotlib axes of ``plt`` on which to plot the annotations.
    plt : `matplotlib.pyplot`
       Instance of matplotlib pyplot to plot ``textStr``.
    butler : `lsst.daf.persistence.Butler`, optional
       The butler associated with the dataset.
    dataId : `lsst.daf.persistence.DataId`, optional
       An instance of `lsst.daf.persistence.DataId` from which to extract the
       filter name.
    camera : `lsst.afw.cameraGeom.Camera`, optional
       The camera associated with the observation from which to build the
       visit image.
    xPlotMin, xPlotMax, yPlotMin, yPlotMax : `float`, optional
       Plotting limits for the \"x\" and \"y\" coordinates.
    """
    if raDecProj:
        axes.set_xlabel("RA (deg) {0:s}".format(filterLabelStr))
        axes.set_ylabel("Dec (deg) {0:s}".format(filterLabelStr))
    else:
        axes.set_xlabel("x (pixels) {0:s}".format(filterLabelStr), size=7, labelpad=6)
        axes.set_ylabel("y (pixels) {0:s}".format(filterLabelStr), size=7)
        # Get RA and Dec tract/visit limits to add to plot axis labels
        if tractInfo is not None and ccdList is None:
            tract00 = tractInfo.getWcs().pixelToSky(xPlotMin, yPlotMin).getPosition(units=geom.degrees)
            tract0N = tractInfo.getWcs().pixelToSky(xPlotMin, yPlotMax).getPosition(units=geom.degrees)
            tractN0 = tractInfo.getWcs().pixelToSky(xPlotMax, yPlotMin).getPosition(units=geom.degrees)
            plot00 = (tract00.getX(), tract00.getY())
            plot0N = (tract0N.getX(), tract0N.getY())
            plotN0 = (tractN0.getX(), tractN0.getY())

        xCoordStr = "RA (deg)"
        yCoordStr = "Dec (deg)"
        if ccdList is not None and not raDecProj:
            axes = plotDirectionArrows(axes, butler, dataId, camera, xPlotMin, xPlotMax,
                                       yPlotMin, yPlotMax)
            # If boresight is rotated a factor of 90 deg, print the RA and Dec
            # at the min/max positions of the data
            calexp = butler.get("calexp", dataId)
            boresightRotAng = calexp.getInfo().getVisitInfo().getBoresightRotAngle().asRadians()
            if np.degrees(boresightRotAng)%90 == 0:
                if np.degrees(boresightRotAng)%180 != 0:
                    xCoordStr = "Dec (deg)"
                    yCoordStr = "RA (deg)"
                    plot00 = geom.Point2D(raDecMin[1], raDecMin[0])
                    plot0N = geom.Point2D(raDecMin[1], raDecMax[0])
                    plotN0 = geom.Point2D(raDecMax[1], raDecMin[0])
                else:
                    plot00 = raDecMin
                    plot0N = geom.Point2D(raDecMin[0], raDecMax[1])
                    plotN0 = geom.Point2D(raDecMax[0], raDecMin[1])

        textKwargs = dict(ha="center", va="center", transform=axes.transAxes, fontsize=6, color="blue")
        plt.text(0.00, -0.06, str("{:.2f}".format(plot00[0])), **textKwargs)
        plt.text(-0.16, 0.00, str("{:.2f}".format(plot00[1])), **textKwargs)
        plt.text(0.99, -0.06, str("{:.2f}".format(plotN0[0])), **textKwargs)
        plt.text(-0.16, 0.99, str("{:.2f}".format(plot0N[1])), **textKwargs)
        plt.text(0.5, -0.11, xCoordStr, **textKwargs)
        plt.text(-0.18, 0.5, yCoordStr, rotation=90, **textKwargs)
    return axes
