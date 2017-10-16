import matplotlib.patches as patches
import numpy as np

import lsst.afw.cameraGeom as cameraGeom
import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable
from lsst.meas.mosaic.updateExposure import applyMosaicResultsExposure

from .utils import checkHscStack

__all__ = ["AllLabeller", "StarGalaxyLabeller", "OverlapsStarGalaxyLabeller", "MatchesStarGalaxyLabeller",
           "CosmosLabeller", "labelZp", "annotateAxes", "labelVisit", "labelCamera",
           "filterStrFromFilename", "plotCameraOutline", "plotTractOutline", "plotPatchOutline",
           "plotCcdOutline", "rotatePixelCoords", "bboxToRaDec", "percent", "setPtSize", "getQuiver"]

class AllLabeller(object):
    labels = {"all": 0}
    plot = ["all"]
    def __call__(self, catalog):
        return np.zeros(len(catalog))

class StarGalaxyLabeller(object):
    labels = {"star": 0, "galaxy": 1}
    plot = ["star"]
    _column = "base_ClassificationExtendedness_value"
    def __call__(self, catalog):
        return np.where(catalog[self._column] < 0.5, 0, 1)

class OverlapsStarGalaxyLabeller(StarGalaxyLabeller):
    labels = {"star": 0, "galaxy": 1, "split": 2}
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


def labelZp(zpLabel, plt, axis, xLoc, yLoc, rotation=0, fontSize=9, color="k"):
    fontSize = int(fontSize - min(3, len(zpLabel)/10))
    plt.text(xLoc, yLoc, "zp: " + zpLabel, ha="center", va="center", fontsize=fontSize, rotation=rotation,
             transform=axis.transAxes, color=color)

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
        if any (ss in filename for ss in ["_ra", "_dec", "distance"]):
            statsUnitStr = " (mas)"
        if any (ss in filename for ss in ["mag_", "_photometry", "matches_mag"]):
            statsUnitStr = " (mmag)"
    lenStr = 0.12 + 0.017*(max(len(meanStr), len(stdevStr)))

    axes.annotate("mean = ", xy=(x0 + 0.12, y0 -yOff),
                  xycoords="axes fraction", ha="right", va=va, fontsize=fontSize, color="k")
    axes.annotate(meanStr, xy=(x0 + lenStr, y0 - yOff),
                  xycoords="axes fraction", ha="right", va=va, fontsize=fontSize, color="k")
    if statsUnitStr is not None:
        axes.annotate(statsUnitStr, xy=(x0 + lenStr + 0.006, y0 - yOff),
                      xycoords="axes fraction", ha="left", va=va, fontsize=fontSize, color="k")
    axes.annotate("stdev = ", xy=(x0 + 0.12, y0- 2*yOff),
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
        l3 = axes.axvline(stats[dataSet].median-stats[dataSet].clip, linestyle="dashdot", color="0.7")
    else:
        l1 = axes.axhline(stats[dataSet].median, linestyle="dotted", color="0.7", label="median")
        l2 = axes.axhline(stats[dataSet].median+stats[dataSet].clip, linestyle="dashdot", color="0.7",
                          label="clip")
        l3 = axes.axhline(stats[dataSet].median-stats[dataSet].clip, linestyle="dashdot", color="0.7")
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
    axes.tick_params(which="both", direction="in", labelleft="off", labelbottom="off")
    axes.locator_params(nbins=6)
    axes.ticklabel_format(useOffset=False)
    camRadius = max(camera.getFpBBox().getWidth(), camera.getFpBBox().getHeight())/2
    camRadius = np.round(camRadius, -2)
    camLimits = np.round(1.25*camRadius, -2)
    for ccd in camera:
        if ccd.getId() in ccdList:
            ccdCorners = ccd.getCorners(cameraGeom.FOCAL_PLANE)
            ccdCenter = ccd.getCenter(cameraGeom.FOCAL_PLANE).getPoint()
            plt.gca().add_patch(patches.Rectangle(ccdCorners[0], *list(ccdCorners[2] - ccdCorners[0]),
                                                  fill=True, facecolor="y", edgecolor="k", ls="solid"))
    axes.set_xlim(-camLimits, camLimits)
    axes.set_ylim(-camLimits, camLimits)
    axes.add_patch(patches.Circle((0, 0), radius=camRadius, color="k", alpha=0.2))
    for x, y, t in ([-1, 0, "N"], [0, 1, "W"], [1, 0, "S"], [0, -1, "E"]):
        axes.text(1.085*camRadius*x, 1.085*camRadius*y, t, ha="center", va="center", fontsize=fontSize - 1)
    axes.text(-0.82*camRadius, 0.95*camRadius, "%s" % camera.getName(), ha="center", fontsize=fontSize,
               color=color)

def plotTractOutline(axes, tractInfo, patchList, fontSize=6):
    buff = 0.02
    axes.tick_params(labelsize=6)
    axes.locator_params(nbins=6)
    axes.ticklabel_format(useOffset=False)

    tractRa, tractDec = bboxToRaDec(tractInfo.getBBox(), tractInfo.getWcs())
    xlim = max(tractRa) + buff, min(tractRa) - buff
    ylim = min(tractDec) - buff, max(tractDec) + buff
    axes.fill(tractRa, tractDec, fill=True, edgecolor='k', lw=1, linestyle='dashed',
              color="black", alpha=0.2)
    for ip, patch in enumerate(tractInfo):
        color = "k"
        alpha = 0.05
        if str(patch.getIndex()[0])+","+str(patch.getIndex()[1]) in patchList:
            color = ("r", "b", "c", "g", "m")[ip%5]
            alpha = 0.5
        ra, dec = bboxToRaDec(patch.getOuterBBox(), tractInfo.getWcs())
        axes.fill(ra, dec, fill=True, color=color, lw=1, linestyle="solid", alpha=alpha)
        ra, dec = bboxToRaDec(patch.getInnerBBox(), tractInfo.getWcs())
        axes.fill(ra, dec, fill=False, color=color, lw=1, linestyle="dashed", alpha=0.5*alpha)
        axes.text(percent(ra), percent(dec, 0.5), str(patch.getIndex()),
                  fontsize=fontSize - 1, horizontalalignment="center", verticalalignment="center")
    axes.text(percent(tractRa, 0.5), 2.0*percent(tractDec, 0.0) - percent(tractDec, 0.18), "RA (deg)",
              fontsize=fontSize, horizontalalignment="center", verticalalignment="center")
    axes.text(2*percent(tractRa, 1.0) - percent(tractRa, 0.78), percent(tractDec, 0.5), "Dec (deg)",
              fontsize=fontSize, horizontalalignment="center", verticalalignment="center",
              rotation="vertical")
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)

def plotCcdOutline(axes, butler, dataId, ccdList, zpLabel=None, fontSize=8):
    """!Plot outlines of CCDs in ccdList
    """
    dataIdCopy = dataId.copy()
    for ccd in ccdList:
        dataIdCopy["ccd"] = ccd
        calexp = butler.get("calexp", dataIdCopy)
        dataRef = butler.dataRef("raw", dataId=dataIdCopy)
        # Check metadata to see if stack used was HSC
        metadata = butler.get("calexp_md", dataIdCopy)
        hscRun = checkHscStack(metadata)
        if zpLabel == "MEAS_MOSAIC":
            result = applyMosaicResultsExposure(dataRef, calexp=calexp)

        wcs = calexp.getWcs()
        w = calexp.getWidth()
        h = calexp.getHeight()
        if hscRun and zpLabel == "MEAS_MOSAIC":
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
        axes.text(centerX, centerY, "%i" % ccd, ha="center", va= "center", fontsize=fontSize)

def plotPatchOutline(axes, tractInfo, patchList):
    """!Plot outlines of patches in patchList
    """
    idFontSize = max(5, 10 - int(0.4*len(patchList)))
    for ip, patch in enumerate(tractInfo):
        if str(patch.getIndex()[0])+","+str(patch.getIndex()[1]) in patchList:
            if len(patchList) < 9:
                ra, dec = bboxToRaDec(patch.getOuterBBox(), tractInfo.getWcs())
                ras = ra + (ra[0], )
                decs = dec + (dec[0], )
                axes.plot(ras, decs, color="black", lw=1, linestyle="solid")
            ra, dec = bboxToRaDec(patch.getInnerBBox(), tractInfo.getWcs())
            ras = ra + (ra[0], )
            decs = dec + (dec[0], )
            axes.plot(ras, decs, color="black", lw=1, linestyle="dashed")
            axes.text(percent(ras), percent(decs, 0.5), str(patch.getIndex()),
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

def bboxToRaDec(bbox, wcs):
    """Get the corners of a BBox and convert them to lists of RA and Dec."""
    corners = []
    for corner in bbox.getCorners():
        p = afwGeom.Point2D(corner.getX(), corner.getY())
        coord = wcs.pixelToSky(p).toIcrs()
        corners.append([coord.getRa().asDegrees(), coord.getDec().asDegrees()])
    ra, dec = zip(*corners)
    return ra, dec

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
    e = np.sqrt(e1**2 +e2**2)
    c1 = e*np.cos(theta)
    c2 = e*np.sin(theta)
    if color is None:
        color = e
    q = ax.quiver(x, y, c1, c2, color=color, angles='uv', scale=scale,
                  units='width', pivot='middle', width=width, headwidth=0.0,
                  headlength=0.0, headaxislength=0.0, label=label)

    return q
