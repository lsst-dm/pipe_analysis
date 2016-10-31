import matplotlib.patches as patches
import numpy as np

import lsst.afw.cameraGeom as cameraGeom
import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable
from lsst.meas.mosaic.updateExposure import applyMosaicResultsExposure

__all__ = ["AllLabeller", "StarGalaxyLabeller", "OverlapsStarGalaxyLabeller",
           "MatchesStarGalaxyLabeller", "CosmosLabeller", "labelZp", "annotateAxes", "labelVisit",
           "plotCameraOutline", "plotTractOutline", "plotPatchOutline", "plotCcdOutline",
           "rotatePixelCoords", "bboxToRaDec", "percent", "setPtSize"]

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


def labelZp(zpLabel, plt, axis, xLoc, yLoc, color="k"):
    plt.text(xLoc, yLoc, "zp: " + zpLabel, ha="center", va="center", fontsize=11,
             transform=axis.transAxes, color=color)

def annotateAxes(plt, axes, stats, dataSet, magThreshold, x0=0.03, y0=0.96, yOff=0.045,
                 ha="left", va="top", color="blue", isHist=False, hscRun=None, matchRadius=None):
    xOffFact = 0.64*len(" N = {0.num:d} (of {0.total:d})".format(stats[dataSet]))
    axes.annotate(dataSet+r" N = {0.num:d} (of {0.total:d})".format(stats[dataSet]),
                  xy=(x0, y0), xycoords="axes fraction", ha=ha, va=va, fontsize=10, color="blue")
    axes.annotate(r"[mag<{0:.1f}]".format(magThreshold), xy=(x0*xOffFact, y0), xycoords="axes fraction",
                  ha=ha, va=va, fontsize=10, color="k", alpha=0.55)
    axes.annotate("mean = {0.mean:.4f}".format(stats[dataSet]), xy=(x0, y0-yOff),
                  xycoords="axes fraction", ha=ha, va=va, fontsize=10)
    axes.annotate("stdev = {0.stdev:.4f}".format(stats[dataSet]), xy=(x0, y0-2*yOff),
                  xycoords="axes fraction", ha=ha, va=va, fontsize=10)
    yOffMult = 3
    if matchRadius is not None:
        axes.annotate("Match radius = {0:.2f}\"".format(matchRadius), xy=(x0, y0-yOffMult*yOff),
                      xycoords="axes fraction", ha=ha, va=va, fontsize=10)
        yOffMult += 1
    if hscRun is not None:
        axes.annotate("HSC stack run: {0:s}".format(hscRun), xy=(x0, y0-yOffMult*yOff),
                      xycoords="axes fraction", ha=ha, va=va, fontsize=10, color="#800080")
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

def labelVisit(filename, plt, axis, xLoc, yLoc, color="k"):
    labelStr = None
    if filename.find("visit-") >= 0:
        labelStr = "visit"
    if filename.find("tract-") >= 0:
        labelStr = "tract"
    if labelStr is not None:
        i1 = filename.find(labelStr + "-") + len(labelStr + "-")
        i2 = filename.find("/", i1)
        visitNumber = filename[i1:i2]
        plt.text(xLoc, yLoc, labelStr + ": " + str(visitNumber), ha="center", va="center", fontsize=12,
                 transform=axis.transAxes, color=color)

def plotCameraOutline(plt, axes, camera, ccdList):
    axes.tick_params(labelsize=6)
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
            # axes.text(ccdCenter.getX(), ccdCenter.getY(), ccd.getId(),
            #                ha="center", fontsize=2)
    axes.set_title("%s CCDs" % camera.getName(), fontsize=6)
    axes.set_xlim(-camLimits, camLimits)
    axes.set_ylim(-camLimits, camLimits)
    axes.add_patch(patches.Circle((0, 0), radius=camRadius, color="black", alpha=0.2))
    for x, y, t in ([-1, 0, "N"], [0, 1, "W"], [1, 0, "S"], [0, -1, "E"]):
        axes.text(1.08*camRadius*x, 1.08*camRadius*y, t, ha="center", va="center", fontsize=6)

def plotTractOutline(axes, tractInfo, patchList):
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
                  fontsize=5, horizontalalignment="center", verticalalignment="center")
    axes.text(percent(tractRa, 0.5), 2.0*percent(tractDec, 0.0) - percent(tractDec, 0.18), "RA (deg)",
              fontsize=6, horizontalalignment="center", verticalalignment="center")
    axes.text(2*percent(tractRa, 1.0) - percent(tractRa, 0.78), percent(tractDec, 0.5), "Dec (deg)",
              fontsize=6, horizontalalignment="center", verticalalignment="center",
              rotation="vertical")
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)

def plotCcdOutline(axes, butler, dataId, ccdList, zpLabel=None):
    """!Plot outlines of CCDs in ccdList
    """
    dataIdCopy = dataId.copy()
    for ccd in ccdList:
        dataIdCopy["ccd"] = ccd
        calexp = butler.get("calexp", dataIdCopy)
        dataRef = butler.dataRef("raw", dataId=dataIdCopy)
        if zpLabel == "MEAS_MOSAIC":
            result = applyMosaicResultsExposure(dataRef, calexp=calexp)

        wcs = calexp.getWcs()
        w = calexp.getWidth()
        h = calexp.getHeight()

        ras = list()
        decs = list()
        for x, y in zip([0, w, w, 0, 0], [0, 0, h, h, 0]):
            xy = afwGeom.Point2D(x, y)
            ra = np.rad2deg(np.float64(wcs.pixelToSky(xy)[0]))
            dec = np.rad2deg(np.float64(wcs.pixelToSky(xy)[1]))
            ras.append(ra)
            decs.append(dec)
        axes.plot(ras, decs, "k-")
        xy = afwGeom.Point2D(w/2, h/2)
        centerX = np.rad2deg(np.float64(wcs.pixelToSky(xy)[0]))
        centerY = np.rad2deg(np.float64(wcs.pixelToSky(xy)[1]))
        axes.text(centerX, centerY, "%i" % ccd, ha="center", va= "center", fontsize=9)

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
        x0 = s.get(xKey)
        y0 = s.get(yKey)
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
        ptSize = min(12, max(4, int(25/np.log10(num))))
    return ptSize
