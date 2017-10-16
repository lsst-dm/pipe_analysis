#!/usr/bin/env python

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all="ignore")
import functools

from collections import defaultdict

from lsst.pex.config import Config, Field, ConfigField, ListField, DictField, ConfigDictField
from lsst.pipe.base import CmdLineTask, ArgumentParser, TaskRunner
from lsst.coadd.utils import TractDataIdContainer
from .analysis import Analysis, AnalysisConfig
from .coaddAnalysis import CoaddAnalysisTask
from .utils import *
from .plotUtils import *

import lsst.afw.table as afwTable

__all__ = ["ColorTransform", "ivezicTransforms", "straightTransforms", "NumStarLabeller",
           "ColorValueInRange", "GalaxyColor", "ColorAnalysisConfig", "ColorAnalysisRunner",
           "ColorAnalysisTask", "ColorColorDistance", "SkyAnalysisRunner", "SkyAnalysisTask"]


class ColorTransform(Config):
    description = Field(dtype=str, doc="Description of the color transform")
    plot = Field(dtype=bool, default=True, doc="Plot this color?")
    coeffs = DictField(keytype=str, itemtype=float, doc="Coefficients for each filter")
    requireGreater = DictField(keytype=str, itemtype=float, default={},
                               doc="Minimum values for colors so that this is useful")
    requireLess = DictField(keytype=str, itemtype=float, default={},
                            doc="Maximum values for colors so that this is useful")

    @classmethod
    def fromValues(cls, description, plot, coeffs, requireGreater={}, requireLess={}):
        self = cls()
        self.description = description
        self.plot = plot
        self.coeffs = coeffs
        self.requireGreater = requireGreater
        self.requireLess = requireLess
        return self

ivezicTransforms = {
    "wPerp": ColorTransform.fromValues("Ivezic w perpendicular", True,
                                       {"HSC-G": -0.227, "HSC-R": 0.792, "HSC-I": -0.567, "": 0.050},
                                       {"wPara": -0.2}, {"wPara": 0.6}),
    "xPerp": ColorTransform.fromValues("Ivezic x perpendicular", True,
                                       {"HSC-G": 0.707, "HSC-R": -0.707, "": -0.988},
                                       {"xPara": 0.8}, {"xPara": 1.6}),
    "yPerp": ColorTransform.fromValues("Ivezic y perpendicular", True,
                                       {"HSC-R": -0.270, "HSC-I": 0.800, "HSC-Z": -0.534, "": 0.054},
                                       {"yPara": 0.1}, {"yPara": 1.2}),
    "wPara": ColorTransform.fromValues("Ivezic w parallel", False,
                                       {"HSC-G": 0.928, "HSC-R": -0.556, "HSC-I": -0.372, "": -0.425}),
    "xPara": ColorTransform.fromValues("Ivezic x parallel", False, {"HSC-R": 1.0, "HSC-I": -1.0}),
    "yPara": ColorTransform.fromValues("Ivezic y parallel", False,
                                       {"HSC-R": 0.895, "HSC-I": -0.448, "HSC-Z": -0.447, "": -0.600}),
}

straightTransforms = {
    "g": ColorTransform.fromValues("HSC-G", True, {"HSC-G": 1.0}),
    "r": ColorTransform.fromValues("HSC-R", True, {"HSC-R": 1.0}),
    "i": ColorTransform.fromValues("HSC-I", True, {"HSC-I": 1.0}),
    "z": ColorTransform.fromValues("HSC-Z", True, {"HSC-Z": 1.0}),
    "y": ColorTransform.fromValues("HSC-Y", True, {"HSC-Y": 1.0}),
    "n921": ColorTransform.fromValues("NB0921", True, {"NB0921": 1.0}),
}


class NumStarLabeller(object):
    labels = {"star": 0, "maybe": 1, "notStar": 2}
    plot = ["star"]
    def __init__(self, numBands):
        self.numBands = numBands
    def __call__(self, catalog):
        return np.array([0 if nn == self.numBands else 2 if nn == 0 else 1 for nn in catalog["numStarFlags"]])


class ColorValueInRange(object):
    """Functor to produce color value if in the appropriate range"""
    def __init__(self, column, requireGreater, requireLess):
        self.column = column
        self.requireGreater = requireGreater
        self.requireLess = requireLess

    def __call__(self, catalog):
        good = np.ones(len(catalog), dtype=bool)
        for col, value in self.requireGreater.items():
            good &= catalog[col] > value
        for col, value in self.requireLess.items():
            good &= catalog[col] < value
        return np.where(good, catalog[self.column], np.nan)


class GalaxyColor(object):
    """Functor to produce difference between galaxy color calculated by different algorithms"""
    def __init__(self, alg1, alg2, prefix1, prefix2):
        self.alg1 = alg1
        self.alg2 = alg2
        self.prefix1 = prefix1
        self.prefix2 = prefix2

    def __call__(self, catalog):
        color1 = -2.5*np.log10(catalog[self.prefix1 + self.alg1]/catalog[self.prefix2 + self.alg1])
        color2 = -2.5*np.log10(catalog[self.prefix1 + self.alg2]/catalog[self.prefix2 + self.alg2])
        return color1 - color2


class ColorAnalysisConfig(Config):
    coaddName = Field(dtype=str, default="deep", doc="Name for coadd")
    flags = ListField(dtype=str, doc="Flags of objects to ignore",
                      default=["base_SdssCentroid_flag", "base_PixelFlags_flag_saturatedCenter",
                               "base_PixelFlags_flag_interpolatedCenter", "base_PsfFlux_flag"])
    analysis = ConfigField(dtype=AnalysisConfig, doc="Analysis plotting options")
    transforms = ConfigDictField(keytype=str, itemtype=ColorTransform, default={},
                                 doc="Color transformations to analyse")
    fluxFilter = Field(dtype=str, default="HSC-I", doc="Filter to use for plotting against magnitude")
    srcSchemaMap = DictField(keytype=str, itemtype=str, default=None, optional=True,
                             doc="Mapping between different stack (e.g. HSC vs. LSST) schema names")

    def setDefaults(self):
        Config.setDefaults(self)
        self.transforms = ivezicTransforms
        self.analysis.flags = []  # We remove bad source ourself


class ColorAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        FilterRefsDict = functools.partial(defaultdict, list)  # Dict for filter-->dataRefs
        tractFilterRefs = defaultdict(FilterRefsDict)  # tract-->filter-->dataRefs
        for patchRef in sum(parsedCmd.id.refList, []):
            if patchRef.datasetExists("deepCoadd_forced_src"):
                tract = patchRef.dataId["tract"]
                filterName = patchRef.dataId["filter"]
                tractFilterRefs[tract][filterName].append(patchRef)

        # Find tract,patch with full colour coverage (makes combining catalogs easier)
        bad = []
        for tract in tractFilterRefs:
            filterRefs = tractFilterRefs[tract]
            patchesForFilters = [set(patchRef.dataId["patch"] for patchRef in patchRefList) for
                                 patchRefList in filterRefs.itervalues()]
            if len(patchesForFilters) == 0:
                parsedCmd.log.warn("No input data found for tract {:d}".format(tract))
                bad.append(tract)
                continue
            keep = set.intersection(*patchesForFilters)  # Patches with full colour coverage
            tractFilterRefs[tract] = {ff: [patchRef for patchRef in filterRefs[ff] if
                                           patchRef.dataId["patch"] in keep] for ff in filterRefs}
        for tract in bad:
            del tractFilterRefs[tract]

        return [(filterRefs, kwargs) for filterRefs in tractFilterRefs.itervalues()]


class ColorAnalysisTask(CmdLineTask):
    ConfigClass = ColorAnalysisConfig
    RunnerClass = ColorAnalysisRunner
    AnalysisClass = Analysis
    _DefaultName = "colorAnalysis"

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "deepCoadd_forced_src",
                               help="data ID, e.g. --id tract=12345 patch=1,2 filter=HSC-X",
                               ContainerClass=TractDataIdContainer)
        return parser

    def run(self, patchRefsByFilter):
        patchList = []
        for patchRefList in patchRefsByFilter.itervalues():
            for dataRef in patchRefList:
                if dataRef.dataId["filter"] == self.config.fluxFilter:
                    patchList.append(dataRef.dataId["patch"] )

        for patchRefList in patchRefsByFilter.itervalues():
            patchRef = patchRefList[0]
            butler = patchRef.getButler()
            dataId = patchRef.dataId
            break
        camera = butler.get("camera")

        # Check to see if stack used was HSC...need to use deepCoadd_forced_src cat because
        # deepCoadd_calexp is persisted to a different location in HSC stack...ughhh!
        forcedMd = butler.get("deepCoadd_forced_src", patchRef.dataId).getMetadata()
        hscRun = checkHscStack(forcedMd)

        skymap = butler.get("deepCoadd_skyMap")
        tractInfo = skymap[dataRef.dataId["tract"]]
        filenamer = Filenamer(butler, "plotColor", dataId)
        unforcedCatalogsByFilter = {ff: self.readCatalogs(patchRefList, "deepCoadd_meas") for
                            ff, patchRefList in patchRefsByFilter.items()}
        for cat in unforcedCatalogsByFilter.itervalues():
            calibrateCoaddSourceCatalog(cat, self.config.analysis.coaddZp)
        unforced = self.transformCatalogs(unforcedCatalogsByFilter, self.config.transforms, hscRun=hscRun)
        forcedCatalogsByFilter = {ff: self.readCatalogs(patchRefList, "deepCoadd_forced_src") for
                            ff, patchRefList in patchRefsByFilter.items()}
        for cat in forcedCatalogsByFilter.itervalues():
            calibrateCoaddSourceCatalog(cat, self.config.analysis.coaddZp)
        # self.plotGalaxyColors(catalogsByFilter, filenamer, dataId)
        forced = self.transformCatalogs(forcedCatalogsByFilter, self.config.transforms,
                                        flagsCats=unforcedCatalogsByFilter, hscRun=hscRun)

        # Create and write parquet tables
        tableFilenamer = Filenamer(butler, 'qaTableColor', dataId)
        writeParquet(forced, tableFilenamer(dataId, description='forced'))

        self.plotStarColors(forced, filenamer, NumStarLabeller(len(forcedCatalogsByFilter)), dataId,
                            camera=camera, tractInfo=tractInfo, patchList=patchList, hscRun=hscRun)
        self.plotStarColorColor(forcedCatalogsByFilter, filenamer, dataId, camera=camera, tractInfo=tractInfo,
                                patchList=patchList, hscRun=hscRun)

    def readCatalogs(self, patchRefList, dataset):
        catList = [patchRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS) for
                   patchRef in patchRefList if patchRef.datasetExists(dataset)]
        return concatenateCatalogs(catList)

    def transformCatalogs(self, catalogs, transforms, flagsCats=None, hscRun=None):
        """
        flagsCats: The forced catalogs do not contain the flags for selecting against objects,
                   so add this optional argument to provide a catalog list that does have these
                   flags.  The indentity of the object lists of flagsCat vs. the catalogs will
                   be checked.
        """
        if flagsCats is None:
            flagsCats = catalogs

        template = catalogs.values()[0]
        num = len(template)
        assert all(len(cat) == num for cat in catalogs.itervalues())

        mapper = afwTable.SchemaMapper(template.schema)
        mapper.addMinimalSchema(afwTable.SourceTable.makeMinimalSchema())
        schema = mapper.getOutputSchema()

        # Only adjust the schema names necessary here (rather than attaching the full alias schema map)
        self.fluxColumn = self.config.analysis.fluxColumn
        self.classificationColumn = "base_ClassificationExtendedness_value"
        self.flags = self.config.flags
        if hscRun is not None:
            self.fluxColumn = self.config.srcSchemaMap[self.config.analysis.fluxColumn] + "_flux"
            self.classificationColumn = self.config.srcSchemaMap[self.classificationColumn]
            self.flags = [self.config.srcSchemaMap[flag] for flag in self.flags]

        for col in transforms:
            doAdd = True
            for ff in transforms[col].coeffs:
                if ff != "" and ff not in catalogs:
                    doAdd = False
                    # if all(ff in catalogs for ff in transforms[col].coeffs):
            if doAdd:
                schema.addField(col, float, transforms[col].description)
        schema.addField("numStarFlags", type=np.int32, doc="Number of times source was flagged as star")
        badKey = schema.addField("bad", type="Flag", doc="Is this a bad source?")
        schema.addField(self.fluxColumn, type=np.float64, doc="Flux from filter " + self.config.fluxFilter)

        # Copy basics (id, RA, Dec)
        new = afwTable.SourceCatalog(schema)
        new.reserve(num)
        new.extend(template, mapper)

        # Set transformed colors
        for col, transform in transforms.items():
            if col not in schema:
                continue
            value = np.ones(num)*transform.coeffs[""] if "" in transform.coeffs else np.zeros(num)
            for ff, coeff in transform.coeffs.items():
                if ff == "":  # Constant: already done
                    continue
                cat = catalogs[ff]
                mag = -2.5*np.log10(cat[self.fluxColumn])
                value += mag*coeff
            new[col][:] = value

        # Flag bad values
        bad = np.zeros(num, dtype=bool)
        for dataCat, flagsCat in zip(catalogs.itervalues(), flagsCats.itervalues()):
            if not checkIdLists(dataCat, flagsCat):
                raise RuntimeError(
                    "Catalog being used for flags does not have the same object list as the data catalog")
            for flag in self.flags:
                if flag in flagsCat.schema:
                    bad |= flagsCat[flag]
        # Can't set column for flags; do row-by-row
        for row, badValue in zip(new, bad):
            row.setFlag(badKey, bool(badValue))

        # Star/galaxy
        numStarFlags = np.zeros(num)
        for cat in catalogs.itervalues():
            numStarFlags += np.where(cat[self.classificationColumn] < 0.5, 1, 0)
        new["numStarFlags"][:] = numStarFlags

        new[self.fluxColumn][:] = catalogs[self.config.fluxFilter][self.fluxColumn]

        return new

    def plotGalaxyColors(self, catalogs, filenamer, dataId):
        filters = set(catalogs.keys())
        if filters.issuperset(set(("HSC-G", "HSC-I"))):
            gg = catalogs["HSC-G"]
            ii = catalogs["HSC-I"]
            assert len(gg) == len(ii)
            mapperList = afwTable.SchemaMapper.join([gg.schema, ii.schema],
                                                    ["g_", "i_"])
            catalog = afwTable.BaseCatalog(mapperList[0].getOutputSchema())
            catalog.reserve(len(gg))
            for gRow, iRow in zip(gg, ii):
                row = catalog.addNew()
                row.assign(gRow, mapperList[0])
                row.assign(iRow, mapperList[1])

            catalog.writeFits("gi.fits")
            shortName = "galaxy-TEST"
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(catalog, GalaxyColor("modelfit_CModel_flux", "slot_CalibFlux_flux", "g_", "i_"),
                               "(g-i)_cmodel - (g-i)_CalibFlux", shortName, self.config.analysis,
                               flags=["modelfit_CModel_flag", "slot_CalibFlux_flag"], prefix="i_",
                               labeller=OverlapsStarGalaxyLabeller("g_", "i_"),
                               qMin=-0.5, qMax=0.5,).plotAll(dataId, filenamer, self.log)

    def plotStarColors(self, catalog, filenamer, labeller, dataId, butler=None, camera=None, tractInfo=None,
                       patchList=None, hscRun=None):
        for col, transform in self.config.transforms.items():
            if not transform.plot or col not in catalog.schema:
                continue
            shortName = "color_" + col
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(catalog, ColorValueInRange(col, transform.requireGreater,
                                                          transform.requireLess),
                               col, shortName, self.config.analysis, flags=["bad"], labeller=labeller,
                               qMin=-0.2, qMax=0.2,
                               ).plotAll(dataId, filenamer, self.log, butler=butler, camera=camera,
                                         tractInfo=tractInfo, patchList=patchList, hscRun=hscRun)

    def plotStarColorColor(self, catalogs, filenamer, dataId, butler=None, camera=None, tractInfo=None,
                           patchList=None, hscRun=None):
        num = len(catalogs.values()[0])
        zp = 0.0
        mags = {ff: zp - 2.5*np.log10(catalogs[ff][self.fluxColumn]) for ff in catalogs}

        bad = np.zeros(num, dtype=bool)
        for cat in catalogs.itervalues():
            for flag in self.flags:
                if flag in cat.schema:
                    bad |= cat[flag]

        bright = np.ones(num, dtype=bool)
        for mm in mags.itervalues():
            bright &= mm < self.config.analysis.magThreshold

        numStarFlags = np.zeros(num)
        for cat in catalogs.itervalues():
            numStarFlags += np.where(cat[self.classificationColumn] < 0.5, 1, 0)

        good = (numStarFlags == len(catalogs)) & np.logical_not(bad) & bright

        combined = self.transformCatalogs(catalogs, straightTransforms, hscRun=hscRun)[good].copy(True)
        filters = set(catalogs.keys())
        color = lambda c1, c2: (mags[c1] - mags[c2])[good]
        if filters.issuperset(set(("HSC-G", "HSC-R", "HSC-I"))):
            # Lower branch only; upper branch is noisy due to astrophysics
            poly = colorColorPlot(dataId, filenamer(dataId, description="gri", style="fit"), self.log,
                                  color("HSC-G", "HSC-R"), color("HSC-R", "HSC-I"), "g - r", "r - i",
                                  xRange=(-0.5, 2.0), yRange=(-0.5, 2.0), order=3, xFitRange=(0.3, 1.1),
                                  camera=camera, hscRun=hscRun)
            shortName = "gri"
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(combined, ColorColorDistance("g", "r", "i", poly, 0.3, 1.1), "griPerp",
                               shortName, self.config.analysis, flags=["bad"], qMin=-0.1, qMax=0.1,
                               labeller=NumStarLabeller(len(catalogs)),
                               ).plotAll(dataId, filenamer, self.log,
                                         Enforcer(requireLess={"star": {"stdev": 0.05}}), camera=camera,
                                         tractInfo=tractInfo, patchList=patchList, hscRun=hscRun)
        if filters.issuperset(set(("HSC-R", "HSC-I", "HSC-Z"))):
            poly = colorColorPlot(dataId, filenamer(dataId, description="riz", style="fit"), self.log,
                                  color("HSC-R", "HSC-I"), color("HSC-I", "HSC-Z"), "r - i", "i - z",
                                  xRange=(-0.5, 2.0), yRange=(-0.4, 0.8), order=3, hscRun=hscRun)
            shortName = "riz"
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(combined, ColorColorDistance("r", "i", "z", poly), "rizPerp", shortName,
                               self.config.analysis, flags=["bad"], qMin=-0.1, qMax=0.1,
                               labeller=NumStarLabeller(len(catalogs)),
                               ).plotAll(dataId, filenamer, self.log,
                                         Enforcer(requireLess={"star": {"stdev": 0.02}}), camera=camera,
                                         tractInfo=tractInfo, patchList=patchList, hscRun=hscRun)
        if filters.issuperset(set(("HSC-I", "HSC-Z", "HSC-Y"))):
            poly = colorColorPlot(dataId, filenamer(dataId, description="izy", style="fit"), self.log,
                                  color("HSC-I", "HSC-Z"), color("HSC-Z", "HSC-Y"), "i - z", "z - y",
                                  xRange=(-0.4, 0.8), yRange=(-0.3, 0.5), order=3, hscRun=hscRun)
            shortName = "izy"
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(combined, ColorColorDistance("i", "z", "y", poly), "izyPerp", shortName,
                               self.config.analysis, flags=["bad"], qMin=-0.1, qMax=0.1,
                               labeller=NumStarLabeller(len(catalogs)),
                               ).plotAll(dataId, filenamer, self.log,
                                         Enforcer(requireLess={"star": {"stdev": 0.02}}), camera=camera,
                                         tractInfo=tractInfo, patchList=patchList, hscRun=hscRun)

        if filters.issuperset(set(("HSC-Z", "NB0921", "HSC-Y"))):
            poly = colorColorPlot(dataId, filenamer(dataId, description="z9y", style="fit"), self.log,
                                  color("HSC-Z", "NB0921"), color("NB0921", "HSC-Y"), "z-n921", "n921-y",
                                  xRange=(-0.2, 0.2), yRange=(-0.1, 0.2), order=2, xFitRange=(-0.05, 0.15),
                                  hscRun=hscRun)
            shortName = "z9y"
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(combined, ColorColorDistance("z", "n921", "y", poly), "z9yPerp", shortName,
                               self.config.analysis, flags=["bad"], qMin=-0.1, qMax=0.1,
                               labeller=NumStarLabeller(len(catalogs)),
                               ).plotAll(dataId, filenamer, self.log,
                                         Enforcer(requireLess={"star": {"stdev": 0.02}}), camera=camera,
                                         tractInfo=tractInfo, patchList=patchList, hscRun=hscRun)


    def _getConfigName(self):
        return None
    def _getMetadataName(self):
        return None
    def _getEupsVersionsName(self):
        return None


def colorColorPlot(dataId, filename, log, xx, yy, xLabel, yLabel, xRange=None, yRange=None, order=1,
                   iterations=1, rej=3.0, xFitRange=None, numBins=51, hscRun=None, logger=None, camera=None):
    fig, axes = plt.subplots(1, 2)
    axes[0].tick_params(which="both", direction="in", labelsize=9)
    axes[1].tick_params(which="both", direction="in", labelsize=9)

    if xRange:
        axes[0].set_xlim(*xRange)
    else:
        xRange = (0.9*xx.min(), 1.1*xx.max())
    if yRange:
        axes[0].set_ylim(*yRange)

    select = np.ones_like(xx, dtype=bool) if not xFitRange else ((xx > xFitRange[0]) & (xx < xFitRange[1]))
    keep = np.ones_like(xx, dtype=bool)
    for ii in range(iterations):
        keep &= select
        poly = np.polyfit(xx[keep], yy[keep], order)
        dy = yy - np.polyval(poly, xx)
        q1, q3 = np.percentile(dy[keep], [25, 75])
        clip = rej*0.74*(q3 - q1)
        keep = np.logical_not(np.abs(dy) > clip)

    keep &= select
    nKeep = np.sum(keep)
    if nKeep < order:
        raise RuntimeError(
            "Not enough good data points ({0:d}) for polynomial fit of order {1:d}".format(nKeep, order))

    poly = np.polyfit(xx[keep], yy[keep], order)
    xLine = np.linspace(xRange[0], xRange[1], 1000)
    yLine = np.polyval(poly, xLine)

    kwargs = dict(s=3, marker="o", lw=0, alpha=0.4)
    axes[0].scatter(xx[keep], yy[keep], c="blue", label="used", **kwargs)
    axes[0].scatter(xx[~keep], yy[~keep], c="black", label="other", **kwargs)
    axes[0].set_xlabel(xLabel)
    axes[0].set_ylabel(yLabel, labelpad=-1)
    axes[0].legend(loc="upper left", fontsize=7)
    axes[0].plot(xLine, yLine, "r-")

    # Label total number of objects of each data type
    xLoc, yLoc = xRange[0] + 0.34*(xRange[1] - xRange[0]), yRange[1] - 0.036*(yRange[1] - yRange[0])
    axes[0].text(xLoc, yLoc, "Nused  = " + str(len(xx[keep])), ha="left", va="center", fontsize=8,
                 color="blue")
    yLoc -= 0.044*(yRange[1] - yRange[0])
    axes[0].text(xLoc, yLoc, "Nother = " + str(len(xx[~keep])), ha="left", va="center", fontsize=8,
                 color="black")

    # Determine quality of locus
    distance2 = []
    poly = np.poly1d(poly)
    polyDeriv = np.polyder(poly)
    calculateDistance2 = lambda x1, y1, x2: (x2 - x1)**2 + (poly(x2) - y1)**2
    for x, y in zip(xx[select], yy[select]):
        roots = np.roots(np.poly1d((1, -x)) + (poly - y)*polyDeriv)
        distance2.append(min(calculateDistance2(x, y, np.real(rr)) for rr in roots if np.real(rr) == rr))
    distance = np.sqrt(distance2)
    distance *= np.where(yy[select] >= poly(xx[select]), 1.0, -1.0)

    q1, median, q3 = np.percentile(distance, [25, 50, 75])
    good = np.logical_not(np.abs(distance - median) > 3.0*0.74*(q3 - q1))
    log.info(("Statistics from {0:s} of Distance to polynomial (mmag): {7:s}\'star\': " +
              "Stats(mean={1:.4f}; stdev={2:.4f}; num={3:d}; total={4:d}; " +
              "median={5:.4f}; clip={6:.4f}; forcedMean=None){8:s}").format(
            dataId, 1000*distance[good].mean(), 1000*distance[good].std(), len(xx[keep]), len(xx),
            1000*np.median(distance[good]),  1000*3.0*0.74*(q3 - q1), "{", "}"))
    meanStr = "mean = {:5.2f} (mmag)".format(1000.0*distance[good].mean())
    stdStr = "  std = {:5.2f} (mmag)".format(1000.0*distance[good].std())
    tractStr = "tract: {:d}".format(dataId["tract"])
    axes[1].set_xlabel("Distance to polynomial fit (mag)")
    axes[1].set_ylabel("Number")
    axes[1].set_yscale("log", nonposy="clip")
    axes[1].hist(distance[good], numBins, range=(-0.05, 0.05), normed=False)
    axes[1].annotate(meanStr, xy=(0.6, 0.96), xycoords="axes fraction", ha="right", va="center",
                     fontsize=8, color="black")
    axes[1].annotate(stdStr, xy=(0.6, 0.92), xycoords="axes fraction", ha="right", va="center",
                     fontsize=8, color="black")
    axes[1].annotate(tractStr, xy=(0.5, 1.04), xycoords="axes fraction", ha="center", va="center",
                     fontsize=10, color="green")

    if camera is not None:
        labelCamera(camera, plt, axes[0], 0.5, 1.04)
    if hscRun is not None:
        axes[0].set_title("HSC stack run: " + hscRun, color="#800080")

    plt.tight_layout(pad=2.5, w_pad=0.5, h_pad=1.0)
    fig.savefig(filename)
    plt.close(fig)

    return poly


class ColorColorDistance(object):
    """Functor to calculate distance from stellar locus in color-color plot"""
    def __init__(self, band1, band2, band3, poly, xMin=None, xMax=None):
        self.band1 = band1
        self.band2 = band2
        self.band3 = band3
        self.poly = poly
        self.xMin = xMin
        self.xMax = xMax

    def __call__(self, catalog):
        xx = catalog[self.band1] - catalog[self.band2]
        yy = catalog[self.band2] - catalog[self.band3]
        polyDeriv = np.polyder(self.poly)
        calculateDistance2 = lambda x1, y1, x2: (x2 - x1)**2 + (self.poly(x2) - y1)**2
        distance2 = np.ones_like(xx)*np.nan
        for i, (x, y) in enumerate(zip(xx, yy)):
            if (not np.isfinite(x) or not np.isfinite(y) or (self.xMin is not None and x < self.xMin) or
                (self.xMax is not None and x > self.xMax)):
                distance2[i] = np.nan
                continue
            roots = np.roots(np.poly1d((1, -x)) + (self.poly - y)*polyDeriv)
            distance2[i] = min(calculateDistance2(x, y, np.real(rr)) for
                               rr in roots if np.real(rr) == rr)
        return np.sqrt(distance2)*np.where(yy >= self.poly(xx), 1.0, -1.0)


class SkyAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        kwargs["cosmos"] = parsedCmd.cosmos

        # Partition all inputs by filter
        filterRefs = defaultdict(list)  # filter-->dataRefs
        for patchRef in sum(parsedCmd.id.refList, []):
            if patchRef.datasetExists("deepCoadd_meas"):
                filterName = patchRef.dataId["filter"]
                filterRefs[filterName].append(patchRef)

        return [(refList, kwargs) for refList in filterRefs.itervalues()]


class SkyAnalysisTask(CoaddAnalysisTask):
    """Version of CoaddAnalysisTask that runs on all inputs simultaneously

    This is most useful for utilising overlaps between tracts.
    """
    _DefaultName = "skyAnalysis"
    RunnerClass = SkyAnalysisRunner
    outputDataset = "plotSky"
