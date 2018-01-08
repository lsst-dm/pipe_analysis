#!/usr/bin/env python

import matplotlib
matplotlib.use("Agg")  # noqa #402
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all="ignore")  # noqa #402
import functools

from collections import defaultdict

from lsst.pex.config import Config, Field, ConfigField, ListField, DictField, ConfigDictField
from lsst.pipe.base import CmdLineTask, ArgumentParser, TaskRunner
from lsst.coadd.utils import TractDataIdContainer
from .analysis import Analysis, AnalysisConfig
from .coaddAnalysis import CoaddAnalysisTask
from .utils import (Filenamer, Enforcer, concatenateCatalogs, checkIdLists, addPatchColumn,
                    calibrateCoaddSourceCatalog, writeParquet, getRepoInfo)
from .plotUtils import OverlapsStarGalaxyLabeller, labelCamera

import lsst.afw.table as afwTable

__all__ = ["ColorTransform", "ivezicTransforms", "straightTransforms", "NumStarLabeller",
           "ColorValueInRange", "GalaxyColor", "ColorAnalysisConfig", "ColorAnalysisRunner",
           "ColorAnalysisTask", "ColorColorDistance", "SkyAnalysisRunner", "SkyAnalysisTask"]


class ColorTransform(Config):
    description = Field(dtype=str, doc="Description of the color transform")
    subDescription = Field(dtype=str, doc="Sub-description of the color transform (added detail)")
    plot = Field(dtype=bool, default=True, doc="Plot this color?")
    coeffs = DictField(keytype=str, itemtype=float, doc="Coefficients for each filter")
    requireGreater = DictField(keytype=str, itemtype=float, default={},
                               doc="Minimum values for colors so that this is useful")
    requireLess = DictField(keytype=str, itemtype=float, default={},
                            doc="Maximum values for colors so that this is useful")

    @classmethod
    def fromValues(cls, description, subDescription, plot, coeffs, requireGreater={}, requireLess={}):
        self = cls()
        self.description = description
        self.subDescription = subDescription
        self.plot = plot
        self.coeffs = coeffs
        self.requireGreater = requireGreater
        self.requireLess = requireLess
        return self


ivezicTransforms = {
    "wPerp": ColorTransform.fromValues("Ivezic w perpendicular", " (griBlue)", True,
                                       {"HSC-G": -0.227, "HSC-R": 0.792, "HSC-I": -0.567, "": 0.050},
                                       {"wPara": -0.2}, {"wPara": 0.6}),
    "xPerp": ColorTransform.fromValues("Ivezic x perpendicular", " (griRed)", True,
                                       {"HSC-G": 0.707, "HSC-R": -0.707, "": -0.988},
                                       {"xPara": 0.8}, {"xPara": 1.6}),
    "yPerp": ColorTransform.fromValues("Ivezic y perpendicular", " (rizRed)", True,
                                       {"HSC-R": -0.270, "HSC-I": 0.800, "HSC-Z": -0.534, "": 0.054},
                                       {"yPara": 0.1}, {"yPara": 1.2}),
    "wPara": ColorTransform.fromValues("Ivezic w parallel", " (griBlue)", False,
                                       {"HSC-G": 0.928, "HSC-R": -0.556, "HSC-I": -0.372, "": -0.425}),
    "xPara": ColorTransform.fromValues("Ivezic x parallel", " (griRed)", False,
                                       {"HSC-R": 1.0, "HSC-I": -1.0}),
    "yPara": ColorTransform.fromValues("Ivezic y parallel", " (rizRed)", False,
                                       {"HSC-R": 0.895, "HSC-I": -0.448, "HSC-Z": -0.447, "": -0.600}),
}

straightTransforms = {
    "g": ColorTransform.fromValues("HSC-G", "", True, {"HSC-G": 1.0}),
    "r": ColorTransform.fromValues("HSC-R", "", True, {"HSC-R": 1.0}),
    "i": ColorTransform.fromValues("HSC-I", "", True, {"HSC-I": 1.0}),
    "z": ColorTransform.fromValues("HSC-Z", "", True, {"HSC-Z": 1.0}),
    "y": ColorTransform.fromValues("HSC-Y", "", True, {"HSC-Y": 1.0}),
    "n921": ColorTransform.fromValues("NB0921", "", True, {"NB0921": 1.0}),
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
    def __init__(self, column, requireGreater, requireLess, unitScale=1.0):
        self.column = column
        self.requireGreater = requireGreater
        self.requireLess = requireLess
        self.unitScale = unitScale

    def __call__(self, catalog):
        good = np.ones(len(catalog), dtype=bool)
        for col, value in self.requireGreater.items():
            good &= catalog[col] > value
        for col, value in self.requireLess.items():
            good &= catalog[col] < value
        return np.where(good, catalog[self.column], np.nan)*self.unitScale


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
                      default=["base_SdssCentroid_flag", "slot_Centroid_flag", "base_PsfFlux_flag",
                               "base_PixelFlags_flag_saturatedCenter",
                               "base_PixelFlags_flag_interpolatedCenter",
                               "base_ClassificationExtendedness_flag"])
    analysis = ConfigField(dtype=AnalysisConfig, doc="Analysis plotting options")
    transforms = ConfigDictField(keytype=str, itemtype=ColorTransform, default={},
                                 doc="Color transformations to analyse")
    fluxFilter = Field(dtype=str, default="HSC-I", doc="Filter to use for plotting against magnitude")
    srcSchemaMap = DictField(keytype=str, itemtype=str, default=None, optional=True,
                             doc="Mapping between different stack (e.g. HSC vs. LSST) schema names")
    toMilli = Field(dtype=bool, default=True, doc="Print stats in milli units (i.e. mas, mmag)?")
    writeParquetOnly = Field(dtype=bool, default=False,
                             doc="Only write out Parquet tables (i.e. do not produce any plots)?")
    doWriteParquetTables = Field(dtype=bool, default=True,
                                 doc=("Write out Parquet tables (for subsequent interactive analysis)?"
                                      "\nNOTE: if True but fastparquet package is unavailable, a warning "
                                      "is issued and table writing is skipped."))

    def setDefaults(self):
        Config.setDefaults(self)
        self.transforms = ivezicTransforms
        self.analysis.flags = []  # We remove bad source ourself
        self.analysis.magThreshold = 22.0  # RHL requested this limit

    def validate(self):
        Config.validate(self)
        if self.writeParquetOnly and not self.doWriteParquetTables:
            raise ValueError("Cannot writeParquetOnly if doWriteParquetTables is False")


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
                                 patchRefList in filterRefs.values()]
            if not patchesForFilters:
                parsedCmd.log.warn("No input data found for tract {:d}".format(tract))
                bad.append(tract)
                continue
            keep = set.intersection(*patchesForFilters)  # Patches with full colour coverage
            tractFilterRefs[tract] = {ff: [patchRef for patchRef in filterRefs[ff] if
                                           patchRef.dataId["patch"] in keep] for ff in filterRefs}
        for tract in bad:
            del tractFilterRefs[tract]

        return [(filterRefs, kwargs) for filterRefs in tractFilterRefs.values()]


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

    def __init__(self, *args, **kwargs):
        CmdLineTask.__init__(self, *args, **kwargs)
        self.unitScale = 1000.0 if self.config.toMilli else 1.0

    def run(self, patchRefsByFilter):
        patchList = []
        repoInfo = None
        for patchRefList in patchRefsByFilter.values():
            for dataRef in patchRefList:
                if dataRef.dataId["filter"] == self.config.fluxFilter:
                    patchList.append(dataRef.dataId["patch"])
                    if repoInfo is None:
                        repoInfo = getRepoInfo(dataRef, coaddName=self.config.coaddName,
                                               coaddDataset="Coadd_forced_src")

        filenamer = Filenamer(repoInfo.butler, "plotColor", repoInfo.dataId)
        unforcedCatalogsByFilter = {ff: self.readCatalogs(patchRefList,
                                                          self.config.coaddName + "Coadd_meas") for
                                    ff, patchRefList in patchRefsByFilter.items()}
        for cat in unforcedCatalogsByFilter.values():
            calibrateCoaddSourceCatalog(cat, self.config.analysis.coaddZp)
        unforced = self.transformCatalogs(unforcedCatalogsByFilter, self.config.transforms,
                                          hscRun=repoInfo.hscRun)
        forcedCatalogsByFilter = {ff: self.readCatalogs(patchRefList,
                                                        self.config.coaddName + "Coadd_forced_src") for
                                  ff, patchRefList in patchRefsByFilter.items()}
        for cat in forcedCatalogsByFilter.values():
            calibrateCoaddSourceCatalog(cat, self.config.analysis.coaddZp)
        # self.plotGalaxyColors(catalogsByFilter, filenamer, dataId)
        forced = self.transformCatalogs(forcedCatalogsByFilter, self.config.transforms,
                                        flagsCats=unforcedCatalogsByFilter, hscRun=repoInfo.hscRun)

        # Create and write parquet tables
        if self.config.doWriteParquetTables:
            tableFilenamer = Filenamer(repoInfo.butler, 'qaTableColor', repoInfo.dataId)
            writeParquet(forced, tableFilenamer(repoInfo.dataId, description='forced'))
            if self.config.writeParquetOnly:
                self.log.info("Exiting after writing Parquet tables.  No plots generated.")
                return

        self.plotStarColors(forced, filenamer, NumStarLabeller(len(forcedCatalogsByFilter)), repoInfo.dataId,
                            camera=repoInfo.camera, tractInfo=repoInfo.tractInfo, patchList=patchList,
                            hscRun=repoInfo.hscRun)
        self.plotStarColorColor(forcedCatalogsByFilter, filenamer, repoInfo.dataId, camera=repoInfo.camera,
                                tractInfo=repoInfo.tractInfo, patchList=patchList, hscRun=repoInfo.hscRun)

    def readCatalogs(self, patchRefList, dataset):
        """Read in and concatenate catalogs of type dataset in lists of data references

        Before appending each catalog to a single list, an extra column indicating the
        patch is added to the catalog.  This is useful for the subsequent interactive
        QA analysis.

        Parameters
        ----------
        patchRefList : `list` of `lsst.daf.persistence.butlerSubset.ButlerDataRef`
           A list of butler data references whose catalogs of dataset type are to be read in
        dataset : `str`
           Name of the catalog dataset to be read in

        Raises
        ------
        `TaskError`
           If no data is read in for the dataRefList

        Returns
        -------
        `list` of concatenated `lsst.afw.table.source.source.SourceCatalog`s
        """
        catList = []
        for patchRef in patchRefList:
            if patchRef.datasetExists(dataset):
                cat = patchRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_HEAVY_FOOTPRINTS)
                cat = addPatchColumn(cat, patchRef.dataId["patch"])
                catList.append(cat)
        if not catList:
            raise TaskError("No catalogs read: %s" % ([patchRef.dataId for patchRef in patchRefList]))
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

        template = list(catalogs.values())[0]
        num = len(template)
        assert all(len(cat) == num for cat in catalogs.values())

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
            if doAdd:
                schema.addField(col, float, transforms[col].description + transforms[col].subDescription)
        schema.addField("numStarFlags", type=np.int32, doc="Number of times source was flagged as star")
        badKey = schema.addField("qaBad_flag", type="Flag", doc="Is this a bad source for color qa analyses?")
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
        for dataCat, flagsCat in zip(catalogs.values(), flagsCats.values()):
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
        for cat in catalogs.values():
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
            self.AnalysisClass(catalog,
                               GalaxyColor("modelfit_CModel_flux", "slot_CalibFlux_flux", "g_", "i_"),
                               "(g-i)_cmodel - (g-i)_CalibFlux", shortName, self.config.analysis,
                               flags=["modelfit_CModel_flag", "slot_CalibFlux_flag"], prefix="i_",
                               labeller=OverlapsStarGalaxyLabeller("g_", "i_"),
                               qMin=-0.5, qMax=0.5,).plotAll(dataId, filenamer, self.log)

    def plotStarColors(self, catalog, filenamer, labeller, dataId, butler=None, camera=None, tractInfo=None,
                       patchList=None, hscRun=None):
        unitStr = "mmag" if self.config.toMilli else "mag"
        for col, transform in self.config.transforms.items():
            if not transform.plot or col not in catalog.schema:
                continue
            shortName = "color_" + col
            self.log.info("shortName = {:s}".format(shortName + transform.subDescription))
            self.AnalysisClass(catalog, ColorValueInRange(col, transform.requireGreater,
                                                          transform.requireLess, unitScale=self.unitScale),
                               "%s (%s)" % (col + transform.subDescription, unitStr), shortName,
                               self.config.analysis, flags=["qaBad_flag"], labeller=labeller,
                               qMin=-0.2, qMax=0.2,
                               ).plotAll(dataId, filenamer, self.log, butler=butler, camera=camera,
                                         tractInfo=tractInfo, patchList=patchList, hscRun=hscRun)

    def plotStarColorColor(self, catalogs, filenamer, dataId, butler=None, camera=None, tractInfo=None,
                           patchList=None, hscRun=None):
        num = len(list(catalogs.values())[0])
        zp = 0.0
        mags = {ff: zp - 2.5*np.log10(catalogs[ff][self.fluxColumn]) for ff in catalogs}

        bad = np.zeros(num, dtype=bool)
        for cat in catalogs.values():
            for flag in self.flags:
                if flag in cat.schema:
                    bad |= cat[flag]

        bright = np.ones(num, dtype=bool)
        for mm in mags.values():
            bright &= mm < self.config.analysis.magThreshold

        numStarFlags = np.zeros(num)
        for cat in catalogs.values():
            numStarFlags += np.where(cat[self.classificationColumn] < 0.5, 1, 0)

        good = (numStarFlags == len(catalogs)) & np.logical_not(bad) & bright

        combined = self.transformCatalogs(catalogs, straightTransforms, hscRun=hscRun)[good].copy(True)
        filters = set(catalogs.keys())
        color = lambda c1, c2: (mags[c1] - mags[c2])[good]
        unitStr = "mmag" if self.config.toMilli else "mag"
        if filters.issuperset(set(("HSC-G", "HSC-R", "HSC-I"))):
            # Do a linear fit to regions defined in Ivezic transforms
            transform = self.config.transforms["wPerp"]
            xFitRange1 = transform.requireGreater["wPara"]
            xFitRange2 = transform.requireLess["wPara"]
            wPerpFit = colorColorPolyFitPlot(dataId, filenamer(dataId, description="gri-wFit", style="fit"),
                                             self.log, color("HSC-G", "HSC-R"), color("HSC-R", "HSC-I"),
                                             "g - r", "r - i", xRange=(-0.5, 2.0), yRange=(-0.5, 2.0),
                                             order=1, xFitRange=(xFitRange1, xFitRange2), camera=camera,
                                             hscRun=hscRun, unitScale=self.unitScale)
            transform = self.config.transforms["xPerp"]
            xFitRange1 = transform.requireGreater["xPara"]
            xFitRange2 = transform.requireLess["xPara"]
            xPerpFit = colorColorPolyFitPlot(dataId, filenamer(dataId, description="gri-xFit", style="fit"),
                                             self.log, color("HSC-G", "HSC-R"), color("HSC-R", "HSC-I"),
                                             "g - r", "r - i", xRange=(-0.5, 2.0), yRange=(-0.5, 2.0),
                                             order=1, xFitRange=(xFitRange1, xFitRange2), camera=camera,
                                             hscRun=hscRun, unitScale=self.unitScale)
            # Lower branch only; upper branch is noisy due to astrophysics
            poly = colorColorPolyFitPlot(dataId, filenamer(dataId, description="gri", style="fit"), self.log,
                                  color("HSC-G", "HSC-R"), color("HSC-R", "HSC-I"), "g - r", "r - i",
                                  xRange=(-0.5, 2.0), yRange=(-0.5, 2.0), order=3, xFitRange=(0.3, 1.1),
                                  camera=camera, hscRun=hscRun, unitScale=self.unitScale)
            shortName = "griDistance"
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(combined, ColorColorDistance("g", "r", "i", poly, 0.3, 1.1),
                               "griDistance (%s)" % unitStr, shortName, self.config.analysis,
                               flags=["qaBad_flag"], qMin=-0.1, qMax=0.1,
                               labeller=NumStarLabeller(len(catalogs)),
                               ).plotAll(dataId, filenamer, self.log,
                                         Enforcer(requireLess={"star": {"stdev": 0.05}}), camera=camera,
                                         tractInfo=tractInfo, patchList=patchList, hscRun=hscRun)
        if filters.issuperset(set(("HSC-R", "HSC-I", "HSC-Z"))):
            # Do a linear fit to regions defined in Ivezic transforms
            transform = self.config.transforms["yPerp"]
            xFitRange1 = transform.requireGreater["yPara"]
            xFitRange2 = transform.requireLess["yPara"]
            yPerpFit = colorColorPolyFitPlot(dataId, filenamer(dataId, description="riz-yFit", style="fit"),
                                             self.log, color("HSC-R", "HSC-I"), color("HSC-I", "HSC-Z"),
                                             "r - i", "i - z", xRange=(-0.5, 2.0), yRange=(-0.4, 0.8),
                                             order=1, xFitRange=(xFitRange1, xFitRange2), camera=camera,
                                             hscRun=hscRun, unitScale=self.unitScale)
            poly = colorColorPolyFitPlot(dataId, filenamer(dataId, description="riz", style="fit"), self.log,
                                         color("HSC-R", "HSC-I"), color("HSC-I", "HSC-Z"), "r - i", "i - z",
                                         xRange=(-0.5, 2.0), yRange=(-0.4, 0.8), order=3, camera=camera,
                                         hscRun=hscRun, unitScale=self.unitScale)
            shortName = "rizDistance"
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(combined, ColorColorDistance("r", "i", "z", poly),
                               "rizDistance (%s)" % unitStr, shortName, self.config.analysis,
                               flags=["qaBad_flag"], qMin=-0.1, qMax=0.1,
                               labeller=NumStarLabeller(len(catalogs)),
                               ).plotAll(dataId, filenamer, self.log,
                                         Enforcer(requireLess={"star": {"stdev": 0.02}}), camera=camera,
                                         tractInfo=tractInfo, patchList=patchList, hscRun=hscRun)
        if filters.issuperset(set(("HSC-I", "HSC-Z", "HSC-Y"))):
            poly = colorColorPolyFitPlot(dataId, filenamer(dataId, description="izy", style="fit"), self.log,
                                         color("HSC-I", "HSC-Z"), color("HSC-Z", "HSC-Y"), "i - z", "z - y",
                                         xRange=(-0.4, 0.8), yRange=(-0.3, 0.5), order=3, camera=camera,
                                         hscRun=hscRun, unitScale=self.unitScale)
            shortName = "izyDistance"
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(combined, ColorColorDistance("i", "z", "y", poly),
                               "izyDistance (%s)" % unitStr, shortName, self.config.analysis,
                               flags=["qaBad_flag"], qMin=-0.1, qMax=0.1,
                               labeller=NumStarLabeller(len(catalogs)),
                               ).plotAll(dataId, filenamer, self.log,
                                         Enforcer(requireLess={"star": {"stdev": 0.02}}), camera=camera,
                                         tractInfo=tractInfo, patchList=patchList, hscRun=hscRun)

        if filters.issuperset(set(("HSC-Z", "NB0921", "HSC-Y"))):
            poly = colorColorPolyFitPlot(dataId, filenamer(dataId, description="z9y", style="fit"), self.log,
                                         color("HSC-Z", "NB0921"), color("NB0921", "HSC-Y"),
                                         "z-n921", "n921-y", xRange=(-0.2, 0.2), yRange=(-0.1, 0.2),
                                         order=2, xFitRange=(-0.05, 0.15), camera=camera, hscRun=hscRun,
                                         unitScale=self.unitScale)
            shortName = "z9yDistance"
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(combined, ColorColorDistance("z", "n921", "y", poly),
                               "z9yDistance (%s)" % unitStr, shortName, self.config.analysis,
                               flags=["qaBad_flag"], qMin=-0.1, qMax=0.1,
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


def colorColorPolyFitPlot(dataId, filename, log, xx, yy, xLabel, yLabel, xRange=None, yRange=None, order=1,
                          iterations=1, rej=3.0, xFitRange=None, numBins=51, hscRun=None, logger=None,
                          camera=None, unitScale=1.0):
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

    # Label polynomial fit parameters to 2 decimal places
    polyStr = "y = {:.2f}".format(poly[len(poly) - 1])
    for i in range(1, len(poly)):
        index = len(poly) - 1 - i
        plusMinus = " - " if poly[index] < 0.0 else " + "
        exponent = "$^{" + str(i) + "}$" if i > 1 else ""
        polyStr += plusMinus + "{:.2f}".format(abs(poly[index])) + "x" + exponent
    xLoc = xRange[0] + 0.05*(xRange[1] - xRange[0])
    yLoc -= 0.055*(yRange[1] - yRange[0])
    axes[0].text(xLoc, yLoc, polyStr, ha="left", va="center", fontsize=8, color="green")

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
    unitStr = "mmag" if unitScale == 1000 else "mag"
    distance *= unitScale
    q1, median, q3 = np.percentile(distance, [25, 50, 75])
    good = np.logical_not(np.abs(distance - median) > 3.0*0.74*(q3 - q1))
    log.info(("Statistics from {0:} of Distance to polynomial ({9:s}): {7:s}\'star\': " +
              "Stats(mean={1:.4f}; stdev={2:.4f}; num={3:d}; total={4:d}; " +
              "median={5:.4f}; clip={6:.4f}; forcedMean=None){8:s}").format(
            dataId, distance[good].mean(), distance[good].std(), len(xx[keep]), len(xx),
            np.median(distance[good]), 3.0*0.74*(q3 - q1), "{", "}", unitStr))
    # Get rid of LaTeX-specific characters for log message printing
    log.info("Polynomial fit: {:2}".format("".join(x for x in polyStr if x not in "{}$")))
    meanStr = "mean = {0:5.2f} ({1:s})".format(distance[good].mean(), unitStr)
    stdStr = "  std = {0:5.2f} ({1:s})".format(distance[good].std(), unitStr)
    tractStr = "tract: {:d}".format(dataId["tract"])
    axes[1].set_xlabel("Distance to polynomial fit ({:s})".format(unitStr))
    axes[1].set_ylabel("Number")
    axes[1].set_yscale("log", nonposy="clip")
    axes[1].hist(distance[good], numBins, range=(-0.05*unitScale, 0.05*unitScale), normed=False)
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

        return [(refList, kwargs) for refList in filterRefs.values()]


class SkyAnalysisTask(CoaddAnalysisTask):
    """Version of CoaddAnalysisTask that runs on all inputs simultaneously

    This is most useful for utilising overlaps between tracts.
    """
    _DefaultName = "skyAnalysis"
    RunnerClass = SkyAnalysisRunner
    outputDataset = "plotSky"
