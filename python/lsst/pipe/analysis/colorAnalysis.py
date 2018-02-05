#!/usr/bin/env python

import matplotlib
matplotlib.use("Agg")  # noqa #402
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all="ignore")  # noqa #402
import functools

from collections import defaultdict

from lsst.pex.config import Config, Field, ConfigField, ListField, DictField, ConfigDictField
from lsst.pipe.base import CmdLineTask, ArgumentParser, TaskRunner, TaskError
from lsst.coadd.utils import TractDataIdContainer
from .analysis import Analysis, AnalysisConfig
from .coaddAnalysis import CoaddAnalysisTask
from .utils import (Filenamer, Enforcer, concatenateCatalogs, checkIdLists, addPatchColumn,
                    calibrateCoaddSourceCatalog, fluxToPlotString, writeParquet, getRepoInfo)
from .plotUtils import OverlapsStarGalaxyLabeller, labelCamera, setPtSize

import lsst.afw.geom as afwGeom
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
    fitLineUpper = ListField(dtype=float, doc="List of [intercept, slope] for upper fit line limits")
    fitLineLower = ListField(dtype=float, doc="List of [intercept, slope] for lower fit line limits")

    @classmethod
    def fromValues(cls, description, subDescription, plot, coeffs, requireGreater={}, requireLess={},
                   fitLineUpper=[], fitLineLower=[]):
        self = cls()
        self.description = description
        self.subDescription = subDescription
        self.plot = plot
        self.coeffs = coeffs
        self.requireGreater = requireGreater
        self.requireLess = requireLess
        self.fitLineUpper = fitLineUpper
        self.fitLineLower = fitLineLower
        return self


ivezicTransforms = {
    "wPerp": ColorTransform.fromValues("Ivezic w perpendicular", " (griBlue)", True,
                                       {"HSC-G": -0.227, "HSC-R": 0.792, "HSC-I": -0.567, "": 0.050},
                                       {"wPara": -0.2}, {"wPara": 0.6},
                                       fitLineUpper=[0.66, -0.46], fitLineLower=[0.15, -0.46]),
    "xPerp": ColorTransform.fromValues("Ivezic x perpendicular", " (griRed)", True,
                                       {"HSC-G": 0.707, "HSC-R": -0.707, "": -0.988},
                                       {"xPara": 0.8}, {"xPara": 1.6},
                                       fitLineUpper = [0.82, -0.4], fitLineLower=[0.51, -0.4]),
    "yPerp": ColorTransform.fromValues("Ivezic y perpendicular", " (rizRed)", True,
                                       {"HSC-R": -0.270, "HSC-I": 0.800, "HSC-Z": -0.534, "": 0.054},
                                       {"yPara": 0.1}, {"yPara": 1.2},
                                       fitLineUpper=[1.08, -0.435], fitLineLower=[0.06, -0.435]),
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
                      default=["slot_Centroid_flag", "slot_Shape_flag",
                               "base_PsfFlux_flag", "modelfit_CModel_flag",
                               "base_PixelFlags_flag_saturated", "base_ClassificationExtendedness_flag"])
    analysis = ConfigField(dtype=AnalysisConfig, doc="Analysis plotting options")
    transforms = ConfigDictField(keytype=str, itemtype=ColorTransform, default={},
                                 doc="Color transformations to analyse")
    fluxFilter = Field(dtype=str, default="HSC-I", doc=("Filter to use for plotting against magnitude and "
                                                        "setting star/galaxy classification"))
    fluxFilterGeneric = Field(dtype=str, default="i", doc=("Filter to use for plotting against magnitude "
                                                           "and setting star/galaxy classification"))
    srcSchemaMap = DictField(keytype=str, itemtype=str, default=None, optional=True,
                             doc="Mapping between different stack (e.g. HSC vs. LSST) schema names")
    toMilli = Field(dtype=bool, default=True, doc="Print stats in milli units (i.e. mas, mmag)?")
    doPlotPcaColors = Field(dtype=bool, default=True, doc="Create the Ivezic PCA color offset plots?")
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
        self.fluxFilter = None
        for patchRefList in patchRefsByFilter.values():
            for dataRef in patchRefList:
                if dataRef.dataId["filter"] == self.config.fluxFilter:
                    self.fluxFilter = self.config.fluxFilter
                    break
                if dataRef.dataId["filter"] == self.config.fluxFilterGeneric:
                    self.fluxFilter = self.config.fluxFilterGeneric
                    break
        if self.fluxFilter is None:
            raise TaskError("Flux filter from config not found (neither {0:s} nor the generic {1:s}".
                            format(self.config.fluxFilter, self.config.fluxFilterGeneric))
        self.log.info("Flux filter for plotting and primary star/galaxy classifiation is: {0:s}".
                      format(self.fluxFilter))
        for patchRefList in patchRefsByFilter.values():
            for dataRef in patchRefList:
                if dataRef.dataId["filter"] == self.fluxFilter:
                    patchList.append(dataRef.dataId["patch"])
                    if repoInfo is None:
                        repoInfo = getRepoInfo(dataRef, coaddName=self.config.coaddName,
                                               coaddDataset="Coadd_forced_src")
        self.log.info("Size of patchList with full color coverage: {:d}".format(len(patchList)))

        # Only adjust the schema names necessary here (rather than attaching the full alias schema map)
        self.fluxColumn = self.config.analysis.fluxColumn
        self.classificationColumn = "base_ClassificationExtendedness_value"
        self.flags = self.config.flags
        if repoInfo.hscRun is not None:
            self.fluxColumn = self.config.srcSchemaMap[self.config.analysis.fluxColumn] + "_flux"
            self.classificationColumn = self.config.srcSchemaMap[self.classificationColumn]
            self.flags = [self.config.srcSchemaMap[flag] for flag in self.flags]

        filenamer = Filenamer(repoInfo.butler, "plotColor", repoInfo.dataId)
        forcedCatalogsByFilter = {ff: self.readCatalogs(patchRefList,
                                                        self.config.coaddName + "Coadd_forced_src") for
                                  ff, patchRefList in patchRefsByFilter.items()}

        for cat in forcedCatalogsByFilter.values():
            calibrateCoaddSourceCatalog(cat, self.config.analysis.coaddZp)
        forcedCatalogsByFilter = self.correctForGalacticExtinction(forcedCatalogsByFilter, repoInfo.tractInfo)
        # self.plotGalaxyColors(catalogsByFilter, filenamer, dataId)
        if self.config.doPlotPcaColors or self.config.doWriteParquetTables:
            forced = self.transformCatalogs(forcedCatalogsByFilter, self.config.transforms,
                                            hscRun=repoInfo.hscRun)

        # Create and write parquet tables
        if self.config.doWriteParquetTables:
            tableFilenamer = Filenamer(repoInfo.butler, 'qaTableColor', repoInfo.dataId)
            writeParquet(forced, tableFilenamer(repoInfo.dataId, description='forced'))
            if self.config.writeParquetOnly:
                self.log.info("Exiting after writing Parquet tables.  No plots generated.")
                return

        if self.config.doPlotPcaColors:
            self.plotStarColors(forced, filenamer, NumStarLabeller(len(forcedCatalogsByFilter)),
                                repoInfo.dataId, camera=repoInfo.camera, tractInfo=repoInfo.tractInfo,
                                patchList=patchList, hscRun=repoInfo.hscRun)
        for fluxColumn in ["base_PsfFlux_flux", "modelfit_CModel_flux"]:
            self.plotStarColorColor(forcedCatalogsByFilter, filenamer, repoInfo.dataId, fluxColumn,
                                    camera=repoInfo.camera, tractInfo=repoInfo.tractInfo,
                                    patchList=patchList, hscRun=repoInfo.hscRun)

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

    def correctForGalacticExtinction(self, catalog, tractInfo):
        """Apply a per-field correction for Galactic Extinction using hard-wired values

        These numbers come from:
        http://irsa.ipac.caltech.edu/applications/DUST/
        Filt LamEff A/E(B-V) A(mag)
               (um) S&F2011
        UD_COSMOS_9813: 150.25, 2.23  WIDE_VVDS_9796: 337.78, 0.74  WIDE_GAMMA15H_9615: 216.30, 0.74
        SDSS g 0.4717 3.303 0.054     SDSS g 0.4717 3.303 0.247     SDSS g 0.4717 3.303 0.093
        SDSS r 0.6165 2.285 0.038     SDSS r 0.6165 2.285 0.171     SDSS r 0.6165 2.285 0.064
        SDSS i 0.7476 1.698 0.028     SDSS i 0.7476 1.698 0.127     SDSS i 0.7476 1.698 0.048
        SDSS z 0.8923 1.263 0.021     SDSS z 0.8923 1.263 0.094     SDSS z 0.8923 1.263 0.035
        WIDE_8766: 35.70, -3.72       WIDE_8767: 37.19, -3.72
        SDSS g 0.4717 3.303 0.079     SDSS g 0.4717 3.303 0.095
        SDSS r 0.6165 2.285 0.055     SDSS r 0.6165 2.285 0.066
        SDSS i 0.7476 1.698 0.041     SDSS i 0.7476 1.698 0.049
        SDSS z 0.8923 1.263 0.030     SDSS z 0.8923 1.263 0.036

        Note that they are derived for SDSS filters, so are not quite right for HSC filters
        and do not include values for bands redder than z.

        Also note that the only fields included are the 5 tracts in the RC + RC2 datasets.
        This is just a placeholder until a per-object implementation is added in DM-13519.
        """
        galacticExtinction = {
            "UD_COSMOS_9813": {"centerCoord": afwGeom.SpherePoint(150.25, 2.23, afwGeom.degrees),
                               "HSC-G": 0.054, "HSC-R": 0.038, "HSC-I": 0.028, "HSC-Z": 0.021},
            "WIDE_VVDS_9796": {"centerCoord": afwGeom.SpherePoint(337.78, 0.74, afwGeom.degrees),
                               "HSC-G": 0.247, "HSC-R": 0.171, "HSC-I": 0.127, "HSC-Z": 0.094},
            "WIDE_GAMMA15H_9615": {"centerCoord": afwGeom.SpherePoint(216.3, 0.74, afwGeom.degrees),
                                   "HSC-G": 0.093, "HSC-R": 0.064, "HSC-I": 0.048, "HSC-Z": 0.035},
            "WIDE_8766": {"centerCoord": afwGeom.SpherePoint(35.70, -3.72, afwGeom.degrees),
                          "HSC-G": 0.079, "HSC-R": 0.055, "HSC-I": 0.041, "HSC-Z": 0.030},
            "WIDE_8767": {"centerCoord": afwGeom.SpherePoint(37.19, -3.72, afwGeom.degrees),
                          "HSC-G": 0.095, "HSC-R": 0.066, "HSC-I": 0.049, "HSC-Z": 0.036}}

        geFound = False
        for fieldName, geEntry in galacticExtinction.items():
            if tractInfo.contains(geEntry["centerCoord"]):
                geFound = True
                break
        if geFound:
            for ff in catalog.keys():
                if ff in galacticExtinction[fieldName]:
                    fluxKeys, errKeys = getFluxKeys(catalog[ff].schema)
                    factor = 10.0**(0.4*galacticExtinction[fieldName][ff])
                    for name, key in list(fluxKeys.items()) + list(errKeys.items()):
                        catalog[ff][key] *= factor
                    self.log.info("Applying Galactic Extinction correction A_{0:s} = {1:.3f}".
                                  format(ff, galacticExtinction[fieldName][ff]))
                else:
                    self.log.warn("Do not have A_X for filter {0:s}.  "
                                  "No Galactic Extinction correction applied for that filter".format(ff))
        else:
            self.log.warn("Do not have Galactic Extinction for tract {0:d} at {1:s}.  "
                          "No Galactic Extinction correction applied".
                          format(tractInfo.getId(), str(tractInfo.getCtrCoord())))
        return catalog

    def transformCatalogs(self, catalogs, transforms, flagsCats=None, hscRun=None):
        """
        Transform catalog entries according to the color transform given

        Parameters
        ----------
        catalogs : `dict` of `lsst.afw.table.source.source.SourceCatalog`s
           One dict entry per filter
        transforms : `dict` of `lsst.pipe.analysis.colorAnalysis.ColorTransform`s
           One dict entry per filter-dependent transform definition
        flagsCats : `dict` of `lsst.afw.table.source.source.SourceCatalog`s
           One dict entry per filter.  Source lists must be identical to those in catalogs.
           This is to provide a way to use a different catalog containing the flags of interest
           for source filtering (e.g. forced catalogs do not have all the flags defined in unforced
           catalogs, but the source lists are identical)
        hscRun : `str` or `NoneType`
           A string representing "HSCPIPE_VERSION" fits header if the data were processed with
           the (now obsolete, but old reruns still exist) "HSC stack", None otherwise
        """
        if flagsCats is None:
            flagsCats = catalogs

        template = list(catalogs.values())[0]
        num = len(template)
        assert all(len(cat) == num for cat in catalogs.values())

        mapper = afwTable.SchemaMapper(template.schema)
        mapper.addMinimalSchema(afwTable.SourceTable.makeMinimalSchema())
        schema = mapper.getOutputSchema()

        for col in transforms:
            doAdd = True
            for ff in transforms[col].coeffs:
                if ff != "" and ff not in catalogs:
                    doAdd = False
            if doAdd:
                schema.addField(col, float, transforms[col].description + transforms[col].subDescription)
        schema.addField("numStarFlags", type=np.int32, doc="Number of times source was flagged as star")
        badKey = schema.addField("qaBad_flag", type="Flag", doc="Is this a bad source for color qa analyses?")
        schema.addField(self.fluxColumn, type=np.float64, doc="Flux from filter " + self.fluxFilter)

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

        new[self.fluxColumn][:] = catalogs[self.fluxFilter][self.fluxColumn]

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

    def plotStarColorColor(self, catalogs, filenamer, dataId, fluxColumn, butler=None, camera=None,
                           tractInfo=None, patchList=None, hscRun=None):
        num = len(list(catalogs.values())[0])
        zp = 0.0
        mags = {ff: zp - 2.5*np.log10(catalogs[ff][fluxColumn]) for ff in catalogs}

        bad = np.zeros(num, dtype=bool)
        for cat in catalogs.values():
            for flag in self.flags:
                if flag in cat.schema:
                    bad |= cat[flag]

        bright = mags[self.fluxFilter] < self.config.analysis.magThreshold
        prettyBrightThreshold = self.config.analysis.magThreshold
        prettyBright = mags[self.fluxFilter] < prettyBrightThreshold

        # Determine number of filters object is classified as a star
        numStarFlags = np.zeros(num)
        for cat in catalogs.values():
            numStarFlags += np.where(cat[self.classificationColumn] < 0.5, 1, 0)

        # Select as a star if classified as such in self.config.fluxFilter
        isStarFlag = catalogs[self.fluxFilter][self.classificationColumn] < 0.5
        # Require stellar classification in self.fluxFilter and at least one other filter for fits
        good = isStarFlag & (numStarFlags >= 2) & ~bad & bright
        decentStars = isStarFlag & ~bad & prettyBright
        decentGalaxies = ~isStarFlag & ~bad & prettyBright

        combined = self.transformCatalogs(catalogs, straightTransforms, hscRun=hscRun)[good].copy(True)
        filters = set(catalogs.keys())
        color = lambda c1, c2: (mags[c1] - mags[c2])[good]
        decentColorStars = lambda c1, c2: (mags[c1] - mags[c2])[decentStars]
        decentStarsMag = mags[self.fluxFilter][decentStars]
        decentColorGalaxies = lambda c1, c2: (mags[c1] - mags[c2])[decentGalaxies]
        decentGalaxiesMag = mags[self.fluxFilter][decentGalaxies]
        unitStr = "mmag" if self.config.toMilli else "mag"
        fluxColStr = fluxToPlotString(fluxColumn)
        if filters.issuperset(set(("HSC-G", "HSC-R", "HSC-I"))):
            # Do a linear fit to regions defined in Ivezic transforms
            transform = self.config.transforms["wPerp"]
            xFitRange1 = transform.requireGreater["wPara"]
            xFitRange2 = transform.requireLess["wPara"]
            fitLineUpper = transform.fitLineUpper
            fitLineLower = transform.fitLineLower
            xRange = (-0.6, 2.0)
            yRange = (-0.6, 3.0)
            nameStr = "gri" + fluxColStr + "-wFit"
            self.log.info("nameStr = {:s}".format(nameStr))
            wPerpFit = colorColorPolyFitPlot(dataId, filenamer(dataId, description=nameStr, style="fit"),
                                             self.log, color("HSC-G", "HSC-R"), color("HSC-R", "HSC-I"),
                                             "g - r  [{0:s}]".format(fluxColStr),
                                             "r - i  [{0:s}]".format(fluxColStr), self.fluxFilter,
                                             xRange=xRange, yRange=yRange, order=1,
                                             xFitRange=(xFitRange1, xFitRange2), yFitRange=(0.05, 0.35),
                                             fitLineUpper=fitLineUpper, fitLineLower=fitLineLower,
                                             magThreshold=self.config.analysis.magThreshold, camera=camera,
                                             hscRun=hscRun, unitScale=self.unitScale)
            transform = self.config.transforms["xPerp"]
            xFitRange1 = transform.requireGreater["xPara"]
            xFitRange2 = transform.requireLess["xPara"]
            fitLineUpper = transform.fitLineUpper
            fitLineLower = transform.fitLineLower
            nameStr = "gri" + fluxColStr + "-xFit"
            self.log.info("nameStr = {:s}".format(nameStr))
            xPerpFit = colorColorPolyFitPlot(dataId, filenamer(dataId, description=nameStr, style="fit"),
                                             self.log, color("HSC-G", "HSC-R"), color("HSC-R", "HSC-I"),
                                             "g - r  [{0:s}]".format(fluxColStr),
                                             "r - i  [{0:s}]".format(fluxColStr), self.fluxFilter,
                                             xRange=xRange, yRange=yRange, order=1,
                                             xFitRange=(xFitRange1, xFitRange2), yFitRange=(0.2, 0.5),
                                             fitLineUpper=fitLineUpper, fitLineLower=fitLineLower,
                                             magThreshold=self.config.analysis.magThreshold, camera=camera,
                                             hscRun=hscRun, unitScale=self.unitScale)
            # Lower branch only; upper branch is noisy due to astrophysics
            nameStr = "gri" + fluxColStr
            self.log.info("nameStr = {:s}".format(nameStr))
            fitLineUpper = [1.32, -0.65]
            fitLineLower = [0.24, -0.49]
            poly = colorColorPolyFitPlot(dataId, filenamer(dataId, description=nameStr, style="fit"),
                                         self.log, color("HSC-G", "HSC-R"), color("HSC-R", "HSC-I"),
                                         "g - r  [{0:s}]".format(fluxColStr),
                                         "r - i  [{0:s}]".format(fluxColStr), self.fluxFilter,
                                         xRange=xRange, yRange=yRange, order=3,
                                         xFitRange=(0.23, 1.2), yFitRange=(0.05, 0.6),
                                         fitLineUpper=fitLineUpper, fitLineLower=fitLineLower,
                                         magThreshold=self.config.analysis.magThreshold, camera=camera,
                                         hscRun=hscRun, unitScale=self.unitScale)
            # Make a color-color plot with both stars and galaxies, less pruning, and no fit
            if fluxColumn is not "base_PsfFlux_flux":
                self.log.info("nameStr: noFit ({1:s}) = {0:s}".format(nameStr, fluxColumn))
                colorColorPlot(dataId, filenamer(dataId, description=nameStr, style="noFit"), self.log,
                               decentColorStars("HSC-G", "HSC-R"), decentColorStars("HSC-R", "HSC-I"),
                               decentColorGalaxies("HSC-G", "HSC-R"), decentColorGalaxies("HSC-R", "HSC-I"),
                               decentStarsMag, decentGalaxiesMag,
                               "g - r  [{0:s}]".format(fluxColStr),
                               "r - i  [{0:s}]".format(fluxColStr), self.fluxFilter, fluxColStr,
                               xRange=(xRange[0], xRange[1] + 0.6), yRange=yRange,
                               magThreshold=prettyBrightThreshold, camera=camera, hscRun=hscRun,
                               unitScale=self.unitScale)
            shortName = "griDistance" + fluxColStr
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(combined, ColorColorDistance("g", "r", "i", poly, xMin=0.2, xMax=1.2),
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
            fitLineUpper = transform.fitLineUpper
            fitLineLower = transform.fitLineLower
            xRange = (-0.6, 2.7)
            yRange = (-0.4, 1.2)
            nameStr = "riz" + fluxColStr + "-yFit"
            self.log.info("nameStr = {:s}".format(nameStr))
            yPerpFit = colorColorPolyFitPlot(dataId, filenamer(dataId, description=nameStr, style="fit"),
                                             self.log, color("HSC-R", "HSC-I"), color("HSC-I", "HSC-Z"),
                                             "r - i  [{0:s}]".format(fluxColStr),
                                             "i - z  [{0:s}]".format(fluxColStr), self.fluxFilter,
                                             xRange=xRange, yRange=yRange, order=1,
                                             xFitRange=(0.4, 1.7), yFitRange=(0.17, 0.63),
                                             fitLineUpper=fitLineUpper, fitLineLower=fitLineLower,
                                             magThreshold=self.config.analysis.magThreshold, camera=camera,
                                             hscRun=hscRun, unitScale=self.unitScale)
            nameStr = "riz" + fluxColStr
            fitLineUpper = [0.92, -0.275]
            fitLineLower = [0.048, -0.55]
            self.log.info("nameStr = {:s}".format(nameStr))
            poly = colorColorPolyFitPlot(dataId, filenamer(dataId, description=nameStr, style="fit"),
                                         self.log, color("HSC-R", "HSC-I"), color("HSC-I", "HSC-Z"),
                                         "r - i  [{0:s}]".format(fluxColStr),
                                         "i - z  [{0:s}]".format(fluxColStr), self.fluxFilter,
                                         xRange=xRange, yRange=yRange, order=2,
                                         xFitRange=(0.0, 1.45), yFitRange=(-0.03, 0.58),
                                         fitLineUpper=fitLineUpper, fitLineLower=fitLineLower,
                                         magThreshold=self.config.analysis.magThreshold, camera=camera,
                                         hscRun=hscRun, unitScale=self.unitScale)
            # Make a color-color plot with both stars and galaxies, less pruning, and no fit
            if fluxColumn is not "base_PsfFlux_flux":
                self.log.info("nameStr: noFit ({1:s}) = {0:s}".format(nameStr, fluxColumn))
                colorColorPlot(dataId, filenamer(dataId, description=nameStr, style="noFit"), self.log,
                               decentColorStars("HSC-R", "HSC-I"), decentColorStars("HSC-I", "HSC-Z"),
                               decentColorGalaxies("HSC-R", "HSC-I"), decentColorGalaxies("HSC-I", "HSC-Z"),
                               decentStarsMag, decentGalaxiesMag,
                               "r - i  [{0:s}]".format(fluxColStr),
                               "i - z  [{0:s}]".format(fluxColStr), self.fluxFilter, fluxColStr,
                               xRange=xRange, yRange=(yRange[0], yRange[1] + 0.2),
                               magThreshold=prettyBrightThreshold, camera=camera, hscRun=hscRun,
                               unitScale=self.unitScale)
            shortName = "rizDistance" + fluxColStr
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(combined, ColorColorDistance("r", "i", "z", poly, xMin=0.0, xMax=0.7),
                               "rizDistance (%s)" % unitStr, shortName, self.config.analysis,
                               flags=["qaBad_flag"], qMin=-0.1, qMax=0.1,
                               labeller=NumStarLabeller(len(catalogs)),
                               ).plotAll(dataId, filenamer, self.log,
                                         Enforcer(requireLess={"star": {"stdev": 0.02}}), camera=camera,
                                         tractInfo=tractInfo, patchList=patchList, hscRun=hscRun)
        if filters.issuperset(set(("HSC-I", "HSC-Z", "HSC-Y"))):
            nameStr = "izy" + fluxColStr
            self.log.info("nameStr = {:s}".format(nameStr))
            fitLineUpper = [0.42, -0.15]
            fitLineLower = [-0.012, -0.53]
            xRange = (-0.5, 1.3)
            yRange = (-0.4, 0.8)
            poly = colorColorPolyFitPlot(dataId, filenamer(dataId, description=nameStr, style="fit"),
                                         self.log, color("HSC-I", "HSC-Z"), color("HSC-Z", "HSC-Y"),
                                         "i - z  [{0:s}]".format(fluxColStr),
                                         "z - y  [{0:s}]".format(fluxColStr), self.fluxFilter,
                                         xRange=xRange, yRange=yRange, order=2,
                                         xFitRange=(-0.05, 0.8), yFitRange=(-0.06, 0.3),
                                         fitLineUpper=fitLineUpper, fitLineLower=fitLineLower,
                                         magThreshold=self.config.analysis.magThreshold, camera=camera,
                                         hscRun=hscRun, unitScale=self.unitScale)
            # Make a color-color plot with both stars and galaxies, less pruning, and no fit
            if fluxColumn is not "base_PsfFlux_flux":
                self.log.info("nameStr: noFit ({1:s}) = {0:s}".format(nameStr, fluxColumn))
                colorColorPlot(dataId, filenamer(dataId, description=nameStr, style="noFit"), self.log,
                               decentColorStars("HSC-I", "HSC-Z"), decentColorStars("HSC-Z", "HSC-Y"),
                               decentColorGalaxies("HSC-I", "HSC-Z"), decentColorGalaxies("HSC-Z", "HSC-Y"),
                               decentStarsMag, decentGalaxiesMag,
                               "i - z  [{0:s}]".format(fluxColStr),
                               "z - y  [{0:s}]".format(fluxColStr), self.fluxFilter, fluxColStr,
                               xRange=xRange, yRange=(yRange[0], yRange[1] + 0.2),
                               magThreshold=prettyBrightThreshold, camera=camera, hscRun=hscRun,
                               unitScale=self.unitScale)
            shortName = "izyDistance" + fluxColStr
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(combined, ColorColorDistance("i", "z", "y", poly, xMin=0.0, xMax=0.3),
                               "izyDistance (%s)" % unitStr, shortName, self.config.analysis,
                               flags=["qaBad_flag"], qMin=-0.1, qMax=0.1,
                               labeller=NumStarLabeller(len(catalogs)),
                               ).plotAll(dataId, filenamer, self.log,
                                         Enforcer(requireLess={"star": {"stdev": 0.02}}), camera=camera,
                                         tractInfo=tractInfo, patchList=patchList, hscRun=hscRun)

        if filters.issuperset(set(("HSC-Z", "NB0921", "HSC-Y"))):
            nameStr = "z9y" + fluxColStr
            self.log.info("nameStr = {:s}".format(nameStr))
            fitLineUpper = [0.195, -0.4]
            fitLineLower = [-0.018, -0.86]
            xRange = (-0.3, 0.45)
            yRange = (-0.2, 0.5)
            poly = colorColorPolyFitPlot(dataId, filenamer(dataId, description=nameStr, style="fit"),
                                         self.log, color("HSC-Z", "NB0921"), color("NB0921", "HSC-Y"),
                                         "z-n921  [{0:s}]".format(fluxColStr),
                                         "n921-y  [{0:s}]".format(fluxColStr), self.fluxFilter,
                                         xRange=xRange, yRange=yRange,
                                         order=2, xFitRange=(-0.07, 0.16), yFitRange=(-0.002, 0.15),
                                         fitLineUpper=fitLineUpper, fitLineLower=fitLineLower,
                                         magThreshold=self.config.analysis.magThreshold, camera=camera,
                                         hscRun=hscRun, unitScale=self.unitScale)
            # Make a color-color plot with both stars and galaxies, less pruning, and no fit
            if fluxColumn is not "base_PsfFlux_flux":
                self.log.info("nameStr: noFit ({1:s}) = {0:s}".format(nameStr, fluxColumn))
                colorColorPlot(dataId, filenamer(dataId, description=nameStr, style="noFit"), self.log,
                               decentColorStars("HSC-Z", "NB0921"), decentColorStars("NB0921", "HSC-Y"),
                               decentColorGalaxies("HSC-Z", "NB0921"), decentColorGalaxies("NB0921", "HSC-Y"),
                               decentStarsMag, decentGalaxiesMag,
                               "z-n921  [{0:s}]".format(fluxColStr),
                               "n921-y  [{0:s}]".format(fluxColStr), self.fluxFilter, fluxColStr,
                               xRange=xRange, yRange=(yRange[0] - 0.05, yRange[1] + 0.05),
                               magThreshold=prettyBrightThreshold, camera=camera, hscRun=hscRun,
                               unitScale=self.unitScale)
            shortName = "z9yDistance" + fluxColStr
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(combined, ColorColorDistance("z", "n921", "y", poly, xMin=-0.0, xMax=0.13),
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


def colorColorPolyFitPlot(dataId, filename, log, xx, yy, xLabel, yLabel, filterStr, xRange=None, yRange=None,
                          order=1, iterations=3, rej=3.0, xFitRange=None, yFitRange=None, fitLineUpper=None,
                          fitLineLower=None, numBins="auto", hscRun=None, logger=None, magThreshold=99.9,
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
    else:
        yRange = (0.9*yy.min(), 1.1*yy.max())

    xLine = np.linspace(xRange[0], xRange[1], 1000)
    if fitLineUpper:
        yLineUpper = fitLineUpper[0] + fitLineUpper[1]*xLine
    if fitLineLower:
        yLineLower = fitLineLower[0] + fitLineLower[1]*xLine

    # Include vertical xFitRange for clipping points in the first iteration
    selectXRange = np.ones_like(xx, dtype=bool) if not xFitRange else ((xx > xFitRange[0]) &
                                                                       (xx < xFitRange[1]))
    # Include horizontal yFitRange for clipping points in the first iteration
    selectYRange = np.ones_like(xx, dtype=bool) if not yFitRange else ((yy > yFitRange[0]) &
                                                                       (yy < yFitRange[1]))
    selectUpper = np.ones_like(xx, dtype=bool) if not fitLineUpper else (yy <
                                                                         fitLineUpper[0] + fitLineUpper[1]*xx)
    selectLower = np.ones_like(xx, dtype=bool) if not fitLineLower else (yy >
                                                                         fitLineLower[0] + fitLineLower[1]*xx)

    select = np.ones_like(xx, dtype=bool)
    for sel in [selectXRange, selectYRange, selectUpper, selectLower]:
        select &= sel

    keep = np.ones_like(xx, dtype=bool)
    for ii in range(iterations):
        keep &= select
        poly = np.polyfit(xx[keep], yy[keep], order)
        dy = yy - np.polyval(poly, xx)
        q1, q3 = np.percentile(dy[keep], [25, 75])
        clip = rej*0.74*(q3 - q1)
        keep = np.logical_not(np.abs(dy) > clip)
        # After the first iteration, reset the vertical and horizontal clipping to be less restrictive
        if ii == 0:
            if xFitRange:
                xMin = xFitRange[0] - 0.06*(xFitRange[1] - xFitRange[0])
                xMax = xFitRange[1] + 0.06*(xFitRange[1] - xFitRange[0])
            if yFitRange:
                yMin = yFitRange[0] - 0.06*(yFitRange[1] - yFitRange[0])
                yMax = yFitRange[1] + 0.06*(yFitRange[1] - yFitRange[0])
            selectXRange = np.ones_like(xx, dtype=bool) if not xFitRange else ((xx > xMin) & (xx < xMax))
            selectYRange = np.ones_like(xx, dtype=bool) if not yFitRange else ((yy > yMin) & (yy < yMax))
            for sel in [selectXRange, selectYRange, selectUpper, selectLower]:
                select &= sel

    log.info("Number of iterations in polynomial fit: {:d}".format(ii + 1))
    keep &= select
    nKeep = np.sum(keep)
    if nKeep < order:
        raise RuntimeError(
            "Not enough good data points ({0:d}) for polynomial fit of order {1:d}".format(nKeep, order))

    poly = np.polyfit(xx[keep], yy[keep], order)
    yLine = np.polyval(poly, xLine)

    kwargs = dict(s=3, marker="o", lw=0, alpha=0.4)
    axes[0].scatter(xx[keep], yy[keep], c="blue", label="used", **kwargs)
    axes[0].scatter(xx[~keep], yy[~keep], c="black", label="other", **kwargs)
    axes[0].set_xlabel(xLabel)
    axes[0].set_ylabel(yLabel, labelpad=-1)
    axes[0].plot(xLine, yLine, "r-")

    # Find index where poly and fit range intersect -- to calculate the local slopes of the fit to make
    # sure it is close to the fitLines (log a warning if they are not within 5%)
    message = ("{0:s} branch of the hard-coded lines for object selection does not cross the "
               "current polynomial fit.\nUsing the xFitRange {1:} to compute the local slope")
    try:
        crossIdxUpper = (np.argwhere(np.diff(np.sign(yOrthLine - yLineUpper)) != 0).reshape(-1) + 0)[0]
    except Exception:
        log.warnf(message, "Upper", xFitRange[1])
        crossIdxUpper = (np.abs(xLine - xFitRange[1])).argmin()
    try:
        crossIdxLower = (np.argwhere(np.diff(np.sign(yOrthLine - yLineLower)) != 0).reshape(-1) + 0)[0]
    except Exception:
        log.warnf(message, "Lower", xFitRange[0])
        crossIdxLower = (np.abs(xLine - xFitRange[0])).argmin()

    # Compute the slope of the two pixels +/-1% of line length from crossing point
    yOffset = int(0.01*len(yOrthLine))
    mUpper = ((yOrthLine[crossIdxUpper + yOffset] - yOrthLine[crossIdxUpper - yOffset])/
              (xLine[crossIdxUpper + yOffset] - xLine[crossIdxUpper - yOffset]))
    mLower = ((yOrthLine[crossIdxLower + yOffset] - yOrthLine[crossIdxLower - yOffset])/
              (xLine[crossIdxLower + yOffset] - xLine[crossIdxLower - yOffset]))
    bUpper = -yOrthLine[crossIdxUpper] - mUpper*xLine[crossIdxUpper]
    bLower = -yOrthLine[crossIdxLower] - mLower*xLine[crossIdxLower]
    # Rotate slope by 90 degrees for source selection lines
    mUpper = -1.0/mUpper
    mLower = -1.0/mLower
    bUpper = yOrthLine[crossIdxUpper] - mUpper*xLine[crossIdxUpper]
    bLower = yOrthLine[crossIdxLower] - mLower*xLine[crossIdxLower]
    message = ("{0:s} branch of the hard-coded lines for object selection does not match the local\nslope of "
               "the current polynomial fit.\n  --> Consider replacing {1:} with [{2:.3f}, {3:.3f}] "
               "(Line crosses fit at x = {4:.2f})")
    if (abs(200*(fitLineUpper[0] - bUpper)/(fitLineUpper[0] + bUpper)) > 5.0 or
            abs(200*(fitLineUpper[1] - mUpper)/(fitLineUpper[1] + mUpper)) > 5.0):
        log.warn(message.format("Upper", fitLineUpper, bUpper, mUpper, xLine[crossIdxUpper]))
    if (abs(200*(fitLineLower[0] - bLower)/(fitLineLower[0] + bLower)) > 5.0 or
            abs(200*(fitLineLower[1] - mLower)/(fitLineLower[1] + mLower)) > 5.0):
        log.warn(message.format("Lower", fitLineLower, bLower, mLower, xLine[crossIdxLower]))
    deltaX = abs(xRange[1] - xRange[0])
    deltaY = abs(yRange[1] - yRange[0])

    if xFitRange:
        # Shade region outside xFitRange
        kwargs = dict(facecolor="k", edgecolor="none", alpha=0.1)
        axes[0].axvspan(axes[0].get_xlim()[0], xFitRange[0], **kwargs)
        axes[0].axvspan(xFitRange[1], axes[0].get_xlim()[1], **kwargs)
    if yFitRange:
        # Shade region outside yFitRange
        xMin = abs(xFitRange[0] - xRange[0])/deltaX if xFitRange else 1
        xMax = abs(xFitRange[1] - xRange[0])/deltaX if xFitRange else 1
        kwargs = dict(xmin=xMin, xmax=xMax, facecolor="k", edgecolor="none", alpha=0.1)
        axes[0].axhspan(axes[0].get_ylim()[0], yFitRange[0], **kwargs)
        axes[0].axhspan(yFitRange[1], axes[0].get_ylim()[1], **kwargs)
    if fitLineUpper:
        scaleLine = 0.05*deltaX*max(1.0, min(3.0, abs(1.0/fitLineUpper[1])))
        xLineUpper = np.linspace(xLine[crossIdxUpper] - scaleLine, xLine[crossIdxUpper] + scaleLine, 100)
        yLineUpper = fitLineUpper[0] + fitLineUpper[1]*xLineUpper
        axes[0].plot(xLineUpper, yLineUpper, "g--")
    if fitLineLower:
        scaleLine = 0.05*deltaX*max(1.0, min(3.0, abs(1.0/fitLineLower[1])))
        xLineLower = np.linspace(xLine[crossIdxLower] - scaleLine, xLine[crossIdxLower] + scaleLine, 100)
        yLineLower = fitLineLower[0] + fitLineLower[1]*xLineLower
        axes[0].plot(xLineLower, yLineLower, "g--")

    # Label total number of objects of each data type
    kwargs = dict(va="center", fontsize=7)
    lenNumObj = max(len(str(len(xx[keep]))), len(str(len(xx[~keep]))))
    fdx = max((min(0.08*lenNumObj, 0.6), 0.33))
    xLoc, yLoc = xRange[0] + 0.05*deltaX, yRange[1] - 0.036*deltaY
    axes[0].text(xLoc, yLoc, "Nused  =", ha="left", color="blue", **kwargs)
    axes[0].text(xLoc + fdx*deltaX, yLoc, str(len(xx[keep])), ha="right", color="blue", **kwargs)
    axes[0].text(xRange[1] - 0.03*deltaX, yLoc, " [" + filterStr + " < " + str(magThreshold) + "]",
                 ha="right", color="blue", **kwargs)
    yLoc -= 0.044*(yRange[1] - yRange[0])
    axes[0].text(xLoc, yLoc, "Nother =", ha="left", color="black", **kwargs)
    axes[0].text(xLoc + fdx*deltaX,  yLoc, str(len(xx[~keep])), ha="right", color="black", **kwargs)

    # Label polynomial fit parameters to 2 decimal places
    polyStr = "y = {:.2f}".format(poly[len(poly) - 1])
    for i in range(1, len(poly)):
        index = len(poly) - 1 - i
        plusMinus = " - " if poly[index] < 0.0 else " + "
        exponent = "$^{" + str(i) + "}$" if i > 1 else ""
        polyStr += plusMinus + "{:.2f}".format(abs(poly[index])) + "x" + exponent
    xLoc = xRange[0] + 0.05*deltaX
    yLoc -= 0.055*deltaY
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
    mean = distance[good].mean()
    stdDev = distance[good].std()
    rms = np.sqrt(np.mean(distance[good]**2))
    log.info(("Statistics from {0:} of Distance to polynomial ({9:s}): {7:s}\'star\': " +
              "Stats(mean={1:.4f}; stdev={2:.4f}; num={3:d}; total={4:d}; " +
              "median={5:.4f}; clip={6:.4f}; forcedMean=None){8:s}").format(
             dataId, mean, stdDev, len(xx[keep]), len(xx), np.median(distance[good]),
             3.0*0.74*(q3 - q1), "{", "}", unitStr))
    # Get rid of LaTeX-specific characters for log message printing
    log.info("Polynomial fit: {:2}".format("".join(x for x in polyStr if x not in "{}$")))
    meanStr = "mean = {0:5.2f} ({1:s})".format(mean, unitStr)
    stdStr = "  std = {0:5.2f}".format(stdDev)
    rmsStr = "  rms = {0:5.2f}".format(rms)
    tractStr = "tract: {:d}".format(dataId["tract"])
    axes[1].set_xlabel("Distance to polynomial fit ({:s})".format(unitStr))
    axes[1].set_ylabel("Number")
    axes[1].set_yscale("log", nonposy="clip")
    axes[1].axvline(x=mean, color="black", linestyle="--")
    count, bins, ignored = axes[1].hist(distance[good], bins=numBins, range=(-4.0*stdDev, 4.0*stdDev),
                                        normed=True, color="green", alpha=0.8)
    axes[1].plot(bins, 1/(stdDev*np.sqrt(2*np.pi))*np.exp(-(bins-mean)**2/(2*stdDev**2)), color="red")
    kwargs = dict(xycoords="axes fraction", ha="right", va="center", fontsize=8, color="black")
    axes[1].annotate(meanStr, xy=(0.6, 0.96), **kwargs)
    axes[1].annotate(stdStr, xy=(0.4, 0.92), **kwargs)
    axes[1].annotate(rmsStr, xy=(0.4, 0.88), **kwargs)
    axes[1].annotate(tractStr, xy=(0.5, 1.04), xycoords="axes fraction", ha="center", va="center",
                     fontsize=10, color="green")

    if camera is not None:
        labelCamera(camera, plt, axes[0], 0.5, 1.04)
    if hscRun is not None:
        axes[0].set_title("HSC stack run: " + hscRun, color="#800080")

    plt.tight_layout(pad=2.5, w_pad=0.5, h_pad=1.0)
    fig.savefig(filename, dpi=120)
    plt.close(fig)

    return poly

def colorColorPlot(dataId, filename, log, xStars, yStars, xGalaxies, yGalaxies, magStars, magGalaxies,
                   xLabel, yLabel, filterStr, fluxColStr, xRange=None, yRange=None, hscRun=None,
                   logger=None, magThreshold=99.9, camera=None, unitScale=1.0):
    fig, axes = plt.subplots(1, 1)
    axes.tick_params(which="both", direction="in", labelsize=9)

    if xRange:
        axes.set_xlim(*xRange)
    else:
        xRange = (0.9*xStars.min(), 1.1*xStars.max())
    if yRange:
        axes.set_ylim(*yRange)

    vMin = min(magStars.min(), magGalaxies.min())
    vMax = min(magStars.max(), magGalaxies.max())

    ptSize = max(1, setPtSize(len(xGalaxies)) - 2)

    kwargs = dict(s=ptSize, marker="o", lw=0, vmin=vMin, vmax=vMax)
    axes.scatter(xGalaxies, yGalaxies, c=magGalaxies, cmap="autumn", label="galaxies", **kwargs)
    axes.scatter(xStars, yStars, c=magStars, cmap="winter", label="stars", **kwargs)
    axes.set_xlabel(xLabel)
    axes.set_ylabel(yLabel, labelpad=-1)

    # Label total number of objects of each data type
    deltaX = abs(xRange[1] - xRange[0])
    deltaY = abs(yRange[1] - yRange[0])
    lenNumObj = max(len(str(len(xStars))), len(str(len(xGalaxies))))
    fdx = max((min(0.095*lenNumObj, 0.9), 0.42))
    xLoc, yLoc = xRange[0] + 0.03*deltaX, yRange[1] - 0.038*deltaY
    kwargs = dict(va="center", fontsize=8)
    axes.text(xLoc, yLoc, "Ngals  =", ha="left", color="red", **kwargs)
    axes.text(xLoc + fdx*deltaX, yLoc, str(len(xGalaxies)) +
              " [" + filterStr + " < " + str(magThreshold) + "]", ha="right", color="red", **kwargs)
    axes.text(xLoc, 0.94*yLoc, "Nstars =", ha="left", color="blue", **kwargs)
    axes.text(xLoc + fdx*deltaX, 0.94*yLoc, str(len(xStars)) +
              " [" + filterStr + " < " + str(magThreshold) + "]", ha="right", color="blue", **kwargs)
    if camera is not None:
        labelCamera(camera, plt, axes, 0.5, 1.09)
    if hscRun is not None:
        axes.set_title("HSC stack run: " + hscRun, color="#800080")

    tractStr = "tract: {:d}".format(dataId["tract"])
    axes.annotate(tractStr, xy=(0.5, 1.04), xycoords="axes fraction", ha="center", va="center",
                  fontsize=10, color="green")

    mappableStars = plt.cm.ScalarMappable(cmap="winter_r", norm=plt.Normalize(vmin=vMin, vmax=vMax))
    mappableStars._A = []        # fake up the array of the scalar mappable. Urgh...
    cbStars = plt.colorbar(mappableStars, aspect=14, pad=-0.09)
    cbStars.ax.tick_params(labelsize=8)
    cbStars.set_label(filterStr + " [" + fluxColStr + "]: stars", rotation=270, labelpad=-24, fontsize=9)
    mappableGalaxies = plt.cm.ScalarMappable(cmap="autumn_r", norm=plt.Normalize(vmin=vMin, vmax=vMax))
    mappableGalaxies._A = []      # fake up the array of the scalar mappable. Urgh...
    cbGalaxies = plt.colorbar(mappableGalaxies, aspect=14)
    cbGalaxies.set_ticks([])
    cbGalaxies.set_label(filterStr + " [" + fluxColStr + "]: galaxies", rotation=270, labelpad=-6, fontsize=9)

    fig.savefig(filename, dpi=120)
    plt.close(fig)

    return None


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
