#!/usr/bin/env python

import matplotlib
matplotlib.use("Agg")  # noqa #402
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
np.seterr(all="ignore")  # noqa #402
import functools
import os
import scipy.stats as scipyStats
import astropy.units as u

from collections import defaultdict

from lsst.pex.config import Config, Field, ConfigField, ListField, DictField, ConfigDictField
from lsst.pipe.base import CmdLineTask, ArgumentParser, TaskRunner, TaskError, Struct
from lsst.pipe.drivers.utils import TractDataIdContainer
from .analysis import Analysis, AnalysisConfig
from .coaddAnalysis import CoaddAnalysisTask
from .utils import (Enforcer, concatenateCatalogs, getFluxKeys, addColumnsToSchema,
                    makeBadArray, addFlag, addIntFloatOrStrColumn, calibrateCoaddSourceCatalog,
                    fluxToPlotString, writeParquet, getRepoInfo, orthogonalRegression,
                    distanceSquaredToPoly, p2p1CoeffsFromLinearFit, linesFromP2P1Coeffs,
                    makeEqnStr, catColors, addMetricMeasurement, updateVerifyJob, computeMeanOfFrac,
                    calcQuartileClippedStats, savePlots)
from .plotUtils import (AllLabeller, OverlapsStarGalaxyLabeller, plotText, labelCamera, setPtSize,
                        determineExternalCalLabel, getPlotInfo)

import lsst.afw.image as afwImage
import lsst.geom as geom
import lsst.afw.table as afwTable
import lsst.verify as verify

__all__ = ["ColorTransform", "ivezicTransformsSDSS", "ivezicTransformsHSC", "straightTransforms",
           "NumStarLabeller", "ColorValueInFitRange", "ColorValueInPerpRange", "GalaxyColor",
           "ColorAnalysisConfig", "ColorAnalysisRunner", "ColorAnalysisTask", "ColorColorDistance",
           "SkyAnalysisRunner", "SkyAnalysisTask"]


class ColorTransform(Config):
    description = Field(dtype=str, doc="Description of the color transform")
    subDescription = Field(dtype=str, doc="Sub-description of the color transform (added detail)")
    plot = Field(dtype=bool, default=True, doc="Plot this color?")
    coeffs = DictField(keytype=str, itemtype=float, doc="Coefficients for each filter")
    x0 = Field(dtype=float, default=None, optional=True,
               doc="x Origin of P1/P2 axis on the color-color plane")
    y0 = Field(dtype=float, default=None, optional=True,
               doc="y Origin of P1/P2 axis on the color-color plane")
    requireGreater = DictField(keytype=str, itemtype=float, default=None, optional=True,
                               doc="Minimum values for colors so that this is useful")
    requireLess = DictField(keytype=str, itemtype=float, default=None, optional=True,
                            doc="Maximum values for colors so that this is useful")
    fitLineSlope = Field(dtype=float, default=None, optional=True, doc="Slope for fit line limits")
    fitLineUpperIncpt = Field(dtype=float, default=None, optional=True,
                              doc="Intercept for upper fit line limits")
    fitLineLowerIncpt = Field(dtype=float, default=None, optional=True,
                              doc="Intercept for lower fit line limits")

    @classmethod
    def fromValues(cls, description, subDescription, plot, coeffs, x0=None, y0=None, requireGreater=None,
                   requireLess=None, fitLineSlope=None, fitLineUpperIncpt=None, fitLineLowerIncpt=None):
        for require in [requireGreater, requireLess]:
            require = {} if require is None else require
        self = cls()
        self.description = description
        self.subDescription = subDescription
        self.plot = plot
        self.coeffs = coeffs
        self.x0 = x0
        self.y0 = y0
        self.requireGreater = requireGreater
        self.requireLess = requireLess
        self.fitLineSlope = fitLineSlope
        self.fitLineUpperIncpt = fitLineUpperIncpt
        self.fitLineLowerIncpt = fitLineLowerIncpt
        return self


ivezicTransformsSDSS = {
    "wPerp": ColorTransform.fromValues("Ivezic w perpendicular", " (griBlue)", True,
                                       {"SDSS-G": -0.227, "SDSS-R": 0.792, "SDSS-I": -0.567, "": 0.050},
                                       x0=0.4250, y0=0.0818,
                                       requireGreater={"wPara": -0.2}, requireLess={"wPara": 0.6}),
    "xPerp": ColorTransform.fromValues("Ivezic x perpendicular", " (griRed)", True,
                                       {"SDSS-G": 0.707, "SDSS-R": -0.707, "": -0.988},
                                       requireGreater={"xPara": 0.8}, requireLess={"xPara": 1.6}),
    "yPerp": ColorTransform.fromValues("Ivezic y perpendicular", " (rizRed)", True,
                                       {"SDSS-R": -0.270, "SDSS-I": 0.800, "SDSS-Z": -0.534, "": 0.054},
                                       x0=0.5763, y0=0.1900,
                                       requireGreater={"yPara": 0.1}, requireLess={"yPara": 1.2}),
    "wPara": ColorTransform.fromValues("Ivezic w parallel", " (griBlue)", False,
                                       {"SDSS-G": 0.928, "SDSS-R": -0.556, "SDSS-I": -0.372, "": -0.425}),
    "xPara": ColorTransform.fromValues("Ivezic x parallel", " (griRed)", False,
                                       {"SDSS-R": 1.0, "SDSS-I": -1.0}),
    "yPara": ColorTransform.fromValues("Ivezic y parallel", " (rizRed)", False,
                                       {"SDSS-R": 0.895, "SDSS-I": -0.448, "SDSS-Z": -0.447, "": -0.600}),
}

ivezicTransformsHSC = {
    "wPerp": ColorTransform.fromValues("Ivezic w perpendicular", " (griBlue)", True,
                                       {"HSC-G": -0.274, "HSC-R": 0.803, "HSC-I": -0.529, "": 0.041},
                                       x0=0.4481, y0=0.1546,
                                       requireGreater={"wPara": -0.2}, requireLess={"wPara": 0.6},
                                       fitLineSlope=-1/0.52, fitLineUpperIncpt=2.40, fitLineLowerIncpt=0.68),
    "xPerp": ColorTransform.fromValues("Ivezic x perpendicular", " (griRed)", True,
                                       {"HSC-G": -0.680, "HSC-R": 0.731, "HSC-I": -0.051, "": 0.792},
                                       x0=1.2654, y0=1.3675,
                                       requireGreater={"xPara": 0.8}, requireLess={"xPara": 1.6},
                                       fitLineSlope=-1/13.35, fitLineUpperIncpt=1.73, fitLineLowerIncpt=0.87),
    "yPerp": ColorTransform.fromValues("Ivezic y perpendicular", " (rizRed)", True,
                                       {"HSC-R": -0.227, "HSC-I": 0.793, "HSC-Z": -0.566, "": -0.017},
                                       x0=1.2219, y0=0.5183,
                                       requireGreater={"yPara": 0.1}, requireLess={"yPara": 1.2},
                                       fitLineSlope=-1/0.40, fitLineUpperIncpt=5.5, fitLineLowerIncpt=2.6),
    # The following still default to the SDSS values.  HSC coeffs will be derived on a subsequent
    # commit
    "wPara": ColorTransform.fromValues("Ivezic w parallel", " (griBlue)", False,
                                       {"HSC-G": 0.888, "HSC-R": -0.427, "HSC-I": -0.461, "": -0.478}),
    "xPara": ColorTransform.fromValues("Ivezic x parallel", " (griRed)", False,
                                       {"HSC-G": 0.075, "HSC-R": 0.922, "HSC-I": -0.997, "": -1.442}),
    "yPara": ColorTransform.fromValues("Ivezic y parallel", " (rizRed)", False,
                                       {"HSC-R": 0.928, "HSC-I": -0.557, "HSC-Z": -0.372, "": -1.332}),
    # The following three entries were derived in the process of calibrating the above coeffs (all three
    # RC2 tracts gave effectively the same fits).  May remove later if deemed no longer useful.
    "wFit": ColorTransform.fromValues("Straight line fit for wPerp range", " (griBlue)", False,
                                      {"HSC-G": 0.52, "HSC-R": -0.52, "": -0.08}),
    "xFit": ColorTransform.fromValues("Straight line fit for xperp range", " (griRed)", False,
                                      {"HSC-G": 13.35, "HSC-R": -13.35, "": -15.54}),
    "yFit": ColorTransform.fromValues("Straight line fit for yPerp range", " (rizRed)", False,
                                      {"HSC-R": 0.40, "HSC-I": -0.40, "": 0.03}),
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
    plot = ["star", "maybe"]

    def __init__(self, numBands):
        self.numBands = numBands

    def __call__(self, catalog):
        return np.array([0 if nn >= self.numBands else 2 if nn == 0 else 1 for nn in catalog["numStarFlags"]])


class ColorValueInFitRange(object):
    """Functor to produce color value if in the appropriate range

    Here the range is set by upper and lower lines roughly perpendicular to the
    fit where those lines cross the fit.  These numbers were previously determined
    and are (currently) hard-wired in the fitLineSlope, fitLineUpperIncpt, and
    fitLineLowerIncpt parameters in the ivezicTransformsHSC dict.
    """

    def __init__(self, column, xColor, yColor, fitLineSlope=None, fitLineUpperIncpt=None,
                 fitLineLowerIncpt=None, unitScale=1.0):
        self.column = column
        self.xColor = xColor
        self.yColor = yColor
        self.fitLineSlope = fitLineSlope
        self.fitLineUpperIncpt = fitLineUpperIncpt
        self.fitLineLowerIncpt = fitLineLowerIncpt
        self.unitScale = unitScale

    def __call__(self, principalColCats):
        good = self.yColor > self.fitLineLowerIncpt + self.fitLineSlope*self.xColor
        good &= self.yColor < self.fitLineUpperIncpt + self.fitLineSlope*self.xColor
        return np.where(good, principalColCats[self.column], np.nan)*self.unitScale


class ColorValueInPerpRange(object):
    """Functor to produce color value if in the appropriate range

    Here the range is set by the Ivezic etal. P1 limits provided in the requireGreater
    and requireLess parameters in the ivezicTransforms dict.
    """
    def __init__(self, column, requireGreater, requireLess, unitScale=1.0):
        self.column = column
        self.requireGreater = requireGreater
        self.requireLess = requireLess
        self.unitScale = unitScale

    def __call__(self, principalColCats):
        good = np.ones(len(principalColCats), dtype=bool)
        for col, value in self.requireGreater.items():
            good &= principalColCats[col] > value
        for col, value in self.requireLess.items():
            good &= principalColCats[col] < value
        return np.where(good, principalColCats[self.column], np.nan)*self.unitScale


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
                      default=["slot_Centroid_flag", "slot_Shape_flag", "base_PsfFlux_flag",
                               "modelfit_CModel_flag", "base_PixelFlags_flag_saturatedCenter",
                               "base_ClassificationExtendedness_flag"])
    analysis = ConfigField(dtype=AnalysisConfig, doc="Analysis plotting options")
    transforms = ConfigDictField(keytype=str, itemtype=ColorTransform, default={},
                                 doc="Color transformations to analyse")
    fluxFilter = Field(dtype=str, default="HSC-I", doc=("Filter to use for plotting against magnitude and "
                                                        "setting star/galaxy classification"))
    fluxFilterGeneric = Field(dtype=str, default="i", doc=("Filter to use for plotting against magnitude "
                                                           "and setting star/galaxy classification"))
    srcSchemaMap = DictField(keytype=str, itemtype=str, default=None, optional=True,
                             doc="Mapping between different stack (e.g. HSC vs. LSST) schema names")
    columnsToCopyFromRef = ListField(dtype=str,
                                     default=["detect_", "merge_peak_", "merge_measurement_", ],
                                     doc="List of \"startswith\" strings of column names to copy from "
                                         "*_ref to *_forced_src catalog.  All columns that start with one "
                                         "of these strings will be copied from the *_ref into the "
                                         "*_forced_src catalog.")
    extinctionCoeffs = DictField(keytype=str, itemtype=float, default=None, optional=True,
                                 doc="Dictionary of extinction coefficients for conversion from E(B-V) "
                                     "to extinction, A_filter")
    correctForGalacticExtinction = Field(dtype=bool, default=True,
                                         doc="Correct flux fields for Galactic Extinction?  Must have "
                                             "extinctionCoeffs config setup.")
    toMilli = Field(dtype=bool, default=True, doc="Print stats in milli units (i.e. mas, mmag)?")
    doPlotPrincipalColors = Field(dtype=bool, default=True,
                                  doc="Create the Ivezic Principal Color offset plots?")
    doPlotGalacticExtinction = Field(dtype=bool, default=True, doc="Create Galactic Extinction plots?")
    writeParquetOnly = Field(dtype=bool, default=False,
                             doc="Only write out Parquet tables (i.e. do not produce any plots)?")
    doWriteParquetTables = Field(dtype=bool, default=True,
                                 doc=("Write out Parquet tables (for subsequent interactive analysis)?"
                                      "\nNOTE: if True but fastparquet package is unavailable, a warning "
                                      "is issued and table writing is skipped."))
    plotRanges = DictField(keytype=str, itemtype=float,
                           default={"griX0": -0.6, "griX1": 2.0, "griY0": -0.6, "griY1": 3.0,
                                    "rizX0": -0.4, "rizX1": 3.0, "rizY0": -0.2, "rizY1": 1.5,
                                    "izyX0": -0.5, "izyX1": 1.3, "izyY0": -0.4, "izyY1": 0.8,
                                    "z9yX0": -0.3, "z9yX1": 0.45, "z9yY0": -0.2, "z9yY1": 0.5},
                           doc="Plot Ranges for various color-color combinations")

    def setDefaults(self):
        Config.setDefaults(self)
        self.transforms = ivezicTransformsHSC
        self.analysis.flags = []  # We remove bad source ourself
        self.analysis.fluxColumn = "base_PsfFlux_instFlux"
        self.analysis.magThreshold = 22.0  # RHL requested this limit
        if self.correctForGalacticExtinction:
            self.flags += ["galacticExtinction_flag"]

    def validate(self):
        Config.validate(self)
        if self.writeParquetOnly and not self.doWriteParquetTables:
            raise ValueError("Cannot writeParquetOnly if doWriteParquetTables is False")
        if self.correctForGalacticExtinction and self.extinctionCoeffs is None:
            raise ValueError("Must set appropriate extinctionCoeffs config.  See "
                             "config/hsc/extinctionCoeffs.py in obs_subaru for an example.")

        # If a wired origin was included in the config, check that it actually
        # lies on the wired P1 line (allowing for round-off error for precision
        # of coefficients specified)
        if self.transforms:
            for col, transform in self.transforms.items():
                if transform.plot and transform.x0 and transform.y0:
                    transformPerp = self.transforms[col]
                    transformPara = self.transforms[col[0] + "Para"]
                    p1p2Lines = linesFromP2P1Coeffs(list(transformPerp.coeffs.values()),
                                                    list(transformPara.coeffs.values()))
                    # Threshold of 2e-2 provides sufficient allowance for round-off error
                    if (np.abs((p1p2Lines.mP1 - p1p2Lines.mP2)*transformPerp.x0 +
                               (p1p2Lines.bP1 - p1p2Lines.bP2)) > 2e-2):
                        raise ValueError(("Wired origin for {} does not lie on line associated with wired "
                                          "PCA coefficients.  Check that the wired values are correct.").
                                         format(col))


class ColorAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        kwargs["subdir"] = parsedCmd.subdir
        FilterRefsDict = functools.partial(defaultdict, list)  # Dict for filter-->dataRefs
        tractFilterRefs = defaultdict(FilterRefsDict)  # tract-->filter-->dataRefs
        for patchRef in sum(parsedCmd.id.refList, []):
            # Make sure the actual input file requested exists (i.e. do not follow the parent chain)
            inputDataFile = patchRef.get("deepCoadd_forced_src_filename")[0]
            if parsedCmd.input not in parsedCmd.output:
                inputDataFile = inputDataFile.replace(parsedCmd.output, parsedCmd.input)
            if os.path.exists(inputDataFile):
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
            tractFilterRefs[tract] = {
                filterName: [patchRef for patchRef in filterRefs[filterName]
                             if patchRef.dataId["patch"] in keep] for filterName in filterRefs}
        for tract in bad:
            del tractFilterRefs[tract]

        # List of filters included on the command line
        parsedFilterList = [dataId["filter"] for dataId in parsedCmd.id.idList]
        for tract in tractFilterRefs:
            numFilters = 0
            for filterName in parsedFilterList:
                if filterName in tractFilterRefs[tract].keys():
                    numFilters += 1
                else:
                    parsedCmd.log.warn("No input data found for filter {0:s} of tract {1:d}".
                                       format(filterName, tract))
            if numFilters < 3:
                parsedCmd.log.warn("Must have at least 3 filters with data existing in the input repo. "
                                   "Only {0:d} exist of those requested ({1:}) for tract {2:d}. "
                                   "Skipping tract.".format(numFilters, parsedFilterList, tract))
                del tractFilterRefs[tract]
            if not tractFilterRefs[tract]:
                raise RuntimeError("No suitable datasets found.")

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
        parser.add_argument("--subdir", type=str, default="",
                            help=("Subdirectory below plots/color/tract-NNNN/ (useful for, "
                                  "e.g., subgrouping of Patches.  Ignored if only one Patch is "
                                  "specified, in which case the subdir is set to patch-NNN"))
        return parser

    def __init__(self, *args, **kwargs):
        CmdLineTask.__init__(self, *args, **kwargs)
        self.unitScale = 1000.0 if self.config.toMilli else 1.0

        self.verifyJob = verify.Job.load_metrics_package(subset="pipe_analysis")

    def runDataRef(self, patchRefsByFilter, subdir=""):
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
        uberCalLabel = determineExternalCalLabel(repoInfo, patchList[0], coaddName=self.config.coaddName)
        self.log.info(f"External calibration(s) used: {uberCalLabel}")
        subdir = "patch-" + str(patchList[0]) if len(patchList) == 1 else subdir
        repoInfo.dataId["subdir"] = "/" + subdir

        # Only adjust the schema names necessary here (rather than attaching the full alias schema map)
        self.fluxColumn = self.config.analysis.fluxColumn
        self.classificationColumn = "base_ClassificationExtendedness_value"
        self.flags = self.config.flags
        if repoInfo.hscRun:
            self.fluxColumn = self.config.srcSchemaMap[self.config.analysis.fluxColumn] + "_flux"
            self.classificationColumn = self.config.srcSchemaMap[self.classificationColumn]
            self.flags = [self.config.srcSchemaMap[flag] for flag in self.flags]

        byFilterForcedCats = {}
        areaDictAll = {}
        for (filterName, patchRefList) in patchRefsByFilter.items():
            coaddType = self.config.coaddName + "Coadd_forced_src"
            cat, areaDict = self.readCatalogs(patchRefList, coaddType, repoInfo)
            byFilterForcedCats[filterName] = cat
            areaDictAll.update(areaDict)

        self.forcedStr = "forced"
        for cat in byFilterForcedCats.values():
            calibrateCoaddSourceCatalog(cat, self.config.analysis.coaddZp)

        geLabel = "None"
        doPlotGalacticExtinction = False
        if self.correctForGalacticExtinction:
            # The per-object Galactic Extinction correction currently requires sims_catUtils to be setup
            # as it uses the EBVbase class to obtain E(B-V).  Putting this in a try/except to fall back
            # to the per-field correction until we can access the EBVbase class from an lsst_distrib
            # installation.
            try:
                byFilterForcedCats = self.correctForGalacticExtinction(byFilterForcedCats, repoInfo.tractInfo)
                doPlotGalacticExtinction = True
                geLabel = "Per Object"
            except Exception:
                byFilterForcedCats = self.correctFieldForGalacticExtinction(byFilterForcedCats,
                                                                            repoInfo.tractInfo)
                geLabel = "Per Field"

        plotInfoDict = getPlotInfo(repoInfo)
        plotInfoDict.update(dict(patchList=patchList, plotType="plotColor", subdir=subdir,
                                 hscRun=repoInfo.hscRun, tractInfo=repoInfo.tractInfo,
                                 dataId=repoInfo.dataId))

        geLabel = "GalExt: " + geLabel
        plotList = []
        if self.config.doPlotGalacticExtinction and doPlotGalacticExtinction:
            plotList.append(self.plotGalacticExtinction(byFilterForcedCats, plotInfoDict, areaDictAll,
                                                        geLabel=geLabel))

            # self.plotGalaxyColors(catalogsByFilter, filenamer, dataId)
        principalColCatsPsf = self.transformCatalogs(byFilterForcedCats, self.config.transforms,
                                                     "base_PsfFlux_instFlux", hscRun=repoInfo.hscRun)
        principalColCatsCModel = self.transformCatalogs(byFilterForcedCats, self.config.transforms,
                                                        "modelfit_CModel_instFlux", hscRun=repoInfo.hscRun)
        # Create and write parquet tables
        if self.config.doWriteParquetTables:
            dataRef_color = repoInfo.butler.dataRef('analysisColorTable', dataId=repoInfo.dataId)
            writeParquet(dataRef_color, principalColCatsPsf)
            if self.config.writeParquetOnly:
                self.log.info("Exiting after writing Parquet tables.  No plots generated.")
                return

        if self.config.doPlotPrincipalColors:
            principalColCats = (principalColCatsCModel if "CModel" in self.fluxColumn else
                                principalColCatsPsf)
            plotList.append(self.plotStarPrincipalColors(principalColCats, byFilterForcedCats, plotInfoDict,
                                                         areaDict, NumStarLabeller(3), geLabel=geLabel,
                                                         uberCalLabel=uberCalLabel))

        for fluxColumn in ["base_PsfFlux_instFlux", "modelfit_CModel_instFlux"]:
            if fluxColumn == "base_PsfFlux_instFlux":
                principalColCats = principalColCatsPsf
            elif fluxColumn == "modelfit_CModel_instFlux":
                principalColCats = principalColCatsCModel
            else:
                raise RuntimeError("Have not computed transformations for: {:s}".format(fluxColumn))

            plotList.append(self.plotStarColorColor(principalColCats, byFilterForcedCats, plotInfoDict,
                                                    areaDictAll, fluxColumn, forcedStr=self.forcedStr,
                                                    geLabel=geLabel, uberCalLabel=uberCalLabel))

        self.allStats, self.allStatsHigh = savePlots(plotList, "plotColor", repoInfo.dataId,
                                                     repoInfo.butler, subdir=subdir)

        # Update the verifyJob with relevant metadata
        metaDict = {"tract": int(plotInfoDict["tract"])}
        if geLabel:
            metaDict.update({"galacticExtinctionCorrection": geLabel})
        if repoInfo.camera:
            metaDict.update({"camera": repoInfo.camera.getName()})
        self.verifyJob = updateVerifyJob(self.verifyJob, metaDict=metaDict)
        # TODO: this should become a proper butler.put once we can persist the json files
        verifyJobFilename = repoInfo.butler.get("colorAnalysis_verify_job_filename",
                                                dataId=repoInfo.dataId)[0]
        self.verifyJob.write(verifyJobFilename)

    def readCatalogs(self, patchRefList, dataset, repoInfo):
        """Read in and concatenate catalogs of type dataset in lists of data references

        If self.config.doWriteParquetTables is True, before appending each catalog to a single
        list, an extra column indicating the patch is added to the catalog.  This is useful for
        the subsequent interactive QA analysis.

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
        `areaDict` : `dict`
            A dict containing the area of each ccd and the locations of their corners.
        """
        catList = []
        areaDict = {}
        for patchRef in patchRefList:
            if patchRef.datasetExists(dataset):
                cat = patchRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_HEAVY_FOOTPRINTS)
                if dataset != self.config.coaddName + "Coadd_meas":
                    refCat = patchRef.get(self.config.coaddName + "Coadd_ref", immediate=True,
                                          flags=afwTable.SOURCE_IO_NO_HEAVY_FOOTPRINTS)
                    refColList = [s for s in refCat.schema.getNames() if
                                  s.startswith(tuple(self.config.columnsToCopyFromRef))]
                    cat = addColumnsToSchema(refCat, cat,
                                             [col for col in refColList if col not in cat.schema and
                                              col in refCat.schema])

                fname = repoInfo.butler.getUri("deepCoadd_calexp", patchRef.dataId)
                reader = afwImage.ExposureFitsReader(fname)
                wcs = reader.readWcs()
                mask = reader.readMask()
                maskBad = mask.array & 2**mask.getMaskPlaneDict()["BAD"]
                maskNoData = mask.array & 2**mask.getMaskPlaneDict()["NO_DATA"]
                maskedPixels = maskBad + maskNoData
                numGoodPix = np.count_nonzero(maskedPixels == 0)
                pixScale = wcs.getPixelScale(reader.readBBox().getCenter()).asArcseconds()
                area = numGoodPix*pixScale**2
                areaDict[patchRef.dataId["patch"]] = area

                patchCorners = wcs.pixelToSky(geom.Box2D(reader.readBBox()).getCorners())
                areaDict["corners_" + str(patchRef.dataId["patch"])] = patchCorners

                if self.config.doWriteParquetTables:
                    cat = addIntFloatOrStrColumn(cat, patchRef.dataId["patch"], "patchId",
                                                 "Patch on which source was detected")
                catList.append(cat)
        if not catList:
            raise TaskError("No catalogs read: %s" % ([patchRef.dataId for patchRef in patchRefList]))
        return concatenateCatalogs(catList), areaDict

    def correctForGalacticExtinction(self, catalog, tractInfo):
        """Correct all fluxes for each object for Galactic Extinction

        This function uses the EBVbase class from lsst.sims.catUtils.dust.EBV, so lsst.sims.catUtils must
        be setup and accessible for use.

        Parameters
        ----------
        catalog : `lsst.afw.table.SourceCatalog`
           The source catalog for which to apply the per-object Galactic Extinction correction to all fluxes.
           Catalog is corrected in place and a Galactic Extinction applied and flag columns are added.
        tractInfo : `lsst.skymap.tractInfo.ExplicitTractInfo`
           TractInfo object associated with catalog

        Raises
        ------
        `ImportError`
           If lsst.sims.catUtils.dust.EBV could not be imported.

        Returns
        -------
        Updated `lsst.afw.table.source.source.SourceCatalog` catalog with fluxes corrected for
        Galactic Extinction with a column added indicating correction applied and a flag indicating
        if the correction failed (in the context having a non-np.isfinite value).
        """
        try:
            from lsst.sims.catUtils.dust.EBV import EBVbase as ebv
        except ImportError:
            raise ImportError("lsst.sims.catUtils.dust.EBV could not be imported.  Cannot use "
                              "correctForGalacticExtinction function without it.")

        for filterName in catalog.keys():
            if filterName in self.config.extinctionCoeffs:
                raList = catalog[filterName]["coord_ra"]
                decList = catalog[filterName]["coord_dec"]
                ebvObject = ebv()
                ebvValues = ebvObject.calculateEbv(equatorialCoordinates=np.array([raList, decList]))
                galacticExtinction = ebvValues*self.config.extinctionCoeffs[filterName]
                bad = ~np.isfinite(galacticExtinction)
                if ~np.isfinite(galacticExtinction).all():
                    self.log.warn("Could not compute {0:s} band Galactic Extinction for "
                                  "{1:d} out of {2:d} sources.  Flag will be set.".
                                  format(filterName, len(raList[bad]), len(raList)))
                factor = 10.0**(0.4*galacticExtinction)
                fluxKeys, errKeys = getFluxKeys(catalog[filterName].schema)
                self.log.info("Applying per-object Galactic Extinction correction for filter {0:s}.  "
                              "Catalog mean A_{0:s} = {1:.3f}".
                              format(filterName, galacticExtinction[~bad].mean()))
                for name, key in list(fluxKeys.items()) + list(errKeys.items()):
                    catalog[filterName][key] *= factor
            else:
                self.log.warn("Do not have A_X/E(B-V) for filter {0:s}.  "
                              "No Galactic Extinction correction applied for that filter.  "
                              "Flag will be set".format(filterName))
                bad = np.ones(len(catalog[list(catalog.keys())[0]]), dtype=bool)
            # Add column of Galactic Extinction value applied to the catalog and a flag for the sources
            # for which it could not be computed
            catalog[filterName] = addIntFloatOrStrColumn(catalog[filterName], galacticExtinction,
                                                         "A_" + str(filterName),
                                                         "Galactic Extinction (in mags) applied "
                                                         "(based on SFD 1998 maps)")
            catalog[filterName] = addFlag(catalog[filterName], bad, "galacticExtinction_flag",
                                          "True if Galactic Extinction failed")

        return catalog

    def correctFieldForGalacticExtinction(self, catalog, tractInfo):
        """Apply a per-field correction for Galactic Extinction using hard-wired values

        These numbers for E(B-V) are based on the Schlegel et al. 1998 (ApJ 500, 525, SFD98)
        dust maps and were obtained from:
        http://irsa.ipac.caltech.edu/applications/DUST/

        Note that the only fields included are the 5 tracts in the RC + RC2 datasets.
        This is just a placeholder until a per-object implementation is added in DM-13519
        """
        ebvValues = {"UD_COSMOS_9813": {"centerCoord": geom.SpherePoint(150.25, 2.23, geom.degrees),
                                        "EBmV": 0.0165},
                     "WIDE_VVDS_9796": {"centerCoord": geom.SpherePoint(337.78, 0.74, geom.degrees),
                                        "EBmV": 0.0748},
                     "WIDE_GAMMA15H_9615": {"centerCoord": geom.SpherePoint(216.3, 0.74, geom.degrees),
                                            "EBmV": 0.0281},
                     "WIDE_8766": {"centerCoord": geom.SpherePoint(35.70, -3.72, geom.degrees),
                                   "EBmV": 0.0246},
                     "WIDE_8767": {"centerCoord": geom.SpherePoint(37.19, -3.72, geom.degrees),
                                   "EBmV": 0.0268}}

        geFound = False
        for fieldName, geEntry in ebvValues.items():
            if tractInfo.contains(geEntry["centerCoord"]):
                ebvValue = ebvValues[fieldName]["EBmV"]
                geFound = True
                break
        if geFound:
            for filterName in catalog.keys():
                if filterName in self.config.extinctionCoeffs:
                    fluxKeys, errKeys = getFluxKeys(catalog[filterName].schema)
                    galacticExtinction = ebvValue*self.config.extinctionCoeffs[filterName]
                    self.log.info("Applying Per-Field Galactic Extinction correction A_{0:s} = {1:.3f}".
                                  format(filterName, galacticExtinction))
                    factor = 10.0**(0.4*galacticExtinction)
                    for name, key in list(fluxKeys.items()) + list(errKeys.items()):
                        catalog[filterName][key] *= factor
                    # Add column of Galactic Extinction value applied to the catalog
                    catalog[filterName] = addIntFloatOrStrColumn(catalog[filterName], galacticExtinction,
                                                                 "A_" + str(filterName),
                                                                 "Galactic Extinction applied "
                                                                 "(based on SFD 1998 maps)")
                    bad = np.zeros(len(catalog[list(catalog.keys())[0]]), dtype=bool)
                    catalog[filterName] = addFlag(catalog[filterName], bad, "galacticExtinction_flag",
                                                  "True if Galactic Extinction not found (so not applied)")
                else:
                    self.log.warn("Do not have A_X/E(B-V) for filter {0:s}.  "
                                  "No Galactic Extinction correction applied for that filter".
                                  format(filterName))
                    bad = np.ones(len(catalog[list(catalog.keys())[0]]), dtype=bool)
                    catalog[filterName] = addFlag(catalog[filterName], bad, "galacticExtinction_flag",
                                                  "True if Galactic Extinction not found (so not applied)")
        else:
            self.log.warn("Do not have Galactic Extinction for tract {0:d} at {1:s}.  "
                          "No Galactic Extinction correction applied".
                          format(tractInfo.getId(), str(tractInfo.getCtrCoord())))
        return catalog

    def transformCatalogs(self, catalogs, transforms, fluxColumn, hscRun=None):
        """
        Transform catalog entries according to the color transform given

        Parameters
        ----------
        catalogs : `dict` of `lsst.afw.table.source.source.SourceCatalog`s
           One dict entry per filter
        transforms : `dict` of `lsst.pipe.analysis.colorAnalysis.ColorTransform`s
           One dict entry per filter-dependent transform definition
        hscRun : `str` or `NoneType`
           A string representing "HSCPIPE_VERSION" fits header if the data were processed with
           the (now obsolete, but old reruns still exist) "HSC stack", None otherwise
        """
        template = list(catalogs.values())[0]
        num = len(template)
        assert all(len(cat) == num for cat in catalogs.values())

        mapper = afwTable.SchemaMapper(template.schema)
        mapper.addMinimalSchema(afwTable.SourceTable.makeMinimalSchema())
        schema = mapper.getOutputSchema()

        for col in transforms:
            doAdd = True
            for filterName in transforms[col].coeffs:
                if filterName != "" and filterName not in catalogs:
                    doAdd = False
            if doAdd:
                schema.addField(col, float, transforms[col].description + transforms[col].subDescription)
        schema.addField("numStarFlags", type=np.int32, doc="Number of times source was flagged as star")
        badKey = schema.addField("qaBad_flag", type="Flag", doc="Is this a bad source for color qa analyses?")
        schema.addField(fluxColumn, type=np.float64, doc="Flux from filter " + self.fluxFilter)
        schema.addField(fluxColumn + "Err", type=np.float64, doc="Flux error for flux from filter " +
                        self.fluxFilter)
        schema.addField("base_InputCount_value", type=np.int32,
                        doc="Input visit count for " + self.fluxFilter)

        # Copy basics (id, RA, Dec)
        new = afwTable.SourceCatalog(schema)
        new.reserve(num)
        new.extend(template, mapper)

        # Set transformed colors
        for col, transform in transforms.items():
            if col not in schema:
                continue
            value = np.ones(num)*transform.coeffs[""] if "" in transform.coeffs else np.zeros(num)
            for filterName, coeff in transform.coeffs.items():
                if filterName == "":  # Constant: already done
                    continue
                cat = catalogs[filterName]
                mag = -2.5*np.log10(cat[fluxColumn])
                value += mag*coeff
            new[col][:] = value

        # Flag bad values
        bad = np.zeros(num, dtype=bool)
        for dataCat in catalogs.values():
            bad |= makeBadArray(dataCat, flagList=self.flags)
        # Can't set column for flags; do row-by-row
        for row, badValue in zip(new, bad):
            row.setFlag(badKey, bool(badValue))

        # Star/galaxy
        numStarFlags = np.zeros(num)
        for cat in catalogs.values():
            numStarFlags += np.where(cat[self.classificationColumn] < 0.5, 1, 0)
        new["numStarFlags"][:] = numStarFlags

        new[fluxColumn][:] = catalogs[self.fluxFilter][fluxColumn]
        new[fluxColumn + "Err"][:] = catalogs[self.fluxFilter][fluxColumn + "Err"]
        new["base_InputCount_value"][:] = catalogs[self.fluxFilter]["base_InputCount_value"]

        return new

    def plotGalacticExtinction(self, byFilterCats, plotInfoDict, areaDict, geLabel=None):
        yield
        for filterName in byFilterCats:
            qMin = (np.nanmean(byFilterCats[filterName]["A_" + filterName]) -
                    6.0*np.nanstd(byFilterCats[filterName]["A_" + filterName]))
            qMax = (np.nanmean(byFilterCats[filterName]["A_" + filterName]) +
                    6.0*np.nanstd(byFilterCats[filterName]["A_" + filterName]))
            shortName = "galacticExtinction_" + filterName
            self.log.info("shortName = {:s}".format(shortName))
            yield from self.AnalysisClass(byFilterCats[filterName],
                                          byFilterCats[filterName]["A_" + filterName],
                                          "%s (%s)" % ("Galactic Extinction:  A_" + filterName, "mag"),
                                          shortName, self.config.analysis, flags=["galacticExtinction_flag"],
                                          labeller=AllLabeller(), qMin=qMin, qMax=qMax,
                                          magThreshold=99.0).plotAll(shortName, plotInfoDict, areaDict,
                                                                     self.log, zpLabel=geLabel,
                                                                     plotRunStats=False)

    def plotGalaxyColors(self, catalogs, plotInfoDict, areaDict):
        yield
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
            yield from self.AnalysisClass(
                catalog, GalaxyColor("modelfit_CModel_instFlux", "slot_CalibFlux_flux", "g_", "i_"),
                "(g-i)_cmodel - (g-i)_CalibFlux", shortName, self.config.analysis,
                flags=["modelfit_CModel_flag", "slot_CalibFlux_flag"], prefix="i_",
                labeller=OverlapsStarGalaxyLabeller("g_", "i_"),
                qMin=-0.5, qMax=0.5,).plotAll(shortName, plotInfoDict, areaDict, self.log)

    def plotStarPrincipalColors(self, principalColCats, byFilterCats, plotInfoDict, areaDict, labeller,
                                geLabel=None, uberCalLabel=None):
        yield
        mags = {filterName: -2.5*np.log10(byFilterCats[filterName]["base_PsfFlux_instFlux"]) for
                filterName in byFilterCats}
        fluxColumn = ("base_PsfFlux_instFlux" if "base_PsfFlux_instFlux" in principalColCats.schema else
                      "modelfit_CModel_instFlux")
        signalToNoise = {filterName:
                         byFilterCats[filterName][fluxColumn]/byFilterCats[filterName][fluxColumn + "Err"]
                         for filterName in byFilterCats}
        unitStr = "mmag" if self.config.toMilli else "mag"
        for col, transform in self.config.transforms.items():
            if not transform.plot or col not in principalColCats.schema:
                continue
            if self.config.transforms == ivezicTransformsHSC:
                if col == "wPerp" or col == "xPerp":
                    colStr1, colStr2, colStr3 = "HSC-G", "HSC-R", "HSC-I"
                    filterStrList = ["g", "r", "i", ""]
                elif col == "yPerp":
                    colStr1, colStr2, colStr3 = "HSC-R", "HSC-I", "HSC-Z"
                    filterStrList = ["r", "i", "z", ""]
                else:
                    raise RuntimeError("Unknown transformation name: {:s}.  Either set transform.plot "
                                       "to False for that transform or provide accommodations for "
                                       "plotting it in the plotStarPrincipalColors function".format(col))
                xColor = catColors(colStr1, colStr2, mags)
                yColor = catColors(colStr2, colStr3, mags)
                filtersStr = filterStrList[0] + filterStrList[1] + filterStrList[2]
                xRange = (self.config.plotRanges[filtersStr + "X0"],
                          self.config.plotRanges[filtersStr + "X1"])
                yRange = (self.config.plotRanges[filtersStr + "Y0"],
                          self.config.plotRanges[filtersStr + "Y1"])
                paraCol = col[0] + "Para"
                principalColorStrs = []
                for pColStr in [paraCol, col]:
                    transformForStr = self.config.transforms[pColStr]
                    pColStr = makeEqnStr(pColStr, transformForStr.coeffs.values(), filterStrList)
                    principalColorStrs.append(pColStr)
                colorsInFitRange = ColorValueInFitRange(col, xColor, yColor,
                                                        transform.fitLineSlope, transform.fitLineUpperIncpt,
                                                        transform.fitLineLowerIncpt, unitScale=self.unitScale)
                colorsInPerpRange = ColorValueInPerpRange(col, transform.requireGreater,
                                                          transform.requireLess, unitScale=self.unitScale)
                colorsInRange = colorsInFitRange
            elif self.config.transforms == ivezicTransformsSDSS:
                colorsInRange = ColorValueInPerpRange(col, transform.requireGreater, transform.requireLess,
                                                      unitScale=self.unitScale)
            else:
                raise RuntimeError("Unknown transformation: {:s}".format(self.config.transforms))

            catLabel = "noDuplicates"
            forcedStr = self.forcedStr + " " + catLabel
            shortName = "color_" + col
            self.log.info("shortName = {:s}".format(shortName + transform.subDescription))
            yield from self.AnalysisClass(
                principalColCats, colorsInRange, "%s (%s)" % (col + transform.subDescription, unitStr),
                shortName, self.config.analysis, flags=["qaBad_flag"], labeller=labeller, qMin=-0.2, qMax=0.2,
                magThreshold=self.config.analysis.magThreshold).plotAll(shortName, plotInfoDict,
                                                                        areaDict, self.log,
                                                                        zpLabel=geLabel, forcedStr=forcedStr,
                                                                        plotRunStats=False,
                                                                        extraLabels=principalColorStrs,
                                                                        uberCalLabel=uberCalLabel)

            # Plot selections of stars for different criteria
            if self.config.transforms == ivezicTransformsHSC:
                description = filtersStr + fluxToPlotString("base_PsfFlux_instFlux")
                qaGood = np.logical_and(np.logical_not(principalColCats["qaBad_flag"]),
                                        principalColCats["numStarFlags"] >= 3)
                if self.config.analysis.useSignalToNoiseThreshold:
                    if "base_InputCount_value" in byFilterCats[self.fluxFilter].schema:
                        inputCounts = byFilterCats[self.fluxFilter]["base_InputCount_value"]
                        scaleFactor = computeMeanOfFrac(inputCounts, tailStr="upper", fraction=0.1,
                                                        floorFactor=10)
                        if scaleFactor == 0.0:
                            scaleFactor = computeMeanOfFrac(inputCounts, tailStr="upper", fraction=0.1,
                                                            floorFactor=1)
                        signalToNoiseThreshold = np.floor(
                            np.sqrt(scaleFactor)*self.config.analysis.signalToNoiseThreshold/100 + 0.49)*100
                        configSNThresh = self.config.analysis.signalToNoiseHighThreshold
                        signalToNoiseHighThreshold = (np.floor(
                            np.sqrt(scaleFactor)*configSNThresh/100 + 0.49)*100)
                    qaGood = np.logical_and(qaGood, signalToNoise[self.fluxFilter] >= signalToNoiseThreshold)
                    # Set self.magThreshold to represent approximately that which corresponds to the
                    # S/N threshold.  Computed as the mean magnitude of the lower 5% of the
                    # S/N > signalToNoiseThreshold subsample
                    ptFrac = max(2, int(0.05*len(mags[self.fluxFilter][qaGood])))
                    magThreshold = np.floor(mags[self.fluxFilter][qaGood][
                        mags[self.fluxFilter][qaGood].argsort()[-ptFrac:]].mean()*10 + 0.5)/10
                    thresholdStr = [r" [S/N$\geqslant$" + str(signalToNoiseHighThreshold) + "]",
                                    " [" + self.fluxFilter + r"$\lesssim$" + str(magThreshold) + "]"]
                else:
                    qaGood = np.logical_and(qaGood, mags[self.fluxFilter] < self.config.analysis.magThreshold)
                    thresholdStr = [" [" + self.fluxFilter + " < " +
                                    str(self.config.analysis.magThreshold) + "]", ]
                inFitGood = np.logical_and(np.isfinite(colorsInFitRange(principalColCats)), qaGood)
                inPerpGood = np.logical_and(np.isfinite(colorsInPerpRange(principalColCats)), qaGood)
                xColor = catColors(colStr1, colStr2, mags)
                yColor = catColors(colStr2, colStr3, mags)
                fig, axes = plt.subplots(1, 1)
                axes.tick_params(which="both", direction="in", labelsize=9)
                axes.set_xlim(*xRange)
                axes.set_ylim(*yRange)

                deltaX = abs(xRange[1] - xRange[0])
                deltaY = abs(yRange[1] - yRange[0])
                lineOffset = [0.15, 0.15]
                if col == "wPerp":
                    lineOffset = [0.30, 0.30]
                if col == "xPerp":
                    lineOffset = [0.60, 0.15]
                if col == "yPerp":
                    lineOffset = [0.30, 0.30]
                lineFitSlope = self.config.transforms[col[0] + "Fit"].coeffs[colStr1]
                lineFitIncpt = self.config.transforms[col[0] + "Fit"].coeffs[""]
                xLine = np.linspace(xRange[0] + lineOffset[0]*deltaX, xRange[1]-lineOffset[1]*deltaX, 100)
                yLineUpper = transform.fitLineUpperIncpt + transform.fitLineSlope*xLine
                yLineLower = transform.fitLineLowerIncpt + transform.fitLineSlope*xLine
                yLineFit = lineFitSlope*xLine + lineFitIncpt
                axes.plot(xLine, yLineUpper, "g--", alpha=0.5)
                axes.plot(xLine, yLineLower, "g--", alpha=0.5)
                axes.plot(xLine, yLineFit, "m--", alpha=0.5)

                ptSize = max(1, setPtSize(len(xColor)) - 2)

                axes.scatter(xColor[qaGood], yColor[qaGood], label="all", color="black", alpha=0.4,
                             marker="o", s=ptSize + 2)
                axes.scatter(xColor[inFitGood], yColor[inFitGood], label="inFit", color="blue",
                             marker="o", s=ptSize + 1)
                axes.scatter(xColor[inPerpGood], yColor[inPerpGood], label="inPerp", color="red",
                             marker="x", s=ptSize, lw=0.5)
                axes.set_xlabel(colStr1 + " $-$ " + colStr2)
                axes.set_ylabel(colStr2 + " $-$ " + colStr3, labelpad=-1)

                # Label total number of objects of each data type
                lenNumObj = max(len(str(len(xColor[qaGood]))), len(str(len(xColor[inFitGood]))),
                                len(str(len(xColor[inPerpGood]))))
                fdx = max((min(0.07*lenNumObj, 0.8), 0.28))
                xLoc, yLoc = xRange[0] + 0.03*deltaX, yRange[1] - 0.038*deltaY
                kwargs = dict(va="center", fontsize=8)
                axes.text(xLoc, yLoc, "NqaGood  =", ha="left", color="black", **kwargs)
                axes.text(xLoc + fdx*deltaX, yLoc, str(len(xColor[qaGood])), ha="right", color="black",
                          **kwargs)
                for threshStr in list(thresholdStr):
                    axes.text(xLoc + 1.54*fdx*deltaX, yLoc, threshStr, ha="right", color="black", **kwargs)
                    yLoc -= 0.05*deltaY
                axes.text(xLoc, yLoc, "NinFitGood =", ha="left", color="blue", **kwargs)
                axes.text(xLoc + fdx*deltaX, yLoc, str(len(xColor[inFitGood])), ha="right", color="blue",
                          **kwargs)
                yLoc -= 0.05*deltaY
                axes.text(xLoc, yLoc, "NinPerpGood =", ha="left", color="red", **kwargs)
                axes.text(xLoc + fdx*deltaX, yLoc, str(len(xColor[inPerpGood])), ha="right", color="red",
                          **kwargs)
                if plotInfoDict["cameraName"]:
                    labelCamera(plotInfoDict, fig, axes, 0.5, 1.09)
                if catLabel:
                    plotText(catLabel, fig, axes, 0.91, -0.09, color="green", fontSize=10)
                if geLabel:
                    plotText(geLabel, fig, axes, 0.11, -0.09, color="green", fontSize=10)
                if plotInfoDict["hscRun"]:
                    axes.set_title("HSC stack run: " + plotInfoDict["hscRun"], color="#800080")

                tractStr = "tract: {:s}".format(plotInfoDict["tract"])
                axes.annotate(tractStr, xy=(0.5, 1.04), xycoords="axes fraction", ha="center", va="center",
                              fontsize=10, color="green")

                yield Struct(fig=fig, description=description, stats=None, statsHigh=None,
                             style=col + "Selections")

    def plotStarColorColor(self, principalColCats, byFilterCats, plotInfoDict, areaDict, fluxColumn,
                           forcedStr=None, geLabel=None, uberCalLabel=None):
        yield
        num = len(list(byFilterCats.values())[0])
        zp = 0.0
        mags = {filterName: zp - 2.5*np.log10(byFilterCats[filterName][fluxColumn]) for
                filterName in byFilterCats}
        signalToNoise = {filterName:
                         byFilterCats[filterName][fluxColumn]/byFilterCats[filterName][fluxColumn + "Err"]
                         for filterName in byFilterCats}

        bad = np.zeros(num, dtype=bool)
        for cat in byFilterCats.values():
            bad |= makeBadArray(cat, flagList=self.flags)
        catLabel = "noDuplicates"

        if self.config.analysis.useSignalToNoiseThreshold:
            if "base_InputCount_value" in byFilterCats[self.fluxFilter].schema:
                inputCounts = byFilterCats[self.fluxFilter]["base_InputCount_value"]
                scaleFactor = computeMeanOfFrac(inputCounts, tailStr="upper", fraction=0.1, floorFactor=10)
                if scaleFactor == 0.0:
                    scaleFactor = computeMeanOfFrac(inputCounts, tailStr="upper", fraction=0.1, floorFactor=1)
                signalToNoiseThreshold = np.floor(
                    np.sqrt(scaleFactor)*self.config.analysis.signalToNoiseThreshold/100 + 0.49)*100
                signalToNoiseHighThreshold = (np.floor(
                    np.sqrt(scaleFactor)*self.config.analysis.signalToNoiseHighThreshold/100 + 0.49)*100)
            bright = signalToNoise[self.fluxFilter] >= signalToNoiseThreshold
            ptFrac = max(2, int(0.05*len(mags[self.fluxFilter][bright])))
            # Set self.magThreshold to represent approximately that which corresponds to the
            # S/N threshold.  Computed as the mean magnitude of the lower 5% of the
            # S/N > signalToNoiseThreshold subsample
            magThreshold = np.floor(mags[self.fluxFilter][bright][
                mags[self.fluxFilter][bright].argsort()[-ptFrac:]].mean()*10 + 0.5)/10
            thresholdStr = [r" [S/N$\geqslant$" + str(signalToNoiseHighThreshold) + "]",
                            " [" + self.fluxFilter + r"$\lesssim$" + str(magThreshold) + "]"]
        else:
            magThreshold = self.config.analysis.magThreshold
            bright = mags[self.fluxFilter] < magThreshold
            thresholdStr = [" [" + self.fluxFilter + " < " + str(self.config.analysis.magThreshold) + "]", ]
        prettyBrightThreshold = self.config.analysis.magThreshold + 4
        prettyBright = mags[self.fluxFilter] < prettyBrightThreshold

        # Determine number of filters object is classified as a star
        numStarFlags = np.zeros(num)
        for cat in byFilterCats.values():
            numStarFlags += np.where(cat[self.classificationColumn] < 0.5, 1, 0)

        # Select as a star if classified as such in self.config.fluxFilter
        isStarFlag = byFilterCats[self.fluxFilter][self.classificationColumn] < 0.5
        # Require stellar classification in self.fluxFilter and at least one other filter for fits
        good = isStarFlag & (numStarFlags >= 2) & ~bad & bright
        goodCombined = isStarFlag & (numStarFlags >= 2) & ~bad
        decentStars = isStarFlag & ~bad & prettyBright
        decentGalaxies = ~isStarFlag & ~bad & prettyBright

        # The combined catalog is only used in the Distance (from the poly fit) AnalysisClass plots
        combined = (self.transformCatalogs(byFilterCats, straightTransforms, "base_PsfFlux_instFlux",
                                           hscRun=plotInfoDict["hscRun"])[goodCombined].copy(True))
        filters = set(byFilterCats.keys())
        goodMags = {filterName: mags[filterName][good] for filterName in byFilterCats}
        decentStarsMag = mags[self.fluxFilter][decentStars]
        decentGalaxiesMag = mags[self.fluxFilter][decentGalaxies]
        unitStr = "mmag" if self.config.toMilli else "mag"
        fluxColStr = fluxToPlotString(fluxColumn)

        polyFitKwargs = dict(thresholdStr=thresholdStr, catLabel=catLabel, geLabel=geLabel,
                             uberCalLabel=uberCalLabel, unitScale=self.unitScale)

        if filters.issuperset(set(("HSC-G", "HSC-R", "HSC-I"))):
            # Do a linear fit to regions defined in Ivezic transforms
            transformPerp = self.config.transforms["wPerp"]
            transformPara = self.config.transforms["wPara"]
            fitLineUpper = [transformPerp.fitLineUpperIncpt, transformPerp.fitLineSlope]
            fitLineLower = [transformPerp.fitLineLowerIncpt, transformPerp.fitLineSlope]
            filtersStr = "gri"
            xRange = (self.config.plotRanges[filtersStr + "X0"],
                      self.config.plotRanges[filtersStr + "X1"])
            yRange = (self.config.plotRanges[filtersStr + "Y0"],
                      self.config.plotRanges[filtersStr + "Y1"])
            nameStr = filtersStr + fluxColStr + "-wFit"
            self.log.info("nameStr = {:s}".format(nameStr))
            if "PSF" in fluxColStr:
                verifyKwargs = dict(verifyJob=self.verifyJob, verifyMetricName="stellar_locus_width_wPerp")
            else:
                verifyKwargs = {}
            yield from colorColorPolyFitPlot(plotInfoDict, nameStr,
                                             self.log, catColors("HSC-G", "HSC-R", mags, good),
                                             catColors("HSC-R", "HSC-I", mags, good),
                                             "g - r  [{0:s}]".format(fluxColStr),
                                             "r - i  [{0:s}]".format(fluxColStr), self.fluxFilter,
                                             transformPerp=transformPerp, transformPara=transformPara,
                                             mags=goodMags, principalCol=principalColCats["wPerp"][good],
                                             xRange=xRange, yRange=yRange, order=1, xFitRange=(0.28, 1.0),
                                             yFitRange=(0.02, 0.48), fitLineUpper=fitLineUpper,
                                             fitLineLower=fitLineLower, **verifyKwargs, **polyFitKwargs)
            transformPerp = self.config.transforms["xPerp"]
            transformPara = self.config.transforms["xPara"]
            fitLineUpper = [transformPerp.fitLineUpperIncpt, transformPerp.fitLineSlope]
            fitLineLower = [transformPerp.fitLineLowerIncpt, transformPerp.fitLineSlope]
            nameStr = filtersStr + fluxColStr + "-xFit"
            self.log.info("nameStr = {:s}".format(nameStr))
            if verifyKwargs:
                verifyKwargs.update({"verifyMetricName": "stellar_locus_width_xPerp"})
            yield from colorColorPolyFitPlot(plotInfoDict, nameStr,
                                             self.log, catColors("HSC-G", "HSC-R", mags, good),
                                             catColors("HSC-R", "HSC-I", mags, good),
                                             "g - r  [{0:s}]".format(fluxColStr),
                                             "r - i  [{0:s}]".format(fluxColStr), self.fluxFilter,
                                             transformPerp=transformPerp, transformPara=transformPara,
                                             mags=goodMags, principalCol=principalColCats["xPerp"][good],
                                             xRange=xRange, yRange=yRange, order=1, xFitRange=(1.05, 1.45),
                                             yFitRange=(0.78, 1.62), fitLineUpper=fitLineUpper,
                                             fitLineLower=fitLineLower,
                                             closeToVertical=True, **verifyKwargs, **polyFitKwargs)
            # Lower branch only; upper branch is noisy due to astrophysics
            nameStr = filtersStr + fluxColStr
            self.log.info("nameStr = {:s}".format(nameStr))
            fitLineUpper = [2.0, -1.31]
            fitLineLower = [0.61, -1.78]
            # TO DO: The return needs to change
            poly = yield from colorColorPolyFitPlot(plotInfoDict, nameStr,
                                                    self.log, catColors("HSC-G", "HSC-R", mags, good),
                                                    catColors("HSC-R", "HSC-I", mags, good),
                                                    "g - r  [{0:s}]".format(fluxColStr),
                                                    "r - i  [{0:s}]".format(fluxColStr), self.fluxFilter,
                                                    xRange=xRange, yRange=yRange, order=3,
                                                    xFitRange=(0.23, 1.2), yFitRange=(0.05, 0.6),
                                                    fitLineUpper=fitLineUpper, fitLineLower=fitLineLower,
                                                    **polyFitKwargs)
            # Make a color-color plot with both stars and galaxies, less pruning, and no fit
            if fluxColumn != "base_PsfFlux_instFlux":
                self.log.info("nameStr: noFit ({1:s}) = {0:s}".format(nameStr, fluxColumn))
                yield from colorColorPlot(plotInfoDict, nameStr,
                                          self.log, catColors("HSC-G", "HSC-R", mags, decentStars),
                                          catColors("HSC-R", "HSC-I", mags, decentStars),
                                          catColors("HSC-G", "HSC-R", mags, decentGalaxies),
                                          catColors("HSC-R", "HSC-I", mags, decentGalaxies),
                                          decentStarsMag, decentGalaxiesMag,
                                          "g - r  [{0:s}]".format(fluxColStr),
                                          "r - i  [{0:s}]".format(fluxColStr), self.fluxFilter, fluxColStr,
                                          xRange=(xRange[0], xRange[1] + 0.6), yRange=yRange,
                                          magThreshold=prettyBrightThreshold, geLabel=geLabel,
                                          unitScale=self.unitScale, uberCalLabel=uberCalLabel)
                yield from colorColor4MagPlots(plotInfoDict, nameStr,
                                               self.log, catColors("HSC-G", "HSC-R", mags, decentStars),
                                               catColors("HSC-R", "HSC-I", mags, decentStars),
                                               catColors("HSC-G", "HSC-R", mags, decentGalaxies),
                                               catColors("HSC-R", "HSC-I", mags, decentGalaxies),
                                               decentStarsMag, decentGalaxiesMag,
                                               "g - r  [{0:s}]".format(fluxColStr),
                                               "r - i  [{0:s}]".format(fluxColStr), self.fluxFilter,
                                               fluxColStr, xRange=(xRange[0], xRange[1] + 0.6), yRange=yRange,
                                               magThreshold=prettyBrightThreshold,
                                               geLabel=geLabel, unitScale=self.unitScale,
                                               uberCalLabel=uberCalLabel)

            shortName = filtersStr + "Distance" + fluxColStr
            self.log.info("shortName = {:s}".format(shortName))
            stdevEnforcer = Enforcer(requireLess={"star": {"stdev": 0.03*self.unitScale}})
            yield from self.AnalysisClass(combined,
                                          ColorColorDistance("g", "r", "i", poly, unitScale=self.unitScale,
                                                             fitLineUpper=fitLineUpper,
                                                             fitLineLower=fitLineLower),
                                          filtersStr + "Distance [%s] (%s)" % (fluxColStr, unitStr),
                                          shortName, self.config.analysis, flags=["qaBad_flag"], qMin=-0.1,
                                          qMax=0.1, magThreshold=prettyBrightThreshold,
                                          labeller=NumStarLabeller(2)).plotAll(shortName, plotInfoDict,
                                                                               areaDict, self.log,
                                                                               stdevEnforcer,
                                                                               forcedStr=forcedStr,
                                                                               zpLabel=geLabel,
                                                                               uberCalLabel=uberCalLabel)
        if filters.issuperset(set(("HSC-R", "HSC-I", "HSC-Z"))):
            # Do a linear fit to regions defined in Ivezic transforms
            transformPerp = self.config.transforms["yPerp"]
            transformPara = self.config.transforms["yPara"]
            fitLineUpper = [transformPerp.fitLineUpperIncpt, transformPerp.fitLineSlope]
            fitLineLower = [transformPerp.fitLineLowerIncpt, transformPerp.fitLineSlope]
            filtersStr = "riz"
            xRange = (self.config.plotRanges[filtersStr + "X0"],
                      self.config.plotRanges[filtersStr + "X1"])
            yRange = (self.config.plotRanges[filtersStr + "Y0"],
                      self.config.plotRanges[filtersStr + "Y1"])
            nameStr = filtersStr + fluxColStr + "-yFit"
            self.log.info("nameStr = {:s}".format(nameStr))
            yield from colorColorPolyFitPlot(plotInfoDict, nameStr,
                                             self.log, catColors("HSC-R", "HSC-I", mags, good),
                                             catColors("HSC-I", "HSC-Z", mags, good),
                                             "r - i  [{0:s}]".format(fluxColStr),
                                             "i - z  [{0:s}]".format(fluxColStr), self.fluxFilter,
                                             transformPerp=transformPerp, transformPara=transformPara,
                                             mags=goodMags, principalCol=principalColCats["yPerp"][good],
                                             xRange=xRange, yRange=yRange, order=1, xFitRange=(0.82, 2.01),
                                             yFitRange=(0.37, 0.81), fitLineUpper=fitLineUpper,
                                             fitLineLower=fitLineLower, **polyFitKwargs)
            nameStr = filtersStr + fluxColStr
            fitLineUpper = [5.9, -3.05]
            fitLineLower = [0.11, -2.07]
            self.log.info("nameStr = {:s}".format(nameStr))
            # TO DO: also change this
            poly = yield from colorColorPolyFitPlot(plotInfoDict, nameStr,
                                                    self.log, catColors("HSC-R", "HSC-I", mags, good),
                                                    catColors("HSC-I", "HSC-Z", mags, good),
                                                    "r - i  [{0:s}]".format(fluxColStr),
                                                    "i - z  [{0:s}]".format(fluxColStr), self.fluxFilter,
                                                    xRange=xRange, yRange=yRange, order=2,
                                                    xFitRange=(-0.01, 1.75), yFitRange=(-0.01, 0.72),
                                                    fitLineUpper=fitLineUpper, fitLineLower=fitLineLower,
                                                    **polyFitKwargs)
            # Make a color-color plot with both stars and galaxies, less pruning, and no fit
            if fluxColumn != "base_PsfFlux_instFlux":
                self.log.info("nameStr: noFit ({1:s}) = {0:s}".format(nameStr, fluxColumn))
                yield from colorColorPlot(plotInfoDict, nameStr,
                                          self.log, catColors("HSC-R", "HSC-I", mags, decentStars),
                                          catColors("HSC-I", "HSC-Z", mags, decentStars),
                                          catColors("HSC-R", "HSC-I", mags, decentGalaxies),
                                          catColors("HSC-I", "HSC-Z", mags, decentGalaxies),
                                          decentStarsMag, decentGalaxiesMag,
                                          "r - i  [{0:s}]".format(fluxColStr),
                                          "i - z  [{0:s}]".format(fluxColStr), self.fluxFilter, fluxColStr,
                                          xRange=xRange, yRange=(yRange[0], yRange[1] + 0.2),
                                          magThreshold=prettyBrightThreshold,
                                          geLabel=geLabel, uberCalLabel=uberCalLabel,
                                          unitScale=self.unitScale)
                yield from colorColor4MagPlots(plotInfoDict, nameStr,
                                               self.log, catColors("HSC-R", "HSC-I", mags, decentStars),
                                               catColors("HSC-I", "HSC-Z", mags, decentStars),
                                               catColors("HSC-R", "HSC-I", mags, decentGalaxies),
                                               catColors("HSC-I", "HSC-Z", mags, decentGalaxies),
                                               decentStarsMag, decentGalaxiesMag,
                                               "r - i  [{0:s}]".format(fluxColStr),
                                               "i - z  [{0:s}]".format(fluxColStr), self.fluxFilter,
                                               fluxColStr, xRange=xRange, yRange=(yRange[0], yRange[1] + 0.2),
                                               magThreshold=prettyBrightThreshold,
                                               geLabel=geLabel, unitScale=self.unitScale,
                                               uberCalLabel=uberCalLabel)
            shortName = filtersStr + "Distance" + fluxColStr
            self.log.info("shortName = {:s}".format(shortName))
            stdevEnforcer = Enforcer(requireLess={"star": {"stdev": 0.03*self.unitScale}})
            yield from self.AnalysisClass(combined, ColorColorDistance("r", "i", "z", poly,
                                                                       unitScale=self.unitScale,
                                                                       fitLineUpper=fitLineUpper,
                                                                       fitLineLower=fitLineLower),
                                          filtersStr + "Distance [%s] (%s)" % (fluxColStr, unitStr),
                                          shortName, self.config.analysis, flags=["qaBad_flag"], qMin=-0.1,
                                          qMax=0.1, magThreshold=prettyBrightThreshold,
                                          labeller=NumStarLabeller(2)).plotAll(shortName, plotInfoDict,
                                                                               areaDict, self.log,
                                                                               stdevEnforcer,
                                                                               forcedStr=forcedStr,
                                                                               zpLabel=geLabel,
                                                                               uberCalLabel=uberCalLabel)
        if filters.issuperset(set(("HSC-I", "HSC-Z", "HSC-Y"))):
            filtersStr = "izy"
            nameStr = filtersStr + fluxColStr
            self.log.info("nameStr = {:s}".format(nameStr))
            fitLineUpper = [2.55, -3.0]
            fitLineLower = [-0.062, -2.35]
            xRange = (self.config.plotRanges[filtersStr + "X0"],
                      self.config.plotRanges[filtersStr + "X1"])
            yRange = (self.config.plotRanges[filtersStr + "Y0"],
                      self.config.plotRanges[filtersStr + "Y1"])
            poly = yield from colorColorPolyFitPlot(plotInfoDict, nameStr,
                                                    self.log, catColors("HSC-I", "HSC-Z", mags, good),
                                                    catColors("HSC-Z", "HSC-Y", mags, good),
                                                    "i - z  [{0:s}]".format(fluxColStr),
                                                    "z - y  [{0:s}]".format(fluxColStr), self.fluxFilter,
                                                    xRange=xRange, yRange=yRange, order=2,
                                                    xFitRange=(-0.05, 0.8), yFitRange=(-0.03, 0.32),
                                                    fitLineUpper=fitLineUpper, fitLineLower=fitLineLower,
                                                    **polyFitKwargs)
            # Make a color-color plot with both stars and galaxies, less pruning, and no fit
            if fluxColumn != "base_PsfFlux_instFlux":
                self.log.info("nameStr: noFit ({1:s}) = {0:s}".format(nameStr, fluxColumn))
                yield from colorColorPlot(plotInfoDict, nameStr, self.log,
                                          catColors("HSC-I", "HSC-Z", mags, decentStars),
                                          catColors("HSC-Z", "HSC-Y", mags, decentStars),
                                          catColors("HSC-I", "HSC-Z", mags, decentGalaxies),
                                          catColors("HSC-Z", "HSC-Y", mags, decentGalaxies),
                                          decentStarsMag, decentGalaxiesMag,
                                          "i - z  [{0:s}]".format(fluxColStr),
                                          "z - y  [{0:s}]".format(fluxColStr), self.fluxFilter, fluxColStr,
                                          xRange=xRange, yRange=(yRange[0], yRange[1] + 0.2),
                                          magThreshold=prettyBrightThreshold,
                                          geLabel=geLabel, unitScale=self.unitScale,
                                          uberCalLabel=uberCalLabel)
                yield from colorColor4MagPlots(plotInfoDict, nameStr,
                                               self.log, catColors("HSC-I", "HSC-Z", mags, decentStars),
                                               catColors("HSC-Z", "HSC-Y", mags, decentStars),
                                               catColors("HSC-I", "HSC-Z", mags, decentGalaxies),
                                               catColors("HSC-Z", "HSC-Y", mags, decentGalaxies),
                                               decentStarsMag, decentGalaxiesMag,
                                               "i - z  [{0:s}]".format(fluxColStr),
                                               "z - y  [{0:s}]".format(fluxColStr), self.fluxFilter,
                                               fluxColStr, xRange=xRange, yRange=(yRange[0], yRange[1] + 0.2),
                                               magThreshold=prettyBrightThreshold,
                                               geLabel=geLabel, uberCalLabel=uberCalLabel,
                                               unitScale=self.unitScale)
            shortName = filtersStr + "Distance" + fluxColStr
            self.log.info("shortName = {:s}".format(shortName))
            stdevEnforcer = Enforcer(requireLess={"star": {"stdev": 0.03*self.unitScale}})
            yield from self.AnalysisClass(combined,
                                          ColorColorDistance("i", "z", "y", poly, unitScale=self.unitScale,
                                                             fitLineUpper=fitLineUpper,
                                                             fitLineLower=fitLineLower),
                                          filtersStr + "Distance [%s] (%s)" % (fluxColStr, unitStr),
                                          shortName, self.config.analysis, flags=["qaBad_flag"], qMin=-0.1,
                                          qMax=0.1, magThreshold=prettyBrightThreshold,
                                          labeller=NumStarLabeller(2)).plotAll(shortName, plotInfoDict,
                                                                               areaDict, self.log,
                                                                               stdevEnforcer,
                                                                               forcedStr=forcedStr,
                                                                               zpLabel=geLabel,
                                                                               uberCalLabel=uberCalLabel)

        if filters.issuperset(set(("HSC-Z", "NB0921", "HSC-Y"))):
            filtersStr = "z9y"
            xRange = (self.config.plotRanges[filtersStr + "X0"],
                      self.config.plotRanges[filtersStr + "X1"])
            yRange = (self.config.plotRanges[filtersStr + "Y0"],
                      self.config.plotRanges[filtersStr + "Y1"])
            nameStr = filtersStr + fluxColStr
            self.log.info("nameStr = {:s}".format(nameStr))
            fitLineUpper = [0.65, -3.5]
            fitLineLower = [-0.01, -0.96]
            poly = yield from colorColorPolyFitPlot(plotInfoDict, nameStr, self.log,
                                                    catColors("HSC-Z", "NB0921", mags, good),
                                                    catColors("NB0921", "HSC-Y", mags, good),
                                                    "z-n921  [{0:s}]".format(fluxColStr),
                                                    "n921-y  [{0:s}]".format(fluxColStr), self.fluxFilter,
                                                    xRange=xRange, yRange=yRange,
                                                    order=2, xFitRange=(-0.08, 0.2), yFitRange=(0.006, 0.19),
                                                    fitLineUpper=fitLineUpper, fitLineLower=fitLineLower,
                                                    **polyFitKwargs)
            # Make a color-color plot with both stars and galaxies, less pruning, and no fit
            if fluxColumn != "base_PsfFlux_instFlux":
                self.log.info("nameStr: noFit ({1:s}) = {0:s}".format(nameStr, fluxColumn))
                yield from colorColorPlot(plotInfoDict, nameStr,
                                          self.log, catColors("HSC-Z", "NB0921", mags, decentStars),
                                          catColors("NB0921", "HSC-Y", mags, decentStars),
                                          catColors("HSC-Z", "NB0921", mags, decentGalaxies),
                                          catColors("NB0921", "HSC-Y", mags, decentGalaxies),
                                          decentStarsMag, decentGalaxiesMag,
                                          "z-n921  [{0:s}]".format(fluxColStr),
                                          "n921-y  [{0:s}]".format(fluxColStr), self.fluxFilter, fluxColStr,
                                          xRange=xRange, yRange=(yRange[0] - 0.05, yRange[1] + 0.05),
                                          magThreshold=prettyBrightThreshold,
                                          geLabel=geLabel, unitScale=self.unitScale,
                                          uberCalLabel=uberCalLabel)
                yield from colorColor4MagPlots(plotInfoDict, nameStr,
                                               self.log, catColors("HSC-Z", "NB0921", mags, decentStars),
                                               catColors("NB0921", "HSC-Y", mags, decentStars),
                                               catColors("HSC-Z", "NB0921", mags, decentGalaxies),
                                               catColors("NB0921", "HSC-Y", mags, decentGalaxies),
                                               decentStarsMag, decentGalaxiesMag,
                                               "z-n921  [{0:s}]".format(fluxColStr),
                                               "n921-y  [{0:s}]".format(fluxColStr), self.fluxFilter,
                                               fluxColStr, xRange=xRange,
                                               yRange=(yRange[0] - 0.05, yRange[1] + 0.05),
                                               magThreshold=prettyBrightThreshold,
                                               geLabel=geLabel, unitScale=self.unitScale,
                                               uberCalLabel=uberCalLabel)
            shortName = filtersStr + "Distance" + fluxColStr
            self.log.info("shortName = {:s}".format(shortName))
            stdevEnforcer = Enforcer(requireLess={"star": {"stdev": 0.03*self.unitScale}})
            yield from self.AnalysisClass(combined,
                                          ColorColorDistance("z", "n921", "y", poly, unitScale=self.unitScale,
                                                             fitLineUpper=fitLineUpper,
                                                             fitLineLower=fitLineLower),
                                          filtersStr + "Distance [%s] (%s)" % (fluxColStr, unitStr),
                                          shortName, self.config.analysis, flags=["qaBad_flag"], qMin=-0.1,
                                          qMax=0.1, magThreshold=prettyBrightThreshold,
                                          labeller=NumStarLabeller(2)).plotAll(shortName, plotInfoDict,
                                                                               areaDict, self.log,
                                                                               stdevEnforcer,
                                                                               forcedStr=forcedStr,
                                                                               zpLabel=geLabel,
                                                                               uberCalLabel=uberCalLabel)

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None

    def _getEupsVersionsName(self):
        return None


def colorColorPolyFitPlot(plotInfoDict, description, log, xx, yy, xLabel, yLabel, filterStr,
                          transformPerp=None, transformPara=None, mags=None, principalCol=None, xRange=None,
                          yRange=None, order=1, iterations=3, rej=3.0, xFitRange=None, yFitRange=None,
                          fitLineUpper=None, fitLineLower=None, numBins="auto", catLabel=None,
                          geLabel=None, uberCalLabel=None, logger=None, thresholdStr=None,
                          unitScale=1.0, closeToVertical=False, verifyJob=None, verifyMetricName=None):
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
    fig.subplots_adjust(wspace=0.46, bottom=0.15, left=0.11, right=0.96, top=0.9)
    axes[0].tick_params(which="both", direction="in", labelsize=9)
    axes[1].tick_params(which="both", direction="in", labelsize=9)

    good = np.logical_and(np.isfinite(xx), np.isfinite(yy))
    xx, yy = xx[good], yy[good]
    numGood = len(xx)
    fitP2 = None
    if mags is not None:
        mags = {filterName: mags[filterName][good] for filterName in mags.keys()}
    if principalCol is not None:
        principalColor = principalCol[good].copy()*unitScale

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

    # Pad vertical and horizontal fit ranges for use after the first fit iteration
    if xFitRange:
        xMinPad = xFitRange[0] - 0.07*(xFitRange[1] - xFitRange[0])
        xMaxPad = xFitRange[1] + 0.07*(xFitRange[1] - xFitRange[0])
    if yFitRange:
        yMinPad = yFitRange[0] - 0.07*(yFitRange[1] - yFitRange[0])
        yMaxPad = yFitRange[1] + 0.07*(yFitRange[1] - yFitRange[0])

    select = np.ones_like(xx, dtype=bool)
    for sel in [selectXRange, selectYRange, selectUpper, selectLower]:
        select &= sel

    keep = np.ones_like(xx, dtype=bool)
    # Perform a polynomial fit using np.polyfit to use as an initial guess for the Orthoganl Regression
    if closeToVertical:
        # Force the initial guess for near-vertical distributions (np.polyfit cannot handle vertical fits)
        keep &= select
        poly = [10.0, -10.0*(xFitRange[0] + (xFitRange[1]-xFitRange[0])/3.0)]
    else:
        for ii in range(iterations):
            keep &= select
            poly = np.polyfit(xx[keep], yy[keep], order)
            dy = yy - np.polyval(poly, xx)
            clippedStats = calcQuartileClippedStats(dy[keep], nSigmaToClip=rej)
            keep = np.logical_not(np.abs(dy) > clippedStats.clipValue)
            # After the first iteration, reset the vertical and horizontal clipping to be less restrictive
            if ii == 0:
                selectXRange = selectXRange if not xFitRange else ((xx > xMinPad) & (xx < xMaxPad))
                selectYRange = selectYRange if not yFitRange else ((yy > yMinPad) & (yy < yMaxPad))
                for sel in [selectXRange, selectYRange, selectUpper, selectLower]:
                    select &= sel

        log.info("Number of iterations in polynomial fit: {:d}".format(ii + 1))
        keep &= select
        nKeep = np.sum(keep)
        if nKeep < order:
            raise RuntimeError(
                "Not enough good data points ({0:d}) for polynomial fit of order {1:d}".format(nKeep, order))

        poly = np.polyfit(xx[keep], yy[keep], order)

    # Calculate the point density
    xyKeep = np.vstack([xx[keep], yy[keep]])
    zKeep = scipyStats.gaussian_kde(xyKeep)(xyKeep)
    xyOther = np.vstack([xx[~keep], yy[~keep]])
    zOther = scipyStats.gaussian_kde(xyOther)(xyOther)
    idxHighDensity = np.argmax(zKeep)
    xHighDensity = xx[keep][idxHighDensity]
    yHighDensity = yy[keep][idxHighDensity]
    log.info("Highest Density point x, y: {0:.4f} {1:.4f}".format(xHighDensity, yHighDensity))

    initialGuess = list(reversed(poly))
    keepOdr = keep.copy()
    orthRegCoeffs = orthogonalRegression(xx[keepOdr], yy[keepOdr], order, initialGuess)
    for ii in range(iterations - 1):
        initialGuess = list(reversed(orthRegCoeffs))
        dy = yy - np.polyval(orthRegCoeffs, xx)
        clippedStats = calcQuartileClippedStats(dy[keepOdr], nSigmaToClip=rej)
        keepOdr = np.logical_not(np.abs(dy) > clippedStats.clipValue) & np.isfinite(xx) & np.isfinite(yy)
        # After the first iteration, reset the vertical and horizontal clipping to be less restrictive
        if ii == 0:
            selectXRange = selectXRange if not xFitRange else ((xx > xMinPad) & (xx < xMaxPad))
            selectYRange = selectYRange if not yFitRange else ((yy > yMinPad) & (yy < yMaxPad))
            for sel in [selectXRange, selectYRange, selectUpper, selectLower]:
                keepOdr &= sel
        nKeepOdr = np.sum(keepOdr)
        if nKeepOdr < order:
            raise RuntimeError(
                "Not enough good data points ({0:d}) for polynomial fit of order {1:d}".
                format(nKeepOdr, order))
        orthRegCoeffs = orthogonalRegression(xx[keepOdr], yy[keepOdr], order, initialGuess)
    yOrthLine = np.polyval(orthRegCoeffs, xLine)

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
    mUpper = ((yOrthLine[crossIdxUpper + yOffset] - yOrthLine[crossIdxUpper - yOffset])
              /(xLine[crossIdxUpper + yOffset] - xLine[crossIdxUpper - yOffset]))
    mLower = ((yOrthLine[crossIdxLower + yOffset] - yOrthLine[crossIdxLower - yOffset])
              /(xLine[crossIdxLower + yOffset] - xLine[crossIdxLower - yOffset]))
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

    # Find some sensible plotting limits for the P1 line fit
    frac = 0.26
    crossIdxMid = crossIdxLower + int(0.5*(crossIdxUpper - crossIdxLower))
    fracIdx = min(int(frac*len(xLine)), len(xLine) - 1 - crossIdxMid)
    yAtCrossIdxMid = yOrthLine[crossIdxMid]
    midCrossPlusFracIdx = np.abs(yOrthLine - (yAtCrossIdxMid + frac*deltaY)).argmin()
    yAtFracIdx = yOrthLine[crossIdxMid + fracIdx]
    idxP1 = (crossIdxMid + fracIdx) if yAtFracIdx < (yAtCrossIdxMid + frac*deltaY) else midCrossPlusFracIdx
    deltaIdxP1 = idxP1 - crossIdxMid
    xP1Line = xLine[crossIdxMid - deltaIdxP1:crossIdxMid + deltaIdxP1]
    yP1Line = yOrthLine[crossIdxMid - deltaIdxP1:crossIdxMid + deltaIdxP1]
    axes[0].plot(xP1Line, yP1Line, "g--", lw=0.75)

    kwargs = dict(s=3, marker="o", lw=0, alpha=0.4)
    axes[0].scatter(xx[~keep], yy[~keep], c=zOther, cmap="gray", label="other", **kwargs)
    axes[0].scatter(xx[keep], yy[keep], c=zKeep, cmap="jet", label="used", **kwargs)
    axes[0].set_xlabel(xLabel)
    axes[0].set_ylabel(yLabel, labelpad=-1)

    mappableKeep = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=zKeep.min(), vmax=zKeep.max()))
    mappableKeep._A = []        # fake up the array of the scalar mappable. Urgh...
    caxKeep = plt.axes([0.46, 0.15, 0.022, 0.75])
    caxKeep.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    cbKeep = plt.colorbar(mappableKeep, cax=caxKeep)
    cbKeep.ax.tick_params(labelsize=6, length=1)
    labelPadShift = len(str(zKeep.max()//10)) if zKeep.max()//10 > 0 else 0
    cbKeep.set_label("Number Density", rotation=270, labelpad=-12 - labelPadShift, fontsize=7)

    if xFitRange:
        # Shade region outside xFitRange
        kwargs = dict(facecolor="k", edgecolor="none", alpha=0.05)
        axes[0].axvspan(axes[0].get_xlim()[0], xFitRange[0], **kwargs)
        axes[0].axvspan(xFitRange[1], axes[0].get_xlim()[1], **kwargs)
        # Looser range after fisrt iteration
        axes[0].axvspan(axes[0].get_xlim()[0], xMinPad, **kwargs)
        axes[0].axvspan(xMaxPad, axes[0].get_xlim()[1], **kwargs)
    if yFitRange:
        # Shade region outside yFitRange
        xMin = abs(xFitRange[0] - xRange[0])/deltaX if xFitRange else 1
        xMax = abs(xFitRange[1] - xRange[0])/deltaX if xFitRange else 1
        axes[0].axhspan(axes[0].get_ylim()[0], yFitRange[0], xmin=xMin, xmax=xMax, **kwargs)
        axes[0].axhspan(yFitRange[1], axes[0].get_ylim()[1], xmin=xMin, xmax=xMax, **kwargs)
        # Looser range after fisrt iteration
        xMin = abs(xMinPad - xRange[0])/deltaX if xFitRange else 1
        xMax = abs(xMaxPad - xRange[0])/deltaX if xFitRange else 1
        axes[0].axhspan(axes[0].get_ylim()[0], yMinPad, xmin=xMin, xmax=xMax, **kwargs)
        axes[0].axhspan(yMaxPad, axes[0].get_ylim()[1], xmin=xMin, xmax=xMax, **kwargs)
    if fitLineUpper:
        # Find some sensible plotting limits for the upper line fit
        frac = 0.1
        yLineUpper = fitLineUpper[0] + fitLineUpper[1]*xLine
        idxAtYlimPlusFrac = np.abs(yLineUpper - (yFitRange[1] + frac*deltaY)).argmin()
        idxAtYlimMinusFrac = np.abs(yLineUpper - (yFitRange[1] - frac*deltaY)).argmin()
        idx0 = min(idxAtYlimPlusFrac, idxAtYlimMinusFrac)
        idx1 = max(idxAtYlimPlusFrac, idxAtYlimMinusFrac)
        idxAtXlimPlusFrac = np.abs(xLine - (xFitRange[1] + frac*deltaX)).argmin()
        idxAtXlimMinusFrac = np.abs(xLine - (xFitRange[1] - frac*deltaX)).argmin()
        idx0 = max(idx0, min(idxAtXlimPlusFrac, idxAtXlimMinusFrac))
        idx1 = min(idx1, max(idxAtXlimPlusFrac, idxAtXlimMinusFrac))
        deltaIdx = max(crossIdxUpper - idx0, idx1 - crossIdxUpper)
        yLineUpper = yLineUpper[crossIdxUpper - deltaIdx:crossIdxUpper + deltaIdx]
        xLineUpper = xLine[crossIdxUpper - deltaIdx:crossIdxUpper + deltaIdx]
        axes[0].plot(xLineUpper, yLineUpper, "r--")
    if fitLineLower:
        # Find some sensible plotting limits for the lower line fit
        frac = 0.1
        yLineLower = fitLineLower[0] + fitLineLower[1]*xLine
        idxAtYlimPlusFrac = np.abs(yLineLower - (yFitRange[0] + frac*deltaY)).argmin()
        idxAtYlimMinusFrac = np.abs(yLineLower - (yFitRange[0] - frac*deltaY)).argmin()
        idx0 = min(idxAtYlimPlusFrac, idxAtYlimMinusFrac)
        idx1 = max(idxAtYlimPlusFrac, idxAtYlimMinusFrac)
        idxAtXlimPlusFrac = np.abs(xLine - (xFitRange[1] + frac*deltaX)).argmin()
        idxAtXlimMinusFrac = np.abs(xLine - (xFitRange[1] - frac*deltaX)).argmin()
        idx0 = max(idx0, min(idxAtXlimPlusFrac, idxAtXlimMinusFrac))
        idx1 = min(idx1, max(idxAtXlimPlusFrac, idxAtXlimMinusFrac))
        deltaIdx = max(crossIdxLower - idx0, idx1 - crossIdxLower)
        yLineLower = yLineLower[crossIdxLower - deltaIdx:crossIdxLower + deltaIdx]
        xLineLower = xLine[crossIdxLower - deltaIdx:crossIdxLower + deltaIdx]
        axes[0].plot(xLineLower, yLineLower, "r--")

    # Label total number of objects of each data type
    kwargs = dict(va="center", fontsize=7)
    lenNumObj = max(len(str(len(xx[keepOdr]))), len(str(len(xx))))
    fdx = max((min(0.08*lenNumObj, 0.6), 0.32))
    xLoc, yLoc = xRange[0] + 0.05*deltaX, yRange[1] - 0.036*deltaY
    axes[0].text(xLoc, yLoc, "N$_{total}$ =", ha="left", color="black", **kwargs)
    axes[0].text(xLoc + fdx*deltaX, yLoc, str(len(xx)), ha="right", color="black", **kwargs)
    yLoc -= 0.044*(yRange[1] - yRange[0])
    axes[0].text(xLoc, yLoc, "N$_{used }$ =", ha="left", color="blue", **kwargs)
    axes[0].text(xLoc + fdx*deltaX, yLoc, str(len(xx[keepOdr])), ha="right", color="blue", **kwargs)
    yLoc += 2*0.044*(yRange[1] - yRange[0])
    for threshStr in list(thresholdStr):
        yLoc -= 0.044*(yRange[1] - yRange[0])
        axes[0].text(xRange[1] - 0.03*deltaX, yLoc, threshStr, ha="right", color="black", **kwargs)
    yLoc = yLoc - 0.044*(yRange[1] - yRange[0]) if len(list(thresholdStr)) == 1 else yLoc

    unitStr = "mmag" if unitScale == 1000 else "mag"
    axes[1].set_xlabel("Dist to poly fit or Pincp Color ({:s})".format(unitStr))
    axes[1].set_ylabel("Number")
    axes[1].set_yscale("log", nonposy="clip")

    # Label orthogonal polynomial fit parameters to 2 decimal places
    xLoc = xRange[0] + 0.045*deltaX
    polyColor = "tab:pink"
    polyFit = orthRegCoeffs
    polyStr = "odr"
    kept = keepOdr

    polyStr = "y$_{" + polyStr + "}$" + " = {:.2f}".format(polyFit[len(polyFit) - 1])
    for i in range(1, len(polyFit)):
        index = len(polyFit) - 1 - i
        exponent = "$^{" + str(i) + "}$" if i > 1 else ""
        coeffStr = "{:.2f}".format(abs(polyFit[index])) + "x" + exponent
        plusMinus = " $-$ " if polyFit[index] < 0.0 else " + "
        if i == 0:
            polyStr += plusMinus.strip(" ") + coeffStr
        else:
            polyStr += plusMinus + coeffStr
    yLoc -= 0.05*deltaY
    kwargs = dict(ha="left", va="center", color=polyColor)
    axes[0].text(xLoc, yLoc, polyStr, fontsize=7, **kwargs)

    if "odr" in polyStr and order == 1:
        m, b = polyFit[0], polyFit[1]
        # Closest point on line to highest density point
        xHighDensity0 = (xHighDensity + m*(yHighDensity - b))/(m**2.0 + 1.0)
        yHighDensity0 = (m*(xHighDensity + m*yHighDensity) + b)/(m**2.0 + 1.0)
        bP2 = yHighDensity0 + (1.0/m)*xHighDensity0
        yP2Line = (-1.0/m)*xLine + bP2
        # Find some sensible plotting limits for the P2 line fit
        frac = 0.15
        idxHd = np.abs(yP2Line - yHighDensity0).argmin()
        idxFrac = idxHd - int(frac*len(xLine))
        fracIdx = max(idxHd - int(frac*len(xLine)), 0)
        yAtIdxFrac = yP2Line[idxFrac]
        idxHdPlusFrac = np.abs(yP2Line - (yHighDensity0 + frac*deltaY)).argmin()
        yAtHdPlusFrac = yP2Line[idxHdPlusFrac]
        idxP2 = idxFrac if yAtIdxFrac < yAtHdPlusFrac else idxHdPlusFrac
        deltaIdxP2 = idxHd - idxP2
        xP2Line = xLine[idxHd - deltaIdxP2:idxHd + deltaIdxP2]
        yP2Line = yP2Line[idxHd - deltaIdxP2:idxHd + deltaIdxP2]
        axes[0].plot(xP2Line, yP2Line, "g--", lw=0.75)
        plotText("P2$^{fit}$", plt, axes[0], xP2Line[0] - 0.022*deltaX, yP2Line[0] + 0.045*deltaY,
                 fontSize=8, color="green", coordSys="data")
        plotText("P1$^{fit}$", plt, axes[0], xP1Line[0] - 0.07*deltaX, yP1Line[0] + 0.035*deltaY,
                 fontSize=8, color="green", coordSys="data")

        # Also plot the effective hard wired lines
        wiredLine = linesFromP2P1Coeffs(list(transformPerp.coeffs.values()),
                                        list(transformPara.coeffs.values()))
        yP2LineWired = wiredLine.mP2*xP2Line + wiredLine.bP2
        yP1LineWired = wiredLine.mP1*xP1Line + wiredLine.bP1
        axes[0].plot(xP2Line, yP2LineWired, "b--", alpha=0.6, lw=0.75)
        axes[0].plot(xP1Line, yP1LineWired, "b--", alpha=0.6, lw=0.75)
        plotText("$_{wired}$", plt, axes[0], xP2Line[0] + 0.032*deltaX, yP2Line[0] + 0.03*deltaY,
                 fontSize=8, color="blue", coordSys="data", alpha=0.6)
        plotText("$_{wired}$", plt, axes[0], xP1Line[0] - 0.02*deltaX, yP1Line[0] + 0.02*deltaY,
                 fontSize=8, color="blue", coordSys="data", alpha=0.6)

        # Derive Ivezic P2 and P1 equations based on linear fit and highest density position (where P1 = 0)
        pColCoeffs = p2p1CoeffsFromLinearFit(m, b, xHighDensity0, yHighDensity0)

        perpIndex = description.find("Fit")
        perpIndexStr = description[perpIndex - 1:perpIndex]
        if perpIndexStr in ("w", "x"):
            perpFilters = ["g", "r", "i", ""]
        elif perpIndexStr == "y":
            perpFilters = ["r", "i", "z", ""]
        else:
            raise RuntimeError("Unknown Principal Color: {0:s}Perp".format(perpIndexStr))

        log.info("  {0:s}Perp_wired: origin x, y: {1:.4f} {2:.4f}".format(perpIndexStr,
                                                                          wiredLine.x0, wiredLine.y0))
        log.info("  {0:s}Perp_fit  : origin x, y: {1:.4f} {2:.4f}".format(perpIndexStr,
                                                                          xHighDensity0, yHighDensity0))

        paraStr = "{0:s}Para{1:s}".format(perpIndexStr, "$_{fit}$")
        paraStr = makeEqnStr(paraStr, pColCoeffs.p1Coeffs, perpFilters)
        perpStr = "{0:s}Perp{1:s}".format(perpIndexStr, "$_{fit}$")
        perpStr = makeEqnStr(perpStr, pColCoeffs.p2Coeffs, perpFilters)

        # Also label plot with hardwired numbers
        principalColorStrs = []
        for transform, pCol in zip([transformPerp, transformPara],
                                   [perpIndexStr + "Perp", perpIndexStr[0:1] + "Para"]):
            principalColorStr = "{0:s}{1:s}".format(pCol, "$_{wired}$")
            principalColorStr = makeEqnStr(principalColorStr, transform.coeffs.values(), perpFilters)
            principalColorStrs.append(principalColorStr)

        xLoc = xRange[1] - 0.03*deltaX
        yLoc -= 0.05*deltaY
        axes[0].text(xLoc, yLoc, perpStr, fontsize=6, ha="right", va="center", color="green")
        yLoc -= 0.04*deltaY
        axes[0].text(xLoc, yLoc, principalColorStrs[0], fontsize=6, ha="right", va="center",
                     color="blue", alpha=0.8)
        yLoc -= 0.05*deltaY
        axes[0].text(xLoc, yLoc, paraStr, fontsize=6, ha="right", va="center", color="green")
        yLoc -= 0.04*deltaY
        axes[0].text(xLoc, yLoc, principalColorStrs[1], fontsize=6, ha="right", va="center",
                     color="blue", alpha=0.8)
        log.info("{0:s}".format("".join(x for x in perpStr if x not in "{}$")))
        log.info("{0:s}".format("".join(x for x in paraStr if x not in "{}$")))

        # Compute fitted P2 for each object
        if transform:
            fitP2 = np.ones(numGood)*pColCoeffs.p2Coeffs[3]
            for i, filterName in enumerate(transform.coeffs.keys()):
                if filterName != "":
                    fitP2 += mags[filterName]*pColCoeffs.p2Coeffs[i]
            fitP2 *= unitScale

    # Determine quality of locus
    distance2 = []
    polyFit = np.poly1d(polyFit)
    polyDeriv = np.polyder(polyFit)
    for x, y in zip(xx[kept], yy[kept]):
        roots = np.roots(np.poly1d((1, -x)) + (polyFit - y)*polyDeriv)
        distance2.append(min(
            distanceSquaredToPoly(x, y, np.real(rr), polyFit) for rr in roots if np.real(rr) == rr))
    distance = np.sqrt(distance2)
    distance *= np.where(yy[kept] >= polyFit(xx[kept]), 1.0, -1.0)
    distance *= unitScale
    clippedStats = calcQuartileClippedStats(distance, nSigmaToClip=3.0)
    good = clippedStats.goodArray
    # Get rid of LaTeX-specific characters for log message printing
    log.info("Polynomial fit: {:2}".format("".join(x for x in polyStr if x not in "{}$")))
    log.info(("Statistics from {0:} of Distance to polynomial ({9:s}): {7:s}\'star\': " +
              "Stats(mean={1:.4f}; stdev={2:.4f}; num={3:d}; total={4:d}; median={5:.4f}; clip={6:.4f})" +
              "{8:s}").format(plotInfoDict["dataId"], clippedStats.mean, clippedStats.stdDev, len(xx[keep]),
                              len(xx), clippedStats.median, clippedStats.clipValue, "{", "}", unitStr))
    meanStr = "mean = {0:5.2f}".format(clippedStats.mean)
    stdStr = "  std = {0:5.2f}".format(clippedStats.stdDev)
    rmsStr = "  rms = {0:5.2f}".format(clippedStats.rms)

    count, bins, ignored = axes[1].hist(distance[good], bins=numBins,
                                        range=(-4.0*clippedStats.stdDev, 4.0*clippedStats.stdDev),
                                        density=True, color=polyColor, alpha=0.6)
    axes[1].plot(bins, (1/(clippedStats.stdDev*np.sqrt(2*np.pi))*np.exp(-(bins - clippedStats.mean)**2
                                                                        /(2*clippedStats.stdDev**2))),
                 color=polyColor)
    axes[1].axvline(x=clippedStats.mean, color=polyColor, linestyle=":")
    kwargs = dict(xycoords="axes fraction", ha="right", va="center", fontsize=7, color=polyColor)
    axes[1].annotate(meanStr, xy=(0.34, 0.965), **kwargs)
    axes[1].annotate(stdStr, xy=(0.34, 0.93), **kwargs)
    axes[1].annotate(rmsStr, xy=(0.34, 0.895), **kwargs)

    axes[1].axvline(x=0.0, color="black", linestyle="--")
    tractStr = "tract: {:s}".format(plotInfoDict["tract"])
    axes[1].annotate(tractStr, xy=(0.5, 1.04), xycoords="axes fraction", ha="center", va="center",
                     fontsize=10, color="green")

    # Plot hardwired principal color distributions
    if principalCol is not None:
        pCkept = principalColor[kept].copy()
        pCmean = pCkept[good].mean()
        pCstdDev = pCkept[good].std()

        count, nBins, ignored = axes[1].hist(pCkept[good], bins=bins,
                                             range=(-4.0*clippedStats.stdDev, 4.0*clippedStats.stdDev),
                                             density=True, color="blue", alpha=0.6)
        axes[1].plot(bins, 1/(pCstdDev*np.sqrt(2*np.pi))*np.exp(-(bins-pCmean)**2/(2*pCstdDev**2)),
                     color="blue")
        axes[1].axvline(x=pCmean, color="blue", linestyle=":")
        pCmeanStr = "{0:s}{1:s} = {2:5.2f}".format(perpStr[0:5], "$_{wired}$", pCmean)
        pCstdStr = "  std = {0:5.2f}".format(pCstdDev)
        kwargs = dict(xycoords="axes fraction", ha="right", va="center", fontsize=7, color="blue")
        axes[1].annotate(pCmeanStr, xy=(0.97, 0.965), **kwargs)
        axes[1].annotate(pCstdStr, xy=(0.97, 0.93), **kwargs)
        log.info(("Statistics from {0:} of {9:s}Perp_wired ({8:s}): {6:s}\'star\': " +
                  "Stats(mean={1:.4f}; stdev={2:.4f}; num={3:d}; total={4:d}; median={5:.4f})" +
                  "{7:s}").format(plotInfoDict["dataId"], pCmean, pCstdDev, len(pCkept[good]), len(pCkept),
                                  np.median(pCkept[good]), "{", "}", unitStr, perpIndexStr))
    # Plot fitted principal color distributions
    if fitP2 is not None:
        fitP2kept = fitP2[kept].copy()
        fitP2mean = fitP2kept[good].mean()
        fitP2stdDev = fitP2kept[good].std()
        count, nBins, ignored = axes[1].hist(fitP2kept[good], bins=bins,
                                             range=(-4.0*clippedStats.stdDev, 4.0*clippedStats.stdDev),
                                             density=True, color="green", alpha=0.6)
        axes[1].plot(bins, 1/(fitP2stdDev*np.sqrt(2*np.pi))*np.exp(-(bins-fitP2mean)**2/(2*fitP2stdDev**2)),
                     color="green")
        axes[1].axvline(x=fitP2mean, color="tab:pink", linestyle=":")
        fitP2meanStr = "{0:s}{1:s} = {2:5.2f}".format(perpStr[0:5], "$_{fit}$", fitP2mean)
        fitP2stdStr = "  std = {0:5.2f}".format(fitP2stdDev)
        kwargs = dict(xycoords="axes fraction", ha="right", va="center", fontsize=7, color="green")
        axes[1].annotate(fitP2meanStr, xy=(0.97, 0.895), **kwargs)
        axes[1].annotate(fitP2stdStr, xy=(0.97, 0.86), **kwargs)
        log.info(("Statistics from {0:} of {9:s}Perp_fit ({8:s}): {6:s}\'star\': " +
                  "Stats(mean={1:.4f}; stdev={2:.4f}; num={3:d}; total={4:d}; median={5:.4f})" +
                  "{7:s}").format(plotInfoDict["dataId"], fitP2mean, fitP2stdDev, len(fitP2kept[good]),
                                  len(fitP2kept), np.median(fitP2kept[good]), "{", "}", unitStr,
                                  perpIndexStr))
        if verifyJob:
            if not verifyMetricName:
                log.warn("A verifyJob was specified, but the metric name was not...skipping metric job")
            else:
                log.info("Adding verify job with metric name: {:}".format(verifyMetricName))
                measExtrasDictList = [{"name": "nUsedInFit", "value": len(fitP2kept[good]),
                                       "label": "nUsed", "description": "Number of points used in the fit"},
                                      {"name": "numberTotal", "value": len(fitP2kept),
                                       "label": "nTot", "description":
                                       "Total number of points considered for use in the fit"},
                                      {"name": "mean", "value": np.around(fitP2mean, decimals=3)*u.mmag,
                                       "label": "mean", "description": "Fit mean"},
                                      {"name": "median", "value":
                                       np.around(np.median(fitP2kept[good]), decimals=3)*u.mmag,
                                       "label": "median", "description": "Fit median"}]
                verifyJob = addMetricMeasurement(verifyJob, "pipe_analysis." + verifyMetricName,
                                                 np.around(fitP2stdDev, decimals=3)*u.mmag,
                                                 measExtrasDictList=measExtrasDictList)

    axes[1].set_ylim(axes[1].get_ylim()[0], axes[1].get_ylim()[1]*2.5)

    if plotInfoDict["cameraName"]:
        labelCamera(plotInfoDict, fig, axes[0], 0.5, 1.04)
    if catLabel:
        plotText(catLabel, fig, axes[0], 0.88, -0.12, fontSize=7, color="green")
    if geLabel:
        plotText(geLabel, fig, axes[0], 0.13, -0.12, fontSize=7, color="green")
    if uberCalLabel:
        plotText(uberCalLabel, fig, axes[0], 0.13, -0.16, fontSize=7, color="green")
    if plotInfoDict["hscRun"]:
        axes[0].set_title("HSC stack run: " + plotInfoDict["hscRun"], color="#800080")

    yield Struct(fig=fig, description=description, stats=None, statsHigh=None, dpi=120, style="fit")

    return orthRegCoeffs


def colorColorPlot(plotInfoDict, description, log, xStars, yStars, xGalaxies, yGalaxies, magStars,
                   magGalaxies, xLabel, yLabel, filterStr, fluxColStr, xRange=None, yRange=None,
                   geLabel=None, uberCalLabel=None, logger=None, magThreshold=99.9, unitScale=1.0):
    fig, axes = plt.subplots(1, 1)
    axes.tick_params(which="both", direction="in", labelsize=9)

    goodStars = np.ones(len(magStars), dtype=bool)
    goodGalaxies = np.ones(len(magGalaxies), dtype=bool)
    if magThreshold < 90.0:
        goodStars = np.logical_and(goodStars, magStars < magThreshold)
        goodGalaxies = np.logical_and(goodGalaxies, magGalaxies < magThreshold)

    if xRange:
        axes.set_xlim(*xRange)
    else:
        xRange = (0.9*xStars[goodStars].min(), 1.1*xStars[goodStars].max())
    if yRange:
        axes.set_ylim(*yRange)

    vMin = min(magStars[goodStars].min(), magGalaxies[goodGalaxies].min())
    vMax = min(magStars[goodStars].max(), magGalaxies[goodGalaxies].max())

    ptSize = max(1, setPtSize(len(xGalaxies[goodGalaxies])) - 2)

    kwargs = dict(s=ptSize, marker="o", lw=0, vmin=vMin, vmax=vMax)
    axes.scatter(xGalaxies[goodGalaxies], yGalaxies[goodGalaxies], c=magGalaxies[goodGalaxies],
                 cmap="autumn", label="galaxies", **kwargs)
    axes.scatter(xStars[goodStars], yStars[goodStars], c=magStars[goodStars],
                 cmap="winter", label="stars", **kwargs)
    axes.set_xlabel(xLabel)
    axes.set_ylabel(yLabel, labelpad=-1)

    # Label total number of objects of each data type
    deltaX = abs(xRange[1] - xRange[0])
    deltaY = abs(yRange[1] - yRange[0])
    lenNumObj = max(len(str(len(xStars[goodStars]))), len(str(len(xGalaxies[goodGalaxies]))))
    fdx = max((min(0.095*lenNumObj, 0.9), 0.42))
    xLoc, yLoc = xRange[0] + 0.03*deltaX, yRange[1] - 0.038*deltaY
    kwargs = dict(va="center", fontsize=8)
    axes.text(xLoc, yLoc, "Ngals  =", ha="left", color="red", **kwargs)
    axes.text(xLoc + fdx*deltaX, yLoc, str(len(xGalaxies[goodGalaxies])) +
              " [" + filterStr + "$<$" + str(magThreshold) + "]", ha="right", color="red", **kwargs)
    axes.text(xLoc, 0.94*yLoc, "Nstars =", ha="left", color="blue", **kwargs)
    axes.text(xLoc + fdx*deltaX, 0.94*yLoc, str(len(xStars[goodStars])) +
              " [" + filterStr + "$<$" + str(magThreshold) + "]", ha="right", color="blue", **kwargs)
    if plotInfoDict["cameraName"]:
        labelCamera(plotInfoDict, fig, axes, 0.5, 1.09)
    if geLabel:
        plotText(geLabel, fig, axes, 0.13, -0.09, fontSize=7, color="green")
    if uberCalLabel:
        plotText(uberCalLabel, fig, axes, 0.89, -0.09, fontSize=7, color="green")
    if plotInfoDict["hscRun"]:
        axes.set_title("HSC stack run: " + plotInfoDict["hscRun"], color="#800080")

    tractStr = "tract: {:s}".format(plotInfoDict["tract"])
    axes.annotate(tractStr, xy=(0.5, 1.04), xycoords="axes fraction", ha="center", va="center",
                  fontsize=10, color="green")

    mappableStars = plt.cm.ScalarMappable(cmap="winter", norm=plt.Normalize(vmin=vMin, vmax=vMax))
    mappableStars._A = []        # fake up the array of the scalar mappable. Urgh...
    cbStars = plt.colorbar(mappableStars, aspect=14, pad=-0.09)
    cbStars.ax.tick_params(labelsize=8)
    cbStars.set_label(filterStr + " [" + fluxColStr + "]: stars", rotation=270, labelpad=-24, fontsize=9)
    mappableGalaxies = plt.cm.ScalarMappable(cmap="autumn", norm=plt.Normalize(vmin=vMin, vmax=vMax))
    mappableGalaxies._A = []      # fake up the array of the scalar mappable. Urgh...
    cbGalaxies = plt.colorbar(mappableGalaxies, aspect=14)
    cbGalaxies.set_ticks([])
    cbGalaxies.set_label(filterStr + " [" + fluxColStr + "]: galaxies", rotation=270, labelpad=-6, fontsize=9)

    yield Struct(fig=fig, description=description, stats=None, statsHigh=None, dpi=120, style="noFit")


def colorColor4MagPlots(plotInfoDict, description, log, xStars, yStars, xGalaxies, yGalaxies, magStars,
                        magGalaxies, xLabel, yLabel, filterStr, fluxColStr, xRange=None, yRange=None,
                        geLabel=None, uberCalLabel=None, logger=None, magThreshold=99.9, unitScale=1.0):

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0, wspace=0, bottom=0.11, right=0.82, top=0.91)

    xRange = ((xRange[0] + 0.01, xRange[1] - 0.01) if xRange
              else (0.9*xStars.min(), 1.1*xStars.max()))
    yRange = ((yRange[0] + 0.01, yRange[1] - 0.01) if yRange
              else (0.9*yStars.min(), 1.1*yStars.max()))
    deltaX = abs(xRange[1] - xRange[0])
    deltaY = abs(yRange[1] - yRange[0])
    vMin = round(min(magStars.min(), magGalaxies.min()) - 0.05, 1)
    vMax = round(max(magStars.max(), magGalaxies.max()) + 0.05, 1)

    ptSize = max(1, setPtSize(len(xGalaxies)) - 2)

    # Divide stars and galaxies into 4 magnitude bins
    binEdges = np.hstack((vMin, np.arange(magThreshold - 3, magThreshold, 1)))
    binIdxStars = np.digitize(magStars, binEdges)
    binIdxGalaxies = np.digitize(magGalaxies, binEdges)
    # The following is for ease of printing the bin ranges in the following loop
    binEdges = [bin for bin in binEdges]
    binEdges.append(magThreshold)

    for i, ax in enumerate(axes.flat[:]):
        index = 4 - i
        ax.tick_params(which="both", direction="in", labelsize=7)
        ax.set_xlim(*xRange)
        ax.set_ylim(*yRange)

        kwargs = dict(s=ptSize, marker="o", lw=0, vmin=vMin, vmax=vMax)
        ax.scatter(xGalaxies[binIdxGalaxies == index], yGalaxies[binIdxGalaxies == index],
                   c=magGalaxies[binIdxGalaxies == index], cmap="autumn", label="galaxies", **kwargs)
        ax.scatter(xStars[binIdxStars == index], yStars[binIdxStars == index],
                   c=magStars[binIdxStars == index], cmap="winter", label="stars", **kwargs)
        if i in (2, 3):
            ax.set_xlabel(xLabel)
        if i in (0, 2):
            ax.set_ylabel(yLabel)

        # Label total number of objects of each data type
        xLoc, yLoc = xRange[0] + 0.05*deltaX, yRange[1] - 0.06*deltaY
        kwargs = dict(va="center", fontsize=7)
        ax.text(xLoc, yLoc, "Ngals  =", ha="left", color="red", **kwargs)
        ax.text(xRange[1] - 0.03*deltaX, yLoc, str(len(xGalaxies[binIdxGalaxies == index])) +
                " [" + str(binEdges[index - 1]) + " <= " + filterStr + " < " + str(binEdges[index]) + "]",
                ha="right", color="red", **kwargs)
        ax.text(xLoc, 0.92*yLoc, "Nstars =", ha="left", va="center", fontsize=7, color="blue")
        ax.text(xRange[1] - 0.03*deltaX, 0.92*yLoc, str(len(xStars[binIdxStars == index])) +
                " [" + str(binEdges[index - 1]) + " <= " + filterStr + " < " + str(binEdges[index]) + "]",
                ha="right", color="blue", **kwargs)

    mappableStars = plt.cm.ScalarMappable(cmap="winter_r", norm=plt.Normalize(vmin=vMin, vmax=vMax))
    mappableStars._A = []        # fake up the array of the scalar mappable. Urgh...
    caxStars = plt.axes([0.88, 0.11, 0.04, 0.8])
    caxGalaxies = plt.axes([0.84, 0.11, 0.04, 0.8])
    cbStars = plt.colorbar(mappableStars, cax=caxStars)
    cbStars.ax.tick_params(labelsize=8)
    cbStars.set_label(filterStr + "[" + fluxColStr + "] :stars", rotation=270, labelpad=-24, fontsize=9)
    mappableGalaxies = plt.cm.ScalarMappable(cmap="autumn_r", norm=plt.Normalize(vmin=vMin, vmax=vMax))
    mappableGalaxies._A = []      # fake up the array of the scalar mappable. Urgh...
    cbGalaxies = plt.colorbar(mappableGalaxies, cax=caxGalaxies)
    cbGalaxies.set_ticks([])
    cbGalaxies.set_label(filterStr + " [" + fluxColStr + "]: galaxies", rotation=270, labelpad=-6, fontsize=9)

    if plotInfoDict["cameraName"]:
        labelCamera(plotInfoDict, fig, axes[0, 0], 1.05, 1.14)
    if geLabel:
        plotText(geLabel, fig, axes[0, 0], 0.12, -1.23, color="green")
    if uberCalLabel:
        plotText(uberCalLabel, fig, axes[0, 1], 0.89, -1.23, fontSize=7, color="green")
    if plotInfoDict["hscRun"]:
        axes.set_title("HSC stack run: " + plotInfoDict["hscRun"], color="#800080")

    tractStr = "tract: {:s}".format(plotInfoDict["tract"])
    axes[0, 0].annotate(tractStr, xy=(1.05, 1.06), xycoords="axes fraction", ha="center", va="center",
                        fontsize=9, color="green")

    yield Struct(fig=fig, description=description, stats=None, statsHigh=None, dpi=120, style="noFitMagBins")


class ColorColorDistance(object):
    """Functor to calculate distance from stellar locus in color-color plot"""
    def __init__(self, band1, band2, band3, poly, unitScale=1.0, xMin=None, xMax=None,
                 fitLineUpper=None, fitLineLower=None):
        self.band1 = band1
        self.band2 = band2
        self.band3 = band3
        if isinstance(poly, np.lib.polynomial.poly1d):
            self.poly = poly
        else:
            self.poly = np.poly1d(poly)
        self.unitScale = unitScale
        self.xMin = xMin
        self.xMax = xMax
        self.fitLineUpper = fitLineUpper
        self.fitLineLower = fitLineLower

    def __call__(self, catalog):
        xx = catalog[self.band1] - catalog[self.band2]
        yy = catalog[self.band2] - catalog[self.band3]
        polyDeriv = np.polyder(self.poly)
        distance2 = np.ones_like(xx)*np.nan
        for i, (x, y) in enumerate(zip(xx, yy)):
            if (not np.isfinite(x) or not np.isfinite(y) or (self.xMin and x < self.xMin) or
                (self.xMax and x > self.xMax) or
                (self.fitLineUpper and y > self.fitLineUpper[0] + self.fitLineUpper[1]*x) or
                    (self.fitLineLower and y < self.fitLineLower[0] + self.fitLineLower[1]*x)):
                distance2[i] = np.nan
                continue
            roots = np.roots(np.poly1d((1, -x)) + (self.poly - y)*polyDeriv)
            distance2[i] = min(distanceSquaredToPoly(x, y, np.real(rr), self.poly) for
                               rr in roots if np.real(rr) == rr)
        return np.sqrt(distance2)*np.where(yy >= self.poly(xx), 1.0, -1.0)*self.unitScale


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
