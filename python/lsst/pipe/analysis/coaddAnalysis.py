# This file is part of pipe_analysis.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import astropy.units as u
import numpy as np
import pandas as pd
np.seterr(all="ignore")  # noqa E402
import functools

from collections import defaultdict

from lsst.daf.persistence.butler import Butler
from lsst.pex.config import (Config, Field, ConfigField, ListField, DictField, ConfigDictField,
                             ConfigurableField)
from lsst.pipe.base import CmdLineTask, ArgumentParser, TaskRunner, TaskError, Struct
from lsst.pipe.drivers.utils import TractDataIdContainer
from lsst.meas.astrom import AstrometryConfig
from lsst.pipe.tasks.parquetTable import MultilevelParquetTable
from lsst.pipe.tasks.colorterms import Colorterm, ColortermLibrary

from lsst.meas.algorithms import LoadIndexedReferenceObjectsTask

from .analysis import AnalysisConfig, Analysis
from .utils import (Enforcer, MagDiff, MagDiffMatches, MagDiffCompare, AstrometryDiff, AngularDistance,
                    TraceSize, PsfTraceSizeDiff, TraceSizeCompare, PercentDiff, E1Resids, E2Resids,
                    E1ResidsHsmRegauss, E2ResidsHsmRegauss, FootAreaDiffCompare, MagDiffCompareErr,
                    CentroidDiff, deconvMom, deconvMomStarGal, concatenateCatalogs, joinMatches,
                    matchAndJoinCatalogs, checkPatchOverlap, addColumnsToSchema, addFpPoint,
                    addFootprintArea, makeBadArray, addElementIdColumn, addIntFloatOrStrColumn,
                    calibrateSourceCatalog, backoutApCorr, matchNanojanskyToAB, fluxToPlotString,
                    andCatalog, writeParquet, getRepoInfo, addAliasColumns, addPreComputedColumns,
                    computeMeanOfFrac, savePlots, updateVerifyJob, getSchema, loadDenormalizeAndUnpackMatches,
                    computeAreaDict, getParquetColumnsList)
from .plotUtils import (CosmosLabeller, AllLabeller, StarGalaxyLabeller, OverlapsStarGalaxyLabeller,
                        MatchesStarGalaxyLabeller, determineExternalCalLabel, getPlotInfo)

import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.verify as verify

__all__ = ["CoaddAnalysisConfig", "CoaddAnalysisRunner", "CoaddAnalysisTask", "CompareCoaddAnalysisConfig",
           "CompareCoaddAnalysisRunner", "CompareCoaddAnalysisTask"]

NANOJANSKYS_PER_AB_FLUX = (0*u.ABmag).to_value(u.nJy)
FLAGCOLORS = ["yellow", "greenyellow", "lime", "aquamarine", "orange", "fuchsia", "gold", "lightseagreen"]


class CoaddAnalysisConfig(Config):
    coaddName = Field(dtype=str, default="deep", doc="Name for coadd")
    matchRadiusRaDec = Field(dtype=float, default=0.5, doc="RaDec Matching radius (arcseconds)")
    matchOverlapRadius = Field(dtype=float, default=0.5, doc="Matching radius for overlaps (arcseconds)")
    matchXy = Field(dtype=bool, default=False, doc="Perform matching based on X/Y pixel values?")
    matchRadiusXy = Field(dtype=float, default=3.0, doc=("X/Y Matching radius (pixels): "
                                                         "ignored unless matchXy=True"))
    colorterms = ConfigField(dtype=ColortermLibrary,
                             doc=("Library of color terms."
                                  "\nNote that the colorterms, if any, need to be loaded in a config "
                                  "override file.  See obs_subaru/config/hsc/coaddAnalysis.py for an "
                                  "example.  If the colorterms for the appropriate reference dataset are "
                                  "loaded, they will be applied.  Otherwise, no colorterms will be applied "
                                  "to the reference catalog."))
    doApplyColorTerms = Field(dtype=bool, default=True, doc="Apply colorterms to reference magnitudes?")
    analysis = ConfigField(dtype=AnalysisConfig, doc="Analysis plotting options")
    analysisMatches = ConfigField(dtype=AnalysisConfig, doc="Analysis plotting options for matches")
    matchesMaxDistance = Field(dtype=float, default=0.15, doc="Maximum plotting distance for matches")
    externalCatalogs = ConfigDictField(keytype=str, itemtype=AstrometryConfig, default={},
                                       doc="Additional external catalogs for matching")
    refObjLoader = ConfigurableField(target=LoadIndexedReferenceObjectsTask, doc="Reference object loader")
    doPlotMags = Field(dtype=bool, default=True, doc="Plot magnitudes? (ignored if plotMatchesOnly is True)")
    doPlotSizes = Field(dtype=bool, default=True, doc="Plot PSF sizes? (ignored if plotMatchesOnly is True)")
    doPlotCentroids = Field(dtype=bool, default=True, doc=("Plot centroids? "
                                                           "(ignored if plotMatchesOnly is True)"))
    doApCorrs = Field(dtype=bool, default=True, doc=("Plot aperture corrections? "
                                                     "(ignored if plotMatchesOnly is True)"))
    doBackoutApCorr = Field(dtype=bool, default=False, doc="Backout aperture corrections?")
    doAddAperFluxHsc = Field(dtype=bool, default=False,
                             doc="Add a field containing 12 pix circular aperture flux to HSC table?")
    doPlotStarGalaxy = Field(dtype=bool, default=True, doc=("Plot star/galaxy? "
                                                            "(ignored if plotMatchesOnly is True)"))
    doPlotOverlaps = Field(dtype=bool, default=True, doc=("Plot overlaps? "
                                                          "(ignored if plotMatchesOnly is True)"))
    plotMatchesOnly = Field(dtype=bool, default=False, doc=("Only make plots related to reference cat"
                                                            "matches?"))
    doPlotMatches = Field(dtype=bool, default=True, doc="Plot matches?")
    doPlotCompareUnforced = Field(dtype=bool, default=True, doc=("Plot difference between forced and unforced"
                                                                 "? (ignored if plotMatchesOnly is True)"))
    doPlotRhoStatistics = Field(dtype=bool, default=True, doc=("Plot Rho statistics?"))
    treecorrParams = DictField(keytype=str, itemtype=None, optional=True,
                               default={"nbins": 11, "min_sep": 0.5, "max_sep": 20,
                                        "sep_units": "arcmin", "verbose": 0},
                               doc=("keyword arguments to be passed to treecorr,"
                                    "if doPlotRhoStatistics is True"))
    doPlotQuiver = Field(dtype=bool, default=True, doc=("Plot ellipticity residuals quiver plot? "
                                                        "(ignored if plotMatchesOnly is True)"))
    doPlotPsfFluxSnHists = Field(dtype=bool, default=True, doc="Plot histograms of raw PSF fluxes and S/N?")
    doPlotFootprintArea = Field(dtype=bool, default=True, doc=("Plot histogram of footprint area? "
                                                               "(ignored if plotMatchesOnly is True)"))
    doPlotInputCounts = Field(dtype=bool, default=True, doc=("Make input counts plot? "
                                                             "(ignored if plotMatchesOnly is True)"))
    doPlotSkyObjects = Field(dtype=bool, default=True, doc="Make sky object plots?")
    doPlotSkyObjectsSky = Field(dtype=bool, default=False, doc="Make sky projection sky object plots?")
    onlyReadStars = Field(dtype=bool, default=False, doc="Only read stars (to save memory)?")
    toMilli = Field(dtype=bool, default=True, doc="Print stats in milli units (i.e. mas, mmag)?")
    srcSchemaMap = DictField(keytype=str, itemtype=str, default=None, optional=True,
                             doc="Mapping between different stack (e.g. HSC vs. LSST) schema names")
    fluxToPlotList = ListField(dtype=str, default=["base_GaussianFlux", "base_CircularApertureFlux_12_0",
                                                   "ext_photometryKron_KronFlux", "modelfit_CModel"],
                               doc="List of fluxes to plot: mag(flux)-mag(base_PsfFlux) vs mag(fluxColumn)")
    # We want the following to come from the *_meas catalogs as they reflect
    # what happened in SFP calibration.
    columnsToCopyFromMeas = ListField(dtype=str, default=["calib_", ],
                                      doc="List of string \"prefixes\" to identify the columns to copy.  "
                                      "All columns with names that start with one of these strings will be "
                                      "copied from the *_meas catalogs into the *_forced_src catalogs "
                                      "UNLESS the full column name contains one of the strings listed "
                                      "in the notInColumnStrList config.")
    # We want the following to come from the *_ref catalogs as they reflect
    # the forced measurement states.
    columnsToCopyFromRef = ListField(dtype=str,
                                     default=["detect_", "merge_peak_sky", "merge_measurement_", ],
                                     doc="List of string \"prefixes\" to identify the columns to copy.  "
                                     "All columns with names that start with one of these strings will be "
                                     "copied from the *_ref catalogs into the *_forced_src catalogs "
                                     "UNLESS the full column name contains one of the strings listed "
                                     "in the notInColumnStrList config.")
    baseColStrList = ListField(
        dtype=str,
        default=["coord", "tract", "patch", "visit", "ccd", "base_PixelFlags", "base_GaussianFlux",
                 "base_PsfFlux", "base_CircularApertureFlux_9_0_instFlux", "base_CircularApertureFlux_12_0",
                 "base_CircularApertureFlux_25_0", "ext_photometryKron_KronFlux", "modelfit_CModel",
                 "base_Sdss", "slot_Centroid", "slot_Shape", "ext_shapeHSM_HsmSourceMoments_",
                 "ext_shapeHSM_HsmPsfMoments_", "ext_shapeHSM_HsmShapeRegauss_", "base_Footprint",
                 "base_FPPosition", "base_ClassificationExtendedness", "parent", "detect", "deblend_nChild",
                 "base_Blendedness_abs", "base_Blendedness_flag", "base_InputCount",
                 "merge_peak_sky", "merge_measurement", "calib", "sky_source"],
        doc=("List of \"startswith\" strings of column names to load from deepCoadd_obj parquet table. "
             "All columns that start with one of these strings will be loaded UNLESS the full column "
             "name contains one of the strings listed in the notInColumnStrList config."))
    notInColStrList = ListField(
        dtype=str,
        default=["flag_bad", "flag_no", "missingDetector_flag", "_region_", "Truncated", "_radius",
                 "_bad_", "initial", "_exp_", "_dev_", "fracDev", "objective", "SdssCentroid_flag_",
                 "SdssShape_flag_u", "SdssShape_flag_m", "_Cov", "_child_", "_parent_"],
        doc=("List of substrings to select against when creating list of columns to load from the "
             "deepCoadd_obj parquet table."))
    flagsToAlias = DictField(keytype=str, itemtype=str,
                             default={"calib_psf_used": "calib_psfUsed",
                                      "calib_psf_candidate": "calib_psfCandidate",
                                      "calib_astrometry_used": "calib_astrometryUsed"},
                             doc=("List of flags to alias to old, pre-RFC-498, names for backwards "
                                  "compatibility with old processings"))
    doReadParquetTables = Field(dtype=bool, default=True,
                                doc=("Read parquet tables from postprocessing (e.g. deepCoadd_obj) as "
                                     "input data instead of afwTable catalogs."))
    doWriteParquetTables = Field(dtype=bool, default=True,
                                 doc=("Write out Parquet tables (for subsequent interactive analysis)?"
                                      "\nNOTE: if True but fastparquet package is unavailable, a warning is "
                                      "issued and table writing is skipped."))
    writeParquetOnly = Field(dtype=bool, default=False,
                             doc="Only write out Parquet tables (i.e. do not produce any plots)?")
    hasFakes = Field(dtype=bool, default=False, doc="Include the analysis of the added fake sources?")
    readFootprintsAs = Field(dtype=str, default=None, optional=True,
                             doc=("What type of Footprint to read in along with the catalog: "
                                  "\n  None : do not read in Footprints."
                                  "\n\"light\": read in regular Footprints (include SpanSet and list of"
                                  "\n         peaks per Footprint)."
                                  "\n\"heavy\": read in HeavyFootprints (include regular Footprint plus"
                                  "\n         flux values per Footprint)."))

    def saveToStream(self, outfile, root="root"):
        """Required for loading colorterms from a Config outside the "lsst"
        namespace.
        """
        print("import lsst.meas.photocal.colorterms", file=outfile)
        return Config.saveToStream(self, outfile, root)

    def setDefaults(self):
        Config.setDefaults(self)
        # self.externalCatalogs = {"sdss-dr9-fink-v5b": astrom}
        self.analysisMatches.magThreshold = 21.0  # External catalogs like PS1 & SDSS used smaller telescopes
        self.refObjLoader.ref_dataset_name = "ps1_pv3_3pi_20170110"

    def validate(self):
        Config.validate(self)
        if self.writeParquetOnly and not self.doWriteParquetTables:
            raise ValueError("Cannot writeParquetOnly if doWriteParquetTables is False")
        if self.plotMatchesOnly:
            self.doPlotMatches = True
        if self.plotMatchesOnly or self.writeParquetOnly:
            self.doPlotOverlaps = False
            self.doPlotCompareUnforced = False
            self.doPlotPsfFluxSnHists = False
            self.doPlotSkyObjectsSky = False
            self.doPlotSkyObjects = False
            self.doPlotFootprintArea = False
            self.doPlotRhoStatistics = False
            self.doPlotQuiver = False
            self.doPlotInputCounts = False
            self.doPlotMags = False
            self.doPlotStarGalaxy = False
            self.doPlotSizes = False
            self.doPlotCentroids = False


class CoaddAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        kwargs["cosmos"] = parsedCmd.cosmos
        kwargs["subdir"] = parsedCmd.subdir

        # Check for existence of appropriate dataset: parquet obj vs. afwTable
        # catalogs.
        datasetList = ["obj"] if parsedCmd.config.doReadParquetTables else ["forced_src", "meas"]
        # Partition all inputs by tract,filter
        FilterRefsDict = functools.partial(defaultdict, list)  # Dict for filter-->dataRefs
        # Make sure the actual input files requested exist (i.e. do not follow
        # the parent chain).  If reading afwTable catalogs, first check for
        # forced catalogs.  Break out of datasets loop if forced catalogs were
        # found, otherwise continue search for existence of unforced (i.e.
        # meas) catalogs.
        for dataset in datasetList:
            tractFilterRefs = defaultdict(FilterRefsDict)  # tract-->filter-->dataRefs
            for patchRef in sum(parsedCmd.id.refList, []):
                tract = patchRef.dataId["tract"]
                filterName = patchRef.dataId["filter"]
                inputDataFile = patchRef.get("deepCoadd_" + dataset + "_filename")[0]
                if parsedCmd.input not in parsedCmd.output:
                    inputDataFile = inputDataFile.replace(parsedCmd.output, parsedCmd.input)
                if os.path.exists(inputDataFile):
                    tractFilterRefs[tract][filterName].append(patchRef)
            if tractFilterRefs:
                break

        if not tractFilterRefs:
            raise RuntimeError("No suitable datasets found.")

        return [(tractFilterRefs[tract][filterName], kwargs) for tract in tractFilterRefs for
                filterName in tractFilterRefs[tract]]


class CoaddAnalysisTask(CmdLineTask):
    _DefaultName = "coaddAnalysis"
    ConfigClass = CoaddAnalysisConfig
    RunnerClass = CoaddAnalysisRunner
    AnalysisClass = Analysis
    outputDataset = "plotCoadd"

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_argument("--cosmos", default=None, help="Filename for Leauthaud Cosmos catalog")
        parser.add_id_argument("--id", "deepCoadd_meas",
                               help="data ID, e.g. --id tract=12345 patch=1,2 filter=HSC-X",
                               ContainerClass=TractDataIdContainer)
        parser.add_argument("--subdir", type=str, default="",
                            help=("Subdirectory below plots/filter/tract-NNNN/ (useful for, "
                                  "e.g., subgrouping of Patches.  Ignored if only one Patch is "
                                  "specified, in which case the subdir is set to patch-NNN"))
        return parser

    def __init__(self, *args, **kwargs):
        CmdLineTask.__init__(self, *args, **kwargs)
        self.zpLabel = None
        self.unitScale = 1000.0 if self.config.toMilli else 1.0
        self.matchRadius = self.config.matchRadiusXy if self.config.matchXy else self.config.matchRadiusRaDec
        self.matchRadiusUnitStr = " (pixels)" if self.config.matchXy else "\""

        self.verifyJob = verify.Job.load_metrics_package(subset="pipe_analysis")

    def runDataRef(self, patchRefList, subdir="", cosmos=None):
        plotList = []

        haveForced = False  # do forced datasets exits (may not for single band datasets)
        dataset = "Coadd_obj" if self.config.doReadParquetTables else "Coadd_forced_src"
        # Explicit input file was checked in CoaddAnalysisRunner, so a check
        # on datasetExists is sufficient here (modulo the case where a forced
        # dataset exists higher up the parent tree than the specified input,
        # but does not exist in the input directory as the former will be
        # found).
        if patchRefList[0].datasetExists(self.config.coaddName + dataset):
            haveForced = True
        forcedStr = "forced" if haveForced else "unforced"
        if self.config.doBackoutApCorr:
            self.log.info("Backing out aperture corrections from all fluxes")
            forcedStr += "\n (noApCorr)"
        if not haveForced:
            self.log.warn("No forced dataset exists for, e.g.,: {:} (only showing first dataId in "
                          "patchRefList).\nPlotting unforced results only.".format(patchRefList[0].dataId))
            dataset = "Coadd_meas"
            if not patchRefList[0].datasetExists(self.config.coaddName + dataset):
                raise TaskError("No data exists in patRefList: %s" %
                                ([patchRef.dataId for patchRef in patchRefList]))
        patchList = [patchRef.dataId["patch"] for patchRef in patchRefList]
        self.log.info("patchList size: {:d}".format(len(patchList)))
        repoInfo = getRepoInfo(patchRefList[0], coaddName=self.config.coaddName, coaddDataset=dataset)
        subdir = "patch-" + str(patchList[0]) if len(patchList) == 1 else subdir
        repoInfo.dataId["subdir"] = "/" + subdir
        patchRefExistsList = []
        for patchRef in patchRefList:
            if patchRef.datasetExists(self.config.coaddName + dataset):
                patchRefExistsList.append(patchRef)

        plotInfoDict = getPlotInfo(repoInfo)
        plotInfoDict.update(dict(plotType="plotCoadd", subdir=subdir, patchList=patchList,
                                 hscRun=repoInfo.hscRun, tractInfo=repoInfo.tractInfo,
                                 dataId=repoInfo.dataId, ccdList=None))
        # Find a visit/ccd input so that you can check for meas_mosaic input
        # (i.e. to set uberCalLabel).
        self.uberCalLabel = determineExternalCalLabel(repoInfo, patchList[0], coaddName=self.config.coaddName)
        self.log.info(f"External calibration(s) used: {self.uberCalLabel}")

        # Set some aliases for differing schema naming conventions
        aliasDictList = [self.config.flagsToAlias, ]
        if repoInfo.hscRun and self.config.srcSchemaMap is not None:
            aliasDictList += [self.config.srcSchemaMap]

        # Always highlight points with x-axis flag set (for cases where
        # they do not get explicitly filtered out).
        highlightList = [(self.config.analysis.fluxColumn.replace("_instFlux", "_flag"), 0, "turquoise")]
        for ih, flagName in enumerate(list(self.config.analysis.flags)):
            if not any(flagName in highlight for highlight in highlightList):
                highlightList += [(flagName, 0, FLAGCOLORS[ih%len(FLAGCOLORS)]), ]
        # Dict of all parameters common to plot* functions
        plotKwargs = dict(zpLabel=self.zpLabel, uberCalLabel=self.uberCalLabel)

        if any(doPlot for doPlot in
               [self.config.doPlotOverlaps, self.config.doPlotCompareUnforced,
                self.config.doPlotPsfFluxSnHists, self.config.doPlotSkyObjects,
                self.config.doPlotSkyObjectsSky, self.config.doPlotFootprintArea,
                self.config.doPlotMags, self.config.doPlotStarGalaxy,
                self.config.doPlotRhoStatistics, cosmos, self.config.externalCatalogs,
                self.config.doWriteParquetTables]) and not self.config.plotMatchesOnly:

            if self.config.doReadParquetTables:
                if haveForced:
                    forced, _ = self.readParquetTables(patchRefExistsList, self.config.coaddName + dataset,
                                                       repoInfo, dfDataset="forced_src")
                unforced, _ = self.readParquetTables(patchRefExistsList, self.config.coaddName + dataset,
                                                     repoInfo, dfDataset="meas")
                areaDict, _ = computeAreaDict(repoInfo, patchRefExistsList,
                                              dataset=self.config.coaddName + "Coadd", fakeCat=None)
            else:
                catalogStruct = self.readAfwCoaddTables(patchRefExistsList, repoInfo, haveForced,
                                                        aliasDictList=aliasDictList)
                unforced = catalogStruct.unforced
                forced = catalogStruct.forced
                areaDict = catalogStruct.areaDict

            plotKwargs.update(dict(zpLabel=self.zpLabel))
            unforcedSchema = getSchema(unforced)
            if haveForced:
                forcedSchema = getSchema(forced)

            # Make sub-catalog of sky objects before flag culling as many of
            # these will have flags set due to measurement difficulties in
            # regions that are really blank sky.
            if self.config.doPlotSkyObjectsSky:
                skyObjCatAll = unforced[unforced["merge_peak_sky"]].copy(deep=True)
            if self.config.doPlotSkyObjects:
                goodSky = (unforced["merge_peak_sky"] & (unforced["base_InputCount_value"] > 0)
                           & (unforced["deblend_nChild"] == 0) & ~unforced["base_PixelFlags_flag_edge"])
                skyObjCat = unforced[goodSky].copy(deep=True)

            # Must do the overlaps before purging the catalogs of non-primary
            # sources.  We only really need one set of these plots and the
            # matching takes a fair amount of time, so only plot for one
            # catalog, favoring the forced catalog if it exists.
            if self.config.doPlotOverlaps:
                # Determine if any patches in the patchList actually overlap
                overlappingPatches = checkPatchOverlap(plotInfoDict["patchList"], plotInfoDict["tractInfo"])
                if not overlappingPatches:
                    self.log.info("No overlapping patches...skipping overlap plots")
                else:
                    if haveForced:
                        forcedOverlaps = self.overlaps(forced, patchList, repoInfo.tractInfo)
                        if forcedOverlaps is not None:
                            plotList.append(self.plotOverlaps(forcedOverlaps, plotInfoDict, areaDict,
                                                              matchRadius=self.config.matchOverlapRadius,
                                                              matchRadiusUnitStr="\"",
                                                              forcedStr=forcedStr, postFix="_forced",
                                                              fluxToPlotList=["modelfit_CModel", ],
                                                              highlightList=highlightList, **plotKwargs))
                            self.log.info("Number of forced overlap objects matched = {:d}".
                                          format(len(forcedOverlaps)))
                        else:
                            self.log.info("No forced overlap objects matched. Overlap plots skipped.")
                    else:
                        unforcedOverlaps = self.overlaps(unforced, patchList, repoInfo.tractInfo)
                        if unforcedOverlaps is not None:
                            plotList.append(
                                self.plotOverlaps(unforcedOverlaps, plotInfoDict, areaDict,
                                                  matchRadius=self.config.matchOverlapRadius,
                                                  matchRadiusUnitStr="\"",
                                                  forcedStr=forcedStr.replace("forced", "unforced"),
                                                  postFix="_unforced", fluxToPlotList=["modelfit_CModel", ],
                                                  highlightList=highlightList, **plotKwargs))
                            self.log.info("Number of unforced overlap objects matched = {:d}".
                                          format(len(unforcedOverlaps)))
                        else:
                            self.log.info("No unforced overlap objects matched. Overlap plots skipped.")

            # Set boolean array indicating sources deemed unsuitable for qa
            # analyses.
            badUnforced = makeBadArray(unforced, onlyReadStars=self.config.onlyReadStars)
            if haveForced:
                badForced = makeBadArray(forced, onlyReadStars=self.config.onlyReadStars)
                badCombined = (badUnforced | badForced)
                unforcedMatched = unforced[~badCombined].copy(deep=True)
                forcedMatched = forced[~badCombined].copy(deep=True)

                if self.config.doPlotCompareUnforced:
                    plotList.append(self.plotCompareUnforced(forcedMatched, unforcedMatched, plotInfoDict,
                                                             areaDict, highlightList=highlightList,
                                                             **plotKwargs))

            # Create and write parquet tables
            if self.config.doWriteParquetTables:
                if haveForced:
                    # Add pre-computed columns for parquet tables
                    forced = addPreComputedColumns(forced, fluxToPlotList=self.config.fluxToPlotList,
                                                   toMilli=self.config.toMilli, unforcedCat=unforced)
                    dataRef_forced = repoInfo.butler.dataRef("analysisCoaddTable_forced",
                                                             dataId=repoInfo.dataId)
                    writeParquet(dataRef_forced, forced, badArray=badForced)
                dataRef_unforced = repoInfo.butler.dataRef("analysisCoaddTable_unforced",
                                                           dataId=repoInfo.dataId)
                # Add pre-computed columns for parquet tables
                unforced = addPreComputedColumns(unforced, fluxToPlotList=self.config.fluxToPlotList,
                                                 toMilli=self.config.toMilli)
                writeParquet(dataRef_unforced, unforced, badArray=badUnforced)
                if self.config.writeParquetOnly and not self.config.doPlotMatches:
                    self.log.info("Exiting after writing Parquet tables.  No plots generated.")
                    return

            if not self.config.writeParquetOnly:
                # Purge the catalogs of flagged sources
                unforced = unforced[~badUnforced].copy(deep=True)
                if haveForced:
                    forced = forced[~badForced].copy(deep=True)
                else:
                    forced = unforced
                self.catLabel = " nChild = 0"
                strIndex = forcedStr.find("\n")
                if strIndex < 0:
                    forcedStr = forcedStr + self.catLabel
                else:
                    forcedStr = forcedStr[:strIndex] + " " + self.catLabel + forcedStr[strIndex:]
                if haveForced:
                    self.log.info("\nNumber of sources in catalogs: unforced = {0:d} and forced = {1:d}".
                                  format(len(unforced), len(forced)))
                else:
                    self.log.info("\nNumber of sources in catalog: unforced = {0:d}".format(len(unforced)))

            if self.config.doPlotPsfFluxSnHists:
                plotList.append(self.plotPsfFluxSnHists(unforced, "base_PsfFlux_cal", plotInfoDict, areaDict,
                                                        forcedStr=forcedStr.replace("forced", "unforced"),
                                                        **plotKwargs))
            if self.config.doPlotSkyObjects:
                plotList.append(self.plotSkyObjects(skyObjCat, "skyObjects", plotInfoDict, areaDict,
                                                    forcedStr=forcedStr.replace("forced", "unforced")))
            if self.config.doPlotSkyObjectsSky:
                plotList.append(self.plotSkyObjectsSky(skyObjCatAll, "skyObjects", plotInfoDict,
                                                       forcedStr=forcedStr.replace("forced", "unforced"),
                                                       alpha=0.7, doPlotTractImage=True,
                                                       doPlotPatchOutline=True, sizeFactor=3.0,
                                                       maxDiamPix=1000))

            if self.config.doPlotFootprintArea:
                if "base_FootprintArea_value" in unforcedSchema:
                    plotList.append(self.plotFootprintHist(unforced, "footArea", plotInfoDict,
                                                           forcedStr=forcedStr.replace("forced", "unforced"),
                                                           **plotKwargs))
                    plotList.append(self.plotFootprint(unforced, plotInfoDict, areaDict,
                                                       forcedStr=forcedStr.replace("forced", "unforced"),
                                                       highlightList=highlightList, **plotKwargs))
                else:
                    self.log.info("config.doPlotFootprintArea is True, but do not have "
                                  "base_FootprintArea_value in schema...skipping footArea plots.")

            if self.config.doPlotRhoStatistics:
                plotList.append(self.plotRhoStatistics(unforced, plotInfoDict,
                                                       forcedStr=forcedStr.replace("forced", "unforced"),
                                                       **plotKwargs))

            if self.config.doPlotQuiver:
                plotList.append(self.plotQuiver(unforced, "ellipResids", plotInfoDict, areaDict,
                                                forcedStr=forcedStr.replace("forced", "unforced"),
                                                scale=2, **plotKwargs))

            if self.config.doPlotInputCounts:
                plotList.append(self.plotInputCounts(unforced, "inputCounts", plotInfoDict,
                                                     forcedStr=forcedStr.replace("forced", "unforced"),
                                                     alpha=0.5, doPlotPatchOutline=True, sizeFactor=5.0,
                                                     maxDiamPix=1000, **plotKwargs))

            plotKwargs.update(dict(highlightList=highlightList))
            if self.config.doPlotMags:
                plotList.append(self.plotMags(unforced, plotInfoDict, areaDict,
                                              forcedStr=forcedStr.replace("forced", "unforced"),
                                              postFix="_unforced", **plotKwargs))
                if haveForced:
                    plotKwargs.update(dict(highlightList=highlightList
                                           + [("merge_measurement_" + repoInfo.genericFilterName, 0,
                                               "yellow")]))

                    plotList.append(self.plotMags(forced, plotInfoDict, areaDict, forcedStr=forcedStr,
                                                  postFix="_forced", **plotKwargs))
                    plotKwargs.update(dict(highlightList=highlightList))
            if self.config.doPlotStarGalaxy:
                if "ext_shapeHSM_HsmSourceMoments_xx" in unforcedSchema:
                    plotList.append(self.plotStarGal(unforced, plotInfoDict, areaDict,
                                                     forcedStr=forcedStr.replace("forced", "unforced"),
                                                     **plotKwargs))
                else:
                    self.log.warn("Cannot run plotStarGal: ext_shapeHSM_HsmSourceMoments_xx not "
                                  "in forcedSchema")

            if self.config.doPlotSizes:
                if all(ss in unforcedSchema for ss in ["base_SdssShape_psf_xx", "calib_psf_used"]):
                    plotList.append(self.plotSizes(unforced, plotInfoDict, areaDict,
                                                   forcedStr=forcedStr.replace("forced", "unforced"),
                                                   postFix="_unforced", **plotKwargs))
                else:
                    self.log.warn("Cannot run plotSizes: base_SdssShape_psf_xx and/or calib_psf_used "
                                  "not in unforcedSchema")
                if haveForced:
                    if all(ss in forcedSchema for ss in ["base_SdssShape_psf_xx", "calib_psf_used"]):
                        plotList.append(self.plotSizes(forced, plotInfoDict, areaDict,
                                                       forcedStr=forcedStr, **plotKwargs))
                    else:
                        self.log.warn("Cannot run plotSizes: base_SdssShape_psf_xx and/or calib_psf_used "
                                      "not in forcedSchema")
            if cosmos:
                plotList.append(self.plotCosmos(forced, plotInfoDict, areaDict, cosmos, repoInfo.dataId))

        if self.config.doPlotMatches or self.config.doWriteParquetTables:
            matchAreaDict = {}
            # First write out unforced match parquet tables
            matches, matchAreaDict = self.readSrcMatches(repoInfo, patchRefExistsList,
                                                         self.config.coaddName + "Coadd_meas",
                                                         aliasDictList=aliasDictList, haveForced=False)
            qaTableSuffix = "_unforced"
            if self.config.doWriteParquetTables:
                matchesDataRef = repoInfo.butler.dataRef("analysisMatchFullRefCoaddTable" + qaTableSuffix,
                                                         dataId=repoInfo.dataId)
                writeParquet(matchesDataRef, matches, badArray=None, prefix="src_")
            # Now write out forced match parquet tables, if present
            if haveForced:
                matches, matchAreaDict = self.readSrcMatches(repoInfo, patchRefExistsList,
                                                             self.config.coaddName + "Coadd_forced_src",
                                                             aliasDictList=aliasDictList,
                                                             haveForced=haveForced)
                qaTableSuffix = "_forced"
                if self.config.doWriteParquetTables:
                    matchesDataRef = repoInfo.butler.dataRef("analysisMatchFullRefCoaddTable" + qaTableSuffix,
                                                             dataId=repoInfo.dataId)
                    writeParquet(matchesDataRef, matches, badArray=None, prefix="src_")

            if self.config.writeParquetOnly:
                self.log.info("Exiting after writing Parquet tables.  No plots generated.")
                return

            if self.config.doPlotMatches:
                # The apCorr backing out, if requested, and the purging of
                # deblend_nChild > 0 objects happens in readSrcMatches, but
                # label won't be set if plotMatchesOnly is True.
                if self.config.doBackoutApCorr and "noApCorr" not in forcedStr:
                    forcedStr += "(noApCorr)"
                forcedStr = forcedStr + " nChild = 0" if "nChild = 0" not in forcedStr else forcedStr

                plotKwargs.update(dict(zpLabel=self.zpLabel))
                matchHighlightList = [("src_" + self.config.analysis.fluxColumn.replace("_instFlux", "_flag"),
                                       0, "turquoise"), ]
                plotKwargs.update(dict(highlightList=matchHighlightList))
                plotList.append(self.plotMatches(matches, plotInfoDict, matchAreaDict, forcedStr=forcedStr,
                                                 **plotKwargs))

                for cat in self.config.externalCatalogs:
                    with andCatalog(cat):
                        matches = self.matchCatalog(forced, repoInfo.filterName,
                                                    self.config.externalCatalogs[cat])
                    if matches is not None:
                        plotList.append(self.plotMatches(matches, plotInfoDict, matchAreaDict,
                                                         forcedStr=forcedStr, matchRadius=self.matchRadius,
                                                         matchRadiusUnitStr=self.matchRadiusUnitStr,
                                                         **plotKwargs))
                    else:
                        self.log.warn("Could not create match catalog for {:}.  Is "
                                      "lsst.meas.extensions.astrometryNet setup?".format(cat))

        self.allStats, self.allStatsHigh = savePlots(plotList, "plotCoadd", repoInfo.dataId,
                                                     repoInfo.butler, subdir=subdir)
        metaDict = {kk: plotInfoDict[kk] for kk in ("filter", "tract", "rerun")
                    if plotInfoDict[kk] is not None}
        if plotInfoDict["cameraName"]:
            metaDict["camera"] = plotInfoDict["cameraName"]
        self.verifyJob = updateVerifyJob(self.verifyJob, metaDict=metaDict, specsList=None)
        # TODO: DM-26758 (or DM-14768) should make the following lines a proper
        # butler.put by directly persisting json files.
        verifyJobFilename = repoInfo.butler.get("coaddAnalysis_verify_job_filename",
                                                dataId=repoInfo.dataId)[0]
        self.verifyJob.write(verifyJobFilename)

    def readParquetTables(self, dataRefList, dataset, repoInfo, dfDataset=None,
                          doApplyExternalPhotoCalib=False, doApplyExternalSkyWcs=False, useMeasMosaic=False,
                          iCat=None):
        """Read in, calibrate, and concatenate parquet tables from a list of
        dataRefs.

        The calibration performed is based on config parameters.  For coadds,
        the only option is the calibration zeropoint.  For visits, the options
        include external calibrations for both photometry (e.g. fgcm) and wcs
        (e.g. jointcal) or simply the zero point from single frame processing.

        Parameters
        ----------
        dataRefList : `list` of
                      `lsst.daf.persistence.butlerSubset.ButlerDataRef`
            A list of butler data references whose catalogs of ``dataset``
            are to be read in.
        dataset : `str`
            Name of the catalog ``dataset`` to be read in, e.g.
            "deepCoadd_obj" (for coadds) or "source" (for visits).
        repoInfo : `lsst.pipe.base.struct.Struct`
            A struct containing elements with repo information needed to
            determine if the catalog data is coadd or visit level and, if the
            latter, to create appropriate dataIds to look for the external
            calibration datasets.
        dfDataset : `str` or `None`, optional
            Name of the dataFrame \"dataset\" to be read in for multilevel
            parquet tables.  For coadd catalogs, which are of type
            `lsst.pipe.tasks.parquetTable.MultilevelParquetTable`, this is
            actually not optional but must be one of, "forced_src", "meas", or
            "ref".  This parameter is not relevant for visit-level catalogs,
            which are of type `lsst.pipe.tasks.parquetTable.ParquetTable`.
        doApplyExternalPhotoCalib : `bool`, optional
            If `True`: Apply the external photometric calibrations specified by
                      ``repoInfo.photoCalibDataset`` to the catalog.
            If `False`: Apply the ``fluxMag0`` photometric calibration from
                        Single Frame Measuerment to the catalog.
        doApplyExternalSkyWcs : `bool`, optional
            If `True`: Apply the external astrometric calibrations specified by
                       ``repoInfo.skyWcsDataset`` the catalog.
            If `False`: Retain the WCS from Single Frame Measurement.
        useMeasMosaic : `bool`, optional
            Use meas_mosaic's applyMosaicResultsCatalog for the external
            calibration (even if photoCalib object exists).  For testing
            implementations.
        iCat : `int` or `None,` optional
            Integer representing whether this is comparison catalog 0 or 1.

        Raises
        ------
        TaskError
            If no data is read in for the ``dataRefList``.
        RuntimeError
            If catalog is of type
            `lsst.pipe.tasks.parquetTable.MultilevelParquetTable` but no
            ``dfDataset`` is provided.

        Returns
        -------
        allCats : `pandas.core.frame.DataFrame`
            The concatenated catalogs as a pandas DataFrame.
        """
        # It is much faster to concatenate a list of DataFrames than to
        # concatenate successively within the for loop.
        commonZpCatList = []
        catList = []
        colsToLoadList = None
        dfLoadColumns = None
        refColsToLoadList = None
        measColsToLoadList = None
        for dataRef in dataRefList:
            if not dataRef.datasetExists(dataset):
                self.log.info("Dataset does not exist: {0:r}, {1:s}".format(dataRef.dataId, dataset))
                continue
            parquetCat = dataRef.get(dataset, immediate=True)
            if isinstance(parquetCat, MultilevelParquetTable) and not any(
                    dfDataset == dfName for dfName in ["forced_src", "meas", "ref"]):
                raise RuntimeError("Must specify a dfDataset for multilevel parquet tables")
            if dfLoadColumns is None and isinstance(parquetCat, MultilevelParquetTable):
                dfLoadColumns = {"dataset": dfDataset, "filter": dataRef.dataId["filter"]}
            # On the first dataRef read in, create list of columns to load
            # based on config lists and their existence in the catalog table.
            if colsToLoadList is None:
                catColumns = getParquetColumnsList(parquetCat, dfDataset=dfDataset,
                                                   filterName=dataRef.dataId["filter"])
                colsToLoadList = [col for col in catColumns if
                                  (col.startswith(tuple(self.config.baseColStrList))
                                   and not any(s in col for s in self.config.notInColStrList))]
                if dfLoadColumns is None:
                    dfLoadColumns = colsToLoadList
                else:
                    dfLoadColumns.update(column=colsToLoadList)
            cat = parquetCat.toDataFrame(columns=dfLoadColumns)
            cat = addElementIdColumn(cat, dataRef.dataId, repoInfo=repoInfo)
            if dfDataset == "forced_src":  # insert some columns from the ref and meas cats for forced cats
                if refColsToLoadList is None:
                    refColumns = getParquetColumnsList(parquetCat, dfDataset="ref",
                                                       filterName=dataRef.dataId["filter"])
                    refColsToLoadList = [col for col in refColumns if
                                         (col.startswith(tuple(self.config.columnsToCopyFromRef))
                                          and not any(s in col for s in self.config.notInColStrList))]
                ref = parquetCat.toDataFrame(columns={"dataset": "ref", "filter": dataRef.dataId["filter"],
                                                      "column": refColsToLoadList})
                cat = pd.concat([cat, ref], axis=1)
                if measColsToLoadList is None:
                    measColumns = getParquetColumnsList(parquetCat, dfDataset="meas",
                                                        filterName=dataRef.dataId["filter"])
                    measColsToLoadList = [col for col in measColumns if
                                          (col.startswith(tuple(self.config.columnsToCopyFromMeas))
                                           and not any(s in col for s in self.config.notInColStrList))]
                meas = parquetCat.toDataFrame(columns={"dataset": "meas", "filter": dataRef.dataId["filter"],
                                                       "column": measColsToLoadList})
                cat = pd.concat([cat, meas], axis=1)

            if "patch" in repoInfo.dataId:  # This is a coadd catalog
                cat = self.calibrateCatalogs(cat, wcs=repoInfo.wcs)
                catList.append(cat)
            else:  # This is a visit catalog
                # Scale fluxes to common zeropoint to make basic comparison
                # plots without calibrated ZP influence.
                commonZpCat = cat.copy(True)
                commonZpCat = calibrateSourceCatalog(commonZpCat, self.config.analysis.commonZp)
                if doApplyExternalPhotoCalib:
                    if not dataRef.datasetExists(repoInfo.photoCalibDataset):
                        continue
                if doApplyExternalSkyWcs:
                    if not dataRef.datasetExists(repoInfo.skyWcsDataset):
                        continue
                fluxMag0 = None
                if not doApplyExternalPhotoCalib:
                    photoCalib = repoInfo.butler.get("calexp_photoCalib", dataRef.dataId)
                    fluxMag0 = photoCalib.getInstFluxAtZeroMagnitude()
                cat = self.calibrateCatalogs(dataRef, cat, fluxMag0, repoInfo, doApplyExternalPhotoCalib,
                                             doApplyExternalSkyWcs, useMeasMosaic, iCat=iCat)
                catList.append(cat)
                commonZpCatList.append(commonZpCat)
        if not catList:
            raise TaskError("No catalogs read: %s" % ([dataRef.dataId for dataRef in dataRefList]))
        allCats = pd.concat(catList, axis=0)
        # The object "id" is associated with the dataframe index.  Add a
        # column that is the id so that it is available for operations on it,
        # e.g. cat["id"].
        allCats["id"] = allCats.index
        # Optionally backout aperture corrections
        if self.config.doBackoutApCorr:
            allCats = backoutApCorr(allCats)
        if commonZpCatList:
            allCommonZpCats = pd.concat(commonZpCatList, axis=0)
            allCommonZpCats["id"] = allCommonZpCats.index
            if self.config.doBackoutApCorr:
                allCommonZpCats = backoutApCorr(allCommonZpCats)
        else:
            allCommonZpCats = None
        return allCats, allCommonZpCats

    def readAfwCoaddTables(self, dataRefList, repoInfo, haveForced, aliasDictList=None):
        """Read in, concatenate, calibrate, and convert to DataFrame a list of
        coadd catalogs that were persisted as afwTables.

        This function delegates to readCatalogs for the actual catalog reading
        and concatenating, in which an extra column indicating the patch ID is
        added to each catalog before appending them all to a single list.  This
        is useful for any subsequent QA analysis using the persisted parquet
        files.  Here these catalogs are calibrated, have useful columns copied
        from the *Coadd_ref to *Coadd_forced_src catalogs, and are converted
        to pandas DataFrames.

        Parameters
        ----------
        dataRefList : `list` of
                      `lsst.daf.persistence.butlerSubset.ButlerDataRef`
            A `list` of butler data references whose coadd catalogs are to be
            read in.
        repoInfo : `lsst.pipe.base.Struct`
            A struct containing relevant information about the repository under
            study.  Elements used here include the dataset names for any
            external calibrations to be applied.
        haveForced : `bool`
            A boolean indicating if a forced_src catalog exists in the
            repository associated with ``repoInfo``.
        aliasDictList : `dict` or `None`, optional
            A `dict` of alias columns to add for backwards compatibility with
            old repositories.

        Raises
        ------
        RuntimeError
            If lengths of *Coadd_forced_src and *Coadd_ref catalogs are not
            equal.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            A struct with attributes:
            ``unforced``
                The concatenated unforced, or "_meas", calibrated catalog
                (`pandas.core.frame.DataFrame`).
            ``forced``
                The concatenated forced, or "_forced_src", calibrated catalog
                (`pandas.core.frame.DataFrame`).
            ``areaDict``
                Contains patch keys that index the patch corners in RA/Dec and
                the effective patch area (i.e. neither the "BAD" nor "NO_DATA"
                mask bit is set) (`dict`).
        """
        if haveForced:
            forcedCatStruct = self.readCatalogs(dataRefList, self.config.coaddName + "Coadd_forced_src",
                                                repoInfo, aliasDictList=aliasDictList,
                                                readFootprintsAs=self.config.readFootprintsAs)
            forced = forcedCatStruct.catalog
            areaDict = forcedCatStruct.areaDict
            forced = self.calibrateCatalogs(forced, wcs=repoInfo.wcs)
            forcedSchema = getSchema(forced)
        else:
            forced = None
        unforcedCatStruct = self.readCatalogs(dataRefList, self.config.coaddName + "Coadd_meas", repoInfo,
                                              aliasDictList=aliasDictList,
                                              readFootprintsAs=self.config.readFootprintsAs)
        unforced = unforcedCatStruct.catalog
        unforced = self.calibrateCatalogs(unforced, wcs=repoInfo.wcs)
        unforcedSchema = getSchema(unforced)
        if not haveForced:
            areaDict = unforcedCatStruct.areaDict
        if haveForced:
            # Copy over some fields from _ref and _meas catalogs to
            # _forced_src catalog.
            refCat = self.readCatalogs(dataRefList, self.config.coaddName + "Coadd_ref", repoInfo).catalog
            if len(forced) != len(refCat):
                raise RuntimeError(("Lengths of forced (N = {0:d}) and ref (N = {0:d}) cats "
                                    "don't match").format(len(forced), len(refCat)))
            refCatSchema = getSchema(refCat)
            refColList = []
            for strPrefix in self.config.columnsToCopyFromRef:
                refColList.extend(refCatSchema.extract(strPrefix + "*"))
            refColsToCopy = [col for col in refColList if col not in forcedSchema
                             and not any(s in col for s in self.config.notInColStrList)
                             and col in refCatSchema
                             and not (repoInfo.hscRun and col == "slot_Centroid_flag")]
            forced = addColumnsToSchema(refCat, forced, refColsToCopy)
            measColList = []
            for strPrefix in self.config.columnsToCopyFromMeas:
                measColList.extend(refCatSchema.extract(strPrefix + "*"))
            measColsToCopy = [col for col in measColList if col not in forcedSchema
                              and not any(s in col for s in self.config.notInColStrList)
                              and col in unforcedSchema
                              and not (repoInfo.hscRun and col == "slot_Centroid_flag")]
            forced = addColumnsToSchema(unforced, forced, measColsToCopy)

        # Convert to pandas DataFrames
        unforced = unforced.asAstropy().to_pandas().set_index("id", drop=False)
        if haveForced:
            forced = forced.asAstropy().to_pandas().set_index("id", drop=False)
        return Struct(unforced=unforced, forced=forced, areaDict=areaDict)

    def readCatalogs(self, dataRefList, dataset, repoInfo, aliasDictList=None, fakeCat=None,
                     raFakesCol="raJ2000", decFakesCol="decJ2000", readFootprintsAs=None,
                     doApplyExternalPhotoCalib=False, doApplyExternalSkyWcs=False, useMeasMosaic=False,
                     iCat=None):
        """Read in and concatenate catalogs of type dataset in lists of
        data references.

        An extra column indicating the patch ID is added to each catalog before
        appending them all to a single list.  This is useful for any subsequent
        QA analysis using the persisted parquet files.

        Parameters
        ----------
        dataRefList : `list` of
                       `lsst.daf.persistence.butlerSubset.ButlerDataRef`
            A list of butler data references whose catalogs of dataset type are
            to be read in.
        dataset : `str`
            Name of the catalog dataset to be read in.
        repoInfo : `lsst.pipe.base.Struct`
            A struct containing relevant information about the repository under
            study.  Elements used here include the dataset names for any
            external calibrations to be applied.
        aliasDictList : `dict` or `None`, optional
            A `dict` of alias columns to add for backwards compatibility with
            old repositories.
        fakeCat : `pandas.core.frame.DataFrame` or `None`, optional
            Catalog of fake sources, used if hasFakes is `True` in which case a
            column (onPatch) is added with the patch number if the fake source
            overlaps a ccd and `np.nan` if it does not.
        raFakesCol : `str`, optional
            The name of the RA column to use from the fakes catalogue.
        decFakesCol : `str`, optional
            The name of the Dec column to use from the fakes catalogue.
        readFootprintsAs : `None` or `str`, optional
            A string dictating if and what type of Footprint to read in along
            with the catalog:
            `None` : do not read in Footprints.
            "light": read in regular Footprints (include SpanSet and list of
                     peaks per Footprint).
            "heavy": read in HeavyFootprints (include regular Footprint plus
                     flux values per Footprint).
        doApplyExternalPhotoCalib : `bool`, optional
            If `True`: Apply the external photometric calibrations specified by
                      ``repoInfo.photoCalibDataset`` to the catalog.
            If `False`: Apply the ``fluxMag0`` photometric calibration from
                        Single Frame Measuerment to the catalog.
        doApplyExternalSkyWcs : `bool`, optional
            If `True`: Apply the external astrometric calibrations specified by
                       ``repoInfo.skyWcsDataset`` the calalog.
            If `False`: Retain the WCS from Single Frame Measurement.
        useMeasMosaic : `bool`, optional
            Use meas_mosaic's applyMosaicResultsCatalog for the external
            calibration (even if photoCalib object exists).  For testing
            implementations.
        iCat : `int` or `None,` optional
            Integer representing whether this is comparison catalog 0 or 1.

        Raises
        ------
        TaskError
            If no data is read in for the dataRefList.
        RuntimeError
            If entry for ``readFootprintsAs`` is not recognized (i.e. not one
            of `None`, \"light\", or \"heavy\").

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            A struct with attributes:
            ``commonZpCatalog``
                The concatenated common zeropoint calibrated catalog
                (`lsst.afw.table.SourceCatalog` or `None` for coadd data).
            ``catalog``
                The concatenated SFM or external calibration calibrated catalog
                (`lsst.afw.table.SourceCatalog`).
            ``areaDict``
                Contains patch keys that index the patch corners in RA/Dec and
                the effective patch area (i.e. neither the "BAD" nor "NO_DATA"
                mask bit is set) (`dict`).
            ``fakeCat``
                The updated catalog of fake sources (None if the config
                parameter hasFakes = `False` (`pandas.core.frame.DataFrame`).
        """
        commonZpCatList = []
        catList = []
        dataRefExistsList = []
        for dataRef in dataRefList:
            if dataRef.datasetExists(dataset):
                dataRefExistsList.append(dataRef)
        calexpPrefix = dataset[:dataset.find("_")] if "_" in dataset else ""
        areaDict, fakeCat = computeAreaDict(repoInfo, dataRefExistsList, dataset=calexpPrefix,
                                            fakeCat=fakeCat, raFakesCol=raFakesCol, decFakesCol=decFakesCol)
        for dataRef in dataRefExistsList:
            if not dataRef.datasetExists(dataset):
                continue
            if not readFootprintsAs:
                catFlags = afwTable.SOURCE_IO_NO_FOOTPRINTS
            elif readFootprintsAs == "light":
                catFlags = afwTable.SOURCE_IO_NO_HEAVY_FOOTPRINTS
            elif readFootprintsAs == "heavy":
                catFlags = 0
            else:
                raise RuntimeError("Unknown entry for readFootprintsAs: {:}.  Only recognize one of: "
                                   "None, \"light\", or \"heavy\"".format(readFootprintsAs))
            cat = dataRef.get(dataset, immediate=True, flags=catFlags)
            # Optionally backout aperture corrections
            if self.config.doBackoutApCorr:
                cat = backoutApCorr(cat)
            schema = getSchema(cat)
            # Old catalogs did not have base_FootprintArea_value so, for
            # backwards compatibility, check if present and add if not.
            if (self.config.doPlotFootprintArea and "base_FootprintArea_value" not in schema
                    and len(schema.extract("merge_footprint*")) > 0):  # to not bother for forced cats
                if self.config.readFootprintsAs != "heavy":
                    self.log.warn("config.doPlotFootprintArea is True, but do not have "
                                  "base_FootprintArea_value in schema.  If reading in an older afw "
                                  "src catalog, may need to run with config.readFootprintsAs=\"heavy\""
                                  "to be able to read in the footprints and compute their area.")
                else:
                    cat = addFootprintArea(cat)
            # Set some "aliases" for differing schema naming conventions.
            # Note: we lose the alias maps when converting to pandas, so now
            # must actually make a copy of the "old" column to a new one with
            # the "new" name. This is really just a backwards-compatibility
            # accommodation for catalogs that are already pretty old, so it
            # will be a no-op in most cases and will likely disappear in the
            #  not-too-distant future.
            if aliasDictList:
                cat = addAliasColumns(cat, aliasDictList)
            # Add elementId column, where element is "patch" for coadd data and
            # "ccd/detector" for visit data (useful to have in Parquet tables
            # for subsequent interactive analysis).
            if "patch" in repoInfo.dataId:  # This is a coadd catalog
                cat = addIntFloatOrStrColumn(cat, dataRef.dataId["patch"], "patchId",
                                             "Patch on which source was detected")
            else:  # This is a visit catalog
                cat = addIntFloatOrStrColumn(cat, dataRef.dataId[repoInfo.ccdKey], "ccdId",
                                             "Id of CCD on which source was detected")
                # Compute Focal Plane coordinates for each source if not
                # already there.
                if (self.config.doPlotCentroids or self.config.analysis.doPlotFP
                        or self.config.analysisMatches.doPlotFP):
                    if "base_FPPosition_x" not in schema and "focalplane_x" not in schema:
                        det = repoInfo.butler.get("calexp_detector", dataRef.dataId)
                        cat = addFpPoint(det, cat)

                # Scale fluxes to common zeropoint to make basic comparison
                # plots without calibrated ZP influence.
                commonZpCat = cat.copy(True)
                commonZpCat = calibrateSourceCatalog(commonZpCat, self.config.analysis.commonZp)
                commonZpCatList.append(commonZpCat)
                if self.config.doApplyExternalPhotoCalib:
                    if repoInfo.hscRun:
                        if not dataRef.datasetExists("fcr_hsc_md") or not dataRef.datasetExists("wcs_hsc"):
                            continue
                    else:
                        # Check for both jointcal_wcs and wcs for compatibility
                        # with old datasets.
                        if not (dataRef.datasetExists(repoInfo.photoCalibDataset)
                                or dataRef.datasetExists("fcr_md")):
                            continue
                if self.config.doApplyExternalSkyWcs:
                    if repoInfo.hscRun:
                        if not dataRef.datasetExists("fcr_hsc_md") or not dataRef.datasetExists("wcs_hsc"):
                            continue
                    else:
                        # Check for both jointcal_wcs and wcs for compatibility
                        # with old datasets.
                        if not (dataRef.datasetExists(repoInfo.skyWcsDataset)
                                or dataRef.datasetExists("wcs")):
                            continue
                fluxMag0 = None
                if not doApplyExternalPhotoCalib:
                    photoCalib = repoInfo.butler.get("calexp_photoCalib", dataRef.dataId)
                    fluxMag0 = photoCalib.getInstFluxAtZeroMagnitude()
                cat = self.calibrateCatalogs(dataRef, cat, fluxMag0, repoInfo, doApplyExternalPhotoCalib,
                                             doApplyExternalSkyWcs, useMeasMosaic, iCat=iCat)
            catList.append(cat)
        if not catList:
            raise TaskError("No catalogs read: %s" % ([dataRef.dataId for dataRef in dataRefList]))

        return Struct(commonZpCatalog=concatenateCatalogs(commonZpCatList),
                      catalog=concatenateCatalogs(catList), areaDict=areaDict, fakeCat=fakeCat)

    def readSrcMatches(self, repoInfo, dataRefList, dataset, aliasDictList=None, haveForced=False,
                       doApplyExternalPhotoCalib=False, doApplyExternalSkyWcs=False, useMeasMosaic=False):
        matchList = []
        matchAreaDict = {}
        dataIdSubList = []
        for dataRef in dataRefList:
            if not dataRef.datasetExists(dataset):
                self.log.info("Dataset does not exist: {0:r}, {1:s}".format(dataRef.dataId, dataset))
                continue
            butler = repoInfo.butler
            # Generate unnormalized match list (from normalized persisted
            # one)  with loadDenormalizeAndUnpackMatches (which requires a
            # refObjLoader to be initialized).
            if self.config.doReadParquetTables:
                if "Coadd" in dataset:
                    datasetType = dataset[:dataset.find("Coadd_") + len("Coadd_")] + "obj"
                    dfDataset = dataset[dataset.find("Coadd_") + len("Coadd_"):]
                    baseDataset = dataset[:dataset.find("Coadd_") + len("Coadd_") - 1]
                else:
                    datasetType = "source"
                    dfDataset = ""
                    baseDataset = ""
                catalog, _ = self.readParquetTables([dataRef, ], datasetType, repoInfo, dfDataset=dfDataset,
                                                    doApplyExternalPhotoCalib=doApplyExternalPhotoCalib,
                                                    doApplyExternalSkyWcs=doApplyExternalSkyWcs,
                                                    useMeasMosaic=useMeasMosaic)
                areaDict, _ = computeAreaDict(repoInfo, [dataRef, ], dataset=baseDataset)
            else:
                if "patch" in repoInfo.dataId:  # This is a coadd catalog
                    catalogStruct = self.readAfwCoaddTables([dataRef, ], repoInfo, haveForced,
                                                            aliasDictList=aliasDictList)
                    if "Coadd_meas" in dataset:
                        catalog = catalogStruct.unforced
                    if "Coadd_forced_src" in dataset:
                        catalog = catalogStruct.forced
                    areaDict = catalogStruct.areaDict
                else:  # This is a visit catalog
                    catStruct = self.readCatalogs(
                        [dataRef, ], dataset, repoInfo, aliasDictList=aliasDictList,
                        readFootprintsAs=self.config.readFootprintsAs,
                        doApplyExternalPhotoCalib=doApplyExternalPhotoCalib,
                        doApplyExternalSkyWcs=doApplyExternalSkyWcs, useMeasMosaic=useMeasMosaic)
                    catalog = catStruct.catalog
                    catalog = catalog.asAstropy().to_pandas().set_index("id", drop=False)
                    areaDict = catStruct.areaDict
            # Set boolean array indicating sources deemed unsuitable for qa
            # analyses.
            schema = getSchema(catalog)
            transCentFlag = "base_TransformedCentroid_flag"
            badFlagList = [transCentFlag, ] if transCentFlag in schema else []
            bad = makeBadArray(catalog, flagList=badFlagList, onlyReadStars=self.config.onlyReadStars)
            if dataset.startswith("deepCoadd_"):
                packedMatches = butler.get("deepCoadd_measMatch", dataRef.dataId)
            else:
                packedMatches = butler.get(dataset + "Match", dataRef.dataId)

            # Purge the match list of sources flagged in the catalog
            badIds = catalog["id"][bad].array
            badMatch = np.zeros(len(packedMatches), dtype=bool)
            for iMat, iMatch in enumerate(packedMatches):
                if iMatch["second"] in badIds:
                    badMatch[iMat] = True

            packedMatches = packedMatches[~badMatch].copy(deep=True)
            if not packedMatches:
                self.log.warn("No good matches for %s" % (dataRef.dataId,))
                continue
            refObjLoader = self.config.refObjLoader.apply(butler=butler)
            matches = loadDenormalizeAndUnpackMatches(catalog, packedMatches, refObjLoader)
            # LSST reads in reference catalogs with flux in "nanojanskys", so
            # must convert to AB.
            matches = matchNanojanskyToAB(matches)
            matchAreaDict.update(areaDict)
            if matches.empty:
                self.log.warn("No matches for %s" % (dataRef.dataId,))
            else:
                if "patch" not in repoInfo.dataId:  # This is a visit catalog
                    if self.config.doApplyExternalSkyWcs:
                        # Update "distance" between reference and source
                        # matches based on external-calibration positions.
                        angularDist = AngularDistance("ref_coord_ra", "src_coord_ra",
                                                      "ref_coord_dec", "src_coord_dec")
                        matches["distance"] = angularDist(matches)

                    # Avoid multi-counting when visit overlaps multiple tracts
                    noTractId = dataRef.dataId.copy()
                    noTractId.pop("tract")
                    if noTractId not in dataIdSubList:
                        matchList.append(matches)
                    dataIdSubList.append(noTractId)
                else:
                    matchList.append(matches)
        if not matchList:
            raise TaskError("No matches read: %s" % ([dataRef.dataId for dataRef in dataRefList]))

        allMatches = pd.concat(matchList, axis=0)
        return allMatches, matchAreaDict

    def calibrateCatalogs(self, catalog, wcs=None):
        self.zpLabel = "common (" + str(self.config.analysis.coaddZp) + ")"
        # My persisted catalogs in lauren/LSST/DM-6816new all have nan for RA
        # and Dec (see DM-9556).
        if np.all(np.isnan(catalog["coord_ra"])):
            if wcs is None:
                self.log.warn("Bad RA, Dec entries but can't update because wcs is None")
            else:
                afwTable.updateSourceCoords(wcs, catalog)
        calibrated = calibrateSourceCatalog(catalog, self.config.analysis.coaddZp)
        return calibrated

    def plotMags(self, catalog, plotInfoDict, areaDict, matchRadius=None,
                 matchRadiusUnitStr=None, zpLabel=None, forcedStr=None, fluxToPlotList=None,
                 postFix="", flagsCat=None, highlightList=None, uberCalLabel=None):
        yield
        schema = getSchema(catalog)
        if not fluxToPlotList:
            fluxToPlotList = self.config.fluxToPlotList
        unitStr = "mmag" if self.config.toMilli else "mag"
        enforcer = Enforcer(requireLess={"star": {"stdev": 0.02*self.unitScale}})
        for col in fluxToPlotList:
            if col + "_instFlux" in schema:
                shortName = "mag_" + col + postFix
                self.log.info("shortName = {:s}".format(shortName))
                yield from self.AnalysisClass(catalog, MagDiff(col + "_instFlux", "base_PsfFlux_instFlux",
                                                               unitScale=self.unitScale),
                                              "Mag(%s) - PSFMag (%s)" % (fluxToPlotString(col), unitStr),
                                              shortName, self.config.analysis, labeller=StarGalaxyLabeller(),
                                              unitScale=self.unitScale,
                                              ).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                                        enforcer=enforcer, matchRadius=matchRadius,
                                                        zpLabel=zpLabel, forcedStr=forcedStr,
                                                        uberCalLabel=uberCalLabel,
                                                        highlightList=highlightList)
                # Also make comparison plots for calib_psf_used only objects
                # for the circular aperture plots.
                if "CircularApertureFlux_12_0" in col:
                    shortName = "mag_" + col + postFix + "_calib_psf_used"
                    self.log.info("shortName = {:s}".format(shortName))
                    calibHighlightList = highlightList.copy()
                    for i, flagName in enumerate([col + "_flag", ] + list(self.config.analysis.flags)):
                        if not any(flagName in highlight for highlight in calibHighlightList):
                            calibHighlightList += [(flagName, 0, FLAGCOLORS[i%len(FLAGCOLORS)]), ]
                    yield from self.AnalysisClass(catalog,
                                                  MagDiff(col + "_instFlux", "base_PsfFlux_instFlux",
                                                          unitScale=self.unitScale),
                                                  ("%s - PSF (calib_psf_used) (%s)" % (fluxToPlotString(col),
                                                   unitStr)),
                                                  shortName, self.config.analysis,
                                                  goodKeys=["calib_psf_used"],
                                                  labeller=StarGalaxyLabeller(), unitScale=self.unitScale,
                                                  fluxColumn="base_CircularApertureFlux_12_0_instFlux"
                                                  ).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                                            enforcer=enforcer, matchRadius=matchRadius,
                                                            zpLabel=zpLabel, forcedStr=forcedStr,
                                                            uberCalLabel=uberCalLabel,
                                                            highlightList=calibHighlightList)

    def plotSizes(self, catalog, plotInfoDict, areaDict, matchRadius=None, zpLabel=None, forcedStr=None,
                  postFix="", highlightList=None, uberCalLabel=None):
        yield
        schema = getSchema(catalog)
        unitStr = " (milli)" if self.config.toMilli else ""
        plotAllKwargs = dict(matchRadius=matchRadius, zpLabel=zpLabel, forcedStr=forcedStr,
                             uberCalLabel=uberCalLabel, highlightList=highlightList)
        calibHighlightList0 = None
        for col in ["base_PsfFlux", ]:
            if col + "_instFlux" in schema:
                if highlightList is not None:
                    calibHighlightList0 = highlightList.copy()
                    if not any(col + "_flag" in highlight for highlight in calibHighlightList0):
                        calibHighlightList0 += [(col + "_flag", 0, "yellow"), ]
                compareCol = "base_SdssShape"
                # Set limits dynamically...can be very different visit-to-visit
                # due to seeing differences.  SDSS and HSM should be similar,
                # so limits based on one should be valid for the other and
                # having the same scale eases comparisons between the two.
                traceSizeFunc = TraceSize(compareCol)

                # First do for calib_psf_used only.
                shortName = "trace" + postFix + "_calib_psf_used"
                psfUsed = catalog[catalog["calib_psf_used"]].copy(deep=True)
                sdssTrace = traceSizeFunc(psfUsed)
                goodVals = np.isfinite(sdssTrace)
                psfUsed = psfUsed[goodVals].copy(deep=True)
                sdssTrace = sdssTrace[goodVals]
                traceMean = np.around(np.nanmean(sdssTrace), 2)
                traceStd = max(0.03, np.around(4.5*np.nanstd(sdssTrace), 2))
                qMin = traceMean - traceStd
                qMax = traceMean + traceStd
                self.log.info("shortName = {:s}".format(shortName))
                if calibHighlightList0 is not None:
                    calibHighlightList = calibHighlightList0.copy()
                    if not any(compareCol + "_flag" in highlight for highlight in calibHighlightList):
                        calibHighlightList += [(compareCol + "_flag", 0, "greenyellow"), ]
                plotAllKwargs.update(highlightList=calibHighlightList)
                yield from self.AnalysisClass(psfUsed, sdssTrace,
                                              ("          SdssShape Trace (calib_psf_used): "
                                               r"$\sqrt{0.5*(I_{xx}+I_{yy})}$ (pixels)"),
                                              shortName, self.config.analysis,
                                              goodKeys=["calib_psf_used"], qMin=qMin, qMax=qMax,
                                              labeller=StarGalaxyLabeller()
                                              ).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                                        **plotAllKwargs)
                if "ext_shapeHSM_HsmSourceMoments_xx" in schema:
                    shortName = "hsmTrace" + postFix + "_calib_psf_used"
                    compareCol = "ext_shapeHSM_HsmSourceMoments"
                    self.log.info("shortName = {:s}".format(shortName))
                    if calibHighlightList0 is not None:
                        calibHighlightList = calibHighlightList0.copy()
                        if not any(compareCol + "_flag" in highlight for highlight in calibHighlightList):
                            calibHighlightList += [(compareCol + "_flag", 0, "greenyellow"), ]
                    plotAllKwargs.update(highlightList=calibHighlightList)
                    yield from self.AnalysisClass(psfUsed, TraceSize(compareCol),
                                                  (r"          HSM Trace (calib_psf_used): $\sqrt{0.5*(I_{xx}"
                                                   r"+I_{yy})}$ (pixels)"), shortName, self.config.analysis,
                                                  goodKeys=["calib_psf_used"], qMin=qMin, qMax=qMax,
                                                  labeller=StarGalaxyLabeller()
                                                  ).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                                            **plotAllKwargs)

                # Now for all stars.
                compareCol = "base_SdssShape"
                shortName = "trace" + postFix
                starsOnly = catalog[catalog["base_ClassificationExtendedness_value"] < 0.5].copy(deep=True)
                sdssTrace = traceSizeFunc(starsOnly)
                self.log.info("shortName = {:s}".format(shortName))
                plotAllKwargs.update(highlightList=highlightList)
                yield from self.AnalysisClass(starsOnly, sdssTrace,
                                              r"  SdssShape Trace: $\sqrt{0.5*(I_{xx}+I_{yy})}$ (pixels)",
                                              shortName, self.config.analysis, qMin=qMin, qMax=qMax,
                                              labeller=StarGalaxyLabeller()).plotAll(shortName, plotInfoDict,
                                                                                     areaDict, self.log,
                                                                                     **plotAllKwargs)
                if "ext_shapeHSM_HsmSourceMoments_xx" in schema:
                    shortName = "hsmTrace" + postFix
                    compareCol = "ext_shapeHSM_HsmSourceMoments"
                    self.log.info("shortName = {:s}".format(shortName))
                    yield from self.AnalysisClass(starsOnly, TraceSize(compareCol),
                                                  r"HSM Trace: $\sqrt{0.5*(I_{xx}+I_{yy})}$ (pixels)",
                                                  shortName, self.config.analysis, qMin=qMin, qMax=qMax,
                                                  labeller=StarGalaxyLabeller()).plotAll(shortName,
                                                                                         plotInfoDict,
                                                                                         areaDict, self.log,
                                                                                         **plotAllKwargs)

            if col + "_instFlux" in schema:
                shortName = "psfTraceDiff" + postFix
                compareCol = "base_SdssShape"
                psfCompareCol = "base_SdssShape_psf"
                if calibHighlightList is not None:
                    if not any(compareCol + "_flag" in highlight for highlight in calibHighlightList):
                        calibHighlightList += [(compareCol + "_flag", 0, "greenyellow"), ]
                    if not any(psfCompareCol + "_flag" in highlight for highlight in calibHighlightList):
                        calibHighlightList += [(psfCompareCol + "_flag", 0, "lime"), ]
                    plotAllKwargs.update(highlightList=calibHighlightList)
                self.log.info("shortName = {:s}".format(shortName))
                yield from self.AnalysisClass(catalog, PsfTraceSizeDiff(compareCol, psfCompareCol),
                                              "    SdssShape Trace % diff (psf_used - PSFmodel)", shortName,
                                              self.config.analysis, flags=[col + "_flag"],
                                              goodKeys=["calib_psf_used"], qMin=-3.0, qMax=3.0,
                                              labeller=StarGalaxyLabeller()
                                              ).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                                        **plotAllKwargs)

                shortName = "e1Resids" + postFix
                self.log.info("shortName = {:s}".format(shortName))
                yield from self.AnalysisClass(catalog, E1Resids(compareCol, psfCompareCol,
                                              unitScale=self.unitScale),
                                              "        SdssShape e1 resids (psf_used - PSFmodel)%s" % unitStr,
                                              shortName, self.config.analysis,
                                              goodKeys=["calib_psf_used"], qMin=-0.05, qMax=0.05,
                                              labeller=StarGalaxyLabeller(),
                                              unitScale=self.unitScale).plotAll(shortName, plotInfoDict,
                                                                                areaDict, self.log,
                                                                                **plotAllKwargs)

                shortName = "e2Resids" + postFix
                self.log.info("shortName = {:s}".format(shortName))
                yield from self.AnalysisClass(catalog, E2Resids(compareCol, psfCompareCol,
                                              unitScale=self.unitScale),
                                              "       SdssShape e2 resids (psf_used - PSFmodel)%s" % unitStr,
                                              shortName, self.config.analysis,
                                              goodKeys=["calib_psf_used"], qMin=-0.05, qMax=0.05,
                                              labeller=StarGalaxyLabeller(),
                                              unitScale=self.unitScale).plotAll(shortName, plotInfoDict,
                                                                                areaDict, self.log,
                                                                                **plotAllKwargs)

                if "ext_shapeHSM_HsmSourceMoments_xx" in schema:
                    shortName = "psfHsmTraceDiff" + postFix
                    compareCol = "ext_shapeHSM_HsmSourceMoments"
                    psfCompareCol = "ext_shapeHSM_HsmPsfMoments"
                    if calibHighlightList0 is not None:
                        calibHighlightList = calibHighlightList0.copy()
                        if not any(compareCol + "_flag" in highlight for highlight in calibHighlightList):
                            calibHighlightList += [(compareCol + "_flag", 0, "greenyellow"), ]
                        if not any(psfCompareCol + "_flag" in highlight for highlight in calibHighlightList):
                            calibHighlightList += [(psfCompareCol + "_flag", 0, "lime"), ]
                    plotAllKwargs.update(highlightList=calibHighlightList)
                    self.log.info("shortName = {:s}".format(shortName))

                    yield from self.AnalysisClass(catalog, PsfTraceSizeDiff(compareCol, psfCompareCol),
                                                  "HSM Trace % diff (psf_used - PSFmodel)", shortName,
                                                  self.config.analysis,
                                                  goodKeys=["calib_psf_used"], qMin=-3.0, qMax=3.0,
                                                  labeller=StarGalaxyLabeller(),
                                                  ).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                                            **plotAllKwargs)
                    shortName = "e1ResidsHsm" + postFix
                    self.log.info("shortName = {:s}".format(shortName))
                    yield from self.AnalysisClass(catalog, E1Resids(compareCol, psfCompareCol,
                                                  unitScale=self.unitScale),
                                                  "   HSM e1 resids (psf_used - PSFmodel)%s" % unitStr,
                                                  shortName, self.config.analysis,
                                                  goodKeys=["calib_psf_used"], qMin=-0.05, qMax=0.05,
                                                  labeller=StarGalaxyLabeller(),
                                                  unitScale=self.unitScale).plotAll(shortName, plotInfoDict,
                                                                                    areaDict, self.log,
                                                                                    **plotAllKwargs)
                    shortName = "e2ResidsHsm" + postFix
                    self.log.info("shortName = {:s}".format(shortName))
                    yield from self.AnalysisClass(catalog, E2Resids(compareCol, psfCompareCol,
                                                  unitScale=self.unitScale),
                                                  "   HSM e2 resids (psf_used - PSFmodel)%s" % unitStr,
                                                  shortName, self.config.analysis,
                                                  goodKeys=["calib_psf_used"], qMin=-0.05, qMax=0.05,
                                                  labeller=StarGalaxyLabeller(),
                                                  unitScale=self.unitScale).plotAll(shortName, plotInfoDict,
                                                                                    areaDict, self.log,
                                                                                    **plotAllKwargs)

                    shortName = "e1ResidsHsmRegauss" + postFix
                    self.log.info("shortName = {:s}".format(shortName))
                    yield from self.AnalysisClass(catalog, E1ResidsHsmRegauss(unitScale=self.unitScale),
                                                  "       HsmRegauss e1 resids (psf_used - HsmPsfMoments)%s" %
                                                  unitStr, shortName, self.config.analysis,
                                                  goodKeys=["calib_psf_used"], qMin=-0.05, qMax=0.05,
                                                  labeller=StarGalaxyLabeller(),
                                                  unitScale=self.unitScale).plotAll(shortName, plotInfoDict,
                                                                                    areaDict, self.log,
                                                                                    **plotAllKwargs)

                    shortName = "e2ResidsHsmRegauss" + postFix
                    self.log.info("shortName = {:s}".format(shortName))
                    yield from self.AnalysisClass(catalog, E2ResidsHsmRegauss(unitScale=self.unitScale),
                                                  "       HsmRegauss e2 resids (psf_used - HsmPsfMoments)%s" %
                                                  unitStr, shortName, self.config.analysis,
                                                  goodKeys=["calib_psf_used"], qMin=-0.05, qMax=0.05,
                                                  labeller=StarGalaxyLabeller(),
                                                  unitScale=self.unitScale).plotAll(shortName, plotInfoDict,
                                                                                    areaDict, self.log,
                                                                                    **plotAllKwargs)

    def plotCentroidXY(self, catalog, plotInfoDict, areaDict, matchRadius=None, zpLabel=None,
                       forcedStr=None, flagsCat=None, uberCalLabel=None, highlightList=None):
        yield
        schema = getSchema(catalog)
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.02*self.unitScale}})
        for col in ["base_SdssCentroid_x", "base_SdssCentroid_y"]:
            if col in schema:
                shortName = col
                self.log.info("shortName = {:s}".format(shortName))
                yield from self.AnalysisClass(catalog, catalog[col], "(%s)" % col, shortName,
                                              self.config.analysis, labeller=StarGalaxyLabeller(),
                                              ).plotFP(shortName, plotInfoDict, self.log, enforcer=enforcer,
                                                       matchRadius=matchRadius, zpLabel=zpLabel,
                                                       forcedStr=forcedStr)

    def plotFootprint(self, catalog, plotInfoDict, areaDict, matchRadius=None, zpLabel=None, forcedStr=None,
                      postFix="", flagsCat=None, plotRunStats=False, highlightList=None, uberCalLabel=None):
        yield
        schema = getSchema(catalog)
        enforcer = None
        plotAllKwargs = dict(matchRadius=matchRadius, zpLabel=zpLabel, forcedStr=forcedStr,
                             uberCalLabel=uberCalLabel, highlightList=highlightList)
        if "calib_psf_used" in schema:
            shortName = "footArea_calib_psf_used"
            self.log.info("shortName = {:s}".format(shortName))

            yield from self.AnalysisClass(catalog, catalog["base_FootprintArea_value"], "%s" % shortName,
                                          shortName, self.config.analysis, goodKeys=["calib_psf_used"],
                                          qMin=-100, qMax=2000, labeller=StarGalaxyLabeller(),
                                          ).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                                    plotRunStats=plotRunStats, **plotAllKwargs)
        shortName = "footArea"
        self.log.info("shortName = {:s}".format(shortName))
        yield from self.AnalysisClass(catalog, catalog["base_FootprintArea_value"], "%s" % shortName,
                                      shortName, self.config.analysis,
                                      qMin=0, qMax=3000, labeller=StarGalaxyLabeller()
                                      ).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                                enforcer=enforcer, plotRunStats=plotRunStats, **plotAllKwargs)

    def plotFootprintHist(self, catalog, description, plotInfoDict, matchRadius=None, zpLabel=None,
                          postFix="", flagsCat=None, uberCalLabel=None, forcedStr=None):
        yield
        stats = None
        shortName = "footArea"
        self.log.info("shortName = {:s}".format(shortName + "Hist"))
        yield from self.AnalysisClass(catalog, catalog["base_FootprintArea_value"], "%s" % shortName,
                                      shortName, self.config.analysis, qMin=0, qMax=3000,
                                      labeller=StarGalaxyLabeller()
                                      ).plotHistogram(description, plotInfoDict, stats=stats,
                                                      matchRadius=matchRadius, zpLabel=zpLabel,
                                                      filterStr=plotInfoDict["filter"],
                                                      uberCalLabel=uberCalLabel, forcedStr=forcedStr)

    def plotPsfFluxSnHists(self, catalog, description, plotInfoDict, areaDict, matchRadius=None,
                           zpLabel=None, forcedStr=None, uberCalLabel=None, postFix="",
                           logPlot=True, density=True, cumulative=-1):
        yield
        schema = getSchema(catalog)
        stats = None
        shortName = "psfInstFlux" if zpLabel == "raw" else "psfCalFlux"
        self.log.info("shortName = {:s}".format(shortName))
        # want "raw" flux
        factor = 10.0**(0.4*self.config.analysis.commonZp) if zpLabel == "raw" else NANOJANSKYS_PER_AB_FLUX
        psfFlux = catalog["base_PsfFlux_instFlux"]*factor
        psfFluxErr = catalog["base_PsfFlux_instFluxErr"]*factor
        # Cull here so that all subsets get the same culling
        bad = makeBadArray(catalog, flagList=self.config.analysis.flags)
        psfFlux = psfFlux[~bad]
        psfFluxErr = psfFluxErr[~bad]
        psfSn = psfFlux/psfFluxErr

        # Scale S/N threshold by ~sqrt(#exposures) if catalog is coadd data
        if "base_InputCount_value" in schema:
            inputCounts = catalog["base_InputCount_value"]
            scaleFactor = computeMeanOfFrac(inputCounts, tailStr="upper", fraction=0.1, floorFactor=10)
            if scaleFactor == 0.0:
                scaleFactor = computeMeanOfFrac(inputCounts, tailStr="upper", fraction=0.1, floorFactor=1)
            highSn = np.floor(
                np.sqrt(scaleFactor)*self.config.analysis.signalToNoiseThreshold/100 + 0.49)*100
        else:
            highSn = self.config.analysis.signalToNoiseThreshold

        lowSn = 20.0
        lowFlux, highFlux = 4000.0, 12500.0
        goodSn = psfSn > lowSn
        psfFluxSnGtLow = psfFlux[goodSn]
        goodSn = psfSn > highSn
        psfFluxSnGtHigh = psfFlux[goodSn]
        goodFlux = psfFlux > lowFlux
        psfSnFluxGtLow = psfSn[goodFlux]
        goodFlux = psfFlux > highFlux
        psfSnFluxGtHigh = psfSn[goodFlux]
        psfUsedCat = catalog[catalog["calib_psf_used"]]
        psfUsedPsfFlux = psfUsedCat["base_PsfFlux_instFlux"]*factor
        psfUsedPsfFluxErr = psfUsedCat["base_PsfFlux_instFluxErr"]*factor
        psfUsedPsfSn = psfUsedPsfFlux/psfUsedPsfFluxErr

        if "lsst" in plotInfoDict["cameraName"]:
            filterStr = "[" + plotInfoDict["cameraName"] + "-" + plotInfoDict["filter"] + "]"
        else:
            filterStr = plotInfoDict["filter"]
        yield from self.AnalysisClass(catalog[~bad], psfFlux, "%s" % shortName, shortName,
                                      self.config.analysis, qMin=0,
                                      qMax=int(min(99999, max(4.0*np.median(psfFlux), 0.25*np.max(psfFlux)))),
                                      labeller=AllLabeller()
                                      ).plotHistogram(description, plotInfoDict, numBins="sqrt", stats=stats,
                                                      zpLabel=zpLabel, forcedStr=forcedStr,
                                                      filterStr=filterStr,
                                                      uberCalLabel=uberCalLabel,
                                                      vertLineList=[lowFlux, highFlux],
                                                      logPlot=logPlot, density=False, cumulative=cumulative,
                                                      addDataList=[psfFluxSnGtLow, psfFluxSnGtHigh,
                                                                   psfUsedPsfFlux],
                                                      addDataLabelList=["S/N>{:.1f}".format(lowSn),
                                                                        "S/N>{:.1f}".format(highSn),
                                                                        "psf_used"])
        shortName = "psfInstFlux/psfInstFluxErr" if zpLabel == "raw" else "psfCalFlux/psfCalFluxErr"
        description = description.replace("Flux", "FluxSn")
        self.log.info("shortName = {:s}".format(shortName))
        yield from self.AnalysisClass(catalog[~bad], psfSn, "%s" % "S/N = " + shortName, shortName,
                                      self.config.analysis, qMin=0, qMax=4*highSn, labeller=AllLabeller()
                                      ).plotHistogram(description, plotInfoDict, numBins="sqrt", stats=stats,
                                                      zpLabel=zpLabel, forcedStr=forcedStr,
                                                      filterStr=filterStr,
                                                      uberCalLabel=uberCalLabel, vertLineList=[lowSn, highSn],
                                                      logPlot=logPlot, density=False, cumulative=cumulative,
                                                      addDataList=[psfSnFluxGtLow, psfSnFluxGtHigh,
                                                                   psfUsedPsfSn],
                                                      addDataLabelList=["Flux>{:.1f}".format(lowFlux),
                                                                        "Flux>{:.1f}".format(highFlux),
                                                                        "psf_used"])

        skyplotKwargs = dict(stats=stats, matchRadius=matchRadius, matchRadiusUnitStr=None, zpLabel=zpLabel)

        yield from self.AnalysisClass(catalog[~bad], psfSn, "%s" % "S/N = " + shortName, shortName,
                                      self.config.analysis, qMin=0, qMax=1.25*highSn, labeller=AllLabeller(),
                                      ).plotSkyPosition(description, plotInfoDict, areaDict,
                                                        dataName="all", **skyplotKwargs)

    def plotStarGal(self, catalog, plotInfoDict, areaDict, matchRadius=None, zpLabel=None, forcedStr=None,
                    highlightList=None, uberCalLabel=None):
        yield
        schema = getSchema(catalog)
        plotAllKwargs = dict(matchRadius=matchRadius, zpLabel=zpLabel, forcedStr=forcedStr,
                             uberCalLabel=uberCalLabel)
        shortName = "pStar"
        self.log.info("shortName = {:s}".format(shortName))
        yield from self.AnalysisClass(catalog, deconvMomStarGal, "P(star) from deconvolved moments",
                                      shortName, self.config.analysis, qMin=-0.1, qMax=1.39,
                                      labeller=StarGalaxyLabeller()
                                      ).plotAll(shortName, plotInfoDict, areaDict, self.log, **plotAllKwargs)
        shortName = "deconvMom"
        self.log.info("shortName = {:s}".format(shortName))
        yield from self.AnalysisClass(catalog, deconvMom, "Deconvolved moments", shortName,
                                      self.config.analysis, qMin=-1.0, qMax=3.0,
                                      labeller=StarGalaxyLabeller()
                                      ).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                                enforcer=Enforcer(requireLess={"star": {"stdev": 0.2}}),
                                                **plotAllKwargs)

        if "ext_shapeHSM_HsmShapeRegauss_resolution" in schema:
            shortName = "resolution"
            self.log.info("shortName = {:s}".format(shortName))
            yield from self.AnalysisClass(catalog, catalog["ext_shapeHSM_HsmShapeRegauss_resolution"],
                                          "Resolution Factor from HsmRegauss",
                                          shortName, self.config.analysis, qMin=-0.1, qMax=1.15,
                                          labeller=StarGalaxyLabeller()
                                          ).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                                    **plotAllKwargs)

    def plotCompareUnforced(self, forced, unforced, plotInfoDict, areaDict, zpLabel=None, fluxToPlotList=None,
                            uberCalLabel=None, matchRadius=None, matchRadiusUnitStr=None, highlightList=None):
        yield
        forcedSchema = getSchema(forced)
        fluxToPlotList = fluxToPlotList if fluxToPlotList else self.config.fluxToPlotList
        unitStr = "mmag" if self.config.toMilli else "mag"
        enforcer = None
        for col in fluxToPlotList:
            magDiffFunc = MagDiff(col + "_instFlux", col + "_instFlux", unitScale=self.unitScale)
            shortName = "compareUnforced_" + col
            self.log.info("shortName = {:s}".format(shortName))
            if col + "_instFlux" in forcedSchema:
                yield from self.AnalysisClass(forced, magDiffFunc(forced, unforced),
                                              "  Forced - Unforced mag [%s] (%s)" %
                                              (fluxToPlotString(col), unitStr),
                                              shortName, self.config.analysis, prefix="",
                                              labeller=OverlapsStarGalaxyLabeller(first="", second=""),
                                              unitScale=self.unitScale, compareCat=unforced,
                                              ).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                                        enforcer=enforcer, matchRadius=matchRadius,
                                                        matchRadiusUnitStr=matchRadiusUnitStr,
                                                        highlightList=highlightList,
                                                        zpLabel=zpLabel, uberCalLabel=uberCalLabel)

    def isBad(self, source):
        """Return True if any of config.badFlags are set for this source.
        """
        for flag in self.config.analysis.flags:
            if source.get(flag):
                return True
        return False

    def overlaps(self, catalog, patchList, tractInfo):
        badForOverlap = makeBadArray(catalog, flagList=self.config.analysis.flags,
                                     onlyReadStars=self.config.onlyReadStars, patchInnerOnly=False)
        goodCat = catalog[~badForOverlap].copy(deep=True)
        overlapPatchList = []
        for patch1 in patchList:
            for patch2 in patchList:
                if patch1 != patch2:
                    overlapping = checkPatchOverlap([patch1, patch2], tractInfo)
                    if overlapping:
                        if {patch1, patch2} not in overlapPatchList:
                            overlapPatchList.append({patch1, patch2})
        matchList = []
        matchRadius = self.config.matchOverlapRadius
        for patchPair in overlapPatchList:
            patchPair = list(patchPair)
            patchCat1 = goodCat[goodCat["patchId"] == patchPair[0]].copy(deep=True)
            patchCat2 = goodCat[goodCat["patchId"] == patchPair[1]].copy(deep=True)
            patchPairMatches = matchAndJoinCatalogs(patchCat1, patchCat2, matchRadius, log=self.log)
            if not patchPairMatches.empty:
                matchList.append(patchPairMatches)
        if matchList:
            matches = pd.concat(matchList, axis=0)
        else:
            matches = None
        return matches

    def plotOverlaps(self, overlaps, plotInfoDict, areaDict, matchRadius=None, matchRadiusUnitStr=None,
                     zpLabel=None, forcedStr=None, postFix="", fluxToPlotList=None, highlightList=None,
                     uberCalLabel=None):
        yield
        schema = getSchema(overlaps)
        if not fluxToPlotList:
            fluxToPlotList = self.config.fluxToPlotList
        unitStr = "mmag" if self.config.toMilli else "mag"
        magEnforcer = Enforcer(requireLess={"star": {"stdev": 0.003*self.unitScale}})
        plotAllKwargs = dict(matchRadius=matchRadius, matchRadiusUnitStr=matchRadiusUnitStr, zpLabel=zpLabel,
                             forcedStr=forcedStr, uberCalLabel=uberCalLabel, highlightList=highlightList)
        for col in fluxToPlotList:
            shortName = "overlap_" + col + postFix
            self.log.info("shortName = {:s}".format(shortName))
            if "first_" + col + "_instFlux" in schema:
                yield from self.AnalysisClass(overlaps, MagDiff("first_" + col + "_instFlux",
                                                                "second_" + col + "_instFlux",
                                                                unitScale=self.unitScale),
                                              "  Overlap mag difference (%s) (%s)" %
                                              (fluxToPlotString(col), unitStr),
                                              shortName, self.config.analysis, prefix="first_",
                                              flags=[col + "_flag"], labeller=OverlapsStarGalaxyLabeller(),
                                              magThreshold=23, unitScale=self.unitScale
                                              ).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                                        enforcer=magEnforcer, **plotAllKwargs)
        unitStr = "mas" if self.config.toMilli else "arcsec"
        distEnforcer = Enforcer(requireLess={"star": {"stdev": 0.005*self.unitScale}})
        shortName = "overlap_distance" + postFix
        self.log.info("shortName = {:s}".format(shortName))
        yield from self.AnalysisClass(
            overlaps, lambda cat: cat["distance"]*(1.0*geom.radians).asArcseconds()*self.unitScale,
            "Distance (%s)" % unitStr, shortName, self.config.analysis, prefix="first_", qMin=-0.01,
            qMax=0.11, labeller=OverlapsStarGalaxyLabeller(), forcedMean=0.0,
            unitScale=self.unitScale).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                              enforcer=distEnforcer, doPrintMedian=True, **plotAllKwargs)

    def plotMatches(self, matches, plotInfoDict, areaDict, description="matches", matchRadius=None,
                    matchRadiusUnitStr=None, zpLabel=None, forcedStr=None, highlightList=None,
                    uberCalLabel=None):
        yield
        schema = getSchema(matches)
        unitStr = "mmag" if self.config.toMilli else "mag"
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.030*self.unitScale}}),
        fluxToPlotList = ["base_PsfFlux", "base_CircularApertureFlux_12_0"]
        plotAllKwargs = dict(matchRadius=matchRadius, matchRadiusUnitStr=matchRadiusUnitStr, zpLabel=zpLabel,
                             forcedStr=forcedStr, uberCalLabel=uberCalLabel, highlightList=highlightList)
        if self.config.doApplyColorTerms:
            ct = self.config.colorterms.getColorterm(plotInfoDict["filter"],
                                                     self.config.refObjLoader.ref_dataset_name)
        else:
            # Pass in a null colorterm.
            # Obtain the filter name from the reference loader filter map, if
            # present, otherwise set to the canonical filter name.
            refFilterName = (self.config.refObjLoader.filterMap[plotInfoDict["filter"]] if
                             plotInfoDict["filter"] in self.config.refObjLoader.filterMap.keys() else
                             afwImage.Filter(afwImage.Filter(plotInfoDict["filter"]).getId()).getName())
            ct = Colorterm(primary=refFilterName, secondary=refFilterName)
            self.log.warn("Note: no colorterms loaded for {:s}, thus no colorterms will be applied to "
                          "the reference catalog".format(self.config.refObjLoader.ref_dataset_name))

        # Magnitude difference plots
        for flux in fluxToPlotList:
            fluxName = flux + "_instFlux"
            if highlightList is not None:
                if not any("src_" + flux + "_flag" in highlight for highlight in highlightList):
                    matchHighlightList = highlightList + [("src_" + flux + "_flag", 0, "yellow"), ]
                    plotAllKwargs.update(highlightList=matchHighlightList)
            if "src_calib_psf_used" in schema:
                shortName = description + "_" + fluxToPlotString(fluxName) + "_mag_calib_psf_used"
                self.log.info("shortName = {:s}".format(shortName))
                yield from self.AnalysisClass(matches, MagDiffMatches(fluxName, ct, zp=0.0,
                                                                      unitScale=self.unitScale),
                                              "%s - ref (calib_psf_used) (%s)" %
                                              (fluxToPlotString(fluxName), unitStr), shortName,
                                              self.config.analysisMatches, prefix="src_",
                                              goodKeys=["calib_psf_used"], qMin=-0.15, qMax=0.1,
                                              labeller=MatchesStarGalaxyLabeller(),
                                              unitScale=self.unitScale).plotAll(shortName, plotInfoDict,
                                                                                areaDict, self.log,
                                                                                enforcer=enforcer,
                                                                                **plotAllKwargs)
            if "src_calib_photometry_used" in schema:
                shortName = description + "_" + fluxToPlotString(fluxName) + "_mag_calib_photometry_used"
                self.log.info("shortName = {:s}".format(shortName))
                yield from self.AnalysisClass(matches, MagDiffMatches(fluxName, ct, zp=0.0,
                                                                      unitScale=self.unitScale),
                                              "   %s - ref (calib_photom_used) (%s)" %
                                              (fluxToPlotString(fluxName), unitStr),
                                              shortName, self.config.analysisMatches, prefix="src_",
                                              goodKeys=["calib_photometry_used"], qMin=-0.15, qMax=0.15,
                                              labeller=MatchesStarGalaxyLabeller(),
                                              unitScale=self.unitScale).plotAll(shortName, plotInfoDict,
                                                                                areaDict, self.log,
                                                                                enforcer=enforcer,
                                                                                **plotAllKwargs)
            shortName = description + "_" + fluxToPlotString(fluxName) + "_mag"
            self.log.info("shortName = {:s}".format(shortName))
            yield from self.AnalysisClass(matches, MagDiffMatches(fluxName, ct, zp=0.0,
                                                                  unitScale=self.unitScale),
                                          "%s - ref (%s)" % (fluxToPlotString(fluxName), unitStr), shortName,
                                          self.config.analysisMatches, prefix="src_", qMin=-0.15, qMax=0.5,
                                          labeller=MatchesStarGalaxyLabeller(), unitScale=self.unitScale,
                                          ).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                                    enforcer=enforcer, **plotAllKwargs)

            plotAllKwargs.update(highlightList=highlightList)
        # Astrometry (positional) difference plots
        unitStr = "mas" if self.config.toMilli else "arcsec"
        qMatchScale = matchRadius if matchRadius else self.matchRadius
        if "src_calib_astrometry_used" in schema:
            shortName = description + "_distance_calib_astrometry_used"
            self.log.info("shortName = {:s}".format(shortName))

            yield from self.AnalysisClass(
                matches, lambda cat: cat["distance"]*(1.0*geom.radians).asArcseconds()*self.unitScale,
                "Distance (%s) (calib_astrom_used)" % unitStr, shortName,
                self.config.analysisMatches, prefix="src_", goodKeys=["calib_astrometry_used"],
                qMin=-0.01*qMatchScale, qMax=0.5*qMatchScale, labeller=MatchesStarGalaxyLabeller(),
                unitScale=self.unitScale).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                                  enforcer=enforcer, doPrintMedian=True, **plotAllKwargs)
        shortName = description + "_distance"
        self.log.info("shortName = {:s}".format(shortName))
        stdevEnforcer = Enforcer(requireLess={"star": {"stdev": 0.050*self.unitScale}})
        yield from self.AnalysisClass(
            matches, lambda cat: cat["distance"]*(1.0*geom.radians).asArcseconds()*self.unitScale,
            "Distance (%s)" % unitStr, shortName, self.config.analysisMatches, prefix="src_",
            qMin=-0.05*qMatchScale, qMax=0.3*qMatchScale, labeller=MatchesStarGalaxyLabeller(),
            forcedMean=0.0, unitScale=self.unitScale).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                                              enforcer=stdevEnforcer, doPrintMedian=True,
                                                              **plotAllKwargs)
        if "src_calib_astrometry_used" in schema:
            shortName = description + "_raCosDec_calib_astrometry_used"
            self.log.info("shortName = {:s}".format(shortName))
            yield from self.AnalysisClass(
                matches,
                AstrometryDiff("src_coord_ra", "ref_coord_ra", declination1="src_coord_dec",
                               declination2="ref_coord_dec", unitScale=self.unitScale),
                r"      $\delta_{RA}$ = $\Delta$RA*cos(Dec) (%s) (calib_astrom_used)" % unitStr,
                shortName, self.config.analysisMatches, prefix="src_", goodKeys=["calib_astrometry_used"],
                qMin=-0.2*qMatchScale, qMax=0.2*qMatchScale, labeller=MatchesStarGalaxyLabeller(),
                unitScale=self.unitScale).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                                  enforcer=enforcer, **plotAllKwargs)
        shortName = description + "_raCosDec"
        self.log.info("shortName = {:s}".format(shortName))
        stdevEnforcer = Enforcer(requireLess={"star": {"stdev": 0.050*self.unitScale}})
        yield from self.AnalysisClass(
            matches,
            AstrometryDiff("src_coord_ra", "ref_coord_ra", declination1="src_coord_dec",
                           declination2="ref_coord_dec", unitScale=self.unitScale),
            r"$\delta_{RA}$ = $\Delta$RA*cos(Dec) (%s)" % unitStr, shortName, self.config.analysisMatches,
            prefix="src_", qMin=-0.2*qMatchScale, qMax=0.2*qMatchScale, labeller=MatchesStarGalaxyLabeller(),
            unitScale=self.unitScale).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                              enforcer=stdevEnforcer, **plotAllKwargs)
        if "src_calib_astrometry_used" in schema:
            shortName = description + "_ra_calib_astrometry_used"
            self.log.info("shortName = {:s}".format(shortName))
            yield from self.AnalysisClass(matches, AstrometryDiff("src_coord_ra", "ref_coord_ra",
                                                                  unitScale=self.unitScale),
                                          r"$\Delta$RA (%s) (calib_astrom_used)" % unitStr, shortName,
                                          self.config.analysisMatches, prefix="src_",
                                          goodKeys=["calib_astrometry_used"], qMin=-0.25*qMatchScale,
                                          qMax=0.25*qMatchScale, labeller=MatchesStarGalaxyLabeller(),
                                          unitScale=self.unitScale
                                          ).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                                    enforcer=enforcer, **plotAllKwargs)
        shortName = description + "_ra"
        self.log.info("shortName = {:s}".format(shortName))
        yield from self.AnalysisClass(
            matches, AstrometryDiff("src_coord_ra", "ref_coord_ra", unitScale=self.unitScale),
            r"$\Delta$RA (%s)" % unitStr, shortName, self.config.analysisMatches, prefix="src_",
            qMin=-0.25*qMatchScale, qMax=0.25*qMatchScale, labeller=MatchesStarGalaxyLabeller(),
            unitScale=self.unitScale).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                              enforcer=stdevEnforcer, **plotAllKwargs)
        shortName = description + "_dec_calib_astrometry_used"
        self.log.info("shortName = {:s}".format(shortName))
        yield from self.AnalysisClass(
            matches, AstrometryDiff("src_coord_dec", "ref_coord_dec", unitScale=self.unitScale),
            r"$\delta_{Dec}$ (%s) (calib_astrom_used)" % unitStr, shortName, self.config.analysisMatches,
            prefix="src_", goodKeys=["calib_astrometry_used"], qMin=-0.25*qMatchScale,
            qMax=0.25*qMatchScale, labeller=MatchesStarGalaxyLabeller(),
            unitScale=self.unitScale).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                              enforcer=enforcer, **plotAllKwargs)
        shortName = description + "_dec"
        self.log.info("shortName = {:s}".format(shortName))
        yield from self.AnalysisClass(
            matches, AstrometryDiff("src_coord_dec", "ref_coord_dec", unitScale=self.unitScale),
            r"$\delta_{Dec}$ (%s)" % unitStr, shortName, self.config.analysisMatches, prefix="src_",
            qMin=-0.3*qMatchScale, qMax=0.3*qMatchScale, labeller=MatchesStarGalaxyLabeller(),
            unitScale=self.unitScale).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                              enforcer=stdevEnforcer, **plotAllKwargs)

    def plotCosmos(self, catalog, plotInfoDict, areaDict, cosmos):
        labeller = CosmosLabeller(cosmos, self.config.matchRadiusRaDec*geom.arcseconds)
        self.AnalysisClass(catalog, deconvMom, "Deconvolved moments", "cosmos", self.config.analysis,
                           qMin=-1.0, qMax=6.0, labeller=labeller,
                           ).plotAll("cosmos", plotInfoDict, areaDict, self.log,
                                     enforcer=Enforcer(requireLess={"star": {"stdev": 0.2}}))

    def matchCatalog(self, catalog, filterName, astrometryConfig):
        try:  # lsst.meas.extensions.astrometryNet is not setup by default
            from lsst.meas.extensions.astrometryNet import LoadAstrometryNetObjectsTask  # noqa : F401
        except ImportError:
            return None
        refObjLoader = LoadAstrometryNetObjectsTask(self.config.refObjLoaderConfig)
        center = geom.averageSpherePoint([src.getCoord() for src in catalog])
        radius = max(center.separation(src.getCoord()) for src in catalog)
        filterName = afwImage.Filter(afwImage.Filter(filterName).getId()).getName()  # Get primary name
        refs = refObjLoader.loadSkyCircle(center, radius, filterName).refCat
        matches = afwTable.matchRaDec(refs, catalog, self.config.matchRadiusRaDec*geom.arcseconds)
        matches = matchNanojanskyToAB(matches)
        return joinMatches(matches, "ref_", "src_")

    def plotRhoStatistics(self, catalog, plotInfoDict, zpLabel=None,
                          forcedStr=None, postFix="", uberCalLabel=None):
        """Plot Rho Statistics with stars used for PSF modelling and non-PSF
        stars.
        """
        yield
        stats = None

        # First do for calib_psf_used only.
        shortName = "Rho" + postFix + "_calib_psf_used"
        psfUsed = catalog[catalog["calib_psf_used"]].copy(deep=True)
        self.log.info("shortName = {:s}".format(shortName))
        yield from self.AnalysisClass(psfUsed, None,
                                      ("        Rho Statistics (calib_psf_used): "),
                                      shortName, self.config.analysis,
                                      goodKeys=["calib_psf_used"], labeller=None
                                      ).plotRhoStatistics(shortName, plotInfoDict, self.log,
                                                          treecorrParams=self.config.treecorrParams,
                                                          stats=stats, zpLabel=zpLabel, forcedStr=forcedStr,
                                                          uberCalLabel=uberCalLabel, verifyJob=self.verifyJob)

        # Now for all stars.
        shortName = "Rho" + postFix + "_all_stars"
        starsOnly = catalog[catalog["base_ClassificationExtendedness_value"] < 0.5].copy(deep=True)
        self.log.info("shortName = {:s}".format(shortName))
        yield from self.AnalysisClass(starsOnly, None,
                                      ("        Rho Statistics: "),
                                      shortName, self.config.analysis, flags=[], labeller=None
                                      ).plotRhoStatistics(shortName, plotInfoDict, self.log,
                                                          treecorrParams=self.config.treecorrParams,
                                                          stats=stats, zpLabel=zpLabel, forcedStr=forcedStr,
                                                          uberCalLabel=uberCalLabel, verifyJob=self.verifyJob)

    def plotQuiver(self, catalog, description, plotInfoDict, areaDict, matchRadius=None,
                   zpLabel=None, forcedStr=None, postFix="", flagsCat=None, uberCalLabel=None, scale=1):
        yield
        stats = None
        shortName = "quiver"
        self.log.info("shortName = {:s}".format(shortName))
        yield from self.AnalysisClass(catalog, None, "%s" % shortName, shortName,
                                      self.config.analysis, labeller=None,
                                      ).plotQuiver(catalog, description, plotInfoDict, areaDict, self.log,
                                                   stats=stats, zpLabel=zpLabel, forcedStr=forcedStr,
                                                   uberCalLabel=uberCalLabel, scale=scale)

    def plotSkyObjects(self, catalog, description, plotInfoDict, areaDict, zpLabel=None, forcedStr=None,
                       postFix="", flagsCat=None):
        yield
        stats = None
        shortName = "skyObjects"
        self.log.info("shortName = {:s}".format(shortName))
        yield from self.AnalysisClass(catalog, None, "%s" % shortName, shortName, self.config.analysis,
                                      labeller=None,).plotSkyObjects(catalog, shortName, plotInfoDict,
                                                                     self.log, zpLabel=zpLabel,
                                                                     forcedStr=forcedStr,
                                                                     verifyJob=self.verifyJob)

        skyplotKwargs = dict(stats=stats, zpLabel=zpLabel, forcedStr=forcedStr)
        skyFlux = "base_CircularApertureFlux_9_0_instFlux"
        skyFluxStr = fluxToPlotString(skyFlux)
        skyFluxes = catalog[skyFlux]*1e12
        qMin, qMax = 0.75*np.nanmin(skyFluxes), 0.75*np.nanmax(skyFluxes)
        yield from self.AnalysisClass(catalog, skyFluxes,
                                      "%s" % "flux(*1e+12)= " + shortName + "[" + skyFluxStr + "]", shortName,
                                      self.config.analysis, qMin=qMin, qMax=qMax, labeller=AllLabeller(),
                                      fluxColumn=skyFlux, magThreshold=99.0
                                      ).plotSkyPosition(shortName, plotInfoDict, areaDict,
                                                        dataName="all", **skyplotKwargs)

    def plotSkyObjectsSky(self, catalog, description, plotInfoDict, forcedStr=None, alpha=0.7,
                          doPlotTractImage=True, doPlotPatchOutline=True, sizeFactor=3.0, maxDiamPix=1000,
                          columnName="base_CircularApertureFlux_9_0_instFlux"):
        yield
        shortName = "skyObjectsSky"
        self.log.info("shortName = {:s}".format(shortName))
        yield from self.AnalysisClass(catalog, None, "%s" % shortName, shortName, self.config.analysis,
                                      labeller=None,).plotInputCounts(catalog, plotInfoDict, self.log,
                                                                      forcedStr=forcedStr,
                                                                      alpha=alpha,
                                                                      doPlotTractImage=doPlotTractImage,
                                                                      doPlotPatchOutline=doPlotPatchOutline,
                                                                      sizeFactor=sizeFactor,
                                                                      maxDiamPix=maxDiamPix,
                                                                      columnName=columnName)

    def plotInputCounts(self, catalog, description, plotInfoDict, zpLabel=None, forcedStr=None,
                        uberCalLabel=None, alpha=0.5, doPlotPatchOutline=True, sizeFactor=5.0,
                        maxDiamPix=1000):
        yield
        shortName = "inputCounts"
        self.log.info("shortName = {:s}".format(shortName))
        yield from self.AnalysisClass(catalog, None, "%s" % shortName, shortName,
                                      self.config.analysis, labeller=None,
                                      ).plotInputCounts(catalog, description, plotInfoDict, self.log,
                                                        forcedStr=forcedStr, uberCalLabel=uberCalLabel,
                                                        alpha=alpha, doPlotPatchOutline=doPlotPatchOutline,
                                                        sizeFactor=sizeFactor, maxDiamPix=maxDiamPix)

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None

    def _getEupsVersionsName(self):
        return None


class CompareCoaddAnalysisConfig(CoaddAnalysisConfig):
    doReadParquetTables1 = Field(dtype=bool, default=True,
                                 doc=("Read parquet tables from postprocessing (e.g. deepCoadd_obj) as "
                                      "input1 data instead of afwTable catalogs."))
    doReadParquetTables2 = Field(dtype=bool, default=True,
                                 doc=("Read parquet tables from postprocessing (e.g. deepCoadd_obj) as "
                                      "input2 data instead of afwTable catalogs."))

    def setDefaults(self):
        CoaddAnalysisConfig.setDefaults(self)
        self.matchRadiusRaDec = 0.2
        self.matchRadiusXy = 1.0e-5  # has to be bigger than absolute zero
        if "base_PsfFlux" not in self.fluxToPlotList:
            self.fluxToPlotList.append("base_PsfFlux")  # Add PSF flux to default list for comparison scripts


class CompareCoaddAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        kwargs["subdir"] = parsedCmd.subdir
        rootDir = parsedCmd.input.split("rerun")[0] if len(parsedCmd.rerun) == 2 else parsedCmd.input
        butlerArgs = dict(root=os.path.join(rootDir, "rerun", parsedCmd.rerun2))
        if parsedCmd.calib is not None:
            butlerArgs["calibRoot"] = parsedCmd.calib
        butler2 = Butler(**butlerArgs)
        idParser = parsedCmd.id.__class__(parsedCmd.id.level)
        idParser.idList = parsedCmd.id.idList
        butler = parsedCmd.butler
        parsedCmd.butler = butler2
        idParser.makeDataRefList(parsedCmd)
        parsedCmd.butler = butler

        return [(refList1, dict(patchRefList2=refList2, **kwargs)) for
                refList1, refList2 in zip(parsedCmd.id.refList, idParser.refList)]


class CompareCoaddAnalysisTask(CoaddAnalysisTask):
    ConfigClass = CompareCoaddAnalysisConfig
    RunnerClass = CompareCoaddAnalysisRunner
    _DefaultName = "compareCoaddAnalysis"

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_argument("--rerun2", required=True, help="Second rerun, for comparison")
        parser.add_id_argument("--id", "deepCoadd_forced_src",
                               help="data ID, e.g. --id tract=12345 patch=1,2 filter=HSC-X",
                               ContainerClass=TractDataIdContainer)
        parser.add_argument("--subdir", type=str, default="",
                            help=("Subdirectory below plots/filter/tract-NNNN/ (useful for, "
                                  "e.g., subgrouping of Patches.  Ignored if only one Patch is "
                                  "specified, in which case the subdir is set to patch-NNN"))
        return parser

    def __init__(self, *args, **kwargs):
        CmdLineTask.__init__(self, *args, **kwargs)
        self.unitScale = 1000.0 if self.config.toMilli else 1.0
        self.matchRadius = self.config.matchRadiusXy if self.config.matchXy else self.config.matchRadiusRaDec
        self.matchRadiusUnitStr = " (pixels)" if self.config.matchXy else "\""

    def runDataRef(self, patchRefList1, patchRefList2, subdir=""):
        plotList = []
        haveForced = True  # do forced datasets exist (may not for single band datasets)?
        dataset1 = "Coadd_obj" if self.config.doReadParquetTables1 else "Coadd_forced_src"
        patchRefExistsList1 = [patchRef1 for patchRef1 in patchRefList1 if
                               patchRef1.datasetExists(self.config.coaddName + dataset1)]
        dataset2 = "Coadd_obj" if self.config.doReadParquetTables2 else "Coadd_forced_src"
        patchRefExistsList2 = [patchRef2 for patchRef2 in patchRefList2 if
                               patchRef2.datasetExists(self.config.coaddName + dataset2)]
        if not patchRefExistsList1 or not patchRefExistsList2:
            haveForced = False
        forcedStr = "forced" if haveForced else "unforced"
        if not haveForced:
            self.log.warn("Forced datasets do not exist for both input1 and input2 for tract: {0:d} "
                          "filter: {1:s}.  Plotting unforced results only.".
                          format(patchRefList1[0].dataId["tract"], patchRefList1[0].dataId["filter"]))
            dataset1 = "Coadd_meas"
            dataset2 = "Coadd_meas"
            patchRefExistsList1 = [patchRef1 for patchRef1 in patchRefList1 if
                                   patchRef1.datasetExists(self.config.coaddName + dataset1)]
        if not patchRefExistsList1:
            raise TaskError("No data exists in patRefList1: %s" %
                            ([patchRef1.dataId for patchRef1 in patchRefList1]))
        patchRefList2 = [dataRef2 for dataRef2 in patchRefList2 if
                         dataRef2.datasetExists(self.config.coaddName + dataset2)]

        patchList1 = [dataRef1.dataId["patch"] for dataRef1 in patchRefList1 if
                      dataRef1.datasetExists(self.config.coaddName + dataset1)]
        patchRefList1 = patchRefExistsList1

        repoInfo1 = getRepoInfo(patchRefList1[0], coaddName=self.config.coaddName, coaddDataset=dataset1)
        repoInfo2 = getRepoInfo(patchRefList2[0], coaddName=self.config.coaddName, coaddDataset=dataset2)
        # Find a visit/ccd input so that you can check for meas_mosaic input
        # (i.e. to set uberCalLabel).
        self.uberCalLabel1 = determineExternalCalLabel(repoInfo1, patchList1[0],
                                                       coaddName=self.config.coaddName)
        self.uberCalLabel2 = determineExternalCalLabel(repoInfo2, patchList1[0],
                                                       coaddName=self.config.coaddName)
        self.uberCalLabel1 = self.uberCalLabel1.replace("  wcs", "_1  wcs")
        self.uberCalLabel2 = self.uberCalLabel2.replace("  wcs", "_2  wcs")
        self.uberCalLabel = self.uberCalLabel1 + "_1\n" + self.uberCalLabel2 + "_2"
        self.log.info(f"External calibration(s) used: {self.uberCalLabel}")

        if self.config.doReadParquetTables1 or self.config.doReadParquetTables2:
            if self.config.doReadParquetTables1:
                if haveForced:
                    forced1, _ = self.readParquetTables(patchRefList1, self.config.coaddName + "Coadd_obj",
                                                        repoInfo1, "forced_src")
                unforced1, _ = self.readParquetTables(patchRefList1, self.config.coaddName + "Coadd_obj",
                                                      repoInfo1, "meas")
                unforced1 = self.calibrateCatalogs(unforced1, wcs=repoInfo1.wcs)
                areaDict1, _ = computeAreaDict(repoInfo1, patchRefList1,
                                               dataset=self.config.coaddName + "Coadd", fakeCat=None)
            if self.config.doReadParquetTables2:
                if haveForced:
                    forced2, _ = self.readParquetTables(patchRefList2, self.config.coaddName + "Coadd_obj",
                                                        repoInfo2, "forced_src")
                unforced2, _ = self.readParquetTables(patchRefList2, self.config.coaddName + "Coadd_obj",
                                                      repoInfo2, "meas")
                unforced2 = self.calibrateCatalogs(unforced2, wcs=repoInfo2.wcs)

        if not self.config.doReadParquetTables1 or not self.config.doReadParquetTables2:
            hscRun = repoInfo1.hscRun if repoInfo1.hscRun else repoInfo2.hscRun
            aliasDictList = [self.config.flagsToAlias, ]
            if hscRun and self.config.srcSchemaMap is not None:
                aliasDictList += [self.config.srcSchemaMap]

            if not self.config.doReadParquetTables1:
                catStruct1 = self.readAfwCoaddTables(patchRefList1, repoInfo1, haveForced,
                                                     aliasDictList=aliasDictList)
                unforced1 = catStruct1.unforced
                forced1 = catStruct1.forced
                areaDict1 = catStruct1.areaDict

            if not self.config.doReadParquetTables2:
                catStruct2 = self.readAfwCoaddTables(patchRefList2, repoInfo2, haveForced,
                                                     aliasDictList=aliasDictList)
                unforced2 = catStruct2.unforced
                forced2 = catStruct2.forced

        forcedStr = "forced" if haveForced else "unforced"
        # Set boolean array indicating sources deemed unsuitable for qa
        # analyses.
        badUnforced1 = makeBadArray(unforced1, onlyReadStars=self.config.onlyReadStars)
        badUnforced2 = makeBadArray(unforced2, onlyReadStars=self.config.onlyReadStars)
        if haveForced:
            badForced1 = makeBadArray(forced1, onlyReadStars=self.config.onlyReadStars)
            badForced2 = makeBadArray(forced2, onlyReadStars=self.config.onlyReadStars)

        # Purge the catalogs of flagged sources
        unforced1 = unforced1[~badUnforced1].copy(deep=True)
        unforced2 = unforced2[~badUnforced2].copy(deep=True)
        if haveForced:
            forced1 = forced1[~badForced1].copy(deep=True)
            forced2 = forced2[~badForced2].copy(deep=True)
        else:
            forced1 = unforced1
            forced2 = unforced2
        self.log.info("\nNumber of sources in unforced catalogs: first = {0:d} and second = {1:d}".
                      format(len(unforced1), len(unforced2)))
        self.log.info("\nNumber of sources in forced catalogs: first = {0:d} and second = {1:d}".
                      format(len(forced1), len(forced2)))

        unforced = matchAndJoinCatalogs(unforced1, unforced2, self.matchRadius, matchXy=self.config.matchXy,
                                        camera1=repoInfo1.camera, camera2=repoInfo2.camera)
        forced = matchAndJoinCatalogs(forced1, forced2, self.matchRadius, matchXy=self.config.matchXy,
                                      camera1=repoInfo1.camera, camera2=repoInfo2.camera)
        self.log.info("Number [fraction] of matches (maxDist = {0:.2f}{1:s}) = {2:d} [{3:d}%] (unforced) "
                      "{4:d} [{5:d}%] (forced)".
                      format(self.matchRadius, self.matchRadiusUnitStr,
                             len(unforced), int(100*len(unforced)/len(unforced1)),
                             len(forced), int(100*len(forced)/len(forced1))))

        self.catLabel = "nChild = 0"
        forcedStr = forcedStr + " " + self.catLabel
        schema = getSchema(forced)

        subdir = "patch-" + str(patchList1[0]) if len(patchList1) == 1 else subdir
        # Always highlight points with x-axis flag set (for cases where
        # they do not get explicitly filtered out).
        highlightList = [(self.config.analysis.fluxColumn.replace("_instFlux", "_flag"), 0, "turquoise"), ]
        # Dict of all parameters common to plot* functions
        plotKwargs1 = dict(matchRadius=self.matchRadius, matchRadiusUnitStr=self.matchRadiusUnitStr,
                           zpLabel=self.zpLabel, highlightList=highlightList, uberCalLabel=self.uberCalLabel)
        plotInfoDict = getPlotInfo(repoInfo1)
        plotInfoDict.update(dict(patchList=patchList1, hscRun=hscRun, tractInfo=repoInfo1.tractInfo,
                                 dataId=repoInfo1.dataId, plotType="plotCompareCoadd", subdir=subdir,
                                 hscRun1=repoInfo1.hscRun, hscRun2=repoInfo2.hscRun))

        if self.config.doPlotMags:
            plotList.append(self.plotMags(forced, plotInfoDict, areaDict1, forcedStr=forcedStr,
                                          **plotKwargs1))

        if self.config.doPlotSizes:
            if ("first_base_SdssShape_psf_xx" in schema and "second_base_SdssShape_psf_xx" in schema):
                plotList.append(self.plotSizes(forced, plotInfoDict, areaDict1, forcedStr=forcedStr,
                                               **plotKwargs1))
            else:
                self.log.warn("Cannot run plotSizes: base_SdssShape_psf_xx not in schema")

        if self.config.doApCorrs:
            plotList.append(self.plotApCorrs(unforced, plotInfoDict, areaDict1,
                                             forcedStr="unforced " + self.catLabel, **plotKwargs1))
        if self.config.doPlotCentroids:
            plotList.append(self.plotCentroids(forced, plotInfoDict, areaDict1, forcedStr=forcedStr,
                                               **plotKwargs1))
        if self.config.doPlotStarGalaxy:
            plotList.append(self.plotStarGal(forced, plotInfoDict, areaDict1, forcedStr=forcedStr,
                                             **plotKwargs1))

        self.allStats, self.allStatsHigh = savePlots(plotList, "plotCompareCoadd", repoInfo1.dataId,
                                                     repoInfo1.butler, subdir=subdir)

    def plotMags(self, catalog, plotInfoDict, areaDict, matchRadius=None, matchRadiusUnitStr=None,
                 zpLabel=None, forcedStr=None, fluxToPlotList=None, postFix="",
                 highlightList=None, uberCalLabel=None):
        yield
        schema = getSchema(catalog)
        if not fluxToPlotList:
            fluxToPlotList = self.config.fluxToPlotList
        unitStr = "mmag" if self.config.toMilli else "mag"
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.02*self.unitScale}})
        for col in fluxToPlotList:
            if ("first_" + col + "_instFlux" in schema and "second_" + col + "_instFlux" in schema):
                shortName = "diff_" + col + postFix
                self.log.info("shortName = {:s}".format(shortName))
                yield from Analysis(catalog, MagDiffCompare(col + "_instFlux", unitScale=self.unitScale),
                                    "      Run Comparison: %s mag diff (%s)" %
                                    (fluxToPlotString(col), unitStr), shortName, self.config.analysis,
                                    prefix="first_", qMin=-0.05, qMax=0.05,
                                    errFunc=MagDiffCompareErr(col + "_instFlux", unitScale=self.unitScale),
                                    labeller=OverlapsStarGalaxyLabeller(),
                                    unitScale=self.unitScale,).plotAll(shortName, plotInfoDict, areaDict,
                                                                       self.log, enforcer=enforcer,
                                                                       matchRadius=matchRadius,
                                                                       matchRadiusUnitStr=matchRadiusUnitStr,
                                                                       zpLabel=zpLabel,
                                                                       uberCalLabel=uberCalLabel,
                                                                       forcedStr=forcedStr,
                                                                       highlightList=highlightList)

    def plotCentroids(self, catalog, plotInfoDict, areaDict, matchRadius=None, matchRadiusUnitStr=None,
                      zpLabel=None, forcedStr=None, highlightList=None, uberCalLabel=None):
        yield
        unitStr = "milliPixels" if self.config.toMilli else "pixels"
        distEnforcer = None
        centroidStr1, centroidStr2 = "base_SdssCentroid", "base_SdssCentroid"
        if bool(plotInfoDict["hscRun1"]) ^ bool(plotInfoDict["hscRun2"]):
            if not plotInfoDict["hscRun1"]:
                centroidStr1 = "base_SdssCentroid_Rot"
            if not plotInfoDict["hscRun2"]:
                centroidStr2 = "base_SdssCentroid_Rot"
        plotAllKwargs = dict(matchRadius=matchRadius, matchRadiusUnitStr=matchRadiusUnitStr, zpLabel=zpLabel,
                             forcedStr=forcedStr, uberCalLabel=uberCalLabel)

        shortName = "diff_x"
        self.log.info("shortName = {:s}".format(shortName))
        yield from Analysis(catalog, CentroidDiff("x", centroid1=centroidStr1, centroid2=centroidStr2,
                                                  unitScale=self.unitScale),
                            "Run Comparison: x offset (%s)" % unitStr, shortName,
                            self.config.analysis, prefix="first_",
                            qMin=-0.08, qMax=0.08, errFunc=None,
                            labeller=OverlapsStarGalaxyLabeller()).\
            plotAll(shortName, plotInfoDict, areaDict, self.log, enforcer=distEnforcer, **plotAllKwargs)
        shortName = "diff_y"
        self.log.info("shortName = {:s}".format(shortName))
        yield from Analysis(catalog, CentroidDiff("y", centroid1=centroidStr1, centroid2=centroidStr2,
                                                  unitScale=self.unitScale),
                            "Run Comparison: y offset (%s)" % unitStr, shortName, self.config.analysis,
                            prefix="first_", qMin=-0.08, qMax=0.08, errFunc=None,
                            labeller=OverlapsStarGalaxyLabeller()).plotAll(shortName, plotInfoDict,
                                                                           areaDict, self.log,
                                                                           enforcer=distEnforcer,
                                                                           **plotAllKwargs)

        unitStr = "mas" if self.config.toMilli else "arcsec"
        shortName = "diff_raCosDec"
        self.log.info("shortName = {:s}".format(shortName))
        yield from Analysis(catalog, AstrometryDiff("first_coord_ra", "second_coord_ra",
                                                    declination1="first_coord_dec",
                                                    declination2="second_coord_dec",
                                                    unitScale=self.unitScale),
                            r"   Run Comparison: $\delta_{RA}$ = $\Delta$RA*cos(Dec) (%s)" % unitStr,
                            shortName, self.config.analysisMatches, prefix="first_", qMin=-0.2*matchRadius,
                            qMax=0.2*matchRadius, labeller=OverlapsStarGalaxyLabeller(),
                            unitScale=self.unitScale,
                            ).plotAll(shortName, plotInfoDict, areaDict, self.log, **plotAllKwargs)
        shortName = "diff_ra"
        self.log.info("shortName = {:s}".format(shortName))
        yield from Analysis(catalog, AstrometryDiff("first_coord_ra", "second_coord_ra", declination1=None,
                                                    declination2=None, unitScale=self.unitScale),
                            r"Run Comparison: $\Delta$RA (%s)" % unitStr, shortName,
                            self.config.analysisMatches, prefix="first_", qMin=-0.25*matchRadius,
                            qMax=0.25*matchRadius, labeller=OverlapsStarGalaxyLabeller(),
                            unitScale=self.unitScale,
                            ).plotAll(shortName, plotInfoDict, areaDict, self.log, **plotAllKwargs)
        shortName = "diff_dec"
        self.log.info("shortName = {:s}".format(shortName))
        yield from Analysis(catalog, AstrometryDiff("first_coord_dec", "second_coord_dec",
                                                    unitScale=self.unitScale),
                            r"$\delta_{Dec}$ (%s)" % unitStr, shortName, self.config.analysisMatches,
                            prefix="first_", qMin=-0.3*matchRadius, qMax=0.3*matchRadius,
                            labeller=OverlapsStarGalaxyLabeller(),
                            unitScale=self.unitScale,
                            ).plotAll(shortName, plotInfoDict, areaDict, self.log, **plotAllKwargs)

    def plotFootprint(self, catalog, plotInfoDict, areaDict, matchRadius=None, matchRadiusUnitStr=None,
                      zpLabel=None, forcedStr=None, postFix="", highlightList=None,
                      uberCalLabel=None):
        yield
        enforcer = None
        plotAllKwargs = dict(matchRadius=matchRadius, matchRadiusUnitStr=matchRadiusUnitStr,
                             zpLabel=zpLabel, forcedStr=forcedStr, uberCalLabel=uberCalLabel, postFix=postFix)
        shortName = "diff_footArea"
        col = "base_FootprintArea_value"
        self.log.info("shortName = {:s}".format(shortName))
        yield from Analysis(catalog, FootAreaDiffCompare(col), "  Run Comparison: Footprint Area difference",
                            shortName, self.config.analysis, prefix="first_", qMin=-250, qMax=250,
                            labeller=OverlapsStarGalaxyLabeller()
                            ).plotAll(shortName, plotInfoDict, areaDict, self.log, enforcer=enforcer,
                                      **plotAllKwargs)
        shortName = "diff_footArea_calib_psf_used"
        self.log.info("shortName = {:s}".format(shortName))
        yield from Analysis(catalog, FootAreaDiffCompare(col),
                            "     Run Comparison: Footprint Area diff (psf_used)",
                            shortName, self.config.analysis, prefix="first_", goodKeys=["calib_psf_used"],
                            qMin=-150, qMax=150, labeller=OverlapsStarGalaxyLabeller(),
                            ).plotAll(shortName, plotInfoDict, areaDict, self.log, enforcer=enforcer,
                                      highlightList=highlightList, **plotAllKwargs)

    def plotSizes(self, catalog, plotInfoDict, areaDict, matchRadius=None, matchRadiusUnitStr=None,
                  zpLabel=None, forcedStr=None, highlightList=None, uberCalLabel=None):
        yield
        schema = getSchema(catalog)
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.02*self.unitScale}})
        plotAllKwargs = dict(matchRadius=matchRadius, matchRadiusUnitStr=matchRadiusUnitStr,
                             zpLabel=zpLabel, forcedStr=forcedStr, uberCalLabel=uberCalLabel)
        for col in ["base_PsfFlux"]:
            if ("first_" + col + "_instFlux" in schema and "second_" + col + "_instFlux" in schema):
                # Make comparison plots for all objects and calib_psf_used
                # only objects.
                for goodFlags in [[], ["calib_psf_used"]]:
                    subCatString = " (calib_psf_used)" if "calib_psf_used" in goodFlags else ""
                    shortNameBase = "trace"
                    shortName = (shortNameBase + "_calib_psf_used" if "calib_psf_used" in goodFlags else
                                 shortNameBase)
                    compareCol = "base_SdssShape"
                    self.log.info("shortName = {:s}".format(shortName))
                    yield from Analysis(catalog, TraceSizeCompare(compareCol),
                                        "    SdssShape Trace Radius Diff (%)" + subCatString,
                                        shortName, self.config.analysis, prefix="first_",
                                        goodKeys=goodFlags, qMin=-0.5, qMax=1.5,
                                        labeller=OverlapsStarGalaxyLabeller()).plotAll(shortName,
                                                                                       plotInfoDict, areaDict,
                                                                                       self.log,
                                                                                       enforcer=enforcer,
                                                                                       **plotAllKwargs)

                    shortNameBase = "psfTrace"
                    shortName = (shortNameBase + "_calib_psf_used" if "calib_psf_used" in goodFlags else
                                 shortNameBase)
                    self.log.info("shortName = {:s}".format(shortName))
                    yield from Analysis(catalog, TraceSizeCompare(compareCol + "_psf"),
                                        "       SdssShape PSF Trace Radius Diff (%)" + subCatString,
                                        shortName, self.config.analysis, prefix="first_",
                                        goodKeys=goodFlags, qMin=-1.1, qMax=1.1,
                                        labeller=OverlapsStarGalaxyLabeller(),
                                        ).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                                  enforcer=enforcer, **plotAllKwargs)

                    if "first_ext_shapeHSM_HsmSourceMoments_xx" in schema:
                        shortNameBase = "hsmTrace"
                        shortName = (shortNameBase + "_calib_psf_used" if "calib_psf_used" in goodFlags else
                                     shortNameBase)
                        compareCol = "ext_shapeHSM_HsmSourceMoments"
                        self.log.info("shortName = {:s}".format(shortName))
                        yield from Analysis(catalog, TraceSizeCompare(compareCol),
                                            "   HSM Trace Radius Diff (%)" + subCatString, shortName,
                                            self.config.analysis, prefix="first_",
                                            goodKeys=goodFlags, qMin=-0.5, qMax=1.5,
                                            labeller=OverlapsStarGalaxyLabeller(),
                                            ).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                                      enforcer=enforcer, **plotAllKwargs)
                        shortNameBase = "hsmPsfTrace"
                        shortName = (shortNameBase + "_calib_psf_used" if "calib_psf_used" in goodFlags else
                                     shortNameBase)
                    if "first_ext_shapeHSM_PsfMoments_xx" in schema:
                        compareCol = "ext_shapeHSM_HsmPsfMoments"
                        self.log.info("shortName = {:s}".format(shortName))
                        yield from Analysis(catalog, TraceSizeCompare(compareCol),
                                            "      HSM PSF Trace Radius Diff (%)" + subCatString,
                                            shortName, self.config.analysis, prefix="first_",
                                            goodKeys=goodFlags, qMin=-1.1, qMax=1.1,
                                            labeller=OverlapsStarGalaxyLabeller(),
                                            ).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                                      enforcer=enforcer, **plotAllKwargs)

                compareCol = "base_SdssShape"
                shortName = "sdssXx"
                self.log.info("shortName = {:s}".format(shortName))
                yield from Analysis(catalog, PercentDiff(compareCol + "_xx"), "SdssShape xx Moment Diff (%)",
                                    shortName, self.config.analysis, prefix="first_",
                                    qMin=-0.5, qMax=1.5, labeller=OverlapsStarGalaxyLabeller(),
                                    ).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                              enforcer=enforcer, **plotAllKwargs)
                shortName = "sdssYy"
                self.log.info("shortName = {:s}".format(shortName))
                yield from Analysis(catalog, PercentDiff(compareCol + "_yy"), "SdssShape yy Moment Diff (%)",
                                    shortName, self.config.analysis, prefix="first_",
                                    qMin=-0.5, qMax=1.5, labeller=OverlapsStarGalaxyLabeller(),
                                    ).plotAll(shortName, plotInfoDict, areaDict, self.log, enforcer=enforcer,
                                              **plotAllKwargs)

    def plotStarGal(self, catalog, plotInfoDict, areaDict, matchRadius=None, matchRadiusUnitStr=None,
                    zpLabel=None, forcedStr=None, highlightList=None, uberCalLabel=None):
        yield
        schema = getSchema(catalog)
        enforcer = None
        plotAllKwargs = dict(matchRadius=matchRadius, matchRadiusUnitStr=matchRadiusUnitStr, zpLabel=zpLabel,
                             forcedStr=forcedStr, highlightList=highlightList, uberCalLabel=uberCalLabel)
        baseCol = "ext_shapeHSM_HsmShapeRegauss"
        col = baseCol + "_resolution"
        if "first_" + col in schema:
            shortName = "diff_resolution"
            self.log.info("shortName = {:s}".format(shortName))
            yield from Analysis(catalog, PercentDiff(col),
                                "           Run Comparison: HsmRegauss Resolution (% diff)",
                                shortName, self.config.analysis, prefix="first_",
                                qMin=-0.2, qMax=0.2, labeller=OverlapsStarGalaxyLabeller()
                                ).plotAll(shortName, plotInfoDict, areaDict, self.log, enforcer=enforcer,
                                          **plotAllKwargs)
        col = baseCol + "_e1"
        if "first_" + col in schema:
            shortName = "diff_HsmShapeRegauss_e1"
            self.log.info("shortName = {:s}".format(shortName))
            yield from Analysis(catalog, PercentDiff(col), "    Run Comparison: HsmRegauss e1 (% diff)",
                                shortName, self.config.analysis, prefix="first_",
                                qMin=-0.2, qMax=0.2, labeller=OverlapsStarGalaxyLabeller()
                                ).plotAll(shortName, plotInfoDict, areaDict, self.log, enforcer=enforcer,
                                          **plotAllKwargs)
        col = baseCol + "_e2"
        if "first_" + col in schema:
            shortName = "diff_HsmShapeRegauss_e2"
            self.log.info("shortName = {:s}".format(shortName))
            yield from Analysis(catalog, PercentDiff(col), "    Run Comparison: HsmRegauss e2 (% diff)",
                                shortName, self.config.analysis, prefix="first_",
                                qMin=-0.2, qMax=0.2, labeller=OverlapsStarGalaxyLabeller()
                                ).plotAll(shortName, plotInfoDict, areaDict, self.log, enforcer=enforcer,
                                          **plotAllKwargs)

    def plotApCorrs(self, catalog, plotInfoDict, areaDict, matchRadius=None, matchRadiusUnitStr=None,
                    zpLabel=None, forcedStr=None, fluxToPlotList=None, highlightList=None, uberCalLabel=None):
        yield
        schema = getSchema(catalog)
        if not fluxToPlotList:
            fluxToPlotList = self.config.fluxToPlotList
        unitStr = "mmag" if self.config.toMilli else "mag"
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.02*self.unitScale}})
        for col in fluxToPlotList:
            if "first_" + col + "_apCorr" in schema and "second_" + col + "_apCorr" in schema:
                shortName = "diff_" + col + "_apCorr"
                self.log.info("shortName = {:s}".format(shortName))
                # apCorrs in coadds can be all nan if they weren't run in sfm,
                # so add a check for valid data but here so we don't encounter
                # the fatal error in Analysis.
                if (len(np.where(np.isfinite(catalog["first_" + col + "_apCorr"]))[0]) > 0
                        and len(np.where(np.isfinite(catalog["second_" + col + "_apCorr"]))[0]) > 0):
                    yield from Analysis(catalog, MagDiffCompare(col + "_apCorr", unitScale=self.unitScale),
                                        "  Run Comparison: %s apCorr diff (%s)" %
                                        (fluxToPlotString(col), unitStr),
                                        shortName, self.config.analysis, prefix="first_", qMin=-0.025,
                                        qMax=0.025, labeller=OverlapsStarGalaxyLabeller(),
                                        unitScale=self.unitScale
                                        ).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                                  enforcer=enforcer, matchRadius=matchRadius,
                                                  matchRadiusUnitStr=matchRadiusUnitStr,
                                                  zpLabel=zpLabel, forcedStr=forcedStr,
                                                  highlightList=highlightList
                                                  + [(col + "_flag_apCorr", 0, "lime"), ],
                                                  uberCalLabel=uberCalLabel)
                else:
                    self.log.warn("No valid data points for shortName = {:s}.  Skipping...".format(shortName))

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None

    def _getEupsVersionsName(self):
        return None
