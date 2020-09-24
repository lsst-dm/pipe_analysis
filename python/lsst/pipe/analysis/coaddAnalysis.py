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
np.seterr(all="ignore")  # noqa E402
import functools

from collections import defaultdict

from lsst.daf.persistence.butler import Butler
from lsst.pex.config import (Config, Field, ConfigField, ListField, DictField, ConfigDictField,
                             ConfigurableField)
from lsst.pipe.base import CmdLineTask, ArgumentParser, TaskRunner, TaskError, Struct
from lsst.pipe.drivers.utils import TractDataIdContainer
from lsst.afw.table.catalogMatches import matchesToCatalog
from lsst.meas.astrom import AstrometryConfig
from lsst.pipe.tasks.colorterms import Colorterm, ColortermLibrary

from lsst.meas.algorithms import LoadIndexedReferenceObjectsTask

from .analysis import AnalysisConfig, Analysis
from .utils import (Enforcer, MagDiff, MagDiffMatches, MagDiffCompare,
                    AstrometryDiff, TraceSize, PsfTraceSizeDiff, TraceSizeCompare, PercentDiff,
                    E1Resids, E2Resids, E1ResidsHsmRegauss, E2ResidsHsmRegauss, FootNpixDiffCompare,
                    MagDiffCompareErr, CentroidDiff, deconvMom, deconvMomStarGal,
                    concatenateCatalogs, joinMatches, checkPatchOverlap,
                    addColumnsToSchema, addApertureFluxesHSC, addFpPoint,
                    addFootprintNPix, makeBadArray, addIntFloatOrStrColumn,
                    calibrateCoaddSourceCatalog, backoutApCorr, matchNanojanskyToAB,
                    fluxToPlotString, andCatalog, writeParquet, getRepoInfo, setAliasMaps,
                    addPreComputedColumns, computeMeanOfFrac, savePlots, updateVerifyJob)
from .plotUtils import (CosmosLabeller, AllLabeller, StarGalaxyLabeller, OverlapsStarGalaxyLabeller,
                        MatchesStarGalaxyLabeller, determineExternalCalLabel, getPlotInfo)

import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.verify as verify

__all__ = ["CoaddAnalysisConfig", "CoaddAnalysisRunner", "CoaddAnalysisTask", "CompareCoaddAnalysisConfig",
           "CompareCoaddAnalysisRunner", "CompareCoaddAnalysisTask"]

NANOJANSKYS_PER_AB_FLUX = (0*u.ABmag).to_value(u.nJy)


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
    doPlotFootprintNpix = Field(dtype=bool, default=True, doc=("Plot histogram of footprint nPix? "
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
                                      "copied from the *_meas catalogs into the *_forced_src catalogs.")
    # We want the following to come from the *_ref catalogs as they reflect
    # the forced measurement states.
    columnsToCopyFromRef = ListField(dtype=str,
                                     default=["detect_", "merge_peak_", "merge_measurement_", ],
                                     doc="List of string \"prefixes\" to identify the columns to copy.  "
                                     "All columns with names that start with one of these strings will be "
                                     "copied from the *_ref catalogs into the *_forced_src catalogs.")
    flagsToAlias = DictField(keytype=str, itemtype=str,
                             default={"calib_psf_used": "calib_psfUsed",
                                      "calib_psf_candidate": "calib_psfCandidate",
                                      "calib_astrometry_used": "calib_astrometryUsed"},
                             doc=("List of flags to alias to old, pre-RFC-498, names for backwards "
                                  "compatibility with old processings"))
    doWriteParquetTables = Field(dtype=bool, default=True,
                                 doc=("Write out Parquet tables (for subsequent interactive analysis)?"
                                      "\nNOTE: if True but fastparquet package is unavailable, a warning is "
                                      "issued and table writing is skipped."))
    writeParquetOnly = Field(dtype=bool, default=False,
                             doc="Only write out Parquet tables (i.e. do not produce any plots)?")
    hasFakes = Field(dtype=bool, default=False, doc="Include the analysis of the added fake sources?")

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
            self.doPlotFootprintNpix = False
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

        # Partition all inputs by tract,filter
        FilterRefsDict = functools.partial(defaultdict, list)  # Dict for filter-->dataRefs
        # Make sure the actual input files requested exist (i.e. do not follow
        # the parent chain).  First check for forced catalogs.  Break out of
        # datasets loop if forced catalogs were found, otherwise continue
        # search for existence of unforced (i.e. meas) catalogs.
        for dataset in ["forced_src", "meas"]:
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
        self.matchControl = afwTable.MatchControl()
        self.matchControl.findOnlyClosest = True
        self.matchControl.symmetricMatch = False

        self.verifyJob = verify.Job.load_metrics_package(subset="pipe_analysis")

    def runDataRef(self, patchRefList, subdir="", cosmos=None):
        plotList = []

        haveForced = False  # do forced datasets exits (may not for single band datasets)
        dataset = "Coadd_forced_src"
        # Explicit input file was checked in CoaddAnalysisRunner, so a check
        # on datasetExists is sufficient here (modulo the case where a forced
        # dataset exists higher up the parent tree than the specified input,
        # but does not exist in the input directory as the former will be
        # found).
        if patchRefList[0].datasetExists(self.config.coaddName + dataset):
            haveForced = True
        forcedStr = "forced" if haveForced else "unforced"
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
        plotInfoDict = getPlotInfo(repoInfo)
        subdir = "patch-" + str(patchList[0]) if len(patchList) == 1 else subdir
        repoInfo.dataId["subdir"] = "/" + subdir
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
        highlightList = [(self.config.analysis.fluxColumn.replace("_instFlux", "_flag"), 0, "turquoise"), ]
        # Dict of all parameters common to plot* functions
        plotKwargs = dict(zpLabel=self.zpLabel, uberCalLabel=self.uberCalLabel)

        if any(doPlot for doPlot in
               [self.config.doPlotOverlaps, self.config.doPlotCompareUnforced,
                self.config.doPlotPsfFluxSnHists, self.config.doPlotSkyObjects,
                self.config.doPlotSkyObjectsSky, self.config.doPlotFootprintNpix,
                self.config.doPlotMags, self.config.doPlotStarGalaxy,
                self.config.doPlotRhoStatistics, cosmos, self.config.externalCatalogs,
                self.config.doWriteParquetTables]) and not self.config.plotMatchesOnly:
            if haveForced:
                forcedCatStruct = self.readCatalogs(patchRefList, self.config.coaddName
                                                    + "Coadd_forced_src", repoInfo)
                forced = forcedCatStruct.catalog
                areaDict = forcedCatStruct.areaDict
                forced = self.calibrateCatalogs(forced, wcs=repoInfo.wcs)

            unforcedCatStruct = self.readCatalogs(patchRefList, self.config.coaddName
                                                  + "Coadd_meas", repoInfo)
            unforced = unforcedCatStruct.catalog
            if not haveForced:
                areaDict = unforcedCatStruct.areaDict
            unforced = self.calibrateCatalogs(unforced, wcs=repoInfo.wcs)
            plotKwargs.update(dict(zpLabel=self.zpLabel))
            if haveForced:
                # Copy over some fields from _ref and _meas catalogs to
                # _forced_src catalog.
                refCat = self.readCatalogs(patchRefList, self.config.coaddName + "Coadd_ref",
                                           repoInfo).catalog
                if len(forced) != len(refCat):
                    raise RuntimeError(("Lengths of forced (N = {0:d}) and ref (N = {0:d}) cats don't match").
                                       format(len(forced), len(refCat)))
                refColList = [s for s in refCat.schema.getNames() if
                              s.startswith(tuple(self.config.columnsToCopyFromRef))]
                refColsToCopy = [col for col in refColList if col not in forced.schema
                                 and col in refCat.schema and not
                                 (repoInfo.hscRun and col == "slot_Centroid_flag")]
                forced = addColumnsToSchema(refCat, forced, refColsToCopy)
                measColList = [s for s in unforced.schema.getNames() if
                               s.startswith(tuple(self.config.columnsToCopyFromMeas))]
                measColsToCopy = [col for col in measColList if col not in forced.schema
                                  and col in unforced.schema and not
                                  (repoInfo.hscRun and col == "slot_Centroid_flag")]
                forced = addColumnsToSchema(unforced, forced, measColsToCopy)

            # Set some aliases for differing schema naming conventions
            coaddList = [unforced, ]
            if haveForced:
                coaddList += [forced]
            for cat in coaddList:
                cat = setAliasMaps(cat, aliasDictList)

            if self.config.doPlotFootprintNpix:
                unforced = addFootprintNPix(unforced, fromCat=unforced)
                if haveForced:
                    forced = addFootprintNPix(forced, fromCat=unforced)

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
            # sources.
            if self.config.doPlotOverlaps:
                # Determine if any patches in the patchList actually overlap
                overlappingPatches = checkPatchOverlap(plotInfoDict["patchList"], plotInfoDict["tractInfo"])
                if not overlappingPatches:
                    self.log.info("No overlapping patches...skipping overlap plots")
                else:
                    if haveForced:
                        forcedOverlaps = self.overlaps(forced)
                        if forcedOverlaps:
                            plotList.append(self.plotOverlaps(forcedOverlaps, plotInfoDict, areaDict,
                                                              matchRadius=self.config.matchOverlapRadius,
                                                              matchRadiusUnitStr="\"",
                                                              forcedStr=forcedStr, postFix="_forced",
                                                              fluxToPlotList=["modelfit_CModel", ],
                                                              **plotKwargs))
                            self.log.info("Number of forced overlap objects matched = {:d}".
                                          format(len(forcedOverlaps)))
                    unforcedOverlaps = self.overlaps(unforced)
                    if unforcedOverlaps:
                        plotList.append(self.plotOverlaps(unforcedOverlaps, plotInfoDict, areaDict,
                                                          matchRadius=self.config.matchOverlapRadius,
                                                          matchRadiusUnitStr="\"",
                                                          forcedStr="unforced", postFix="_unforced",
                                                          fluxToPlotList=["modelfit_CModel", ],
                                                          highlightList=highlightList, **plotKwargs))
                        self.log.info("Number of unforced overlap objects matched = {:d}".
                                      format(len(unforcedOverlaps)))

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
                self.catLabel = "nChild = 0"
                forcedStr = forcedStr + " " + self.catLabel
                if haveForced:
                    self.log.info("\nNumber of sources in catalogs: unforced = {0:d} and forced = {1:d}".
                                  format(len(unforced), len(forced)))
                else:
                    self.log.info("\nNumber of sources in catalog: unforced = {0:d}".format(len(unforced)))

            if self.config.doPlotPsfFluxSnHists:
                plotList.append(self.plotPsfFluxSnHists(unforced, "base_PsfFlux_cal", plotInfoDict, areaDict,
                                                        forcedStr="unforced " + self.catLabel, **plotKwargs))
            if self.config.doPlotSkyObjects:
                plotList.append(self.plotSkyObjects(skyObjCat, "skyObjects", plotInfoDict, areaDict,
                                                    forcedStr="unforced"))
            if self.config.doPlotSkyObjectsSky:
                plotList.append(self.plotSkyObjectsSky(skyObjCatAll, "skyObjects", plotInfoDict,
                                                       forcedStr="unforced", alpha=0.7,
                                                       doPlotTractImage=True, doPlotPatchOutline=True,
                                                       sizeFactor=3.0, maxDiamPix=1000))

            if self.config.doPlotFootprintNpix:
                plotList.append(self.plotFootprintHist(forced, "footNpix", plotInfoDict, **plotKwargs))
                plotList.append(self.plotFootprint(forced, plotInfoDict, areaDict, forcedStr=forcedStr,
                                                   **plotKwargs))

            if self.config.doPlotRhoStatistics:
                plotList.append(self.plotRhoStatistics(unforced, plotInfoDict,
                                                       forcedStr="unforced " + self.catLabel, **plotKwargs))

            if self.config.doPlotQuiver:
                plotList.append(self.plotQuiver(unforced, "ellipResids", plotInfoDict, areaDict,
                                                forcedStr="unforced " + self.catLabel, scale=2, **plotKwargs))

            if self.config.doPlotInputCounts:
                plotList.append(self.plotInputCounts(unforced, "inputCounts", plotInfoDict,
                                                     forcedStr="unforced " + self.catLabel, alpha=0.5,
                                                     doPlotPatchOutline=True, sizeFactor=5.0,
                                                     maxDiamPix=1000, **plotKwargs))

            plotKwargs.update(dict(highlightList=highlightList))
            if self.config.doPlotMags:
                plotList.append(self.plotMags(unforced, plotInfoDict, areaDict,
                                              forcedStr="unforced " + self.catLabel,
                                              postFix="_unforced", **plotKwargs))
                if haveForced:
                    plotKwargs.update(dict(highlightList=highlightList
                                           + [("merge_measurement_" + repoInfo.genericFilterName, 0,
                                               "yellow")]))

                    plotList.append(self.plotMags(forced, plotInfoDict, areaDict, forcedStr=forcedStr,
                                                  postFix="_forced", **plotKwargs))
                    plotKwargs.update(dict(highlightList=highlightList))
            if self.config.doPlotStarGalaxy:
                if "ext_shapeHSM_HsmSourceMoments_xx" in unforced.schema:
                    plotList.append(self.plotStarGal(unforced, plotInfoDict, areaDict,
                                                     forcedStr="unforced " + self.catLabel, **plotKwargs))
                else:
                    self.log.warn("Cannot run plotStarGal: ext_shapeHSM_HsmSourceMoments_xx not "
                                  "in forced.schema")

            if self.config.doPlotSizes:
                if all(ss in unforced.schema for ss in ["base_SdssShape_psf_xx", "calib_psf_used"]):
                    plotList.append(self.plotSizes(unforced, plotInfoDict, areaDict,
                                                   forcedStr="unforced " + self.catLabel,
                                                   postFix="_unforced", **plotKwargs))
                else:
                    self.log.warn("Cannot run plotSizes: base_SdssShape_psf_xx and/or calib_psf_used "
                                  "not in unforced.schema")
                if haveForced:
                    if all(ss in forced.schema for ss in ["base_SdssShape_psf_xx", "calib_psf_used"]):
                        plotList.append(self.plotSizes(forced, plotInfoDict, areaDict,
                                                       forcedStr=forcedStr, **plotKwargs))
                    else:
                        self.log.warn("Cannot run plotSizes: base_SdssShape_psf_xx and/or calib_psf_used "
                                      "not in forced.schema")
            if cosmos:
                plotList.append(self.plotCosmos(forced, plotInfoDict, areaDict, cosmos, repoInfo.dataId))

        matchAreaDict = {}
        if self.config.doPlotMatches or self.config.doWriteParquetTables:
            if haveForced:
                matches, matchAreaDict = self.readSrcMatches(patchRefList, self.config.coaddName
                                                             + "Coadd_forced_src",
                                                             hscRun=repoInfo.hscRun, wcs=repoInfo.wcs,
                                                             aliasDictList=aliasDictList)
                qaTableSuffix = "_forced"
                if self.config.doWriteParquetTables:
                    matchesDataRef = repoInfo.butler.dataRef("analysisMatchFullRefCoaddTable" + qaTableSuffix,
                                                             dataId=repoInfo.dataId)
                    writeParquet(matchesDataRef, matches, badArray=None, prefix="src_")

            matches, matchAreaDict = self.readSrcMatches(patchRefList, self.config.coaddName + "Coadd_meas",
                                                         hscRun=repoInfo.hscRun, wcs=repoInfo.wcs,
                                                         aliasDictList=aliasDictList)
            qaTableSuffix = "_unforced"

            if self.config.doWriteParquetTables:
                matchesDataRef = repoInfo.butler.dataRef("analysisMatchFullRefCoaddTable" + qaTableSuffix,
                                                         dataId=repoInfo.dataId)
                writeParquet(matchesDataRef, matches, badArray=None, prefix="src_")
                if self.config.writeParquetOnly:
                    self.log.info("Exiting after writing Parquet tables.  No plots generated.")
                    return

            plotKwargs.update(dict(zpLabel=self.zpLabel))
            matchHighlightList = [("src_" + self.config.analysis.fluxColumn.replace("_instFlux", "_flag"), 0,
                                   "turquoise"), ]
            plotKwargs.update(dict(highlightList=matchHighlightList))
            plotList.append(self.plotMatches(matches, plotInfoDict, matchAreaDict, forcedStr=forcedStr,
                                             **plotKwargs))

        for cat in self.config.externalCatalogs:
            with andCatalog(cat):
                matches = self.matchCatalog(forced, repoInfo.filterName, self.config.externalCatalogs[cat])
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

    def readCatalogs(self, patchRefList, dataset, repoInfo, fakeCat=None, raFakesCol="raJ2000",
                     decFakesCol="decJ2000"):
        """Read in and concatenate catalogs of type dataset in lists of
        data references.

        An extra column indicating the patch ID is added to each catalog before
        appending them all to a single list.  This is useful for any subsequent
        QA analysis using the persisted parquet files.

        Parameters
        ----------
        patchRefList : `list` of
                       `lsst.daf.persistence.butlerSubset.ButlerDataRef`
            A list of butler data references whose catalogs of dataset type are
            to be read in.
        dataset : `str`
            Name of the catalog dataset to be read in.
        repoInfo : `lsst.pipe.base.Struct`
            A struct containing relevant information about the repository under
            study.  Elements used here include the dataset names for any
            external calibrations to be applied.
        fakeCat : `pandas.core.frame.DataFrame`, optional
            Catalog of fake sources, used if hasFakes is `True` in which case a
            column (onPatch) is added with the patch number if the fake source
            overlaps a ccd and `np.nan` if it does not.
        raFakesCol : `str`, optional
            The name of the RA column to use from the fakes catalogue.
        decFakesCol : `str`, optional
            The name of the Dec column to use from the fakes catalogue.

        Raises
        ------
        TaskError
            If no data is read in for the dataRefList.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            A struct with attributes:
            ``commonZpCatalog``
                The concatenated common zeropoint calibrated catalog
                (`lsst.afw.table.SourceCatalog`).
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
        catList = []
        areaDict = {}
        for patchRef in patchRefList:
            if patchRef.datasetExists(dataset):
                cat = patchRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_HEAVY_FOOTPRINTS)
                cat = addIntFloatOrStrColumn(cat, patchRef.dataId["patch"], "patchId",
                                             "Patch on which source was detected")
                if self.config.hasFakes:
                    fname = repoInfo.butler.getUri("fakes_deepCoadd_calexp", patchRef.dataId)
                else:
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

                if self.config.hasFakes:
                    # Check which fake sources fall in the patch
                    cornerRas = [px.asRadians() for (px, py) in patchCorners]
                    cornerDecs = [py.asRadians() for (px, py) in patchCorners]
                    possOnPatch = ((fakeCat[raFakesCol].values > np.min(cornerRas))
                                   & (fakeCat[raFakesCol].values < np.max(cornerRas))
                                   & (fakeCat[decFakesCol].values > np.min(cornerDecs))
                                   & (fakeCat[decFakesCol].values < np.max(cornerDecs)))

                    validPatchPolygon = mask.getInfo().getValidPolygon()

                    def isInPolygon(df):
                        skyCoord = geom.SpherePoint(df.raFakesCol, df.decFakesCol, geom.radians)
                        pixCoord = wcs.skyToPixel(skyCoord)
                        return validPatchPolygon.contains(pixCoord)

                    fakeCat["isOnPatch"] = fakeCat[possOnPatch].apply(isInPolygon, axis=1)
                    fakeCat.loc[fakeCat["isOnPatch"], "onPatch"] = patchRef.dataId["patch"]

                catList.append(cat)
        if not catList:
            raise TaskError("No catalogs read: %s" % ([patchRef.dataId for patchRef in patchRefList]))

        if not self.config.hasFakes:
            fakeCat = None

        return Struct(catalog=concatenateCatalogs(catList), areaDict=areaDict, fakeCat=fakeCat)

    def readSrcMatches(self, dataRefList, dataset, hscRun=None, wcs=None, aliasDictList=None):
        catList = []
        matchAreaDict = {}
        for dataRef in dataRefList:
            if not dataRef.datasetExists(dataset):
                self.log.info("Dataset does not exist: {0:r}, {1:s}".format(dataRef.dataId, dataset))
                continue
            butler = dataRef.getButler()

            # Generate unnormalized match list (from normalized persisted one)
            # with joinMatchListWithCatal (which requires a refObjLoader to be
            # initialized).
            catalog = dataRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS)
            catalog = addIntFloatOrStrColumn(catalog, dataRef.dataId["patch"], "patchId",
                                             "Patch on which source was detected")
            # Set some aliases for differing schema naming conventions
            if aliasDictList:
                catalog = setAliasMaps(catalog, aliasDictList)
            needToAddColumns = any(ss not in catalog.schema for ss in
                                   list(self.config.columnsToCopyFromMeas)
                                   + list(self.config.columnsToCopyFromRef))
            if dataset != self.config.coaddName + "Coadd_meas" and needToAddColumns:
                # Copy over some fields from _ref and _meas catalogs to
                # _forced_src catalog.
                refCat = dataRef.get(self.config.coaddName + "Coadd_ref", immediate=True,
                                     flags=afwTable.SOURCE_IO_NO_FOOTPRINTS)
                unforced = dataRef.get(self.config.coaddName + "Coadd_meas", immediate=True,
                                       flags=afwTable.SOURCE_IO_NO_FOOTPRINTS)
                if len(catalog) != len(refCat):
                    raise RuntimeError(("Lengths of forced (N = {0:d}) and ref (N = {0:d}) cats don't match").
                                       format(len(catalog), len(refCat)))
                refColList = [s for s in refCat.schema.getNames() if
                              s.startswith(tuple(self.config.columnsToCopyFromRef))]
                refColsToCopy = [col for col in refColList if col not in catalog.schema
                                 and col in refCat.schema
                                 and not (hscRun and col == "slot_Centroid_flag")]
                catalog = addColumnsToSchema(refCat, catalog, refColsToCopy)
                measColList = [s for s in unforced.schema.getNames() if
                               s.startswith(tuple(self.config.columnsToCopyFromMeas))]
                measColsToCopy = [col for col in measColList if col not in catalog.schema
                                  and col in unforced.schema
                                  and not (hscRun and col == "slot_Centroid_flag")]
                catalog = addColumnsToSchema(unforced, catalog, measColsToCopy)
                if aliasDictList:
                    catalog = setAliasMaps(catalog, aliasDictList)

            # Set boolean array indicating sources deemed unsuitable for qa
            # analyses.
            transCentFlag = "base_TransformedCentroid_flag"
            badFlagList = [transCentFlag, ] if transCentFlag in catalog.schema else ["slot_Centroid_flag", ]
            bad = makeBadArray(catalog, flagList=badFlagList, onlyReadStars=self.config.onlyReadStars)
            catalog = self.calibrateCatalogs(catalog, wcs=wcs)

            fname = butler.getUri(self.config.coaddName + "Coadd_calexp", dataRef.dataId)
            reader = afwImage.ExposureFitsReader(fname)
            if wcs is None:
                wcs = reader.readWcs()
            patchCorners = wcs.pixelToSky(geom.Box2D(reader.readBBox()).getCorners())
            matchAreaDict["corners_" + str(dataRef.dataId["patch"])] = patchCorners

            if dataset.startswith("deepCoadd_"):
                packedMatches = butler.get("deepCoadd_measMatch", dataRef.dataId)
            else:
                packedMatches = butler.get(dataset + "Match", dataRef.dataId)

            # Purge the match list of sources flagged in the catalog
            badIds = catalog["id"][bad]
            badMatch = np.zeros(len(packedMatches), dtype=bool)
            for iMat, iMatch in enumerate(packedMatches):
                if iMatch["second"] in badIds:
                    badMatch[iMat] = True

            packedMatches = packedMatches[~badMatch].copy(deep=True)
            if not packedMatches:
                self.log.warn("No good matches for %s" % (dataRef.dataId,))
                continue
            # The reference object loader grows the bbox by the config
            # parameter pixelMargin.  This is set to 50 by default but is not
            # reflected by the radius parameter set in the metadata, so some
            # matches may reside outside the circle searched within this
            # radius.  Thus, increase the radius set in the metadata fed into
            # joinMatchListWithCatalog() to accommodate.
            matchmeta = packedMatches.table.getMetadata()
            rad = matchmeta.getDouble("RADIUS")
            matchmeta.setDouble("RADIUS", rad*1.05, "field radius in degrees, approximate, padded")
            refObjLoader = self.config.refObjLoader.apply(butler=butler)
            matches = refObjLoader.joinMatchListWithCatalog(packedMatches, catalog)
            if not hasattr(matches[0].first, "schema"):
                raise RuntimeError("Unable to unpack matches.  "
                                   "Do you have the correct reference catalog setup?")
            # LSST reads in reference catalogs with flux in "nanojanskys", so
            # must convert to AB.
            matches = matchNanojanskyToAB(matches)
            if hscRun and self.config.doAddAperFluxHsc:
                addApertureFluxesHSC(matches, prefix="second_")

            if not matches:
                self.log.warn("No matches for %s" % (dataRef.dataId,))
                continue

            # Set the alias maps for the matches sources (i.e. the .second
            # attribute schema for each match).
            if aliasDictList:
                for mm in matches:
                    mm.second = setAliasMaps(mm.second, aliasDictList)

            matchMeta = butler.get(dataset, dataRef.dataId,
                                   flags=afwTable.SOURCE_IO_NO_FOOTPRINTS).getTable().getMetadata()
            catalog = matchesToCatalog(matches, matchMeta)
            # Compute Focal Plane coordinates for each source if not already
            # there.
            if self.config.analysisMatches.doPlotFP:
                if "src_base_FPPosition_x" not in catalog.schema and "src_focalplane_x" not in catalog.schema:
                    det = butler.get("calexp_detector", dataRef.dataId)
                    catalog = addFpPoint(det, catalog, prefix="src_")
            # Optionally backout aperture corrections
            if self.config.doBackoutApCorr:
                catalog = backoutApCorr(catalog)
            # Set the alias maps for the matched catalog sources
            if aliasDictList:
                catalog = setAliasMaps(catalog, aliasDictList, prefix="src_")

            catList.append(catalog)

        if not catList:
            raise TaskError("No matches read: %s" % ([dataRef.dataId for dataRef in dataRefList]))

        return concatenateCatalogs(catList), matchAreaDict

    def calibrateCatalogs(self, catalog, wcs=None):
        self.zpLabel = "common (" + str(self.config.analysis.coaddZp) + ")"
        # My persisted catalogs in lauren/LSST/DM-6816new all have nan for RA
        # and Dec (see DM-9556).
        if np.all(np.isnan(catalog["coord_ra"])):
            if wcs is None:
                self.log.warn("Bad RA, Dec entries but can't update because wcs is None")
            else:
                for src in catalog:
                    src.updateCoord(wcs)
        # Optionally backout aperture corrections
        if self.config.doBackoutApCorr:
            catalog = backoutApCorr(catalog)
        calibrated = calibrateCoaddSourceCatalog(catalog, self.config.analysis.coaddZp)
        return calibrated

    def plotMags(self, catalog, plotInfoDict, areaDict, matchRadius=None,
                 matchRadiusUnitStr=None, zpLabel=None, forcedStr=None, fluxToPlotList=None,
                 postFix="", flagsCat=None, highlightList=None, uberCalLabel=None):
        yield
        if not fluxToPlotList:
            fluxToPlotList = self.config.fluxToPlotList
        unitStr = "mmag" if self.config.toMilli else "mag"
        enforcer = Enforcer(requireLess={"star": {"stdev": 0.02*self.unitScale}})
        for col in fluxToPlotList:
            if col + "_instFlux" in catalog.schema:
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
                    flagColors = ["yellow", "greenyellow", "lime", "aquamarine", "orange",
                                  "fuchsia", "gold", "lightseagreen"]
                    for i, flagName in enumerate([col + "_flag", ] + list(self.config.analysis.flags)):
                        if not any(flagName in highlight for highlight in calibHighlightList):
                            calibHighlightList += [(flagName, 0, flagColors[i%len(flagColors)]), ]
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
        unitStr = " (milli)" if self.config.toMilli else ""
        plotAllKwargs = dict(matchRadius=matchRadius, zpLabel=zpLabel, forcedStr=forcedStr,
                             uberCalLabel=uberCalLabel, highlightList=highlightList)
        calibHighlightList0 = None
        for col in ["base_PsfFlux", ]:
            if col + "_instFlux" in catalog.schema:
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
                if "ext_shapeHSM_HsmSourceMoments_xx" in catalog.schema:
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
                if "ext_shapeHSM_HsmSourceMoments_xx" in catalog.schema:
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

            if col + "_instFlux" in catalog.schema:
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

                if "ext_shapeHSM_HsmSourceMoments_xx" in catalog.schema:
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
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.02*self.unitScale}})
        for col in ["base_SdssCentroid_x", "base_SdssCentroid_y"]:
            if col in catalog.schema:
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
        enforcer = None
        plotAllKwargs = dict(matchRadius=matchRadius, zpLabel=zpLabel, forcedStr=forcedStr,
                             uberCalLabel=uberCalLabel, highlightList=highlightList)
        if "calib_psf_used" in catalog.schema:
            shortName = "footNpix_calib_psf_used"
            self.log.info("shortName = {:s}".format(shortName))

            yield from self.AnalysisClass(catalog, catalog["base_Footprint_nPix"], "%s" % shortName,
                                          shortName, self.config.analysis, goodKeys=["calib_psf_used"],
                                          qMin=-100, qMax=2000, labeller=StarGalaxyLabeller(),
                                          ).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                                    plotRunStats=plotRunStats, **plotAllKwargs)
        shortName = "footNpix"
        self.log.info("shortName = {:s}".format(shortName))
        yield from self.AnalysisClass(catalog, catalog["base_Footprint_nPix"], "%s" % shortName, shortName,
                                      self.config.analysis, flags=["base_Footprint_nPix_flag"],
                                      qMin=0, qMax=3000, labeller=StarGalaxyLabeller()
                                      ).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                                enforcer=enforcer, plotRunStats=plotRunStats, **plotAllKwargs)

    def plotFootprintHist(self, catalog, description, plotInfoDict, matchRadius=None, zpLabel=None,
                          postFix="", flagsCat=None, uberCalLabel=None):
        yield
        stats = None
        shortName = "footNpix"
        self.log.info("shortName = {:s}".format(shortName + "Hist"))
        yield from self.AnalysisClass(catalog, catalog["base_Footprint_nPix"], "%s" % shortName, shortName,
                                      self.config.analysis, flags=["base_Footprint_nPix_flag"], qMin=0,
                                      qMax=3000, labeller=StarGalaxyLabeller()
                                      ).plotHistogram(description, plotInfoDict, stats=stats,
                                                      matchRadius=matchRadius, zpLabel=zpLabel,
                                                      filterStr=plotInfoDict["filter"],
                                                      uberCalLabel=uberCalLabel)

    def plotPsfFluxSnHists(self, catalog, description, plotInfoDict, areaDict, matchRadius=None,
                           zpLabel=None, forcedStr=None, uberCalLabel=None, postFix="",
                           logPlot=True, density=True, cumulative=-1):
        yield
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
        if "base_InputCount_value" in catalog.schema:
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

        if "ext_shapeHSM_HsmShapeRegauss_resolution" in catalog.schema:
            shortName = "resolution"
            self.log.info("shortName = {:s}".format(shortName))
            yield from self.AnalysisClass(catalog, catalog["ext_shapeHSM_HsmShapeRegauss_resolution"],
                                          "Resolution Factor from HsmRegauss",
                                          shortName, self.config.analysis, qMin=-0.1, qMax=1.15,
                                          labeller=StarGalaxyLabeller()
                                          ).plotAll(shortName, plotInfoDict, areaDict, self.log,
                                                    **plotAllKwargs)

    def plotCompareUnforced(self, forced, unforced, plotInfoDict, areaDict, zpLabel=None, fluxToPlotList=None,
                            uberCalLabel=None, matchRadius=None, matchRadiusUnitStr=None, matchControl=None,
                            highlightList=None):
        yield
        fluxToPlotList = fluxToPlotList if fluxToPlotList else self.config.fluxToPlotList
        unitStr = "mmag" if self.config.toMilli else "mag"
        enforcer = None
        for col in fluxToPlotList:
            magDiffFunc = MagDiff(col + "_instFlux", col + "_instFlux", unitScale=self.unitScale)
            shortName = "compareUnforced_" + col
            self.log.info("shortName = {:s}".format(shortName))
            if col + "_instFlux" in forced.schema:
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

    def overlaps(self, catalog):
        badForOverlap = makeBadArray(catalog, flagList=self.config.analysis.flags,
                                     onlyReadStars=self.config.onlyReadStars, patchInnerOnly=False)
        goodCat = catalog[~badForOverlap]
        matches = afwTable.matchRaDec(goodCat, self.config.matchOverlapRadius*geom.arcseconds)
        if not matches:
            self.log.info("Did not find any overlapping matches")
        return joinMatches(matches, "first_", "second_")

    def plotOverlaps(self, overlaps, plotInfoDict, areaDict, matchRadius=None, matchRadiusUnitStr=None,
                     zpLabel=None, forcedStr=None, postFix="", fluxToPlotList=None, highlightList=None,
                     uberCalLabel=None):
        yield
        if not fluxToPlotList:
            fluxToPlotList = self.config.fluxToPlotList
        unitStr = "mmag" if self.config.toMilli else "mag"
        magEnforcer = Enforcer(requireLess={"star": {"stdev": 0.003*self.unitScale}})
        plotAllKwargs = dict(matchRadius=matchRadius, matchRadiusUnitStr=matchRadiusUnitStr, zpLabel=zpLabel,
                             forcedStr=forcedStr, uberCalLabel=uberCalLabel, highlightList=highlightList)
        for col in fluxToPlotList:
            shortName = "overlap_" + col + postFix
            self.log.info("shortName = {:s}".format(shortName))
            if "first_" + col + "_instFlux" in overlaps.schema:
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
            if "src_calib_psf_used" in matches.schema:
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
            if "src_calib_photometry_used" in matches.schema:
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
        if "src_calib_astrometry_used" in matches.schema:
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
        if "src_calib_astrometry_used" in matches.schema:
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
        if "src_calib_astrometry_used" in matches.schema:
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

        skyplotKwargs = dict(stats=stats, zpLabel=zpLabel)
        skyFlux = "base_CircularApertureFlux_9_0_instFlux"
        skyFluxStr = fluxToPlotString(skyFlux)
        skyFluxes = catalog[skyFlux]*1e12
        qMin, qMax = 0.75*min(skyFluxes), 0.75*max(skyFluxes)
        yield from self.AnalysisClass(catalog, skyFluxes,
                                      "%s" % "flux(*1e+12)= " + shortName + "[" + skyFluxStr + "]", shortName,
                                      self.config.analysis, qMin=qMin, qMax=qMax, labeller=AllLabeller(),
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


class CompareCoaddAnalysisTask(CmdLineTask):
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
        self.matchControl = afwTable.MatchControl()
        self.matchControl.findOnlyClosest = True
        self.matchControl.symmetricMatch = False

    def runDataRef(self, patchRefList1, patchRefList2, subdir=""):
        plotList = []
        haveForced = True  # do forced datasets exits (may not for single band datasets)
        dataset = "Coadd_forced_src"
        patchRefExistsList1 = [patchRef1 for patchRef1 in patchRefList1 if
                               patchRef1.datasetExists(self.config.coaddName + dataset)]
        if not patchRefExistsList1:
            haveForced = False

        if not haveForced:
            self.log.warn("No forced dataset exist for tract: {0:d} filter: {1:s}.  "
                          "Plotting unforced results only.".format(patchRefList1[0].dataId["tract"],
                                                                   patchRefList1[0].dataId["filter"]))
            dataset = "Coadd_meas"
            patchRefExistsList1 = [patchRef1 for patchRef1 in patchRefList1 if
                                   patchRef1.datasetExists(self.config.coaddName + dataset)]
        if not patchRefExistsList1:
            raise TaskError("No data exists in patRefList1: %s" %
                            ([patchRef1.dataId for patchRef1 in patchRefList1]))
        patchRefList2 = [dataRef2 for dataRef2 in patchRefList2 if
                         dataRef2.datasetExists(self.config.coaddName + dataset)]

        patchList1 = [dataRef1.dataId["patch"] for dataRef1 in patchRefList1 if
                      dataRef1.datasetExists(self.config.coaddName + dataset)]
        patchRefList1 = patchRefExistsList1

        repoInfo1 = getRepoInfo(patchRefList1[0], coaddName=self.config.coaddName, coaddDataset=dataset)
        repoInfo2 = getRepoInfo(patchRefList2[0], coaddName=self.config.coaddName, coaddDataset=dataset)
        # Find a visit/ccd input so that you can check for meas_mosaic input
        # (i.e. to set uberCalLabel).
        self.uberCalLabel1 = determineExternalCalLabel(repoInfo1, patchList1[0],
                                                       coaddName=self.config.coaddName)
        self.uberCalLabel2 = determineExternalCalLabel(repoInfo2, patchList1[0],
                                                       coaddName=self.config.coaddName)
        self.uberCalLabel = self.uberCalLabel1 + "_1\n" + self.uberCalLabel2 + "_2"
        self.log.info(f"External calibration(s) used: {self.uberCalLabel}")

        if haveForced:
            forcedStruct1 = self.readCatalogs(patchRefList1, self.config.coaddName + "Coadd_forced_src",
                                              repoInfo1)
            areaDict1 = forcedStruct1.areaDict
            forced1 = self.calibrateCatalogs(forcedStruct1.catalog, wcs=repoInfo1.wcs)
            forcedStruct2 = self.readCatalogs(patchRefList2, self.config.coaddName + "Coadd_forced_src",
                                              repoInfo2)
            forced2 = self.calibrateCatalogs(forcedStruct2.catalog, wcs=repoInfo2.wcs)
        unforcedStruct1 = self.readCatalogs(patchRefList1, self.config.coaddName + "Coadd_meas", repoInfo1)
        unforced1 = unforcedStruct1.catalog
        if not haveForced:
            areaDict1 = unforcedStruct1.areaDict
        unforced1 = self.calibrateCatalogs(unforced1, wcs=repoInfo1.wcs)
        unforcedStruct2 = self.readCatalogs(patchRefList2, self.config.coaddName + "Coadd_meas", repoInfo2)
        unforced2 = self.calibrateCatalogs(unforcedStruct2.catalog, wcs=repoInfo2.wcs)

        forcedStr = "forced" if haveForced else "unforced"

        if haveForced:
            # Copy over some fields from _ref and _meas catalogs to
            # _forced_src catalog.
            refCat1 = self.readCatalogs(patchRefList1, self.config.coaddName + "Coadd_ref", repoInfo1).catalog
            refColList1 = [s for s in refCat1.schema.getNames() if
                           s.startswith(tuple(self.config.columnsToCopyFromRef))]
            refCat2 = self.readCatalogs(patchRefList2, self.config.coaddName + "Coadd_ref", repoInfo2).catalog
            refColList2 = [s for s in refCat2.schema.getNames() if
                           s.startswith(tuple(self.config.columnsToCopyFromRef))]
            refColsToCopy1 = [col for col in refColList1 if col not in forced1.schema
                              and col in refCat1.schema
                              and not (repoInfo1.hscRun and col == "slot_Centroid_flag")]
            refColsToCopy2 = [col for col in refColList2 if col not in forced2.schema
                              and col in refCat2.schema
                              and not (repoInfo2.hscRun and col == "slot_Centroid_flag")]
            forced1 = addColumnsToSchema(refCat1, forced1, refColsToCopy1)
            forced2 = addColumnsToSchema(refCat2, forced2, refColsToCopy2)
            measColList1 = [s for s in unforced1.schema.getNames() if
                            s.startswith(tuple(self.config.columnsToCopyFromMeas))]
            measColList2 = [s for s in unforced2.schema.getNames() if
                            s.startswith(tuple(self.config.columnsToCopyFromMeas))]
            measColsToCopy1 = [col for col in measColList1 if col not in forced1.schema
                               and col in unforced1.schema
                               and not (repoInfo1.hscRun and col == "slot_Centroid_flag")]
            measColsToCopy2 = [col for col in measColList2 if col not in forced2.schema
                               and col in unforced2.schema
                               and not (repoInfo2.hscRun and col == "slot_Centroid_flag")]
            forced1 = addColumnsToSchema(unforced1, forced1, measColsToCopy1)
            forced2 = addColumnsToSchema(unforced2, forced2, measColsToCopy2)

        # Set an alias map for differing schema naming conventions of different
        # stacks (if any).
        repoList = [repoInfo1.hscRun, repoInfo2.hscRun]
        coaddList = [unforced1, unforced2]
        if haveForced:
            repoList += repoList
            coaddList += [forced1, forced2]
        aliasDictList0 = [self.config.flagsToAlias, ]
        for hscRun, catalog in zip(repoList, coaddList):
            aliasDictList = aliasDictList0
            if hscRun and self.config.srcSchemaMap is not None:
                aliasDictList += [self.config.srcSchemaMap]
            if aliasDictList:
                catalog = setAliasMaps(catalog, aliasDictList)

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
        unforced = self.matchCatalogs(unforced1, unforced2)
        forced = self.matchCatalogs(forced1, forced2)

        self.catLabel = "nChild = 0"
        forcedStr = forcedStr + " " + self.catLabel

        aliasDictList = aliasDictList0
        if hscRun and self.config.srcSchemaMap is not None:
            aliasDictList += [self.config.srcSchemaMap]
        if aliasDictList:
            forced = setAliasMaps(forced, aliasDictList)
            unforced = setAliasMaps(unforced, aliasDictList)

        self.log.info("\nNumber of sources in forced catalogs: first = {0:d} and second = {1:d}".format(
                      len(forced1), len(forced2)))

        subdir = "patch-" + str(patchList1[0]) if len(patchList1) == 1 else subdir
        hscRun = repoInfo1.hscRun if repoInfo1.hscRun else repoInfo2.hscRun
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
            if ("first_base_SdssShape_psf_xx" in forced.schema
                    and "second_base_SdssShape_psf_xx" in forced.schema):
                plotList.append(self.plotSizes(forced, plotInfoDict, areaDict1, forcedStr=forcedStr,
                                               **plotKwargs1))
            else:
                self.log.warn("Cannot run plotSizes: base_SdssShape_psf_xx not in catalog.schema")

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

    def readCatalogs(self, patchRefList, dataset, repoInfo):
        catList = []
        areaDict = {}
        for patchRef in patchRefList:
            if not patchRef.datasetExists(dataset):
                continue
            patchCat = patchRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS)
            catList.append(patchCat)
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

        catList = [patchRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS) for
                   patchRef in patchRefList if patchRef.datasetExists(dataset)]
        if not catList:
            raise TaskError("No catalogs read: %s" % ([patchRef.dataId for patchRef in patchRefList]))
        return Struct(catalog=concatenateCatalogs(catList), areaDict=areaDict)

    def matchCatalogs(self, catalog1, catalog2, matchRadius=None, matchControl=None):
        matchRadius = matchRadius if matchRadius else self.matchRadius
        matchControl = matchControl if matchControl else self.matchControl
        if self.config.matchXy:
            matches = afwTable.matchXy(catalog1, catalog2, matchRadius, matchControl)
        else:
            matches = afwTable.matchRaDec(catalog1, catalog2, matchRadius*geom.arcseconds, matchControl)
        if not matches:
            raise TaskError("No matches found")
        return joinMatches(matches, "first_", "second_")

    def calibrateCatalogs(self, catalog, wcs=None):
        self.zpLabel = "common (" + str(self.config.analysis.coaddZp) + ")"
        # For some reason my persisted catalogs in lauren/LSST/DM-6816new all
        # have nan for RA and Dec.
        if np.all(np.isnan(catalog["coord_ra"])):
            if wcs is None:
                self.log.warn("Bad RA, Dec entries but can't update because wcs is None")
            else:
                for src in catalog:
                    src.updateCoord(wcs)
        calibrated = calibrateCoaddSourceCatalog(catalog, self.config.analysis.coaddZp)
        return calibrated

    def plotMags(self, catalog, plotInfoDict, areaDict, matchRadius=None, matchRadiusUnitStr=None,
                 zpLabel=None, forcedStr=None, fluxToPlotList=None, postFix="",
                 highlightList=None, uberCalLabel=None):
        yield
        if not fluxToPlotList:
            fluxToPlotList = self.config.fluxToPlotList
        unitStr = "mmag" if self.config.toMilli else "mag"
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.02*self.unitScale}})
        for col in fluxToPlotList:
            if ("first_" + col + "_instFlux" in catalog.schema and "second_" + col + "_instFlux" in
                    catalog.schema):
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
        shortName = "diff_footNpix"
        col = "base_Footprint_nPix"
        self.log.info("shortName = {:s}".format(shortName))
        yield from Analysis(catalog, FootNpixDiffCompare(col), "  Run Comparison: Footprint nPix difference",
                            shortName, self.config.analysis, prefix="first_", qMin=-250, qMax=250,
                            labeller=OverlapsStarGalaxyLabeller()
                            ).plotAll(shortName, plotInfoDict, areaDict, self.log, enforcer=enforcer,
                                      **plotAllKwargs)
        shortName = "diff_footNpix_calib_psf_used"
        self.log.info("shortName = {:s}".format(shortName))
        yield from Analysis(catalog, FootNpixDiffCompare(col),
                            "     Run Comparison: Footprint nPix diff (psf_used)",
                            shortName, self.config.analysis, prefix="first_", goodKeys=["calib_psf_used"],
                            qMin=-150, qMax=150, labeller=OverlapsStarGalaxyLabeller(),
                            ).plotAll(shortName, plotInfoDict, areaDict, self.log, enforcer=enforcer,
                                      highlightList=highlightList, **plotAllKwargs)

    def plotSizes(self, catalog, plotInfoDict, areaDict, matchRadius=None, matchRadiusUnitStr=None,
                  zpLabel=None, forcedStr=None, highlightList=None, uberCalLabel=None):
        yield
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.02*self.unitScale}})
        plotAllKwargs = dict(matchRadius=matchRadius, matchRadiusUnitStr=matchRadiusUnitStr,
                             zpLabel=zpLabel, forcedStr=forcedStr, uberCalLabel=uberCalLabel)
        for col in ["base_PsfFlux"]:
            if ("first_" + col + "_instFlux" in catalog.schema
                    and "second_" + col + "_instFlux" in catalog.schema):
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

                    if "first_ext_shapeHSM_HsmSourceMoments_xx" in catalog.schema:
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
                    if "first_ext_shapeHSM_PsfMoments_xx" in catalog.schema:
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
        enforcer = None
        plotAllKwargs = dict(matchRadius=matchRadius, matchRadiusUnitStr=matchRadiusUnitStr, zpLabel=zpLabel,
                             forcedStr=forcedStr, highlightList=highlightList, uberCalLabel=uberCalLabel)
        baseCol = "ext_shapeHSM_HsmShapeRegauss"
        col = baseCol + "_resolution"
        if "first_" + col in catalog.schema:
            shortName = "diff_resolution"
            self.log.info("shortName = {:s}".format(shortName))
            yield from Analysis(catalog, PercentDiff(col),
                                "           Run Comparison: HsmRegauss Resolution (% diff)",
                                shortName, self.config.analysis, prefix="first_",
                                qMin=-0.2, qMax=0.2, labeller=OverlapsStarGalaxyLabeller()
                                ).plotAll(shortName, plotInfoDict, areaDict, self.log, enforcer=enforcer,
                                          **plotAllKwargs)
        col = baseCol + "_e1"
        if "first_" + col in catalog.schema:
            shortName = "diff_HsmShapeRegauss_e1"
            self.log.info("shortName = {:s}".format(shortName))
            yield from Analysis(catalog, PercentDiff(col), "    Run Comparison: HsmRegauss e1 (% diff)",
                                shortName, self.config.analysis, prefix="first_",
                                qMin=-0.2, qMax=0.2, labeller=OverlapsStarGalaxyLabeller()
                                ).plotAll(shortName, plotInfoDict, areaDict, self.log, enforcer=enforcer,
                                          **plotAllKwargs)
        col = baseCol + "_e2"
        if "first_" + col in catalog.schema:
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
        if not fluxToPlotList:
            fluxToPlotList = self.config.fluxToPlotList
        unitStr = "mmag" if self.config.toMilli else "mag"
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.02*self.unitScale}})
        for col in fluxToPlotList:
            if "first_" + col + "_apCorr" in catalog.schema and "second_" + col + "_apCorr" in catalog.schema:
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
                                                  zpLabel=None, forcedStr=forcedStr,
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
