#!/usr/bin/env python

from __future__ import print_function

import os
import astropy.coordinates as coord
import astropy.units as units
import numpy as np
np.seterr(all="ignore")  # noqa E402
import pandas as pd
import functools

from collections import defaultdict

from lsst.daf.persistence.butler import Butler
from lsst.pex.config import (Config, Field, ConfigField, ListField, DictField, ConfigDictField,
                             ConfigurableField)
from lsst.pipe.base import CmdLineTask, ArgumentParser, TaskRunner, TaskError
from lsst.pipe.drivers.utils import TractDataIdContainer
from lsst.afw.table.catalogMatches import matchesToCatalog
from lsst.meas.astrom import AstrometryConfig
from lsst.pipe.tasks.colorterms import Colorterm, ColortermLibrary

from lsst.meas.algorithms import LoadIndexedReferenceObjectsTask

from .analysis import AnalysisConfig, Analysis
from .utils import (Filenamer, Enforcer, MagDiff, MagDiffMatches, MagDiffCompare,
                    AstrometryDiff, TraceSize, PsfTraceSizeDiff, TraceSizeCompare, PercentDiff,
                    E1Resids, E2Resids, E1ResidsHsmRegauss, E2ResidsHsmRegauss, FootNpixDiffCompare,
                    MagDiffCompareErr, CentroidDiff, deconvMom,
                    deconvMomStarGal, concatenateCatalogs, joinMatches, checkPatchOverlap,
                    addColumnsToSchema, addApertureFluxesHSC, addFpPoint,
                    addFootprintNPix, makeBadArray, addIntFloatOrStrColumn,
                    calibrateCoaddSourceCatalog, backoutApCorr, matchNanojanskyToAB,
                    fluxToPlotString, andCatalog, writeParquet, getRepoInfo, setAliasMaps,
                    addPreComputedColumns, computeMeanOfFrac, getSchema)
from .plotUtils import (CosmosLabeller, AllLabeller, StarGalaxyLabeller, OverlapsStarGalaxyLabeller,
                        MatchesStarGalaxyLabeller, determineUberCalLabel)

import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable

__all__ = ["CoaddAnalysisConfig", "CoaddAnalysisRunner", "CoaddAnalysisTask", "CompareCoaddAnalysisConfig",
           "CompareCoaddAnalysisRunner", "CompareCoaddAnalysisTask"]

NANOJANSKYS_PER_AB_FLUX = (0*units.ABmag).to_value(units.nJy)

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
    doPlotOverlaps = Field(dtype=bool, default=True, doc="Plot overlaps? (ignored if plotMatchesOnly is True)")
    plotMatchesOnly = Field(dtype=bool, default=False, doc="Only make plots related to reference cat matches?")
    doPlotMatches = Field(dtype=bool, default=True, doc="Plot matches?")
    doPlotCompareUnforced = Field(dtype=bool, default=True, doc=("Plot difference between forced and unforced"
                                                                 "? (ignored if plotMatchesOnly is True)"))
    doPlotQuiver = Field(dtype=bool, default=True, doc=("Plot ellipticity residuals quiver plot? "
                                                        "(ignored if plotMatchesOnly is True)"))
    doPlotPsfFluxSnHists = Field(dtype=bool, default=True, doc="Plot histograms of raw PSF fluxes and S/N?")
    doPlotFootprintNpix = Field(dtype=bool, default=True, doc=("Plot histogram of footprint nPix? "
                                                               "(ignored if plotMatchesOnly is True)"))
    doPlotInputCounts = Field(dtype=bool, default=True, doc=("Make input counts plot? "
                                                             "(ignored if plotMatchesOnly is True)"))
    onlyReadStars = Field(dtype=bool, default=False, doc="Only read stars (to save memory)?")
    toMilli = Field(dtype=bool, default=True, doc="Print stats in milli units (i.e. mas, mmag)?")
    srcSchemaMap = DictField(keytype=str, itemtype=str, default=None, optional=True,
                             doc="Mapping between different stack (e.g. HSC vs. LSST) schema names")
    fluxToPlotList = ListField(dtype=str, default=["base_GaussianFlux", "base_CircularApertureFlux_12_0",
                                                   "ext_photometryKron_KronFlux", "modelfit_CModel"],
                               doc="List of fluxes to plot: mag(flux)-mag(base_PsfFlux) vs mag(fluxColumn)")
    columnsToCopy = ListField(dtype=str,
                              default=["calib_psf_used", "calib_psf_candidate", "calib_psf_reserved",
                                       "calib_astrometry_used", "calib_photometry_used",
                                       "calib_photometry_reserved", "detect_isPatchInner",
                                       "detect_isTractInner", "merge_peak_sky", "calib_psfUsed",
                                       "calib_psfCandidate", ],
                              doc="List of columns to copy from one source catalog to another.")
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

    def saveToStream(self, outfile, root="root"):
        """Required for loading colorterms from a Config outside the 'lsst' namespace"""
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
            self.doPlotMags = False
            self.doPlotSizes = False
            self.doPlotCentroids = False
            self.doPlotStarGalaxy = False
            self.doPlotOverlaps = False
            self.doPlotCompareUnforced = False
            self.doPlotQuiver = False
            self.doPlotFootprintNpix = False
            self.doPlotInputCounts = False


class CoaddAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        kwargs["cosmos"] = parsedCmd.cosmos
        kwargs["subdir"] = parsedCmd.subdir

        # Partition all inputs by tract,filter
        FilterRefsDict = functools.partial(defaultdict, list)  # Dict for filter-->dataRefs
        # Make sure the actual input files requested exist (i.e. do not follow the parent chain)
        # First check for forced catalogs.  Break out of datasets loop if forced catalogs were found,
        # otherwise continue search for existence of unforced catalogs
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

    def runDataRef(self, patchRefList, subdir="", cosmos=None):
        haveForced = False  # do forced datasets exits (may not for single band datasets)
        dataset = "Coadd_forced_src"
        # Explicit input file was checked in CoaddAnalysisRunner, so a check on datasetExists
        # is sufficient here (modulo the case where a forced dataset exists higher up the parent
        # tree than the specified input, but does not exist in the input directory as the former
        # will be found)
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
        subdir = "patch-" + str(patchList[0]) if len(patchList) == 1 else subdir
        filenamer = Filenamer(repoInfo.butler, self.outputDataset, repoInfo.dataId, subdir=subdir)
        # Find a visit/ccd input so that you can check for meas_mosaic input (i.e. to set uberCalLabel)
        self.uberCalLabel = determineUberCalLabel(repoInfo, patchList[0], coaddName=self.config.coaddName)
        self.log.info("Uber-calibration used: {:}".format(self.uberCalLabel))

        # Set some aliases for differing schema naming conventions
        aliasDictList = [self.config.flagsToAlias, ]
        if repoInfo.hscRun and self.config.srcSchemaMap is not None:
            aliasDictList += [self.config.srcSchemaMap]

        # Dict of all parameters common to plot* functions
        plotKwargs = dict(butler=repoInfo.butler, camera=repoInfo.camera, tractInfo=repoInfo.tractInfo,
                          patchList=patchList, hscRun=repoInfo.hscRun, zpLabel=self.zpLabel,
                          uberCalLabel=self.uberCalLabel)

        if any (doPlot for doPlot in [self.config.doPlotMags, self.config.doPlotStarGalaxy,
                                      self.config.doPlotOverlaps, self.config.doPlotCompareUnforced,
                                      cosmos, self.config.externalCatalogs,
                                      self.config.doWriteParquetTables]):
            if haveForced:
                forced = self.readCatalogs(patchRefList, self.config.coaddName + "Coadd_forced_src")
                forced = self.calibrateCatalogs(forced, wcs=repoInfo.wcs)
            unforced = self.readCatalogs(patchRefList, self.config.coaddName + "Coadd_meas")
            unforced = self.calibrateCatalogs(unforced, wcs=repoInfo.wcs)
            plotKwargs.update(dict(zpLabel=self.zpLabel))
            if haveForced:
                # copy over some fields from unforced to forced catalog
                forced = addColumnsToSchema(unforced, forced,
                                            [col for col in list(self.config.columnsToCopy) +
                                             list(self.config.analysis.flags) if
                                             col not in forced.schema and col in unforced.schema and
                                             not (repoInfo.hscRun and col == "slot_Centroid_flag")])
                # Add the reference band flags for forced photometry to forced catalog
                refBandCat = self.readCatalogs(patchRefList, self.config.coaddName + "Coadd_ref")
                if len(forced) != len(refBandCat):
                    raise RuntimeError(("Lengths of forced (N = {0:d}) and ref (N = {0:d}) cats don't match").
                                       format(len(forced), len(refBandCat)))
                refBandList = list(s.field.getName() for s in refBandCat.schema if "merge_measurement_"
                                   in s.field.getName())
                forced = addColumnsToSchema(refBandCat, forced,
                                            [col for col in refBandList if col not in forced.schema and
                                             col in refBandCat.schema])

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

            # Convert to pandas DataFrames
            unforced = unforced.asAstropy().to_pandas()
            if haveForced:
                forced = forced.asAstropy().to_pandas()

            unforcedSchema = getSchema(unforced)
            if haveForced:
                forcedSchema = getSchema(forced)
            plotKwargs.update(dict(zpLabel=self.zpLabel))

            # Must do the overlaps before purging the catalogs of non-primary sources
            if self.config.doPlotOverlaps and not self.config.writeParquetOnly:
                # Determine if any patches in the patchList actually overlap
                overlappingPatches = checkPatchOverlap(patchList, repoInfo.tractInfo)
                if not overlappingPatches:
                    self.log.info("No overlapping patches...skipping overlap plots")
                else:
                    if haveForced:
                        forcedOverlaps = self.overlaps(forced)
                        if not forcedOverlaps.empty:
                            self.plotOverlaps(forcedOverlaps, filenamer, repoInfo.dataId,
                                              matchRadius=self.config.matchOverlapRadius,
                                              matchRadiusUnitStr="\"",
                                              forcedStr=forcedStr, postFix="_forced",
                                              fluxToPlotList=["modelfit_CModel", ], **plotKwargs)
                            self.log.info("Number of forced overlap objects matched = {:d}".
                                          format(len(forcedOverlaps)))
                    unforcedOverlaps = self.overlaps(unforced)
                    if not unforcedOverlaps.empty:
                        self.plotOverlaps(unforcedOverlaps, filenamer, repoInfo.dataId,
                                          matchRadius=self.config.matchOverlapRadius,
                                          matchRadiusUnitStr="\"",
                                          forcedStr="unforced", postFix="_unforced",
                                          fluxToPlotList=["modelfit_CModel", ], **plotKwargs)
                        self.log.info("Number of unforced overlap objects matched = {:d}".
                                      format(len(unforcedOverlaps)))

            # Set boolean array indicating sources deemed unsuitable for qa analyses
            bad = makeBadArray(unforced, flagList=self.config.analysis.flags,
                               onlyReadStars=self.config.onlyReadStars)
            if haveForced:
                bad |= makeBadArray(forced, flagList=self.config.analysis.flags,
                                    onlyReadStars=self.config.onlyReadStars)

            # Create and write parquet tables
            if self.config.doWriteParquetTables:
                if haveForced:
                    # Add pre-computed columns for parquet tables
                    forced = addPreComputedColumns(forced, fluxToPlotList=self.config.fluxToPlotList,
                                                   toMilli=self.config.toMilli, unforcedCat=unforced)
                    dataRef_forced = repoInfo.butler.dataRef('analysisCoaddTable_forced',
                                                             dataId=repoInfo.dataId)
                    writeParquet(dataRef_forced, forced, badArray=bad)
                dataRef_unforced = repoInfo.butler.dataRef('analysisCoaddTable_unforced',
                                                           dataId=repoInfo.dataId)
                # Add pre-computed columns for parquet tables
                unforced = addPreComputedColumns(unforced, fluxToPlotList=self.config.fluxToPlotList,
                                                 toMilli=self.config.toMilli)
                writeParquet(dataRef_unforced, unforced, badArray=bad)
                if self.config.writeParquetOnly:
                    self.log.info("Exiting after writing Parquet tables.  No plots generated.")
                    return

            # Purge the catalogs of flagged sources
            unforced = unforced[~bad].copy(deep=True)
            if haveForced:
                forced = forced[~bad].copy(deep=True)
            else:
                forced = unforced
            self.catLabel = "nChild = 0"
            forcedStr = forcedStr + " " + self.catLabel
            if haveForced:
                self.log.info("\nNumber of sources in catalogs: unforced = {0:d} and forced = {1:d}".format(
                    len(unforced), len(forced)))
            else:
                self.log.info("\nNumber of sources in catalog: unforced = {0:d}".format(len(unforced)))

            flagsCat = unforced

            if self.config.doPlotPsfFluxSnHists:
                self.plotPsfFluxSnHists(unforced,
                                        filenamer(repoInfo.dataId, description="base_PsfFlux_cal",
                                                  style="hist"),
                                        repoInfo.dataId, forcedStr="unforced " + self.catLabel, **plotKwargs)
            if self.config.doPlotFootprintNpix:
                self.plotFootprintHist(forced,
                                       filenamer(repoInfo.dataId, description="footNpix", style="hist"),
                                       repoInfo.dataId, flagsCat=flagsCat, **plotKwargs)
                self.plotFootprint(forced, filenamer, repoInfo.dataId, forcedStr=forcedStr, flagsCat=flagsCat,
                                   **plotKwargs)

            if self.config.doPlotQuiver:
                self.plotQuiver(unforced,
                                filenamer(repoInfo.dataId, description="ellipResids", style="quiver"),
                                dataId=repoInfo.dataId, forcedStr="unforced " + self.catLabel, scale=2,
                                **plotKwargs)

            if self.config.doPlotInputCounts:
                self.plotInputCounts(unforced, filenamer(repoInfo.dataId, description="inputCounts",
                                                         style="tract"), dataId=repoInfo.dataId,
                                     forcedStr="unforced " + self.catLabel, alpha=0.5,
                                     doPlotTractImage=True, doPlotPatchOutline=True, sizeFactor=5.0,
                                     maxDiamPix=1000, **plotKwargs)

            if self.config.doPlotMags:
                self.plotMags(unforced, filenamer, repoInfo.dataId, forcedStr="unforced " + self.catLabel,
                              postFix="_unforced", flagsCat=flagsCat, **plotKwargs)
                if haveForced:
                    self.plotMags(forced, filenamer, repoInfo.dataId, forcedStr=forcedStr, postFix="_forced",
                                  flagsCat=flagsCat,
                                  highlightList=[("merge_measurement_" + repoInfo.genericFilterName, 0,
                                                  "yellow"), ], **plotKwargs)
            if self.config.doPlotStarGalaxy:
                if "ext_shapeHSM_HsmSourceMoments_xx" in unforcedSchema:
                    self.plotStarGal(unforced, filenamer, repoInfo.dataId,
                                     forcedStr="unforced " + self.catLabel, **plotKwargs)
                else:
                    self.log.warn("Cannot run plotStarGal: ext_shapeHSM_HsmSourceMoments_xx not "
                                  "in unforcedSchema")

            if self.config.doPlotSizes:
                if all(ss in unforcedSchema for ss in ["base_SdssShape_psf_xx", "calib_psf_used"]):
                    self.plotSizes(unforced, filenamer, repoInfo.dataId, forcedStr="unforced " + self.catLabel,
                                   postFix="_unforced", **plotKwargs)
                else:
                    self.log.warn("Cannot run plotSizes: base_SdssShape_psf_xx and/or calib_psf_used "
                                  "not in unforcedSchema")
                if haveForced:
                    if all(ss in forcedSchema for ss in ["base_SdssShape_psf_xx", "calib_psf_used"]):
                        self.plotSizes(forced, filenamer, repoInfo.dataId, forcedStr=forcedStr, **plotKwargs)
                    else:
                        self.log.warn("Cannot run plotSizes: base_SdssShape_psf_xx and/or calib_psf_used "
                                      "not in forcedSchema")
            if cosmos:
                self.plotCosmos(forced, filenamer, cosmos, repoInfo.dataId)
            if self.config.doPlotCompareUnforced and haveForced:
                self.plotCompareUnforced(forced, unforced, filenamer, repoInfo.dataId, **plotKwargs)

        if self.config.doPlotMatches:
            if haveForced:
                matches = self.readSrcMatches(patchRefList, self.config.coaddName + "Coadd_forced_src",
                                              hscRun=repoInfo.hscRun, wcs=repoInfo.wcs,
                                              aliasDictList=aliasDictList)
            else:
                matches = self.readSrcMatches(patchRefList, self.config.coaddName + "Coadd_meas",
                                              hscRun=repoInfo.hscRun, wcs=repoInfo.wcs,
                                              aliasDictList=aliasDictList)
            plotKwargs.update(dict(zpLabel=self.zpLabel))
            self.plotMatches(matches, repoInfo.filterName, filenamer, repoInfo.dataId, forcedStr=forcedStr,
                             **plotKwargs)

        for cat in self.config.externalCatalogs:
            with andCatalog(cat):
                matches = self.matchCatalog(forced, repoInfo.filterName, self.config.externalCatalogs[cat])
                if matches is not None:
                    self.plotMatches(matches, repoInfo.filterName, filenamer, repoInfo.dataId,
                                     forcedStr=forcedStr, matchRadius=self.matchRadius,
                                     matchRadiusUnitStr=self.matchRadiusUnitStr, **plotKwargs)
                else:
                    self.log.warn("Could not create match catalog for {:}.  Is "
                                  "lsst.meas.extensions.astrometryNet setup?".format(cat))

    def readCatalogs(self, patchRefList, dataset):
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
        """
        catList = []
        for patchRef in patchRefList:
            if patchRef.datasetExists(dataset):
                cat = patchRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_HEAVY_FOOTPRINTS)
                if self.config.doWriteParquetTables:
                    cat = addIntFloatOrStrColumn(cat, patchRef.dataId["patch"], "patchId",
                                                 "Patch on which source was detected")
                catList.append(cat)
        if not catList:
            raise TaskError("No catalogs read: %s" % ([patchRef.dataId for patchRef in patchRefList]))
        return concatenateCatalogs(catList)

    def readSrcMatches(self, dataRefList, dataset, hscRun=None, wcs=None, aliasDictList=None):
        catList = []
        for dataRef in dataRefList:
            if not dataRef.datasetExists(dataset):
                self.log.info("Dataset does not exist: {0:r}, {1:s}".format(dataRef.dataId, dataset))
                continue
            butler = dataRef.getButler()

            # Generate unnormalized match list (from normalized persisted one) with joinMatchListWithCatalog
            # (which requires a refObjLoader to be initialized).
            catalog = dataRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS)
            # Set some aliases for differing schema naming conventions
            if aliasDictList:
                catalog = setAliasMaps(catalog, aliasDictList)
            schema = getSchema(catalog)
            if dataset != "deepCoadd_meas" and any(ss not in schema for ss in self.config.columnsToCopy):
                unforced = dataRef.get("deepCoadd_meas", immediate=True,
                                       flags=afwTable.SOURCE_IO_NO_FOOTPRINTS)
                unforcedSchema = getSchema(unforced)
                # copy over some fields from unforced to forced catalog
                catalog = addColumnsToSchema(unforced, catalog,
                                             [col for col in list(self.config.columnsToCopy) +
                                              list(self.config.analysis.flags) if
                                              col not in schema and col in unforcedSchema and
                                              not (hscRun and col == "slot_Centroid_flag")])
                if aliasDictList:
                    catalog = setAliasMaps(catalog, aliasDictList)
            catalog = self.calibrateCatalogs(catalog, wcs=wcs)

            # Convert to pandas DataFrames
            catalog = catalog.asAstropy().to_pandas()

            # Set boolean array indicating sources deemed unsuitable for qa analyses
            bad = makeBadArray(catalog, flagList=self.config.analysis.flags,
                               onlyReadStars=self.config.onlyReadStars)

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
            self.zpLabel = self.zpLabel
            packedMatches = packedMatches[~badMatch].copy(deep=True)
            if not packedMatches:
                self.log.warn("No good matches for %s" % (dataRef.dataId,))
                continue
            # The reference object loader grows the bbox by the config parameter pixelMargin.  This
            # is set to 50 by default but is not reflected by the radius parameter set in the
            # metadata, so some matches may reside outside the circle searched within this radius
            # Thus, increase the radius set in the metadata fed into joinMatchListWithCatalog() to
            # accommodate.
            matchmeta = packedMatches.table.getMetadata()
            rad = matchmeta.getDouble("RADIUS")
            matchmeta.setDouble("RADIUS", rad*1.05, "field radius in degrees, approximate, padded")
            refObjLoader = self.config.refObjLoader.apply(butler=butler)
            matches = refObjLoader.joinMatchListWithCatalog(packedMatches, catalog)
            if not hasattr(matches[0].first, "schema"):
                raise RuntimeError("Unable to unpack matches.  "
                                   "Do you have the correct reference catalog setup?")
            # LSST reads in reference catalogs with flux in "nanojanskys", so must convert to AB
            matches = matchNanojanskyToAB(matches)
            if hscRun and self.config.doAddAperFluxHsc:
                addApertureFluxesHSC(matches, prefix="second_")

            if not matches:
                self.log.warn("No matches for %s" % (dataRef.dataId,))
                continue

            # Set the alias maps for the matches sources (i.e. the .second attribute schema for each match)
            if aliasDictList:
                for mm in matches:
                    mm.second = setAliasMaps(mm.second, aliasDictList)

            matchMeta = butler.get(dataset, dataRef.dataId,
                                   flags=afwTable.SOURCE_IO_NO_FOOTPRINTS).getTable().getMetadata()
            catalog = matchesToCatalog(matches, matchMeta)
            # Compute Focal Plane coordinates for each source if not already there
            if self.config.analysisMatches.doPlotFP:
                if "src_base_FPPosition_x" not in catalog.schema and "src_focalplane_x" not in catalog.schema:
                    exp = butler.get("calexp", dataRef.dataId)
                    det = exp.getDetector()
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

        return concatenateCatalogs(catList)

    def calibrateCatalogs(self, catalog, wcs=None):
        self.zpLabel = "common (" + str(self.config.analysis.coaddZp) + ")"
        # My persisted catalogs in lauren/LSST/DM-6816new all have nan for ra dec (see DM-9556)
        if np.all(np.isnan(catalog["coord_ra"])):
            if wcs is None:
                self.log.warn("Bad ra, dec entries but can't update because wcs is None")
            else:
                for src in catalog:
                    src.updateCoord(wcs)
        # Optionally backout aperture corrections
        if self.config.doBackoutApCorr:
            catalog = backoutApCorr(catalog)
        calibrated = calibrateCoaddSourceCatalog(catalog, self.config.analysis.coaddZp)
        return calibrated

    def plotMags(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None, tractInfo=None,
                 patchList=None, hscRun=None, matchRadius=None, matchRadiusUnitStr=None, zpLabel=None,
                 forcedStr=None, fluxToPlotList=None, postFix="", flagsCat=None, highlightList=None,
                 uberCalLabel=None):
        if not fluxToPlotList:
            fluxToPlotList = self.config.fluxToPlotList
        unitStr = "mmag" if self.config.toMilli else "mag"
        enforcer = Enforcer(requireLess={"star": {"stdev": 0.02*self.unitScale}})
        schema = getSchema(catalog)
        for col in fluxToPlotList:
            isForced = False if forcedStr is None else ("forced" in forcedStr)
            if col + "_instFlux" in schema and not (isForced and "CircularAper" in col):
                shortName = "mag_" + col + postFix
                self.log.info("shortName = {:s}".format(shortName))
                self.AnalysisClass(catalog,
                                   MagDiff(col + "_instFlux", "base_PsfFlux_instFlux",
                                           unitScale=self.unitScale),
                                   "Mag(%s) - PSFMag (%s)" % (fluxToPlotString(col), unitStr),
                                   shortName, self.config.analysis,
                                   flags=[col + "_flag"], labeller=StarGalaxyLabeller(),
                                   flagsCat=flagsCat, unitScale=self.unitScale,
                                   ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                             camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                             patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                             zpLabel=zpLabel, forcedStr=forcedStr, uberCalLabel=uberCalLabel,
                                             highlightList=highlightList)

    def plotSizes(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None, tractInfo=None,
                  patchList=None, hscRun=None, matchRadius=None, zpLabel=None, forcedStr=None, postFix="",
                  flagsCat=None, uberCalLabel=None):
        enforcer = None
        unitStr = " (milli)" if self.config.toMilli else ""
        plotAllKwargs = dict(butler=butler, camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                             patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                             zpLabel=zpLabel, forcedStr=forcedStr, uberCalLabel=uberCalLabel)
        schema = getSchema(catalog)
        for col in ["base_PsfFlux", ]:
            if col + "_instFlux" in schema:
                compareCol = "base_SdssShape"
                # Set limits dynamically...can be very different visit-to-visit due to seeing differences
                # SDSS and HSM should be similar, so limits based on one should be valid for the other and
                # having the same scale eases comparisons between the two.
                traceSizeFunc = TraceSize(compareCol)

                # First do for calib_psf_used only.
                shortName = "trace" + postFix + "_calib_psf_used"
                psfUsed = catalog[catalog["calib_psf_used"] & ~catalog["base_SdssShape_flag"]].copy(deep=True)
                sdssTrace = traceSizeFunc(psfUsed)
                sdssTrace = sdssTrace[np.where(np.isfinite(sdssTrace))]
                traceMean = np.around(np.nanmean(sdssTrace), 2)
                traceStd = max(0.03, np.around(4.5*np.nanstd(sdssTrace), 2))
                qMin = traceMean - traceStd
                qMax = traceMean + traceStd
                self.log.info("shortName = {:s}".format(shortName))
                self.AnalysisClass(psfUsed, sdssTrace,
                                   ("          SdssShape Trace (calib_psf_used): "
                                    "$\sqrt{0.5*(I_{xx}+I_{yy})}$ (pixels)"),
                                   shortName, self.config.analysis, flags=[col + "_flag"],
                                   goodKeys=["calib_psf_used"], qMin=qMin, qMax=qMax,
                                   labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                   ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, **plotAllKwargs)
                if "ext_shapeHSM_HsmSourceMoments_xx" in schema:
                    shortName = "hsmTrace" + postFix + "_calib_psf_used"
                    compareCol = "ext_shapeHSM_HsmSourceMoments"
                    self.log.info("shortName = {:s}".format(shortName))
                    self.AnalysisClass(psfUsed, TraceSize(compareCol),
                                       ("          HSM Trace (calib_psf_used): $\sqrt{0.5*(I_{xx}+I_{yy})}$"
                                        " (pixels)"), shortName, self.config.analysis, flags=[col + "_flag"],
                                       goodKeys=["calib_psf_used"], qMin=qMin, qMax=qMax,
                                       labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                       ).plotAll(dataId, filenamer, self.log, enforcer=enforcer,
                                                 **plotAllKwargs)

                # Now for all stars.
                shortName = "trace" + postFix
                starsOnly = catalog[catalog["base_ClassificationExtendedness_value"] < 0.5].copy(deep=True)
                sdssTrace = traceSizeFunc(starsOnly)
                self.log.info("shortName = {:s}".format(shortName))
                self.AnalysisClass(starsOnly, sdssTrace,
                                   "  SdssShape Trace: $\sqrt{0.5*(I_{xx}+I_{yy})}$ (pixels)", shortName,
                                   self.config.analysis, flags=[col + "_flag"], qMin=qMin, qMax=qMax,
                                   labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                   ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, **plotAllKwargs)
                if "ext_shapeHSM_HsmSourceMoments_xx" in schema:
                    shortName = "hsmTrace" + postFix
                    compareCol = "ext_shapeHSM_HsmSourceMoments"
                    self.log.info("shortName = {:s}".format(shortName))
                    self.AnalysisClass(starsOnly, TraceSize(compareCol),
                                       "HSM Trace: $\sqrt{0.5*(I_{xx}+I_{yy})}$ (pixels)", shortName,
                                       self.config.analysis, flags=[col + "_flag"], qMin=qMin, qMax=qMax,
                                       labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                       ).plotAll(dataId, filenamer, self.log, enforcer=enforcer,
                                                 **plotAllKwargs)

            if col + "_instFlux" in schema:
                shortName = "psfTraceDiff" + postFix
                compareCol = "base_SdssShape"
                psfCompareCol = "base_SdssShape_psf"
                self.log.info("shortName = {:s}".format(shortName))
                self.AnalysisClass(catalog, PsfTraceSizeDiff(compareCol, psfCompareCol),
                                   "    SdssShape Trace % diff (psf_used - PSFmodel)", shortName,
                                   self.config.analysis, flags=[col + "_flag"],
                                   goodKeys=["calib_psf_used"], qMin=-3.0, qMax=3.0,
                                   labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                   ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, **plotAllKwargs)

                shortName = "e1Resids" + postFix
                self.log.info("shortName = {:s}".format(shortName))
                self.AnalysisClass(catalog, E1Resids(compareCol, psfCompareCol, unitScale=self.unitScale),
                                   "        SdssShape e1 resids (psf_used - PSFmodel)%s" % unitStr, shortName,
                                   self.config.analysis, flags=[col + "_flag"], goodKeys=["calib_psf_used"],
                                   qMin=-0.05, qMax=0.05, labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                   unitScale=self.unitScale,
                                   ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, **plotAllKwargs)

                shortName = "e2Resids" + postFix
                self.log.info("shortName = {:s}".format(shortName))
                self.AnalysisClass(catalog, E2Resids(compareCol, psfCompareCol, unitScale=self.unitScale),
                                   "       SdssShape e2 resids (psf_used - PSFmodel)%s" % unitStr, shortName,
                                   self.config.analysis, flags=[col + "_flag"], goodKeys=["calib_psf_used"],
                                   qMin=-0.05, qMax=0.05, labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                   unitScale=self.unitScale,
                                   ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, **plotAllKwargs)

                if "ext_shapeHSM_HsmSourceMoments_xx" in schema:
                    shortName = "psfHsmTraceDiff" + postFix
                    compareCol = "ext_shapeHSM_HsmSourceMoments"
                    psfCompareCol = "ext_shapeHSM_HsmPsfMoments"
                    self.log.info("shortName = {:s}".format(shortName))
                    self.AnalysisClass(catalog, PsfTraceSizeDiff(compareCol, psfCompareCol),
                                       "HSM Trace % diff (psf_used - PSFmodel)", shortName,
                                       self.config.analysis, flags=[col + "_flag"],
                                       goodKeys=["calib_psf_used"], qMin=-3.0, qMax=3.0,
                                       labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                       ).plotAll(dataId, filenamer, self.log, enforcer=enforcer,
                                                 **plotAllKwargs)
                    shortName = "e1ResidsHsm" + postFix
                    self.log.info("shortName = {:s}".format(shortName))
                    self.AnalysisClass(catalog, E1Resids(compareCol, psfCompareCol, unitScale=self.unitScale),
                                       "   HSM e1 resids (psf_used - PSFmodel)%s" % unitStr, shortName,
                                       self.config.analysis, flags=[col + "_flag"],
                                       goodKeys=["calib_psf_used"], qMin=-0.05, qMax=0.05,
                                       labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                       unitScale=self.unitScale,
                                       ).plotAll(dataId, filenamer, self.log, enforcer=enforcer,
                                                 **plotAllKwargs)
                    shortName = "e2ResidsHsm" + postFix
                    self.log.info("shortName = {:s}".format(shortName))
                    self.AnalysisClass(catalog, E2Resids(compareCol, psfCompareCol, unitScale=self.unitScale),
                                       "   HSM e2 resids (psf_used - PSFmodel)%s" % unitStr, shortName,
                                       self.config.analysis, flags=[col + "_flag"],
                                       goodKeys=["calib_psf_used"], qMin=-0.05, qMax=0.05,
                                       labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                       unitScale=self.unitScale,
                                       ).plotAll(dataId, filenamer, self.log, enforcer=enforcer,
                                                 **plotAllKwargs)

                    shortName = "e1ResidsHsmRegauss" + postFix
                    self.log.info("shortName = {:s}".format(shortName))
                    self.AnalysisClass(catalog, E1ResidsHsmRegauss(unitScale=self.unitScale),
                                       "       HsmRegauss e1 resids (psf_used - HsmPsfMoments)%s" % unitStr,
                                       shortName, self.config.analysis,
                                       flags=[col + "_flag", "ext_shapeHSM_HsmShapeRegauss_flag"],
                                       goodKeys=["calib_psf_used"], qMin=-0.05, qMax=0.05,
                                       labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                       unitScale=self.unitScale,
                                       ).plotAll(dataId, filenamer, self.log, enforcer=enforcer,
                                                 **plotAllKwargs)

                    shortName = "e2ResidsHsmRegauss" + postFix
                    self.log.info("shortName = {:s}".format(shortName))
                    self.AnalysisClass(catalog, E2ResidsHsmRegauss(unitScale=self.unitScale),
                                       "       HsmRegauss e2 resids (psf_used - HsmPsfMoments)%s" % unitStr,
                                       shortName, self.config.analysis,
                                       flags=[col + "_flag", "ext_shapeHSM_HsmShapeRegauss_flag"],
                                       goodKeys=["calib_psf_used"], qMin=-0.05, qMax=0.05,
                                       labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                       unitScale=self.unitScale,
                                       ).plotAll(dataId, filenamer, self.log, enforcer=enforcer,
                                                 **plotAllKwargs)

    def plotCentroidXY(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None,
                       tractInfo=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None,
                       forcedStr=None, flagsCat=None, uberCalLabel=None):
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.02*self.unitScale}})
        schema = getSchema(catalog)
        for col in ["base_SdssCentroid_x", "base_SdssCentroid_y"]:
            if col in schema:
                shortName = col
                self.log.info("shortName = {:s}".format(shortName))
                self.AnalysisClass(catalog, catalog[col], "(%s)" % col, shortName, self.config.analysis,
                                   labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                   ).plotFP(dataId, filenamer, self.log, enforcer=enforcer, camera=camera,
                                            ccdList=ccdList, hscRun=hscRun, matchRadius=matchRadius,
                                            zpLabel=zpLabel, forcedStr=forcedStr)

    def plotFootprint(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None,
                      tractInfo=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None,
                      forcedStr=None, postFix="", flagsCat=None, plotRunStats=False, highlightList=None,
                      uberCalLabel=None):
        enforcer = None
        plotAllKwargs = dict(butler=butler, camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                             patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                             zpLabel=zpLabel, forcedStr=forcedStr, uberCalLabel=uberCalLabel)
        schema = getSchema(catalog)
        if "calib_psf_used" in schema:
            shortName = "footNpix_calib_psf_used"
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(catalog, catalog["base_Footprint_nPix"], "%s" % shortName, shortName,
                               self.config.analysis, flags=["base_Footprint_nPix_flag"],
                               goodKeys=["calib_psf_used"], qMin=-100, qMax=2000,
                               labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                               ).plotAll(dataId, filenamer, self.log, enforcer=enforcer,
                                         plotRunStats=plotRunStats, highlightList=highlightList,
                                         **plotAllKwargs)
        shortName = "footNpix"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(catalog, catalog["base_Footprint_nPix"], "%s" % shortName, shortName,
                           self.config.analysis, flags=["base_Footprint_nPix_flag"],
                           qMin=0, qMax=3000, labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                           ).plotAll(dataId, filenamer, self.log, enforcer=enforcer,
                                     plotRunStats=plotRunStats, **plotAllKwargs)

    def plotFootprintHist(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None,
                          tractInfo=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None,
                          postFix="", flagsCat=None, uberCalLabel=None):
        stats = None
        shortName = "footNpix"
        self.log.info("shortName = {:s}".format(shortName + "Hist"))
        self.AnalysisClass(catalog, catalog["base_Footprint_nPix"], "%s" % shortName, shortName,
                           self.config.analysis, flags=["base_Footprint_nPix_flag"], qMin=0, qMax=3000,
                           labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                           ).plotHistogram(filenamer, stats=stats, camera=camera, ccdList=ccdList,
                                           tractInfo=tractInfo, patchList=patchList, hscRun=hscRun,
                                           matchRadius=matchRadius, zpLabel=zpLabel,
                                           filterStr=dataId['filter'], uberCalLabel=uberCalLabel)

    def plotPsfFluxSnHists(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None,
                           tractInfo=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None,
                           forcedStr=None, uberCalLabel=None, postFix="", flagsCat=None, logPlot=True,
                           density=True, cumulative=-1):
        schema = getSchema(catalog)
        stats = None
        shortName = "psfInstFlux" if zpLabel == "raw" else "psfCalFlux"
        self.log.info("shortName = {:s}".format(shortName))
        # want "raw" flux
        factor = 10.0**(0.4*self.config.analysis.commonZp) if zpLabel == "raw" else NANOJANSKYS_PER_AB_FLUX
        psfFlux = catalog["base_PsfFlux_instFlux"]*factor
        psfFluxErr = catalog["base_PsfFlux_instFluxErr"]*factor
        psfSn = psfFlux/psfFluxErr

        # Scale S/N threshold by ~sqrt(#exposures) if catalog is coadd data
        if "base_InputCount_value" in schema:
            inputCounts = catalog["base_InputCount_value"]
            scaleFactor = computeMeanOfFrac(inputCounts, tailStr="upper", fraction=0.1, floorFactor=10)
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
        psfSnFluxGtLow= psfSn[goodFlux]
        goodFlux = psfFlux > highFlux
        psfSnFluxGtHigh = psfSn[goodFlux]
        psfUsedCat = catalog[catalog["calib_psf_used"]]
        psfUsedPsfFlux = psfUsedCat["base_PsfFlux_instFlux"]*factor
        psfUsedPsfFluxErr = psfUsedCat["base_PsfFlux_instFluxErr"]*factor
        psfUsedPsfSn = psfUsedPsfFlux/psfUsedPsfFluxErr

        self.AnalysisClass(catalog, psfFlux, "%s" % shortName, shortName,
                           self.config.analysis, flags=["base_PsfFlux_flag"], qMin=0,
                           qMax = int(min(99999, max(4.0*np.median(psfFlux), 0.25*np.max(psfFlux)))),
                           labeller=AllLabeller(), flagsCat=flagsCat,
                           ).plotHistogram(filenamer, numBins="sqrt", stats=stats, camera=camera,
                                           ccdList=ccdList, tractInfo=tractInfo, patchList=patchList,
                                           hscRun=hscRun, zpLabel=zpLabel,
                                           forcedStr=forcedStr, filterStr=dataId["filter"],
                                           uberCalLabel=uberCalLabel, vertLineList=[lowFlux, highFlux],
                                           logPlot=logPlot, density=False, cumulative=cumulative,
                                           addDataList=[psfFluxSnGtLow, psfFluxSnGtHigh, psfUsedPsfFlux],
                                           addDataLabelList=["S/N>{:.1f}".format(lowSn),
                                                             "S/N>{:.1f}".format(highSn), "psf_used"])
        shortName = "psfInstFlux/psfInstFluxErr" if zpLabel == "raw" else "psfCalFlux/psfCalFluxErr"
        filenamer = filenamer.replace("Flux", "FluxSn")
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(catalog, psfSn, "%s" % "S/N = " + shortName, shortName,
                           self.config.analysis, flags=["base_PsfFlux_flag"], qMin=0,
                           qMax = 4*highSn, labeller=AllLabeller(), flagsCat=flagsCat,
                           ).plotHistogram(filenamer, numBins="sqrt", stats=stats, camera=camera,
                                           ccdList=ccdList, tractInfo=tractInfo, patchList=patchList,
                                           hscRun=hscRun, zpLabel=zpLabel,
                                           forcedStr=forcedStr, filterStr=dataId["filter"],
                                           uberCalLabel=uberCalLabel, vertLineList=[lowSn, highSn],
                                           logPlot=logPlot, density=False, cumulative=cumulative,
                                           addDataList=[psfSnFluxGtLow, psfSnFluxGtHigh, psfUsedPsfSn],
                                           addDataLabelList=["Flux>{:.1f}".format(lowFlux),
                                                             "Flux>{:.1f}".format(highFlux), "psf_used"])

        skyplotKwargs = dict(dataId=dataId, butler=butler, stats=stats, camera=camera, ccdList=ccdList,
                             tractInfo=tractInfo, patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                             matchRadiusUnitStr=None, zpLabel=zpLabel)
        filenamer = filenamer.replace("hist", "sky-all")

        self.AnalysisClass(catalog, psfSn, "%s" % "S/N = " + shortName, shortName,
                           self.config.analysis, flags=["base_PsfFlux_flag"], qMin=0,
                           qMax = 1.25*highSn, labeller=AllLabeller(), flagsCat=flagsCat,
                           ).plotSkyPosition(filenamer, dataName="all", **skyplotKwargs)


    def plotStarGal(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None, tractInfo=None,
                    patchList=None, hscRun=None, matchRadius=None, zpLabel=None, forcedStr=None,
                    flagsCat=None, uberCalLabel=None):
        schema = getSchema(catalog)
        enforcer = None
        plotAllKwargs = dict(butler=butler, camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                             patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                             zpLabel=zpLabel, forcedStr=forcedStr, uberCalLabel=uberCalLabel)
        shortName = "pStar"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(catalog, deconvMomStarGal, "P(star) from deconvolved moments",
                           shortName, self.config.analysis, qMin=-0.1, qMax=1.39,
                           labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                           ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, **plotAllKwargs)
        shortName = "deconvMom"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(catalog, deconvMom, "Deconvolved moments", shortName,
                           self.config.analysis, qMin=-1.0, qMax=3.0, labeller=StarGalaxyLabeller(),
                           flagsCat=flagsCat,
                           ).plotAll(dataId, filenamer, self.log,
                                     enforcer=Enforcer(requireLess={"star": {"stdev": 0.2}}),
                                     **plotAllKwargs)

        if "ext_shapeHSM_HsmShapeRegauss_resolution" in schema:
            shortName = "resolution"
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(catalog, catalog["ext_shapeHSM_HsmShapeRegauss_resolution"],
                               "Resolution Factor from HsmRegauss",
                               shortName, self.config.analysis, qMin=-0.1, qMax=1.15,
                               labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                               ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, **plotAllKwargs)

    def plotCompareUnforced(self, forced, unforced, filenamer, dataId, butler=None, camera=None, ccdList=None,
                            tractInfo=None, patchList=None, hscRun=None, zpLabel=None, fluxToPlotList=None,
                            uberCalLabel=None, matchRadius=None, matchRadiusUnitStr=None, matchControl=None):
        forcedSchema = getSchema(forced)
        fluxToPlotList = fluxToPlotList if fluxToPlotList else self.config.fluxToPlotList
        unitStr = "mmag" if self.config.toMilli else "mag"
        enforcer = None
        for col in fluxToPlotList:
            magDiffFunc = MagDiff(col + "_instFlux", col + "_instFlux", unitScale=self.unitScale)
            shortName = "compareUnforced_" + col
            self.log.info("shortName = {:s}".format(shortName))
            if col + "_instFlux" in forcedSchema:
                self.AnalysisClass(forced, magDiffFunc(forced, unforced),
                                   "  Forced - Unforced mag [%s] (%s)" % (fluxToPlotString(col), unitStr),
                                   shortName, self.config.analysis, prefix="", flags=[col + "_flag"],
                                   labeller=OverlapsStarGalaxyLabeller(first="", second=""),
                                   unitScale=self.unitScale, compareCat=unforced,
                                   ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                             camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                             patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                             matchRadiusUnitStr=matchRadiusUnitStr, zpLabel=zpLabel,
                                             uberCalLabel=uberCalLabel)

    def isBad(self, source):
        """Return True if any of config.badFlags are set for this source."""
        for flag in self.config.analysis.flags:
            if source.get(flag):
                return True
        return False

    def overlaps(self, catalog):
        badForOverlap = makeBadArray(catalog, flagList=self.config.analysis.flags,
                                     onlyReadStars=self.config.onlyReadStars, patchInnerOnly=False)
        goodCat = catalog[~badForOverlap].copy(deep=True)
        skyCoords = coord.SkyCoord(goodCat["coord_ra"], goodCat["coord_dec"], unit=units.rad)
        # Setting nthneighbor to 2 to avoid self-matches, but this is not
        # robust agaist other zero distance matches (i.e. if any object other
        # than the source itself has identical coordinates, the returned index
        # may still be that of "self").
        # TODO: file an issue with astropy to robustly preclude self-matches
        inds, dists, _ = coord.match_coordinates_sky(skyCoords, skyCoords, nthneighbor=2)
        selfMatches = [i == ind for i, ind in enumerate(inds)]
        if sum(selfMatches) > 0:
            self.log.warn("There were {} objects self-matched by "
                          "astropy.coordinates.match_coordinates_sky()").format(sum(selfMatches))
        matchedIds = dists < self.config.matchOverlapRadius*units.arcsec
        matchedIndices = inds[matchedIds]
        matchedDistances = dists[matchedIds]
        matchFirst = goodCat[matchedIds].copy(deep=True)
        matchSecond = goodCat.iloc[matchedIndices].copy(deep=True)
        matchFirst.rename(columns=lambda x: "first_" + x, inplace=True)
        matchSecond.rename(columns=lambda x: "second_" + x, inplace=True)
        matchFirst.index = pd.RangeIndex(len(matchFirst.index))
        matchSecond.index = pd.RangeIndex(len(matchSecond.index))
        matches = pd.concat([matchFirst, matchSecond], axis=1)
        matches["distance"] = matchedDistances.rad
        if matches.empty:
            self.log.info("Did not find any overlapping matches")
        return matches

    def plotOverlaps(self, overlaps, filenamer, dataId, butler=None, camera=None, ccdList=None,
                     tractInfo=None, patchList=None, hscRun=None, matchRadius=None, matchRadiusUnitStr=None,
                     zpLabel=None, forcedStr=None, postFix="", fluxToPlotList=None, flagsCat=None,
                     uberCalLabel=None):
        schema = getSchema(overlaps)
        if not fluxToPlotList:
            fluxToPlotList = self.config.fluxToPlotList
        unitStr = "mmag" if self.config.toMilli else "mag"
        magEnforcer = Enforcer(requireLess={"star": {"stdev": 0.003*self.unitScale}})
        plotAllKwargs = dict(butler=butler, camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                             patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                             matchRadiusUnitStr=matchRadiusUnitStr, zpLabel=zpLabel, forcedStr=forcedStr,
                             uberCalLabel=uberCalLabel)
        for col in fluxToPlotList:
            shortName = "overlap_" + col + postFix
            self.log.info("shortName = {:s}".format(shortName))
            if "first_" + col + "_instFlux" in schema:
                self.AnalysisClass(overlaps, MagDiff("first_" + col + "_instFlux",
                                                     "second_" + col + "_instFlux",
                                                     unitScale=self.unitScale),
                                   "  Overlap mag difference (%s) (%s)" % (fluxToPlotString(col), unitStr),
                                   shortName, self.config.analysis, prefix="first_", flags=[col + "_flag"],
                                   labeller=OverlapsStarGalaxyLabeller(), magThreshold=23,
                                   unitScale=self.unitScale,
                                   ).plotAll(dataId, filenamer, self.log, enforcer=magEnforcer,
                                             **plotAllKwargs)
        unitStr = "mas" if self.config.toMilli else "arcsec"
        distEnforcer = Enforcer(requireLess={"star": {"stdev": 0.005*self.unitScale}})
        shortName = "overlap_distance" + postFix
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(overlaps,
                           lambda cat: cat["distance"]*(1.0*afwGeom.radians).asArcseconds()*self.unitScale,
                           "Distance (%s)" % unitStr, shortName, self.config.analysis, prefix="first_",
                           qMin=-0.01, qMax=0.11, labeller=OverlapsStarGalaxyLabeller(), flagsCat=flagsCat,
                           forcedMean=0.0, unitScale=self.unitScale,
                           ).plotAll(dataId, filenamer, self.log, enforcer=distEnforcer, doPrintMedian=True,
                                     **plotAllKwargs)

    def plotMatches(self, matches, filterName, filenamer, dataId, description="matches", butler=None,
                    camera=None, ccdList=None, tractInfo=None, patchList=None, hscRun=None, matchRadius=None,
                    matchRadiusUnitStr=None, zpLabel=None, forcedStr=None, flagsCat=None, uberCalLabel=None):
        schema = getSchema(matches)
        unitStr = "mmag" if self.config.toMilli else "mag"
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.030*self.unitScale}}),
        fluxToPlotList = ["base_PsfFlux_instFlux", "base_CircularApertureFlux_12_0_instFlux"]
        plotAllKwargs = dict(butler=butler, camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                             patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                             matchRadiusUnitStr=matchRadiusUnitStr, zpLabel=zpLabel, forcedStr=forcedStr,
                             uberCalLabel=uberCalLabel)
        if self.config.doApplyColorTerms:
            ct = self.config.colorterms.getColorterm(filterName, self.config.refObjLoader.ref_dataset_name)
        else:
            # Pass in a null colorterm.
            # Obtain the filter name from the reference loader filter map, if present, otherwise set
            # to the canonical filter name.
            refFilterName = (self.config.refObjLoader.filterMap[filterName] if
                             filterName in self.config.refObjLoader.filterMap.keys() else
                             afwImage.Filter(afwImage.Filter(filterName).getId()).getName())
            ct = Colorterm(primary=refFilterName, secondary=refFilterName)
            self.log.warn("Note: no colorterms loaded for {:s}, thus no colorterms will be applied to "
                          "the reference catalog".format(self.config.refObjLoader.ref_dataset_name))

        # Magnitude difference plots
        for fluxName in fluxToPlotList:
            if "src_calib_psf_used" in schema:
                shortName = description + "_" + fluxToPlotString(fluxName) + "_mag_calib_psf_used"
                self.log.info("shortName = {:s}".format(shortName))
                self.AnalysisClass(matches, MagDiffMatches(fluxName, ct, zp=0.0, unitScale=self.unitScale),
                                   "%s - ref (calib_psf_used) (%s)" % (fluxToPlotString(fluxName), unitStr),
                                   shortName,
                                   self.config.analysisMatches, prefix="src_", goodKeys=["calib_psf_used"],
                                   qMin=-0.15, qMax=0.1, labeller=MatchesStarGalaxyLabeller(),
                                   flagsCat=flagsCat, unitScale=self.unitScale,
                                   ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, **plotAllKwargs)
            if "src_calib_photometry_used" in schema:
                shortName = description + "_" + fluxToPlotString(fluxName) + "_mag_calib_photometry_used"
                self.log.info("shortName = {:s}".format(shortName))
                self.AnalysisClass(matches, MagDiffMatches(fluxName, ct, zp=0.0, unitScale=self.unitScale),
                                   "   %s - ref (calib_photom_used) (%s)" % (fluxToPlotString(fluxName),
                                                                             unitStr),
                                   shortName, self.config.analysisMatches, prefix="src_",
                                   goodKeys=["calib_photometry_used"], qMin=-0.15, qMax=0.15,
                                   labeller=MatchesStarGalaxyLabeller(), flagsCat=flagsCat,
                                   unitScale=self.unitScale,
                                   ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, **plotAllKwargs)
            shortName = description + "_" + fluxToPlotString(fluxName) + "_mag"
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(matches, MagDiffMatches(fluxName, ct, zp=0.0, unitScale=self.unitScale),
                               "%s - ref (%s)" % (fluxToPlotString(fluxName), unitStr), shortName,
                               self.config.analysisMatches,
                               prefix="src_", qMin=-0.15, qMax=0.5, labeller=MatchesStarGalaxyLabeller(),
                               unitScale=self.unitScale,
                               ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, **plotAllKwargs)

        # Astrometry (positional) difference plots
        unitStr = "mas" if self.config.toMilli else "arcsec"
        qMatchScale = matchRadius if matchRadius else self.matchRadius
        if "src_calib_astrometry_used" in schema:
            shortName = description + "_distance_calib_astrometry_used"
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(matches,
                               lambda cat:
                                   cat["distance"]*(1.0*afwGeom.radians).asArcseconds()*self.unitScale,
                               "Distance (%s) (calib_astrom_used)" % unitStr, shortName,
                               self.config.analysisMatches, prefix="src_", goodKeys=["calib_astrometry_used"],
                               qMin=-0.01*qMatchScale, qMax=0.5*qMatchScale,
                               labeller=MatchesStarGalaxyLabeller(), flagsCat=flagsCat,
                               unitScale=self.unitScale,
                               ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, doPrintMedian=True,
                                         **plotAllKwargs)
        shortName = description + "_distance"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(matches,
                           lambda cat: cat["distance"]*(1.0*afwGeom.radians).asArcseconds()*self.unitScale,
                           "Distance (%s)" % unitStr, shortName, self.config.analysisMatches, prefix="src_",
                           qMin=-0.05*qMatchScale, qMax=0.3*qMatchScale,
                           labeller=MatchesStarGalaxyLabeller(), flagsCat=flagsCat, forcedMean=0.0,
                           unitScale=self.unitScale,
                           ).plotAll(dataId, filenamer, self.log,
                                     enforcer=Enforcer(requireLess={"star": {"stdev": 0.050*self.unitScale}}),
                                     doPrintMedian=True, **plotAllKwargs)
        if "src_calib_astrometry_used" in schema:
            shortName = description + "_raCosDec_calib_astrometry_used"
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(matches, AstrometryDiff("src_coord_ra", "ref_coord_ra",
                                                       declination1="src_coord_dec",
                                                       declination2="ref_coord_dec",
                                                       unitScale=self.unitScale),
                               "      $\delta_{Ra}$ = $\Delta$RA*cos(Dec) (%s) (calib_astrom_used)" % unitStr,
                               shortName, self.config.analysisMatches, prefix="src_",
                               goodKeys=["calib_astrometry_used"], qMin=-0.2*qMatchScale,
                               qMax=0.2*qMatchScale, labeller=MatchesStarGalaxyLabeller(),
                               flagsCat=flagsCat, unitScale=self.unitScale,
                               ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, **plotAllKwargs)
        shortName = description + "_raCosDec"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(matches, AstrometryDiff("src_coord_ra", "ref_coord_ra",
                                                   declination1="src_coord_dec", declination2="ref_coord_dec",
                                                   unitScale=self.unitScale),
                           "$\delta_{Ra}$ = $\Delta$RA*cos(Dec) (%s)" % unitStr, shortName,
                           self.config.analysisMatches, prefix="src_", qMin=-0.2*qMatchScale,
                           qMax=0.2*qMatchScale, labeller=MatchesStarGalaxyLabeller(),
                           flagsCat=flagsCat, unitScale=self.unitScale,
                           ).plotAll(dataId, filenamer, self.log,
                                     enforcer=Enforcer(requireLess={"star": {"stdev": 0.050*self.unitScale}}),
                                     **plotAllKwargs)
        if "src_calib_astrometry_used" in schema:
            shortName = description + "_ra_calib_astrometry_used"
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(matches,
                               AstrometryDiff("src_coord_ra", "ref_coord_ra", unitScale=self.unitScale),
                               "$\Delta$RA (%s) (calib_astrom_used)" % unitStr, shortName,
                               self.config.analysisMatches, prefix="src_", goodKeys=["calib_astrometry_used"],
                               qMin=-0.25*qMatchScale, qMax=0.25*qMatchScale,
                               labeller=MatchesStarGalaxyLabeller(), flagsCat=flagsCat,
                               unitScale=self.unitScale,
                               ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, **plotAllKwargs)
        shortName = description + "_ra"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(matches, AstrometryDiff("src_coord_ra", "ref_coord_ra", unitScale=self.unitScale),
                           "$\Delta$RA (%s)" % unitStr, shortName, self.config.analysisMatches,
                           prefix="src_", qMin=-0.25*qMatchScale, qMax=0.25*qMatchScale,
                           labeller=MatchesStarGalaxyLabeller(), flagsCat=flagsCat, unitScale=self.unitScale,
                           ).plotAll(dataId, filenamer, self.log,
                                     enforcer=Enforcer(requireLess={"star": {"stdev": 0.050*self.unitScale}}),
                                     **plotAllKwargs)
        if "src_calib_astrometry_used" in schema:
            shortName = description + "_dec_calib_astrometry_used"
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(matches,
                               AstrometryDiff("src_coord_dec", "ref_coord_dec", unitScale=self.unitScale),
                               "$\delta_{Dec}$ (%s) (calib_astrom_used)" % unitStr, shortName,
                               self.config.analysisMatches, prefix="src_", goodKeys=["calib_astrometry_used"],
                               qMin=-0.25*qMatchScale, qMax=0.25*qMatchScale,
                               labeller=MatchesStarGalaxyLabeller(), flagsCat=flagsCat,
                               unitScale=self.unitScale,
                               ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, **plotAllKwargs)
        shortName = description + "_dec"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(matches,
                           AstrometryDiff("src_coord_dec", "ref_coord_dec", unitScale=self.unitScale),
                           "$\delta_{Dec}$ (%s)" % unitStr, shortName, self.config.analysisMatches,
                           prefix="src_", qMin=-0.3*qMatchScale, qMax=0.3*qMatchScale,
                           labeller=MatchesStarGalaxyLabeller(), flagsCat=flagsCat, unitScale=self.unitScale,
                           ).plotAll(dataId, filenamer, self.log,
                                     enforcer=Enforcer(requireLess={"star": {"stdev": 0.050*self.unitScale}}),
                                     **plotAllKwargs)

    def plotCosmos(self, catalog, filenamer, cosmos, dataId):
        labeller = CosmosLabeller(cosmos, self.config.matchRadiusRaDec*afwGeom.arcseconds)
        self.AnalysisClass(catalog, deconvMom, "Deconvolved moments", "cosmos", self.config.analysis,
                           qMin=-1.0, qMax=6.0, labeller=labeller,
                           ).plotAll(dataId, filenamer, self.log,
                                     enforcer=Enforcer(requireLess={"star": {"stdev": 0.2}}))

    def matchCatalog(self, catalog, filterName, astrometryConfig):
        try:  # lsst.meas.extensions.astrometryNet is not setup by default
            from lsst.meas.extensions.astrometryNet import LoadAstrometryNetObjectsTask  # noqa : F401
        except ImportError:
            return None
        refObjLoader = LoadAstrometryNetObjectsTask(self.config.refObjLoaderConfig)
        center = afwGeom.averageSpherePoint([src.getCoord() for src in catalog])
        radius = max(center.separation(src.getCoord()) for src in catalog)
        filterName = afwImage.Filter(afwImage.Filter(filterName).getId()).getName()  # Get primary name
        refs = refObjLoader.loadSkyCircle(center, radius, filterName).refCat
        matches = afwTable.matchRaDec(refs, catalog, self.config.matchRadiusRaDec*afwGeom.arcseconds)
        matches = matchNanojanskyToAB(matches)
        return joinMatches(matches, "ref_", "src_")

    def plotQuiver(self, catalog, filenamer, dataId=None, butler=None, camera=None, ccdList=None,
                   tractInfo=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None,
                   forcedStr=None, postFix="", flagsCat=None, uberCalLabel=None, scale=1):
        stats = None
        shortName = "quiver"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(catalog, None, "%s" % shortName, shortName,
                           self.config.analysis, labeller=None,
                           ).plotQuiver(catalog, filenamer, self.log, stats=stats, dataId=dataId,
                                        butler=butler, camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                        patchList=patchList, hscRun=hscRun, zpLabel=zpLabel,
                                        forcedStr=forcedStr, uberCalLabel=uberCalLabel, scale=scale)

    def plotInputCounts(self, catalog, filenamer, dataId, butler, tractInfo, patchList=None, camera=None,
                        hscRun=None, zpLabel=None, forcedStr=None, uberCalLabel=None, alpha=0.5,
                        doPlotTractImage=True, doPlotPatchOutline=True, sizeFactor=5.0, maxDiamPix=1000):
        shortName = "inputCounts"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(catalog, None, "%s" % shortName, shortName,
                           self.config.analysis, labeller=None,
                           ).plotInputCounts(catalog, filenamer, self.log, dataId, butler, tractInfo,
                                             patchList=patchList, camera=camera, forcedStr=forcedStr,
                                             uberCalLabel=uberCalLabel, alpha=alpha,
                                             doPlotTractImage=doPlotTractImage,
                                             doPlotPatchOutline=doPlotPatchOutline,
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
        # Find a visit/ccd input so that you can check for meas_mosaic input (i.e. to set uberCalLabel)
        self.uberCalLabel1 = determineUberCalLabel(repoInfo1, patchList1[0], coaddName=self.config.coaddName)
        self.uberCalLabel2 = determineUberCalLabel(repoInfo2, patchList1[0], coaddName=self.config.coaddName)
        self.uberCalLabel = self.uberCalLabel1 + "_1 " + self.uberCalLabel2 + "_2"
        self.log.info("Uber-calibration(s) used: {:}".format(self.uberCalLabel))

        if haveForced:
            forced1 = self.readCatalogs(patchRefList1, self.config.coaddName + "Coadd_forced_src")
            forced1 = self.calibrateCatalogs(forced1, wcs=repoInfo1.wcs)
            forced2 = self.readCatalogs(patchRefList2, self.config.coaddName + "Coadd_forced_src")
            forced2 = self.calibrateCatalogs(forced2, wcs=repoInfo2.wcs)
        unforced1 = self.readCatalogs(patchRefList1, self.config.coaddName + "Coadd_meas")
        unforced1 = self.calibrateCatalogs(unforced1, wcs=repoInfo1.wcs)
        unforced2 = self.readCatalogs(patchRefList2, self.config.coaddName + "Coadd_meas")
        unforced2 = self.calibrateCatalogs(unforced2, wcs=repoInfo2.wcs)

        forcedStr = "forced" if haveForced else "unforced"

        unforced1Schema = getSchema(unforced1)
        unforced2Schema = getSchema(unforced2)
        if haveForced:
            forced1Schema = getSchema(forced1)
            forced2Schema = getSchema(forced2)
            # copy over some fields from unforced to forced catalog
            forced1 = addColumnsToSchema(unforced1, forced1,
                                         [col for col in list(self.config.columnsToCopy) +
                                          list(self.config.analysis.flags) if
                                          col not in forced1Schema and col in unforced1Schema and
                                          not (repoInfo1.hscRun and col == "slot_Centroid_flag")])
            forced2 = addColumnsToSchema(unforced2, forced2,
                                         [col for col in list(self.config.columnsToCopy) +
                                          list(self.config.analysis.flags) if
                                          col not in forced2Schema and col in unforced2Schema and
                                          not (repoInfo2.hscRun and col == "slot_Centroid_flag")])

        # Set an alias map for differing schema naming conventions of different stacks (if any)
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

        # Set boolean array indicating sources deemed unsuitable for qa analyses
        bad1 = makeBadArray(unforced1, flagList=self.config.analysis.flags,
                            onlyReadStars=self.config.onlyReadStars)
        bad2 = makeBadArray(unforced2, flagList=self.config.analysis.flags,
                            onlyReadStars=self.config.onlyReadStars)
        if haveForced:
            bad1 |= makeBadArray(forced1, flagList=self.config.analysis.flags,
                                 onlyReadStars=self.config.onlyReadStars)
            bad2 |= makeBadArray(forced2, flagList=self.config.analysis.flags,
                                 onlyReadStars=self.config.onlyReadStars)

        # Purge the catalogs of flagged sources
        unforced1 = unforced1[~bad1].copy(deep=True)
        unforced2 = unforced2[~bad2].copy(deep=True)
        if haveForced:
            forced1 = forced1[~bad1].copy(deep=True)
            forced2 = forced2[~bad2].copy(deep=True)
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
        forcedSchema = getSchema(forced)

        self.log.info("\nNumber of sources in forced catalogs: first = {0:d} and second = {1:d}".format(
                      len(forced1), len(forced2)))

        subdir = "patch-" + str(patchList1[0]) if len(patchList1) == 1 else subdir
        filenamer = Filenamer(repoInfo1.butler, "plotCompareCoadd", repoInfo1.dataId, subdir=subdir)
        hscRun = repoInfo1.hscRun if repoInfo1.hscRun else repoInfo2.hscRun
        # Dict of all parameters common to plot* functions
        plotKwargs1 = dict(butler=repoInfo1.butler, camera=repoInfo1.camera, tractInfo=repoInfo1.tractInfo,
                           patchList=patchList1, hscRun=hscRun, matchRadius=self.matchRadius,
                           matchRadiusUnitStr=self.matchRadiusUnitStr,
                           zpLabel=self.zpLabel, uberCalLabel=self.uberCalLabel)

        if self.config.doPlotMags:
            self.plotMags(forced, filenamer, repoInfo1.dataId, forcedStr=forcedStr, **plotKwargs1)

        if self.config.doPlotSizes:
            if ("first_base_SdssShape_psf_xx" in forcedSchema and
               "second_base_SdssShape_psf_xx" in forcedSchema):
                self.plotSizes(forced, filenamer, repoInfo1.dataId, forcedStr=forcedStr, **plotKwargs1)
            else:
                self.log.warn("Cannot run plotSizes: base_SdssShape_psf_xx not in catalogSchema")

        if self.config.doApCorrs:
            self.plotApCorrs(unforced, filenamer, repoInfo1.dataId, forcedStr="unforced " + self.catLabel,
                             **plotKwargs1)
        if self.config.doPlotCentroids:
            self.plotCentroids(forced, filenamer, repoInfo1.dataId, forcedStr=forcedStr,
                               hscRun1=repoInfo1.hscRun, hscRun2=repoInfo2.hscRun, **plotKwargs1)
        if self.config.doPlotStarGalaxy:
            self.plotStarGal(forced, filenamer, repoInfo1.dataId, forcedStr=forcedStr, **plotKwargs1)

    def readCatalogs(self, patchRefList, dataset):
        catList = [patchRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS) for
                   patchRef in patchRefList if patchRef.datasetExists(dataset)]
        if not catList:
            raise TaskError("No catalogs read: %s" % ([patchRef.dataId for patchRef in patchRefList]))
        return concatenateCatalogs(catList)

    def matchCatalogs(self, catalog1, catalog2, matchRadius=None, matchControl=None):
        matchRadius = matchRadius if matchRadius else self.matchRadius
        matchControl = matchControl if matchControl else self.matchControl
        if self.config.matchXy:
            matches = afwTable.matchXy(catalog1, catalog2, matchRadius, matchControl)
        else:
            matches = afwTable.matchRaDec(catalog1, catalog2, matchRadius*afwGeom.arcseconds, matchControl)
        if not matches:
            raise TaskError("No matches found")
        return joinMatches(matches, "first_", "second_")

    def calibrateCatalogs(self, catalog, wcs=None):
        self.zpLabel = "common (" + str(self.config.analysis.coaddZp) + ")"
        # For some reason my persisted catalogs in lauren/LSST/DM-6816new all have nan for ra dec
        if np.all(np.isnan(catalog["coord_ra"])):
            if wcs is None:
                self.log.warn("Bad ra, dec entries but can't update because wcs is None")
            else:
                for src in catalog:
                    src.updateCoord(wcs)
        calibrated = calibrateCoaddSourceCatalog(catalog, self.config.analysis.coaddZp)
        return calibrated

    def plotMags(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None, tractInfo=None,
                 patchList=None, hscRun=None, matchRadius=None, matchRadiusUnitStr=None, zpLabel=None,
                 forcedStr=None, fluxToPlotList=None, postFix="", flagsCat=None, highlightList=None,
                 uberCalLabel=None):
        schema = getSchema(catalog)
        if not fluxToPlotList:
            fluxToPlotList = self.config.fluxToPlotList
        unitStr = "mmag" if self.config.toMilli else "mag"
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.02*self.unitScale}})
        for col in fluxToPlotList:
            if ("first_" + col + "_instFlux" in schema and "second_" + col + "_instFlux" in schema):
                shortName = "diff_" + col + postFix
                self.log.info("shortName = {:s}".format(shortName))
                Analysis(catalog, MagDiffCompare(col + "_instFlux", unitScale=self.unitScale),
                         "      Run Comparison: %s mag diff (%s)" % (fluxToPlotString(col), unitStr),
                         shortName, self.config.analysis, prefix="first_", qMin=-0.05, qMax=0.05,
                         flags=[col + "_flag"], errFunc=MagDiffCompareErr(col + "_instFlux",
                                                                          unitScale=self.unitScale),
                         labeller=OverlapsStarGalaxyLabeller(), flagsCat=flagsCat, unitScale=self.unitScale,
                         ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                   camera=camera, ccdList=ccdList, tractInfo=tractInfo, patchList=patchList,
                                   hscRun=hscRun, matchRadius=matchRadius,
                                   matchRadiusUnitStr=matchRadiusUnitStr, zpLabel=zpLabel,
                                   uberCalLabel=uberCalLabel, forcedStr=forcedStr, highlightList=highlightList)

    def plotCentroids(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None,
                      tractInfo=None, patchList=None, hscRun=None, hscRun1=None, hscRun2=None,
                      matchRadius=None, matchRadiusUnitStr=None, zpLabel=None, forcedStr=None,
                      flagsCat=None, highlightList=None, uberCalLabel=None):
        unitStr = "milliPixels" if self.config.toMilli else "pixels"
        distEnforcer = None
        centroidStr1, centroidStr2 = "base_SdssCentroid", "base_SdssCentroid"
        if bool(hscRun1) ^ bool(hscRun2):
            if not hscRun1:
                centroidStr1 = "base_SdssCentroid_Rot"
            if not hscRun2:
                centroidStr2 = "base_SdssCentroid_Rot"
        plotAllKwargs = dict(butler=butler, camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                             patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                             matchRadiusUnitStr=matchRadiusUnitStr, zpLabel=zpLabel, forcedStr=forcedStr,
                             uberCalLabel=uberCalLabel)

        shortName = "diff_x"
        self.log.info("shortName = {:s}".format(shortName))
        Analysis(catalog, CentroidDiff("x", centroid1=centroidStr1, centroid2=centroidStr2,
                                       unitScale=self.unitScale),
                 "Run Comparison: x offset (%s)" % unitStr, shortName, self.config.analysis, prefix="first_",
                 qMin=-0.08, qMax=0.08, errFunc=None, labeller=OverlapsStarGalaxyLabeller(),
                 ).plotAll(dataId, filenamer, self.log, enforcer=distEnforcer, **plotAllKwargs)
        shortName = "diff_y"
        self.log.info("shortName = {:s}".format(shortName))
        Analysis(catalog, CentroidDiff("y", centroid1=centroidStr1, centroid2=centroidStr2,
                                       unitScale=self.unitScale),
                 "Run Comparison: y offset (%s)" % unitStr, shortName, self.config.analysis, prefix="first_",
                 qMin=-0.08, qMax=0.08, errFunc=None, labeller=OverlapsStarGalaxyLabeller(),
                 ).plotAll(dataId, filenamer, self.log, enforcer=distEnforcer, **plotAllKwargs)

        unitStr = "mas" if self.config.toMilli else "arcsec"
        shortName = "diff_raCosDec"
        self.log.info("shortName = {:s}".format(shortName))
        Analysis(catalog, AstrometryDiff("first_coord_ra", "second_coord_ra", declination1="first_coord_dec",
                                         declination2="second_coord_dec", unitScale=self.unitScale),
                 "   Run Comparison: $\delta_{Ra}$ = $\Delta$RA*cos(Dec) (%s)" % unitStr, shortName,
                 self.config.analysisMatches, prefix="first_", qMin=-0.2*matchRadius, qMax=0.2*matchRadius,
                 labeller=OverlapsStarGalaxyLabeller(),
                 flagsCat=flagsCat, unitScale=self.unitScale,
                 ).plotAll(dataId, filenamer, self.log, **plotAllKwargs)
        shortName = "diff_ra"
        self.log.info("shortName = {:s}".format(shortName))
        Analysis(catalog, AstrometryDiff("first_coord_ra", "second_coord_ra", declination1=None,
                                         declination2=None, unitScale=self.unitScale),
                 "Run Comparison: $\Delta$RA (%s)" % unitStr, shortName, self.config.analysisMatches,
                 prefix="first_", qMin=-0.25*matchRadius, qMax=0.25*matchRadius,
                 labeller=OverlapsStarGalaxyLabeller(), flagsCat=flagsCat, unitScale=self.unitScale,
                 ).plotAll(dataId, filenamer, self.log, **plotAllKwargs)
        shortName = "diff_dec"
        self.log.info("shortName = {:s}".format(shortName))
        Analysis(catalog, AstrometryDiff("first_coord_dec", "second_coord_dec", unitScale=self.unitScale),
                 "$\delta_{Dec}$ (%s)" % unitStr, shortName, self.config.analysisMatches, prefix="first_",
                 qMin=-0.3*matchRadius, qMax=0.3*matchRadius, labeller=OverlapsStarGalaxyLabeller(),
                 flagsCat=flagsCat, unitScale=self.unitScale,
                 ).plotAll(dataId, filenamer, self.log, **plotAllKwargs)

    def plotFootprint(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None,
                      tractInfo=None, patchList=None, hscRun=None, matchRadius=None, matchRadiusUnitStr=None,
                      zpLabel=None, forcedStr=None, postFix="", flagsCat=None, highlightList=None,
                      uberCalLabel=None):
        enforcer = None
        plotAllKwargs = dict(butler=butler, camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                             patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                             matchRadiusUnitStr=matchRadiusUnitStr, zpLabel=zpLabel,
                             forcedStr=forcedStr, uberCalLabel=uberCalLabel, postFix=postFix)
        shortName = "diff_footNpix"
        col = "base_Footprint_nPix"
        self.log.info("shortName = {:s}".format(shortName))
        Analysis(catalog, FootNpixDiffCompare(col), "  Run Comparison: Footprint nPix difference", shortName,
                 self.config.analysis, prefix="first_", qMin=-250, qMax=250, flags=[col + "_flag"],
                 labeller=OverlapsStarGalaxyLabeller(), flagsCat=flagsCat,
                 ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, **plotAllKwargs)
        shortName = "diff_footNpix_calib_psf_used"
        self.log.info("shortName = {:s}".format(shortName))
        Analysis(catalog, FootNpixDiffCompare(col), "     Run Comparison: Footprint nPix diff (psf_used)",
                 shortName, self.config.analysis, prefix="first_", goodKeys=["calib_psf_used"],
                 qMin=-150, qMax=150, flags=[col + "_flag"], labeller=OverlapsStarGalaxyLabeller(),
                 flagsCat=flagsCat,
                 ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, highlightList=highlightList,
                           **plotAllKwargs)

    def plotSizes(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None, tractInfo=None,
                  patchList=None, hscRun=None, matchRadius=None, matchRadiusUnitStr=None, zpLabel=None,
                  forcedStr=None, uberCalLabel=None):
        schema = getSchema(catalog)
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.02*self.unitScale}})
        plotAllKwargs = dict(butler=butler, camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                             patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                             matchRadiusUnitStr=matchRadiusUnitStr, zpLabel=zpLabel, forcedStr=forcedStr,
                             uberCalLabel=uberCalLabel)
        for col in ["base_PsfFlux"]:
            if ("first_" + col + "_instFlux" in schema and "second_" + col + "_instFlux" in schema):
                # Make comparison plots for all objects and calib_psf_used only objects
                for goodFlags in [[], ["calib_psf_used"]]:
                    badFlags = [col + "_flag", "base_SdssShape_flag"]
                    subCatString = " (calib_psf_used)" if "calib_psf_used" in goodFlags else ""
                    shortNameBase = "trace"
                    shortName = (shortNameBase + "_calib_psf_used" if "calib_psf_used" in goodFlags else
                                 shortNameBase)
                    compareCol = "base_SdssShape"
                    self.log.info("shortName = {:s}".format(shortName))
                    Analysis(catalog, TraceSizeCompare(compareCol),
                             "    SdssShape Trace Radius Diff (%)" + subCatString,
                             shortName, self.config.analysis, flags=badFlags, prefix="first_",
                             goodKeys=goodFlags, qMin=-0.5, qMax=1.5, labeller=OverlapsStarGalaxyLabeller(),
                             ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, **plotAllKwargs)

                    shortNameBase = "psfTrace"
                    shortName = (shortNameBase + "_calib_psf_used" if "calib_psf_used" in goodFlags else
                                 shortNameBase)
                    compareCol = "base_SdssShape_psf"
                    self.log.info("shortName = {:s}".format(shortName))
                    Analysis(catalog, TraceSizeCompare(compareCol),
                             "       SdssShape PSF Trace Radius Diff (%)" + subCatString,
                             shortName, self.config.analysis, flags=badFlags, prefix="first_",
                             goodKeys=goodFlags, qMin=-1.1, qMax=1.1, labeller=OverlapsStarGalaxyLabeller(),
                             ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, **plotAllKwargs)

                    badFlags = [col + "_flag", "ext_shapeHSH_HsmSourceMoments_flag"]
                    if "first_ext_shapeHSM_HsmSourceMoments_xx" in schema:
                        shortNameBase = "hsmTrace"
                        shortName = (shortNameBase + "_calib_psf_used" if "calib_psf_used" in goodFlags else
                                     shortNameBase)
                        compareCol = "ext_shapeHSM_HsmSourceMoments"
                        self.log.info("shortName = {:s}".format(shortName))
                        Analysis(catalog, TraceSizeCompare(compareCol),
                                 "   HSM Trace Radius Diff (%)" + subCatString, shortName,
                                 self.config.analysis, flags=badFlags, prefix="first_",
                                 goodKeys=goodFlags, qMin=-0.5, qMax=1.5,
                                 labeller=OverlapsStarGalaxyLabeller(),
                                 ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, **plotAllKwargs)
                        shortNameBase = "hsmPsfTrace"
                        shortName = (shortNameBase + "_calib_psf_used" if "calib_psf_used" in goodFlags else
                                     shortNameBase)
                    badFlags = [col + "_flag", "ext_shapeHSH_PsfMoments_flag"]
                    if "first_ext_shapeHSM_PsfMoments_xx" in schema:
                        compareCol = "ext_shapeHSM_HsmPsfMoments"
                        self.log.info("shortName = {:s}".format(shortName))
                        Analysis(catalog, TraceSizeCompare(compareCol),
                                 "      HSM PSF Trace Radius Diff (%)" + subCatString,
                                 shortName, self.config.analysis, flags=badFlags, prefix="first_",
                                 goodKeys=goodFlags, qMin=-1.1, qMax=1.1,
                                 labeller=OverlapsStarGalaxyLabeller(),
                                 ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, **plotAllKwargs)

                badFlags = [col + "_flag", "base_SdssShape_flag"]
                shortName = "sdssXx"
                compareCol = "base_SdssShape_xx"
                self.log.info("shortName = {:s}".format(shortName))
                Analysis(catalog, PercentDiff(compareCol), "SdssShape xx Moment Diff (%)", shortName,
                         self.config.analysis, flags=badFlags, prefix="first_",
                         qMin=-0.5, qMax=1.5, labeller=OverlapsStarGalaxyLabeller(),
                         ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, **plotAllKwargs)
                shortName = "sdssYy"
                compareCol = "base_SdssShape_yy"
                self.log.info("shortName = {:s}".format(shortName))
                Analysis(catalog, PercentDiff(compareCol), "SdssShape yy Moment Diff (%)", shortName,
                         self.config.analysis, flags=badFlags, prefix="first_",
                         qMin=-0.5, qMax=1.5, labeller=OverlapsStarGalaxyLabeller(),
                         ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, **plotAllKwargs)

    def plotStarGal(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None, tractInfo=None,
                    patchList=None, hscRun=None, matchRadius=None, matchRadiusUnitStr=None, zpLabel=None,
                    forcedStr=None, flagsCat=None, uberCalLabel=None):
        schema = getSchema(catalog)
        enforcer = None
        plotAllKwargs = dict(butler=butler, camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                             patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                             matchRadiusUnitStr=matchRadiusUnitStr, zpLabel=zpLabel, forcedStr=forcedStr,
                             uberCalLabel=uberCalLabel)
        col = "ext_shapeHSM_HsmShapeRegauss_resolution"
        if "first_" + col in schema:
            shortName = "diff_resolution"
            self.log.info("shortName = {:s}".format(shortName))
            Analysis(catalog, PercentDiff(col),
                     "           Run Comparison: HsmRegauss Resolution (% diff)",
                     shortName, self.config.analysis, flags=[col + "_flag"], prefix="first_",
                     qMin=-0.2, qMax=0.2,
                     labeller=OverlapsStarGalaxyLabeller(), flagsCat=flagsCat,
                     ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, **plotAllKwargs)
        col = "ext_shapeHSM_HsmShapeRegauss_e1"
        if "first_" + col in schema:
            shortName = "diff_HsmShapeRegauss_e1"
            self.log.info("shortName = {:s}".format(shortName))
            Analysis(catalog, PercentDiff(col),
                     "    Run Comparison: HsmRegauss e1 (% diff)",
                     shortName, self.config.analysis, flags=[col + "_flag"], prefix="first_",
                     qMin=-0.2, qMax=0.2,
                     labeller=OverlapsStarGalaxyLabeller(), flagsCat=flagsCat,
                     ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, **plotAllKwargs)
        col = "ext_shapeHSM_HsmShapeRegauss_e2"
        if "first_" + col in schema:
            shortName = "diff_HsmShapeRegauss_e2"
            self.log.info("shortName = {:s}".format(shortName))
            Analysis(catalog, PercentDiff(col),
                     "    Run Comparison: HsmRegauss e2 (% diff)",
                     shortName, self.config.analysis, flags=[col + "_flag"], prefix="first_",
                     qMin=-0.2, qMax=0.2,
                     labeller=OverlapsStarGalaxyLabeller(), flagsCat=flagsCat,
                     ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, **plotAllKwargs)

    def plotApCorrs(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None,
                    tractInfo=None, patchList=None, hscRun=None, matchRadius=None, matchRadiusUnitStr=None,
                    zpLabel=None, forcedStr=None, fluxToPlotList=None, uberCalLabel=None):
        schema = getSchema(catalog)
        if not fluxToPlotList:
            fluxToPlotList = self.config.fluxToPlotList
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.02*self.unitScale}})
        for col in fluxToPlotList:
            if "first_" + col + "_apCorr" in schema and "second_" + col + "_apCorr" in schema:
                shortName = "diff_" + col + "_apCorr"
                self.log.info("shortName = {:s}".format(shortName))
                # apCorrs in coadds can be all nan if they weren't run in sfm, so add a check for valid data
                # but here so we don't encounter the fatal error in Analysis
                if (len(np.where(np.isfinite(catalog["first_" + col + "_apCorr"]))[0]) > 0 and
                   len(np.where(np.isfinite(catalog["second_" + col + "_apCorr"]))[0]) > 0):
                    Analysis(catalog, MagDiffCompare(col + "_apCorr"),
                             "  Run Comparison: %s apCorr diff" % fluxToPlotString(col),
                             shortName, self.config.analysis,
                             prefix="first_", qMin=-0.025, qMax=0.025, flags=[col + "_flag_apCorr"],
                             labeller=OverlapsStarGalaxyLabeller(),
                             ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                       camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                       patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                       matchRadiusUnitStr=matchRadiusUnitStr, zpLabel=None,
                                       forcedStr=forcedStr, uberCalLabel=uberCalLabel)
                else:
                    self.log.warn("No valid data points for shortName = {:s}.  Skipping...".format(shortName))

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None

    def _getEupsVersionsName(self):
        return None
