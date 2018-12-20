#!/usr/bin/env python

from __future__ import print_function

import os
import numpy as np
np.seterr(all="ignore")  # noqa E402
import functools

from collections import defaultdict

from lsst.daf.persistence.butler import Butler
from lsst.pex.config import (Config, Field, ConfigField, ListField, DictField, ConfigDictField,
                             ConfigurableField)
from lsst.pipe.base import CmdLineTask, ArgumentParser, TaskRunner, TaskError
from lsst.coadd.utils import TractDataIdContainer
from lsst.afw.table.catalogMatches import matchesToCatalog
from lsst.meas.astrom import AstrometryConfig
from lsst.meas.extensions.astrometryNet import LoadAstrometryNetObjectsTask
from lsst.pipe.tasks.colorterms import Colorterm, ColortermLibrary

from lsst.meas.algorithms import LoadIndexedReferenceObjectsTask

from .analysis import AnalysisConfig, Analysis
from .utils import (Filenamer, Enforcer, MagDiff, MagDiffMatches, MagDiffCompare,
                    AstrometryDiff, TraceSize, PsfTraceSizeDiff, TraceSizeCompare, PercentDiff,
                    E1Resids, E2Resids, E1ResidsHsmRegauss, E2ResidsHsmRegauss, FootNpixDiffCompare,
                    MagDiffErr, CentroidDiff, deconvMom,
                    deconvMomStarGal, concatenateCatalogs, joinMatches, checkPatchOverlap,
                    addColumnsToSchema, addApertureFluxesHSC, addFpPoint,
                    addFootprintNPix, makeBadArray, addIntFloatOrStrColumn,
                    calibrateCoaddSourceCatalog, backoutApCorr, matchJanskyToDn,
                    fluxToPlotString, andCatalog, writeParquet, getRepoInfo, setAliasMaps)
from .plotUtils import (CosmosLabeller, AllLabeller, StarGalaxyLabeller, OverlapsStarGalaxyLabeller,
                        MatchesStarGalaxyLabeller)

import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable

__all__ = ["CoaddAnalysisConfig", "CoaddAnalysisRunner", "CoaddAnalysisTask", "CompareCoaddAnalysisConfig",
           "CompareCoaddAnalysisRunner", "CompareCoaddAnalysisTask"]


class CoaddAnalysisConfig(Config):
    coaddName = Field(dtype=str, default="deep", doc="Name for coadd")
    matchRadius = Field(dtype=float, default=0.5, doc="Matching radius (arcseconds)")
    matchOverlapRadius = Field(dtype=float, default=0.5, doc="Matching radius for overlaps (arcseconds)")
    colorterms = ConfigField(dtype=ColortermLibrary,
                             doc=("Library of color terms."
                                  "\nNote that the colorterms, if any, need to be loaded in a config "
                                  "override file.  See obs_subaru/config/hsc/coaddAnalysis.py for an "
                                  "example.  If the colorterms for the appropriate reference dataset are "
                                  "loaded, they will be applied.  Otherwise, no colorterms will be applied "
                                  "to the reference catalog."))
    doApplyColorTerms = Field(dtype=bool, default=True, doc="Apply colorterms to refernece magnitudes?")
    analysis = ConfigField(dtype=AnalysisConfig, doc="Analysis plotting options")
    analysisMatches = ConfigField(dtype=AnalysisConfig, doc="Analysis plotting options for matches")
    matchesMaxDistance = Field(dtype=float, default=0.15, doc="Maximum plotting distance for matches")
    externalCatalogs = ConfigDictField(keytype=str, itemtype=AstrometryConfig, default={},
                                       doc="Additional external catalogs for matching")
    refObjLoader = ConfigurableField(target=LoadIndexedReferenceObjectsTask, doc="Reference object loader")
    doPlotMags = Field(dtype=bool, default=True, doc="Plot magnitudes?")
    doPlotSizes = Field(dtype=bool, default=True, doc="Plot PSF sizes?")
    doPlotCentroids = Field(dtype=bool, default=True, doc="Plot centroids?")
    doApCorrs = Field(dtype=bool, default=True, doc="Plot aperture corrections?")
    doBackoutApCorr = Field(dtype=bool, default=False, doc="Backout aperture corrections?")
    doAddAperFluxHsc = Field(dtype=bool, default=False,
                             doc="Add a field containing 12 pix circular aperture flux to HSC table?")
    doPlotStarGalaxy = Field(dtype=bool, default=True, doc="Plot star/galaxy?")
    doPlotOverlaps = Field(dtype=bool, default=True, doc="Plot overlaps?")
    doPlotMatches = Field(dtype=bool, default=True, doc="Plot matches?")
    doPlotCompareUnforced = Field(dtype=bool, default=True,
                                  doc="Plot difference between forced and unforced?")
    doPlotQuiver = Field(dtype=bool, default=True, doc="Plot ellipticity residuals quiver plot?")
    doPlotPsfFluxHist = Field(dtype=bool, default=True, doc="Plot histogram of raw PSF instFluxes?")
    doPlotFootprintNpix = Field(dtype=bool, default=True, doc="Plot histogram of footprint nPix?")
    doPlotInputCounts = Field(dtype=bool, default=True, doc="Make input counts plot?")
    onlyReadStars = Field(dtype=bool, default=False, doc="Only read stars (to save memory)?")
    toMilli = Field(dtype=bool, default=True, doc="Print stats in milli units (i.e. mas, mmag)?")
    srcSchemaMap = DictField(keytype=str, itemtype=str, default=None, optional=True,
                             doc="Mapping between different stack (e.g. HSC vs. LSST) schema names")
    fluxToPlotList = ListField(dtype=str, default=["base_GaussianFlux", "ext_photometryKron_KronFlux",
                                                   "modelfit_CModel"],
                               doc="List of fluxes to plot: mag(flux)-mag(base_PsfFlux) vs mag(fluxColumn)")
    columnsToCopy = ListField(dtype=str,
                              default=["calib_psf_used", "calib_psf_candidate", "detect_isPatchInner",
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


class CoaddAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        kwargs["cosmos"] = parsedCmd.cosmos

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
        return parser

    def __init__(self, *args, **kwargs):
        CmdLineTask.__init__(self, *args, **kwargs)
        self.unitScale = 1000.0 if self.config.toMilli else 1.0

    def runDataRef(self, patchRefList, cosmos=None):
        haveForced = False  # do forced datasets exits (may not for single band datasets)
        dataset = "Coadd_forced_src"
        # Explicit input file was checked in CoaddAnalysisRunner, so a check on datasetExists
        # is sufficient here (modulo the case where a forced dataset exists higher up the parent
        # tree than the specified input, but does not exist in the input directory as the former
        # will be found)
        if patchRefList[0].datasetExists(self.config.coaddName + dataset):
            haveForced = True
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
        filenamer = Filenamer(repoInfo.butler, self.outputDataset, repoInfo.dataId)
        if (self.config.doPlotMags or self.config.doPlotStarGalaxy or self.config.doPlotOverlaps or
                self.config.doPlotCompareUnforced or cosmos or self.config.externalCatalogs):
            if haveForced:
                forced = self.readCatalogs(patchRefList, self.config.coaddName + "Coadd_forced_src")
                forced = self.calibrateCatalogs(forced, wcs=repoInfo.wcs)
            unforced = self.readCatalogs(patchRefList, self.config.coaddName + "Coadd_meas")
            unforced = self.calibrateCatalogs(unforced, wcs=repoInfo.wcs)

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
        aliasDictList = [self.config.flagsToAlias, ]
        if repoInfo.hscRun is not None and self.config.srcSchemaMap is not None:
            aliasDictList += [self.config.srcSchemaMap]
        for cat in coaddList:
            cat = setAliasMaps(cat, aliasDictList)

        forcedStr = "forced" if haveForced else "unforced"

        if self.config.doPlotFootprintNpix:
            unforced = addFootprintNPix(unforced, fromCat=unforced)
            if haveForced:
                forced = addFootprintNPix(forced, fromCat=unforced)

        # Must do the overlaps before purging the catalogs of non-primary sources
        if self.config.doPlotOverlaps:
            # Determine if any patches in the patchList actually overlap
            overlappingPatches = checkPatchOverlap(patchList, repoInfo.tractInfo)
            if not overlappingPatches:
                self.log.info("No overlapping patches...skipping overlap plots")
            else:
                self.catLabel = "nChild = 0"
                if haveForced:
                    forcedOverlaps = self.overlaps(forced)
                    if forcedOverlaps:
                        self.plotOverlaps(forcedOverlaps, filenamer, repoInfo.dataId, butler=repoInfo.butler,
                                          camera=repoInfo.camera, tractInfo=repoInfo.tractInfo,
                                          patchList=patchList, hscRun=repoInfo.hscRun,
                                          matchRadius=self.config.matchOverlapRadius, zpLabel=self.zpLabel,
                                          forcedStr=forcedStr, postFix="_forced",
                                          fluxToPlotList=["modelfit_CModel", ])
                    self.log.info("Number of forced overlap objects matched = {:d}".
                                  format(len(forcedOverlaps)))
                unforcedOverlaps = self.overlaps(unforced)
                if unforcedOverlaps:
                    self.plotOverlaps(unforcedOverlaps, filenamer, repoInfo.dataId, butler=repoInfo.butler,
                                      camera=repoInfo.camera, tractInfo=repoInfo.tractInfo,
                                      patchList=patchList, hscRun=repoInfo.hscRun,
                                      matchRadius=self.config.matchOverlapRadius, zpLabel=self.zpLabel,
                                      forcedStr="unforced", postFix="_unforced",
                                      fluxToPlotList=["modelfit_CModel", ])
                self.log.info("Number of unforced overlap objects matched = {:d}".
                              format(len(unforcedOverlaps)))

        # Set boolean array indicating sources deemed unsuitable for qa analyses
        self.catLabel = "noDuplicates"
        bad = makeBadArray(unforced, flagList=self.config.analysis.flags,
                           onlyReadStars=self.config.onlyReadStars)
        if haveForced:
            bad |= makeBadArray(forced, flagList=self.config.analysis.flags,
                                onlyReadStars=self.config.onlyReadStars)

        # Create and write parquet tables
        if self.config.doWriteParquetTables:
            tableFilenamer = Filenamer(repoInfo.butler, 'qaTableCoadd', repoInfo.dataId)
            if haveForced:
                writeParquet(forced, tableFilenamer(repoInfo.dataId, description='forced'), badArray=bad)
            writeParquet(unforced, tableFilenamer(repoInfo.dataId, description='unforced'), badArray=bad)
            if self.config.writeParquetOnly:
                self.log.info("Exiting after writing Parquet tables.  No plots generated.")
                return

        # Purge the catalogs of flagged sources
        unforced = unforced[~bad].copy(deep=True)
        if haveForced:
            forced = forced[~bad].copy(deep=True)
        else:
            forced = unforced
        self.zpLabel = self.zpLabel + " " + self.catLabel
        if haveForced:
            self.log.info("\nNumber of sources in catalogs: unforced = {0:d} and forced = {1:d}".format(
                len(unforced), len(forced)))
        else:
            self.log.info("\nNumber of sources in catalog: unforced = {0:d}".format(len(unforced)))

        flagsCat = unforced

        if self.config.doPlotFootprintNpix:
            self.plotFootprintHist(forced, filenamer(repoInfo.dataId, description="footNpix", style="hist"),
                                   repoInfo.dataId, butler=repoInfo.butler, camera=repoInfo.camera,
                                   tractInfo=repoInfo.tractInfo, patchList=patchList, hscRun=repoInfo.hscRun,
                                   zpLabel=self.zpLabel, flagsCat=flagsCat)
            self.plotFootprint(forced, filenamer, repoInfo.dataId, butler=repoInfo.butler,
                               camera=repoInfo.camera, tractInfo=repoInfo.tractInfo, patchList=patchList,
                               hscRun=repoInfo.hscRun, zpLabel=self.zpLabel, flagsCat=flagsCat)

        if self.config.doPlotQuiver:
            self.plotQuiver(unforced, filenamer(repoInfo.dataId, description="ellipResids", style="quiver"),
                            dataId=repoInfo.dataId, butler=repoInfo.butler, camera=repoInfo.camera,
                            tractInfo=repoInfo.tractInfo, patchList=patchList, hscRun=repoInfo.hscRun,
                            zpLabel=self.zpLabel, forcedStr="unforced", scale=2)

        if self.config.doPlotInputCounts:
            self.plotInputCounts(unforced, filenamer(repoInfo.dataId, description="inputCounts",
                                                     style="tract"),
                                 dataId=repoInfo.dataId, butler=repoInfo.butler, tractInfo=repoInfo.tractInfo,
                                 patchList=patchList, camera=repoInfo.camera, hscRun=repoInfo.hscRun,
                                 forcedStr="unforced", alpha=0.5, doPlotTractImage=True,
                                 doPlotPatchOutline=True, sizeFactor=5.0, maxDiamPix=1000)

        if self.config.doPlotMags:
            self.plotMags(unforced, filenamer, repoInfo.dataId, butler=repoInfo.butler,
                          camera=repoInfo.camera, tractInfo=repoInfo.tractInfo, patchList=patchList,
                          hscRun=repoInfo.hscRun, zpLabel=self.zpLabel, forcedStr="unforced",
                          postFix="_unforced", flagsCat=flagsCat)
            if haveForced:
                self.plotMags(forced, filenamer, repoInfo.dataId, butler=repoInfo.butler,
                              camera=repoInfo.camera, tractInfo=repoInfo.tractInfo, patchList=patchList,
                              hscRun=repoInfo.hscRun, zpLabel=self.zpLabel, forcedStr=forcedStr,
                              postFix="_forced", flagsCat=flagsCat,
                              highlightList=[("merge_measurement_" + repoInfo.genericFilterName, 0,
                                              "yellow"), ])
        if self.config.doPlotStarGalaxy:
            if "ext_shapeHSM_HsmSourceMoments_xx" in unforced.schema:
                self.plotStarGal(unforced, filenamer, repoInfo.dataId, butler=repoInfo.butler,
                                 camera=repoInfo.camera, tractInfo=repoInfo.tractInfo, patchList=patchList,
                                 hscRun=repoInfo.hscRun, zpLabel=self.zpLabel, forcedStr="unforced")
            else:
                self.log.warn("Cannot run plotStarGal: ext_shapeHSM_HsmSourceMoments_xx not in forced.schema")

        if self.config.doPlotSizes:
            if all(ss in unforced.schema for ss in ["base_SdssShape_psf_xx", "calib_psf_used"]):
                self.plotSizes(unforced, filenamer, repoInfo.dataId, butler=repoInfo.butler,
                               camera=repoInfo.camera, tractInfo=repoInfo.tractInfo, patchList=patchList,
                               hscRun=repoInfo.hscRun, zpLabel=self.zpLabel, forcedStr="unforced",
                               postFix="_unforced")
            else:
                self.log.warn("Cannot run plotSizes: base_SdssShape_psf_xx and/or calib_psf_used "
                              "not in unforced.schema")
            if haveForced:
                if all(ss in forced.schema for ss in ["base_SdssShape_psf_xx", "calib_psf_used"]):
                    self.plotSizes(forced, filenamer, repoInfo.dataId, butler=repoInfo.butler,
                                   camera=repoInfo.camera, tractInfo=repoInfo.tractInfo, patchList=patchList,
                                   hscRun=repoInfo.hscRun, zpLabel=self.zpLabel, forcedStr=forcedStr)
                else:
                    self.log.warn("Cannot run plotSizes: base_SdssShape_psf_xx and/or calib_psf_used "
                                  "not in forced.schema")
        if cosmos:
            self.plotCosmos(forced, filenamer, cosmos, repoInfo.dataId)
        if self.config.doPlotCompareUnforced and haveForced:
            self.plotCompareUnforced(forced, unforced, filenamer, repoInfo.dataId, butler=repoInfo.butler,
                                     camera=repoInfo.camera, tractInfo=repoInfo.tractInfo,
                                     patchList=patchList, hscRun=repoInfo.hscRun,
                                     matchRadius=self.config.matchRadius, zpLabel=self.zpLabel)

        if self.config.doPlotMatches:
            if haveForced:
                matches = self.readSrcMatches(patchRefList, self.config.coaddName + "Coadd_forced_src",
                                              hscRun=repoInfo.hscRun, wcs=repoInfo.wcs,
                                              aliasDictList=aliasDictList)
            else:
                matches = self.readSrcMatches(patchRefList, self.config.coaddName + "Coadd_meas",
                                              hscRun=repoInfo.hscRun, wcs=repoInfo.wcs,
                                              aliasDictList=aliasDictList)
            self.plotMatches(matches, repoInfo.filterName, filenamer, repoInfo.dataId, butler=repoInfo.butler,
                             camera=repoInfo.camera, tractInfo=repoInfo.tractInfo, patchList=patchList,
                             hscRun=repoInfo.hscRun, zpLabel=self.zpLabel, forcedStr=forcedStr)

        for cat in self.config.externalCatalogs:
            with andCatalog(cat):
                matches = self.matchCatalog(forced, repoInfo.filterName, self.config.externalCatalogs[cat])
                self.plotMatches(matches, repoInfo.filterName, filenamer, repoInfo.dataId,
                                 butler=repoInfo.butler, camera=repoInfo.camera, tractInfo=repoInfo.tractInfo,
                                 patchList=patchList, hscRun=repoInfo.hscRun, zpLabel=self.zpLabel,
                                 forcedStr=forcedStr, matchRadius=self.config.matchRadius)

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
            if aliasDictList is not None:
                catalog = setAliasMaps(catalog, aliasDictList)
            if dataset != "deepCoadd_meas" and any(ss not in catalog.schema
                                                   for ss in self.config.columnsToCopy):
                unforced = dataRef.get("deepCoadd_meas", immediate=True,
                                       flags=afwTable.SOURCE_IO_NO_FOOTPRINTS)
                # copy over some fields from unforced to forced catalog
                catalog = addColumnsToSchema(unforced, catalog,
                                             [col for col in list(self.config.columnsToCopy) +
                                              list(self.config.analysis.flags) if
                                              col not in catalog.schema and col in unforced.schema and
                                              not (hscRun and col == "slot_Centroid_flag")])
                if aliasDictList is not None:
                    catalog = setAliasMaps(catalog, aliasDictList)

            # Set boolean array indicating sources deemed unsuitable for qa analyses
            bad = makeBadArray(catalog, flagList=self.config.analysis.flags,
                               onlyReadStars=self.config.onlyReadStars)

            catalog = self.calibrateCatalogs(catalog, wcs=wcs)

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
            self.catLabel = "noDuplicates"
            self.zpLabel = self.zpLabel + " " + self.catLabel
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
                                   "Do you have the correct astrometry_net_data setup?")
            # LSST reads in a_net catalogs with flux in "janskys", so must convert back to DN
            matches = matchJanskyToDn(matches)
            if hscRun and self.config.doAddAperFluxHsc:
                addApertureFluxesHSC(matches, prefix="second_")

            if not matches:
                self.log.warn("No matches for %s" % (dataRef.dataId,))
                continue

            # Set the alias maps for the matches sources (i.e. the .second attribute schema for each match)
            if aliasDictList is not None:
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
            if aliasDictList is not None:
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
                 patchList=None, hscRun=None, matchRadius=None, zpLabel=None, forcedStr=None,
                 fluxToPlotList=None, postFix="", flagsCat=None, highlightList=None):
        if fluxToPlotList is None:
            fluxToPlotList = self.config.fluxToPlotList
        unitStr = "mmag" if self.config.toMilli else "mag"
        enforcer = Enforcer(requireLess={"star": {"stdev": 0.02*self.unitScale}})
        for col in fluxToPlotList:
            if col + "_instFlux" in catalog.schema:
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
                                             zpLabel=zpLabel, forcedStr=forcedStr,
                                             highlightList=highlightList)

    def plotSizes(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None, tractInfo=None,
                  patchList=None, hscRun=None, matchRadius=None, zpLabel=None, forcedStr=None, postFix="",
                  flagsCat=None):
        enforcer = None
        unitStr = " (milli)" if self.config.toMilli else ""
        for col in ["base_PsfFlux", ]:
            if col + "_instFlux" in catalog.schema:
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
                                   ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                             camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                             patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                             zpLabel=zpLabel, forcedStr=forcedStr)
                if "ext_shapeHSM_HsmSourceMoments_xx" in catalog.schema:
                    shortName = "hsmTrace" + postFix + "_calib_psf_used"
                    compareCol = "ext_shapeHSM_HsmSourceMoments"
                    self.log.info("shortName = {:s}".format(shortName))
                    self.AnalysisClass(psfUsed, TraceSize(compareCol),
                                       ("          HSM Trace (calib_psf_used): $\sqrt{0.5*(I_{xx}+I_{yy})}$"
                                        " (pixels)"), shortName, self.config.analysis, flags=[col + "_flag"],
                                       goodKeys=["calib_psf_used"], qMin=qMin, qMax=qMax,
                                       labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                       ).plotAll(dataId, filenamer, self.log, enforcer=enforcer,
                                                 butler=butler, camera=camera, ccdList=ccdList,
                                                 tractInfo=tractInfo, patchList=patchList, hscRun=hscRun,
                                                 matchRadius=matchRadius, zpLabel=zpLabel,
                                                 forcedStr=forcedStr)

                # Now for all stars.
                shortName = "trace" + postFix
                starsOnly = catalog[catalog["base_ClassificationExtendedness_value"] < 0.5].copy(deep=True)
                sdssTrace = traceSizeFunc(starsOnly)
                self.log.info("shortName = {:s}".format(shortName))
                self.AnalysisClass(starsOnly, sdssTrace,
                                   "  SdssShape Trace: $\sqrt{0.5*(I_{xx}+I_{yy})}$ (pixels)", shortName,
                                   self.config.analysis, flags=[col + "_flag"], qMin=qMin, qMax=qMax,
                                   labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                   ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                             camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                             patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                             zpLabel=zpLabel, forcedStr=forcedStr)
                if "ext_shapeHSM_HsmSourceMoments_xx" in catalog.schema:
                    shortName = "hsmTrace" + postFix
                    compareCol = "ext_shapeHSM_HsmSourceMoments"
                    self.log.info("shortName = {:s}".format(shortName))
                    self.AnalysisClass(starsOnly, TraceSize(compareCol),
                                       "HSM Trace: $\sqrt{0.5*(I_{xx}+I_{yy})}$ (pixels)", shortName,
                                       self.config.analysis, flags=[col + "_flag"], qMin=qMin, qMax=qMax,
                                       labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                       ).plotAll(dataId, filenamer, self.log, enforcer=enforcer,
                                                 butler=butler, camera=camera, ccdList=ccdList,
                                                 tractInfo=tractInfo, patchList=patchList, hscRun=hscRun,
                                                 matchRadius=matchRadius, zpLabel=zpLabel,
                                                 forcedStr=forcedStr)

            if col + "_instFlux" in catalog.schema:
                shortName = "psfTraceDiff" + postFix
                compareCol = "base_SdssShape"
                psfCompareCol = "base_SdssShape_psf"
                self.log.info("shortName = {:s}".format(shortName))
                self.AnalysisClass(catalog, PsfTraceSizeDiff(compareCol, psfCompareCol),
                                   "    SdssShape Trace % diff (psf_used - PSFmodel)", shortName,
                                   self.config.analysis, flags=[col + "_flag"],
                                   goodKeys=["calib_psf_used"], qMin=-3.0, qMax=3.0,
                                   labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                   ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                             camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                             patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                             zpLabel=zpLabel, forcedStr=forcedStr)

                shortName = "e1Resids" + postFix
                self.log.info("shortName = {:s}".format(shortName))
                self.AnalysisClass(catalog, E1Resids(compareCol, psfCompareCol, unitScale=self.unitScale),
                                   "        SdssShape e1 resids (psf_used - PSFmodel)%s" % unitStr, shortName,
                                   self.config.analysis, flags=[col + "_flag"], goodKeys=["calib_psf_used"],
                                   qMin=-0.05, qMax=0.05, labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                   unitScale=self.unitScale,
                                   ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                             camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                             patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                             zpLabel=zpLabel, forcedStr=forcedStr)

                shortName = "e2Resids" + postFix
                self.log.info("shortName = {:s}".format(shortName))
                self.AnalysisClass(catalog, E2Resids(compareCol, psfCompareCol, unitScale=self.unitScale),
                                   "       SdssShape e2 resids (psf_used - PSFmodel)%s" % unitStr, shortName,
                                   self.config.analysis, flags=[col + "_flag"], goodKeys=["calib_psf_used"],
                                   qMin=-0.05, qMax=0.05, labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                   unitScale=self.unitScale,
                                   ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                             camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                             patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                             zpLabel=zpLabel, forcedStr=forcedStr)

                if "ext_shapeHSM_HsmSourceMoments_xx" in catalog.schema:
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
                                                 butler=butler, camera=camera, ccdList=ccdList,
                                                 tractInfo=tractInfo, patchList=patchList, hscRun=hscRun,
                                                 matchRadius=matchRadius, zpLabel=zpLabel,
                                                 forcedStr=forcedStr)
                    shortName = "e1ResidsHsm" + postFix
                    self.log.info("shortName = {:s}".format(shortName))
                    self.AnalysisClass(catalog, E1Resids(compareCol, psfCompareCol, unitScale=self.unitScale),
                                       "   HSM e1 resids (psf_used - PSFmodel)%s" % unitStr, shortName,
                                       self.config.analysis, flags=[col + "_flag"],
                                       goodKeys=["calib_psf_used"], qMin=-0.05, qMax=0.05,
                                       labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                       unitScale=self.unitScale,
                                       ).plotAll(dataId, filenamer, self.log, enforcer=enforcer,
                                                 butler=butler, camera=camera, ccdList=ccdList,
                                                 tractInfo=tractInfo, patchList=patchList, hscRun=hscRun,
                                                 matchRadius=matchRadius, zpLabel=zpLabel,
                                                 forcedStr=forcedStr)
                    shortName = "e2ResidsHsm" + postFix
                    self.log.info("shortName = {:s}".format(shortName))
                    self.AnalysisClass(catalog, E2Resids(compareCol, psfCompareCol, unitScale=self.unitScale),
                                       "   HSM e2 resids (psf_used - PSFmodel)%s" % unitStr, shortName,
                                       self.config.analysis, flags=[col + "_flag"],
                                       goodKeys=["calib_psf_used"], qMin=-0.05, qMax=0.05,
                                       labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                       unitScale=self.unitScale,
                                       ).plotAll(dataId, filenamer, self.log, enforcer=enforcer,
                                                 butler=butler, camera=camera, ccdList=ccdList,
                                                 tractInfo=tractInfo, patchList=patchList, hscRun=hscRun,
                                                 matchRadius=matchRadius, zpLabel=zpLabel,
                                                 forcedStr=forcedStr)

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
                                                 butler=butler, camera=camera, ccdList=ccdList,
                                                 tractInfo=tractInfo, patchList=patchList, hscRun=hscRun,
                                                 matchRadius=matchRadius, zpLabel=zpLabel,
                                                 forcedStr=forcedStr)

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
                                                 butler=butler, camera=camera, ccdList=ccdList,
                                                 tractInfo=tractInfo, patchList=patchList, hscRun=hscRun,
                                                 matchRadius=matchRadius, zpLabel=zpLabel,
                                                 forcedStr=forcedStr)

    def plotCentroidXY(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None,
                       tractInfo=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None,
                       forcedStr=None, flagsCat=None):
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.02*self.unitScale}})
        for col in ["base_SdssCentroid_x", "base_SdssCentroid_y"]:
            if col in catalog.schema:
                shortName = col
                self.log.info("shortName = {:s}".format(shortName))
                self.AnalysisClass(catalog, catalog[col], "(%s)" % col, shortName, self.config.analysis,
                                   labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                   ).plotFP(dataId, filenamer, self.log, enforcer=enforcer, camera=camera,
                                            ccdList=ccdList, hscRun=hscRun, matchRadius=matchRadius,
                                            zpLabel=zpLabel, forcedStr=forcedStr)

    def plotFootprint(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None,
                      tractInfo=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None,
                      forcedStr=None, postFix="", flagsCat=None, plotRunStats=False, highlightList=None):
        enforcer = None
        if "calib_psf_used" in catalog.schema:
            shortName = "footNpix_calib_psf_used"
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(catalog, catalog["base_Footprint_nPix"], "%s" % shortName, shortName,
                               self.config.analysis, flags=["base_Footprint_nPix_flag"],
                               goodKeys=["calib_psf_used"], qMin=-100, qMax=2000,
                               labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                               ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                         camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                         patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                         zpLabel=zpLabel, forcedStr=forcedStr, plotRunStats=plotRunStats,
                                         highlightList=highlightList)
        shortName = "footNpix"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(catalog, catalog["base_Footprint_nPix"], "%s" % shortName, shortName,
                           self.config.analysis, flags=["base_Footprint_nPix_flag"],
                           qMin=0, qMax=3000, labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                           ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                     camera=camera, ccdList=ccdList, tractInfo=tractInfo, patchList=patchList,
                                     hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel,
                                     forcedStr=forcedStr, plotRunStats=plotRunStats)

    def plotFootprintHist(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None,
                          tractInfo=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None,
                          postFix="", flagsCat=None):
        stats = None
        shortName = "footNpix"
        self.log.info("shortName = {:s}".format(shortName + "Hist"))
        self.AnalysisClass(catalog, catalog["base_Footprint_nPix"], "%s" % shortName, shortName,
                           self.config.analysis, flags=["base_Footprint_nPix_flag"], qMin=0, qMax=3000,
                           labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                           ).plotHistogram(filenamer, stats=stats, camera=camera, hscRun=hscRun,
                                           matchRadius=matchRadius, zpLabel=zpLabel,
                                           filterStr=dataId['filter'])

    def plotPsfFluxHist(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None,
                        tractInfo=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None,
                        postFix="", flagsCat=None, vertLineList=[4000, 12500], logPlot=True,
                        density=True, cumulative=-1):
        stats = None
        shortName = "rawPsfFlux"
        self.log.info("shortName = {:s}".format(shortName))
        factor = 10.0**(0.4*self.config.analysis.commonZp) # want raw flux
        rawFlux = catalog["base_PsfFlux_instFlux"]*factor
        rawFluxErr = catalog["base_PsfFlux_instFluxErr"]*factor
        rawSn = rawFlux/rawFluxErr
        goodSn = rawSn > 20.0
        rawFluxSnGt20 = rawFlux[goodSn]
        goodSn = rawSn > 50.0
        rawFluxSnGt50 = rawFlux[goodSn]
        goodFlux = rawFlux > 4000.0
        rawSnFluxGt4000 = rawSn[goodFlux]
        goodFlux = rawFlux > 12500.0
        rawSnFluxGt12500 = rawSn[goodFlux]
        psfUsedCat = catalog[catalog["calib_psf_used"]]
        psfUsedRawFlux = psfUsedCat["base_PsfFlux_instFlux"]*factor
        psfUsedRawFluxErr = psfUsedCat["base_PsfFlux_instFluxErr"]*factor
        psfUsedRawSn = psfUsedRawFlux/psfUsedRawFluxErr

        self.AnalysisClass(catalog, rawFlux, "%s" % shortName, shortName,
                           self.config.analysis, flags=["base_PsfFlux_flag"], qMin=0,
                           qMax = int(min(99999, max(5.0*np.median(rawFlux), 0.25*np.max(rawFlux)))),
                           labeller=AllLabeller(), flagsCat=flagsCat,
                           ).plotHistogram(filenamer, numBins="sqrt", stats=stats, camera=camera,
                                           ccdList=ccdList, hscRun=hscRun, zpLabel=zpLabel,
                                           filterStr=dataId['filter'], vertLineList=vertLineList,
                                           logPlot=logPlot, density=False, cumulative=cumulative,
                                           addDataList=[rawFluxSnGt20, rawFluxSnGt50, psfUsedRawFlux],
                                           addDataLabelList=["S/N>20", "S/N>50", "psf_used"])
        shortName = "rawPsfFlux/rawPsfFluxErr"
        filenamer = filenamer.replace("Flux", "FluxSn")
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(catalog, rawSn, "%s" % "S/N = " + shortName, shortName,
                           self.config.analysis, flags=["base_PsfFlux_flag"], qMin=0,
                           # qMax = int(min(100000, max(6.0*np.median(rawSn), 0.25*np.max(rawSn)))),
                           qMax = 449,
                           labeller=AllLabeller(), flagsCat=flagsCat,
                           ).plotHistogram(filenamer, numBins="sqrt", stats=stats, camera=camera,
                                           ccdList=ccdList, hscRun=hscRun, zpLabel=zpLabel,
                                           filterStr=dataId['filter'], vertLineList=[20, 50],
                                           logPlot=logPlot, density=False, cumulative=cumulative,
                                           addDataList=[rawSnFluxGt4000, rawSnFluxGt12500,
                                                        psfUsedRawSn],
                                           addDataLabelList=["Flux>4000", "Flux>12500", "psf_used"])

    def plotStarGal(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None, tractInfo=None,
                    patchList=None, hscRun=None, matchRadius=None, zpLabel=None, forcedStr=None,
                    flagsCat=None):
        enforcer = None
        shortName = "pStar"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(catalog, deconvMomStarGal, "P(star) from deconvolved moments",
                           shortName, self.config.analysis, qMin=-0.1, qMax=1.39,
                           labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                           ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                     camera=camera, ccdList=ccdList, tractInfo=tractInfo, patchList=patchList,
                                     hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel,
                                     forcedStr=forcedStr)
        shortName = "deconvMom"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(catalog, deconvMom, "Deconvolved moments", shortName,
                           self.config.analysis, qMin=-1.0, qMax=3.0, labeller=StarGalaxyLabeller(),
                           flagsCat=flagsCat,
                           ).plotAll(dataId, filenamer, self.log,
                                     enforcer=Enforcer(requireLess={"star": {"stdev": 0.2}}),
                                     butler=butler, camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                     patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                     zpLabel=zpLabel, forcedStr=forcedStr)

        if "ext_shapeHSM_HsmShapeRegauss_resolution" in catalog.schema:
            shortName = "resolution"
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(catalog, catalog["ext_shapeHSM_HsmShapeRegauss_resolution"],
                               "Resolution Factor from HsmRegauss",
                               shortName, self.config.analysis, qMin=-0.1, qMax=1.15,
                               labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                               ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                         camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                         patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                         zpLabel=zpLabel, forcedStr=forcedStr)

    def plotCompareUnforced(self, forced, unforced, filenamer, dataId, butler=None, camera=None, ccdList=None,
                            tractInfo=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None,
                            fluxToPlotList=None):
        if fluxToPlotList is None:
            fluxToPlotList = self.config.fluxToPlotList
        unitStr = "mmag" if self.config.toMilli else "mag"
        enforcer = None
        catalog = joinMatches(afwTable.matchRaDec(forced, unforced, matchRadius*afwGeom.arcseconds),
                              "forced_", "unforced_")
        for col in fluxToPlotList:
            shortName = "compareUnforced_" + col
            self.log.info("shortName = {:s}".format(shortName))
            if "forced_" + col + "_instFlux" in catalog.schema:
                self.AnalysisClass(catalog, MagDiff("forced_" + col + "_instFlux",
                                                    "unforced_" + col + "_instFlux",
                                                    unitScale=self.unitScale),
                                   "  Forced - Unforced mag [%s] (%s)" % (fluxToPlotString(col), unitStr),
                                   shortName, self.config.analysis, prefix="forced_", flags=[col + "_flag"],
                                   labeller=OverlapsStarGalaxyLabeller("forced_", "unforced_"),
                                   unitScale=self.unitScale,
                                   ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                             camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                             patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                             zpLabel=zpLabel)

    def isBad(self, source):
        """Return True if any of config.badFlags are set for this source."""
        for flag in self.config.analysis.flags:
            if source.get(flag):
                return True
        return False

    def overlaps(self, catalog):
        badForOverlap = makeBadArray(catalog, flagList=self.config.analysis.flags,
                                     onlyReadStars=self.config.onlyReadStars, patchInnerOnly=False)
        goodCat = catalog[~badForOverlap]
        matches = afwTable.matchRaDec(goodCat, self.config.matchOverlapRadius*afwGeom.arcseconds)
        if not matches:
            self.log.info("Did not find any overlapping matches")
        return joinMatches(matches, "first_", "second_")

    def plotOverlaps(self, overlaps, filenamer, dataId, butler=None, camera=None, ccdList=None,
                     tractInfo=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None,
                     forcedStr=None, postFix="", fluxToPlotList=None, flagsCat=None):
        if fluxToPlotList is None:
            fluxToPlotList = self.config.fluxToPlotList
        unitStr = "mmag" if self.config.toMilli else "mag"
        magEnforcer = Enforcer(requireLess={"star": {"stdev": 0.003*self.unitScale}})
        for col in fluxToPlotList:
            shortName = "overlap_" + col + postFix
            self.log.info("shortName = {:s}".format(shortName))
            if "first_" + col + "_instFlux" in overlaps.schema:
                self.AnalysisClass(overlaps, MagDiff("first_" + col + "_instFlux",
                                                     "second_" + col + "_instFlux",
                                                     unitScale=self.unitScale),
                                   "  Overlap mag difference (%s) (%s)" % (fluxToPlotString(col), unitStr),
                                   shortName, self.config.analysis, prefix="first_", flags=[col + "_flag"],
                                   labeller=OverlapsStarGalaxyLabeller(), magThreshold=23,
                                   unitScale=self.unitScale,
                                   ).plotAll(dataId, filenamer, self.log, enforcer=magEnforcer, butler=butler,
                                             camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                             patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                             zpLabel=zpLabel, forcedStr=forcedStr)
        unitStr = "mas" if self.config.toMilli else "arcsec"
        distEnforcer = Enforcer(requireLess={"star": {"stdev": 0.005*self.unitScale}})
        shortName = "overlap_distance" + postFix
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(overlaps,
                           lambda cat: cat["distance"]*(1.0*afwGeom.radians).asArcseconds()*self.unitScale,
                           "Distance (%s)" % unitStr, shortName, self.config.analysis, prefix="first_",
                           qMin=-0.01, qMax=0.11, labeller=OverlapsStarGalaxyLabeller(), flagsCat=flagsCat,
                           forcedMean=0.0, unitScale=self.unitScale,
                           ).plotAll(dataId, filenamer, self.log, enforcer=distEnforcer, butler=butler,
                                     camera=camera, ccdList=ccdList, tractInfo=tractInfo, patchList=patchList,
                                     hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel,
                                     forcedStr=forcedStr)

    def plotMatches(self, matches, filterName, filenamer, dataId, description="matches", butler=None,
                    camera=None, ccdList=None, tractInfo=None, patchList=None, hscRun=None, matchRadius=None,
                    zpLabel=None, forcedStr=None, flagsCat=None):
        unitStr = "mmag" if self.config.toMilli else "mag"
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.030*self.unitScale}}),
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
        if "src_calib_psf_used" in matches.schema:
            shortName = description + "_mag_calib_psf_used"
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(matches, MagDiffMatches("base_PsfFlux_instFlux", ct, zp=0.0,
                                                       unitScale=self.unitScale),
                               "MagPsf - ref (calib_psf_used) (%s)" % unitStr, shortName,
                               self.config.analysisMatches, prefix="src_", goodKeys=["calib_psf_used"],
                               qMin=-0.15, qMax=0.1, labeller=MatchesStarGalaxyLabeller(), flagsCat=flagsCat,
                               unitScale=self.unitScale,
                               ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                         camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                         patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                         zpLabel=zpLabel, forcedStr=forcedStr)
        if "src_calib_photometry_used" in matches.schema:
            shortName = description + "_mag_calib_photometry_used"
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(matches, MagDiffMatches("base_PsfFlux_instFlux", ct, zp=0.0,
                                                       unitScale=self.unitScale),
                               "   MagPsf - ref (calib_photom_used) (%s)" % unitStr, shortName,
                               self.config.analysisMatches, prefix="src_", goodKeys=["calib_photometry_used"],
                               qMin=-0.15, qMax=0.15, labeller=MatchesStarGalaxyLabeller(), flagsCat=flagsCat,
                               unitScale=self.unitScale,
                               ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                         camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                         patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                         zpLabel=zpLabel, forcedStr=forcedStr)
        shortName = description + "_mag"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(matches, MagDiffMatches("base_PsfFlux_instFlux", ct, zp=0.0,
                                                   unitScale=self.unitScale),
                           "MagPsf - ref (%s)" % unitStr, shortName, self.config.analysisMatches,
                           prefix="src_", qMin=-0.15, qMax=0.5, labeller=MatchesStarGalaxyLabeller(),
                           unitScale=self.unitScale,
                           ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                     camera=camera, ccdList=ccdList, tractInfo=tractInfo, patchList=patchList,
                                     hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel,
                                     forcedStr=forcedStr)

        unitStr = "mas" if self.config.toMilli else "arcsec"
        if "src_calib_astrometry_used" in matches.schema:
            shortName = description + "_distance_calib_astrometry_used"
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(matches,
                               lambda cat:
                                   cat["distance"]*(1.0*afwGeom.radians).asArcseconds()*self.unitScale,
                               "Distance (%s) (calib_astrom_used)" % unitStr, shortName,
                               self.config.analysisMatches, prefix="src_", goodKeys=["calib_astrometry_used"],
                               qMin=-0.01*self.config.matchRadius, qMax=0.5*self.config.matchRadius,
                               labeller=MatchesStarGalaxyLabeller(), flagsCat=flagsCat,
                               unitScale=self.unitScale,
                               ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                         camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                         patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                         zpLabel=zpLabel, forcedStr=forcedStr)
        shortName = description + "_distance"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(matches,
                           lambda cat: cat["distance"]*(1.0*afwGeom.radians).asArcseconds()*self.unitScale,
                           "Distance (%s)" % unitStr, shortName, self.config.analysisMatches, prefix="src_",
                           qMin=-0.05*self.config.matchRadius, qMax=0.3*self.config.matchRadius,
                           labeller=MatchesStarGalaxyLabeller(), flagsCat=flagsCat, forcedMean=0.0,
                           unitScale=self.unitScale,
                           ).plotAll(dataId, filenamer, self.log,
                                     enforcer=Enforcer(requireLess={"star": {"stdev": 0.050*self.unitScale}}),
                                     butler=butler, camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                     patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                     zpLabel=zpLabel, forcedStr=forcedStr)
        if "src_calib_astrometry_used" in matches.schema:
            shortName = description + "_raCosDec_calib_astrometry_used"
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(matches, AstrometryDiff("src_coord_ra", "ref_coord_ra",
                                                       declination1="src_coord_dec",
                                                       declination2="ref_coord_dec",
                                                       unitScale=self.unitScale),
                               "      $\delta_{Ra}$ = $\Delta$RA*cos(Dec) (%s) (calib_astrom_used)" % unitStr,
                               shortName, self.config.analysisMatches, prefix="src_",
                               goodKeys=["calib_astrometry_used"], qMin=-0.2*self.config.matchRadius,
                               qMax=0.2*self.config.matchRadius, labeller=MatchesStarGalaxyLabeller(),
                               flagsCat=flagsCat, unitScale=self.unitScale,
                               ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                         camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                         patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                         zpLabel=zpLabel, forcedStr=forcedStr)
        shortName = description + "_raCosDec"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(matches, AstrometryDiff("src_coord_ra", "ref_coord_ra",
                                                   declination1="src_coord_dec", declination2="ref_coord_dec",
                                                   unitScale=self.unitScale),
                           "$\delta_{Ra}$ = $\Delta$RA*cos(Dec) (%s)" % unitStr, shortName,
                           self.config.analysisMatches, prefix="src_", qMin=-0.2*self.config.matchRadius,
                           qMax=0.2*self.config.matchRadius, labeller=MatchesStarGalaxyLabeller(),
                           flagsCat=flagsCat, unitScale=self.unitScale,
                           ).plotAll(dataId, filenamer, self.log,
                                     enforcer=Enforcer(requireLess={"star": {"stdev": 0.050*self.unitScale}}),
                                     butler=butler, camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                     patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                     zpLabel=zpLabel, forcedStr=forcedStr)
        if "src_calib_astrometry_used" in matches.schema:
            shortName = description + "_ra_calib_astrometry_used"
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(matches,
                               AstrometryDiff("src_coord_ra", "ref_coord_ra", unitScale=self.unitScale),
                               "$\Delta$RA (%s) (calib_astrom_used)" % unitStr, shortName,
                               self.config.analysisMatches, prefix="src_", goodKeys=["calib_astrometry_used"],
                               qMin=-0.25*self.config.matchRadius, qMax=0.25*self.config.matchRadius,
                               labeller=MatchesStarGalaxyLabeller(), flagsCat=flagsCat,
                               unitScale=self.unitScale,
                               ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                         camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                         patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                         zpLabel=zpLabel, forcedStr=forcedStr)
        shortName = description + "_ra"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(matches, AstrometryDiff("src_coord_ra", "ref_coord_ra", unitScale=self.unitScale),
                           "$\Delta$RA (%s)" % unitStr, shortName, self.config.analysisMatches,
                           prefix="src_", qMin=-0.25*self.config.matchRadius,
                           qMax=0.25*self.config.matchRadius, labeller=MatchesStarGalaxyLabeller(),
                           flagsCat=flagsCat, unitScale=self.unitScale,
                           ).plotAll(dataId, filenamer, self.log,
                                     enforcer=Enforcer(requireLess={"star": {"stdev": 0.050*self.unitScale}}),
                                     butler=butler, camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                     patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                     zpLabel=zpLabel, forcedStr=forcedStr)
        if "src_calib_astrometry_used" in matches.schema:
            shortName = description + "_dec_calib_astrometry_used"
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(matches,
                               AstrometryDiff("src_coord_dec", "ref_coord_dec", unitScale=self.unitScale),
                               "$\delta_{Dec}$ (%s) (calib_astrom_used)" % unitStr, shortName,
                               self.config.analysisMatches, prefix="src_", goodKeys=["calib_astrometry_used"],
                               qMin=-0.25*self.config.matchRadius, qMax=0.25*self.config.matchRadius,
                               labeller=MatchesStarGalaxyLabeller(), flagsCat=flagsCat,
                               unitScale=self.unitScale,
                               ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                         camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                         patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                         zpLabel=zpLabel, forcedStr=forcedStr)
        shortName = description + "_dec"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(matches,
                           AstrometryDiff("src_coord_dec", "ref_coord_dec", unitScale=self.unitScale),
                           "$\delta_{Dec}$ (%s)" % unitStr, shortName, self.config.analysisMatches,
                           prefix="src_", qMin=-0.3*self.config.matchRadius, qMax=0.3*self.config.matchRadius,
                           labeller=MatchesStarGalaxyLabeller(), flagsCat=flagsCat, unitScale=self.unitScale,
                           ).plotAll(dataId, filenamer, self.log,
                                     enforcer=Enforcer(requireLess={"star": {"stdev": 0.050*self.unitScale}}),
                                     butler=butler, camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                     patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                     zpLabel=zpLabel, forcedStr=forcedStr)

    def plotCosmos(self, catalog, filenamer, cosmos, dataId):
        labeller = CosmosLabeller(cosmos, self.config.matchRadius*afwGeom.arcseconds)
        self.AnalysisClass(catalog, deconvMom, "Deconvolved moments", "cosmos", self.config.analysis,
                           qMin=-1.0, qMax=6.0, labeller=labeller,
                           ).plotAll(dataId, filenamer, self.log,
                                     enforcer=Enforcer(requireLess={"star": {"stdev": 0.2}}))

    def matchCatalog(self, catalog, filterName, astrometryConfig):
        refObjLoader = LoadAstrometryNetObjectsTask(self.config.refObjLoaderConfig)
        center = afwGeom.averageSpherePoint([src.getCoord() for src in catalog])
        radius = max(center.separation(src.getCoord()) for src in catalog)
        filterName = afwImage.Filter(afwImage.Filter(filterName).getId()).getName()  # Get primary name
        refs = refObjLoader.loadSkyCircle(center, radius, filterName).refCat
        matches = afwTable.matchRaDec(refs, catalog, self.config.matchRadius*afwGeom.arcseconds)
        matches = matchJanskyToDn(matches)
        return joinMatches(matches, "ref_", "src_")

    def plotQuiver(self, catalog, filenamer, dataId=None, butler=None, camera=None, ccdList=None,
                   tractInfo=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None,
                   forcedStr=None, postFix="", flagsCat=None, scale=1):
        stats = None
        shortName = "quiver"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(catalog, None, "%s" % shortName, shortName,
                           self.config.analysis, labeller=None,
                           ).plotQuiver(catalog, filenamer, self.log, stats=stats, dataId=dataId,
                                        butler=butler, camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                        patchList=patchList, hscRun=hscRun, zpLabel=zpLabel,
                                        forcedStr=forcedStr, scale=scale)

    def plotInputCounts(self, catalog, filenamer, dataId, butler, tractInfo, patchList=None, camera=None,
                        hscRun=None, forcedStr=None, alpha=0.5, doPlotTractImage=True,
                        doPlotPatchOutline=True, sizeFactor=5.0, maxDiamPix=1000):
        shortName = "inputCounts"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(catalog, None, "%s" % shortName, shortName,
                           self.config.analysis, labeller=None,
                           ).plotInputCounts(catalog, filenamer, self.log, dataId, butler, tractInfo,
                                             patchList=patchList, camera=camera, forcedStr=forcedStr,
                                             alpha=alpha, doPlotTractImage=doPlotTractImage,
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
        self.matchRadius = 0.2
        if "base_PsfFlux" not in self.fluxToPlotList:
            self.fluxToPlotList.append("base_PsfFlux")  # Add PSF flux to default list for comparison scripts


class CompareCoaddAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
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
        return parser

    def __init__(self, *args, **kwargs):
        CmdLineTask.__init__(self, *args, **kwargs)
        self.unitScale = 1000.0 if self.config.toMilli else 1.0

    def runDataRef(self, patchRefList1, patchRefList2):
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

        if haveForced:
            # copy over some fields from unforced to forced catalog
            forced1 = addColumnsToSchema(unforced1, forced1,
                                         [col for col in list(self.config.columnsToCopy) +
                                          list(self.config.analysis.flags) if
                                          col not in forced1.schema and col in unforced1.schema and
                                          not (repoInfo1.hscRun and col == "slot_Centroid_flag")])
            forced2 = addColumnsToSchema(unforced2, forced2,
                                         [col for col in list(self.config.columnsToCopy) +
                                          list(self.config.analysis.flags) if
                                          col not in forced2.schema and col in unforced2.schema and
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
            if hscRun is not None and self.config.srcSchemaMap is not None:
                aliasDictList += [self.config.srcSchemaMap]
            if aliasDictList is not None:
                catalog = setAliasMaps(catalog, aliasDictList)

        # Set boolean array indicating sources deemed unsuitable for qa analyses
        self.catLabel = "noDuplicates"
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

        aliasDictList = aliasDictList0
        if hscRun is not None and self.config.srcSchemaMap is not None:
            aliasDictList += [self.config.srcSchemaMap]
        if aliasDictList is not None:
            forced = setAliasMaps(forced, aliasDictList)
            unforced = setAliasMaps(unforced, aliasDictList)

        self.log.info("\nNumber of sources in forced catalogs: first = {0:d} and second = {1:d}".format(
                      len(forced1), len(forced2)))

        filenamer = Filenamer(repoInfo1.butler, "plotCompareCoadd", repoInfo1.dataId)
        hscRun = repoInfo1.hscRun if repoInfo1.hscRun is not None else repoInfo2.hscRun
        if self.config.doPlotMags:
            self.plotMags(forced, filenamer, repoInfo1.dataId, butler=repoInfo1.butler,
                          camera=repoInfo1.camera, tractInfo=repoInfo1.tractInfo, patchList=patchList1,
                          hscRun=hscRun, matchRadius=self.config.matchRadius, zpLabel=self.zpLabel,
                          forcedStr=forcedStr)

        if self.config.doPlotSizes:
            if ("first_base_SdssShape_psf_xx" in forced.schema and
               "second_base_SdssShape_psf_xx" in forced.schema):
                self.plotSizes(forced, filenamer, repoInfo1.dataId, butler=repoInfo1.butler,
                               camera=repoInfo1.camera, tractInfo=repoInfo1.tractInfo, patchList=patchList1,
                               hscRun=hscRun, matchRadius=self.config.matchRadius, zpLabel=self.zpLabel,
                               forcedStr=forcedStr)
            else:
                self.log.warn("Cannot run plotSizes: base_SdssShape_psf_xx not in catalog.schema")

        if self.config.doApCorrs:
            self.plotApCorrs(unforced, filenamer, repoInfo1.dataId, butler=repoInfo1.butler,
                             camera=repoInfo1.camera, tractInfo=repoInfo1.tractInfo, patchList=patchList1,
                             hscRun=hscRun, matchRadius=self.config.matchRadius, zpLabel=self.zpLabel,
                             forcedStr="unforced")
        if self.config.doPlotCentroids:
            self.plotCentroids(forced, filenamer, repoInfo1.dataId, butler=repoInfo1.butler,
                               camera=repoInfo1.camera, tractInfo=repoInfo1.tractInfo, patchList=patchList1,
                               hscRun=hscRun, hscRun1=repoInfo1.hscRun, hscRun2=repoInfo2.hscRun,
                               matchRadius=self.config.matchRadius, zpLabel=self.zpLabel, forcedStr=forcedStr)
        if self.config.doPlotStarGalaxy:
            self.plotStarGal(forced, filenamer, repoInfo1.dataId, butler=repoInfo1.butler,
                             camera=repoInfo1.camera, tractInfo=repoInfo1.tractInfo, patchList=patchList1,
                             hscRun=hscRun, matchRadius=self.config.matchRadius, zpLabel=self.zpLabel,
                             forcedStr=forcedStr)

    def readCatalogs(self, patchRefList, dataset):
        catList = [patchRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS) for
                   patchRef in patchRefList if patchRef.datasetExists(dataset)]
        if not catList:
            raise TaskError("No catalogs read: %s" % ([patchRef.dataId for patchRef in patchRefList]))
        return concatenateCatalogs(catList)

    def matchCatalogs(self, catalog1, catalog2):
        matches = afwTable.matchRaDec(catalog1, catalog2, self.config.matchRadius*afwGeom.arcseconds)
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
                 patchList=None, hscRun=None, matchRadius=None, zpLabel=None, forcedStr=None,
                 fluxToPlotList=None, postFix="", flagsCat=None, highlightList=None):
        if fluxToPlotList is None:
            fluxToPlotList = self.config.fluxToPlotList
        unitStr = "mmag" if self.config.toMilli else "mag"
        if fluxToPlotList is None:
            fluxToPlotList = self.config.fluxToPlotList
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.02*self.unitScale}})
        for col in fluxToPlotList:
            if ("first_" + col + "_instFlux" in catalog.schema and "second_" + col + "_instFlux" in
                catalog.schema):
                shortName = "diff_" + col + postFix
                self.log.info("shortName = {:s}".format(shortName))
                Analysis(catalog, MagDiffCompare(col + "_instFlux", unitScale=self.unitScale),
                         "      Run Comparison: %s mag diff (%s)" % (fluxToPlotString(col), unitStr),
                         shortName, self.config.analysis, prefix="first_", qMin=-0.05, qMax=0.05,
                         flags=[col + "_flag"], errFunc=MagDiffErr(col + "_instFlux",
                                                                   unitScale=self.unitScale),
                         labeller=OverlapsStarGalaxyLabeller(), flagsCat=flagsCat, unitScale=self.unitScale,
                         ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                   camera=camera, ccdList=ccdList, tractInfo=tractInfo, patchList=patchList,
                                   hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel,
                                   forcedStr=forcedStr, highlightList=highlightList)

    def plotCentroids(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None,
                      tractInfo=None, patchList=None, hscRun=None, hscRun1=None, hscRun2=None,
                      matchRadius=None, zpLabel=None, forcedStr=None, flagsCat=None, highlightList=None):
        unitStr = "milliPixels" if self.config.toMilli else "pixels"
        distEnforcer = None
        centroidStr1, centroidStr2 = "base_SdssCentroid", "base_SdssCentroid"
        if bool(hscRun1) ^ bool(hscRun2):
            if hscRun1 is None:
                centroidStr1 = "base_SdssCentroid_Rot"
            if hscRun2 is None:
                centroidStr2 = "base_SdssCentroid_Rot"

        shortName = "diff_x"
        self.log.info("shortName = {:s}".format(shortName))
        Analysis(catalog, CentroidDiff("x", centroid1=centroidStr1, centroid2=centroidStr2,
                                       unitScale=self.unitScale),
                 "Run Comparison: x offset (%s)" % unitStr, shortName, self.config.analysis, prefix="first_",
                 qMin=-0.08, qMax=0.08, errFunc=None, labeller=OverlapsStarGalaxyLabeller(),
                 ).plotAll(dataId, filenamer, self.log, enforcer=distEnforcer, butler=butler, camera=camera,
                           ccdList=ccdList, tractInfo=tractInfo, patchList=patchList, hscRun=hscRun,
                           matchRadius=matchRadius, zpLabel=zpLabel, forcedStr=forcedStr)
        shortName = "diff_y"
        self.log.info("shortName = {:s}".format(shortName))
        Analysis(catalog, CentroidDiff("y", centroid1=centroidStr1, centroid2=centroidStr2,
                                       unitScale=self.unitScale),
                 "Run Comparison: y offset (%s)" % unitStr, shortName, self.config.analysis, prefix="first_",
                 qMin=-0.08, qMax=0.08, errFunc=None, labeller=OverlapsStarGalaxyLabeller(),
                 ).plotAll(dataId, filenamer, self.log, enforcer=distEnforcer, butler=butler, camera=camera,
                           ccdList=ccdList, tractInfo=tractInfo, patchList=patchList, hscRun=hscRun,
                           matchRadius=matchRadius, zpLabel=zpLabel, forcedStr=forcedStr)

        unitStr = "mas" if self.config.toMilli else "arcsec"
        shortName = "diff_raCosDec"
        self.log.info("shortName = {:s}".format(shortName))
        Analysis(catalog, AstrometryDiff("first_coord_ra", "second_coord_ra", declination1="first_coord_dec",
                                         declination2="second_coord_dec", unitScale=self.unitScale),
                 "   Run Comparison: $\delta_{Ra}$ = $\Delta$RA*cos(Dec) (%s)" % unitStr, shortName,
                 self.config.analysisMatches, prefix="first_", qMin=-0.2*self.config.matchRadius,
                 qMax=0.2*self.config.matchRadius, labeller=OverlapsStarGalaxyLabeller(),
                 flagsCat=flagsCat, unitScale=self.unitScale,
                 ).plotAll(dataId, filenamer, self.log, butler=butler, camera=camera, ccdList=ccdList,
                           tractInfo=tractInfo, patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                           zpLabel=zpLabel, forcedStr=forcedStr)
        shortName = "diff_ra"
        self.log.info("shortName = {:s}".format(shortName))
        Analysis(catalog, AstrometryDiff("first_coord_ra", "second_coord_ra", declination1=None,
                                         declination2=None, unitScale=self.unitScale),
                 "Run Comparison: $\Delta$RA (%s)" % unitStr, shortName, self.config.analysisMatches,
                 prefix="first_", qMin=-0.25*self.config.matchRadius, qMax=0.25*self.config.matchRadius,
                 labeller=OverlapsStarGalaxyLabeller(), flagsCat=flagsCat, unitScale=self.unitScale,
                 ).plotAll(dataId, filenamer, self.log, butler=butler, camera=camera, ccdList=ccdList,
                           tractInfo=tractInfo, patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                           zpLabel=zpLabel, forcedStr=forcedStr)
        shortName = "diff_dec"
        self.log.info("shortName = {:s}".format(shortName))
        Analysis(catalog, AstrometryDiff("first_coord_dec", "second_coord_dec", unitScale=self.unitScale),
                 "$\delta_{Dec}$ (%s)" % unitStr, shortName, self.config.analysisMatches, prefix="first_",
                 qMin=-0.3*self.config.matchRadius, qMax=0.3*self.config.matchRadius,
                 labeller=OverlapsStarGalaxyLabeller(), flagsCat=flagsCat, unitScale=self.unitScale,
                 ).plotAll(dataId, filenamer, self.log, butler=butler, camera=camera, ccdList=ccdList,
                           tractInfo=tractInfo, patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                           zpLabel=zpLabel, forcedStr=forcedStr)

    def plotFootprint(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None,
                      tractInfo=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None,
                      forcedStr=None, postFix="", flagsCat=None, highlightList=None):
        enforcer = None
        shortName = "diff_footNpix"
        col = "base_Footprint_nPix"
        self.log.info("shortName = {:s}".format(shortName))
        Analysis(catalog, FootNpixDiffCompare(col), "  Run Comparison: Footprint nPix difference", shortName,
                 self.config.analysis, prefix="first_", qMin=-250, qMax=250, flags=[col + "_flag"],
                 labeller=OverlapsStarGalaxyLabeller(), flagsCat=flagsCat,
                 ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler, camera=camera,
                           ccdList=ccdList, tractInfo=tractInfo, patchList=patchList, hscRun=hscRun,
                           matchRadius=matchRadius, zpLabel=zpLabel, forcedStr=forcedStr, postFix=postFix)
        shortName = "diff_footNpix_calib_psf_used"
        self.log.info("shortName = {:s}".format(shortName))
        Analysis(catalog, FootNpixDiffCompare(col), "     Run Comparison: Footprint nPix diff (psf_used)",
                 shortName, self.config.analysis, prefix="first_", goodKeys=["calib_psf_used"],
                 qMin=-150, qMax=150, flags=[col + "_flag"], labeller=OverlapsStarGalaxyLabeller(),
                 flagsCat=flagsCat,
                 ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler, camera=camera,
                           ccdList=ccdList, tractInfo=tractInfo, patchList=patchList, hscRun=hscRun,
                           matchRadius=matchRadius, zpLabel=zpLabel, forcedStr=forcedStr, postFix=postFix,
                           highlightList=highlightList)

    def plotSizes(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None, tractInfo=None,
                  patchList=None, hscRun=None, matchRadius=None, zpLabel=None, forcedStr=None):
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.02*self.unitScale}})
        for col in ["base_PsfFlux"]:
            if ("first_" + col + "_instFlux" in catalog.schema and
               "second_" + col + "_instFlux" in catalog.schema):
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
                             ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                       camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                       patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                       zpLabel=zpLabel, forcedStr=forcedStr)

                    shortNameBase = "psfTrace"
                    shortName = (shortNameBase + "_calib_psf_used" if "calib_psf_used" in goodFlags else
                                 shortNameBase)
                    compareCol = "base_SdssShape_psf"
                    self.log.info("shortName = {:s}".format(shortName))
                    Analysis(catalog, TraceSizeCompare(compareCol),
                             "       SdssShape PSF Trace Radius Diff (%)" + subCatString,
                             shortName, self.config.analysis, flags=badFlags, prefix="first_",
                             goodKeys=goodFlags, qMin=-1.1, qMax=1.1, labeller=OverlapsStarGalaxyLabeller(),
                             ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                       camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                       patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                       zpLabel=zpLabel, forcedStr=forcedStr)

                    badFlags = [col + "_flag", "ext_shapeHSH_HsmSourceMoments_flag"]
                    if "first_ext_shapeHSM_HsmSourceMoments_xx" in catalog.schema:
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
                                 ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                           camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                           patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                           zpLabel=zpLabel, forcedStr=forcedStr)
                        shortNameBase = "hsmPsfTrace"
                        shortName = (shortNameBase + "_calib_psf_used" if "calib_psf_used" in goodFlags else
                                     shortNameBase)
                    badFlags = [col + "_flag", "ext_shapeHSH_PsfMoments_flag"]
                    if "first_ext_shapeHSM_PsfMoments_xx" in catalog.schema:
                        compareCol = "ext_shapeHSM_HsmPsfMoments"
                        self.log.info("shortName = {:s}".format(shortName))
                        Analysis(catalog, TraceSizeCompare(compareCol),
                                 "      HSM PSF Trace Radius Diff (%)" + subCatString,
                                 shortName, self.config.analysis, flags=badFlags, prefix="first_",
                                 goodKeys=goodFlags, qMin=-1.1, qMax=1.1,
                                 labeller=OverlapsStarGalaxyLabeller(),
                                 ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                           camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                           patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                           zpLabel=zpLabel, forcedStr=forcedStr)

                badFlags = [col + "_flag", "base_SdssShape_flag"]
                shortName = "sdssXx"
                compareCol = "base_SdssShape_xx"
                self.log.info("shortName = {:s}".format(shortName))
                Analysis(catalog, PercentDiff(compareCol), "SdssShape xx Moment Diff (%)", shortName,
                         self.config.analysis, flags=badFlags, prefix="first_",
                         qMin=-0.5, qMax=1.5, labeller=OverlapsStarGalaxyLabeller(),
                         ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                   camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                   patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                   zpLabel=zpLabel, forcedStr=forcedStr)
                shortName = "sdssYy"
                compareCol = "base_SdssShape_yy"
                self.log.info("shortName = {:s}".format(shortName))
                Analysis(catalog, PercentDiff(compareCol), "SdssShape yy Moment Diff (%)", shortName,
                         self.config.analysis, flags=badFlags, prefix="first_",
                         qMin=-0.5, qMax=1.5, labeller=OverlapsStarGalaxyLabeller(),
                         ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                   camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                   patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                   zpLabel=zpLabel, forcedStr=forcedStr)

    def plotStarGal(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None, tractInfo=None,
                    patchList=None, hscRun=None, matchRadius=None, zpLabel=None, forcedStr=None,
                    flagsCat=None):
        enforcer = None
        col = "ext_shapeHSM_HsmShapeRegauss_resolution"
        if "first_" + col in catalog.schema:
            shortName = "diff_resolution"
            self.log.info("shortName = {:s}".format(shortName))
            Analysis(catalog, PercentDiff(col),
                     "           Run Comparison: HsmRegauss Resolution (% diff)",
                     shortName, self.config.analysis, flags=[col + "_flag"], prefix="first_",
                     qMin=-0.2, qMax=0.2,
                     labeller=OverlapsStarGalaxyLabeller(), flagsCat=flagsCat,
                     ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                               camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                               patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                               zpLabel=zpLabel, forcedStr=forcedStr)
        col = "ext_shapeHSM_HsmShapeRegauss_e1"
        if "first_" + col in catalog.schema:
            shortName = "diff_HsmShapeRegauss_e1"
            self.log.info("shortName = {:s}".format(shortName))
            Analysis(catalog, PercentDiff(col),
                     "    Run Comparison: HsmRegauss e1 (% diff)",
                     shortName, self.config.analysis, flags=[col + "_flag"], prefix="first_",
                     qMin=-0.2, qMax=0.2,
                     labeller=OverlapsStarGalaxyLabeller(), flagsCat=flagsCat,
                     ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                               camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                               patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                               zpLabel=zpLabel, forcedStr=forcedStr)
        col = "ext_shapeHSM_HsmShapeRegauss_e2"
        if "first_" + col in catalog.schema:
            shortName = "diff_HsmShapeRegauss_e2"
            self.log.info("shortName = {:s}".format(shortName))
            Analysis(catalog, PercentDiff(col),
                     "    Run Comparison: HsmRegauss e2 (% diff)",
                     shortName, self.config.analysis, flags=[col + "_flag"], prefix="first_",
                     qMin=-0.2, qMax=0.2,
                     labeller=OverlapsStarGalaxyLabeller(), flagsCat=flagsCat,
                     ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                               camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                               patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                               zpLabel=zpLabel, forcedStr=forcedStr)

    def plotApCorrs(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None,
                    tractInfo=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None,
                    forcedStr=None, fluxToPlotList=None):
        if fluxToPlotList is None:
            fluxToPlotList = self.config.fluxToPlotList
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.02*self.unitScale}})
        for col in fluxToPlotList:
            if "first_" + col + "_apCorr" in catalog.schema and "second_" + col + "_apCorr" in catalog.schema:
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
                                       zpLabel=None, forcedStr=forcedStr)
                else:
                    self.log.warn("No valid data points for shortName = {:s}.  Skipping...".format(shortName))

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None

    def _getEupsVersionsName(self):
        return None
