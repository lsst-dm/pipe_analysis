import os
import matplotlib
matplotlib.use("Agg")  # noqa 402
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all="ignore")  # noqa 402

from collections import defaultdict

from lsst.daf.persistence.butler import Butler
from lsst.pex.config import Field, ChoiceField
from lsst.pipe.base import ArgumentParser, TaskRunner, TaskError
from lsst.meas.base.forcedPhotCcd import PerTractCcdDataIdContainer
from lsst.afw.table.catalogMatches import matchesToCatalog
from .analysis import Analysis
from .visitAnalysis import (VisitAnalysisConfig, VisitAnalysisTask, VisitAnalysisRunner,
                            CompareVisitAnalysisTask, CompareVisitAnalysisConfig, CompareVisitAnalysisRunner,
                            CcdAnalysis)
from .utils import (Filenamer, AngularDistance, concatenateCatalogs, addApertureFluxesHSC, addFpPoint,
                    addFootprintNPix, addRotPoint, makeBadArray, addIntFloatOrStrColumn,
                    backoutApCorr, matchNanojanskyToAB, andCatalog, writeParquet,
                    getRepoInfo, getCcdNameRefList, getDataExistsRefList, setAliasMaps,
                    addPreComputedColumns)
from .plotUtils import annotateAxes, labelVisit, labelCamera, plotText
from .fakesAnalysis import (addDegreePositions, matchCatalogs, addNearestNeighbor, fakesPositionCompare,
                            getPlotInfo, fakesAreaDepth, fakesMagnitudeCompare, fakesMagnitudeNearestNeighbor,
                            fakesMagnitudeBlendedness, fakesCompletenessPlot)

import lsst.afw.table as afwTable
import lsst.geom as lsstGeom

class ForcedVisitAnalysisConfig(VisitAnalysisConfig):
    excludePrefixStr = Field(
        dtype=str,
        doc=("Prefix string for flux columns to ignore when applying fluxMag0 calibration "
             "(i.e. those copied over from the coadd catalog)."),
        default="ref_",
    )

    def setDefaults(self):
        VisitAnalysisConfig.setDefaults(self)
        self.doWriteParquetTables = False
        self.doApplyExternalPhotoCalib = False
        self.doApplyExternalSkyWcs = False
        self.doPlotFootprintNpix = False
        self.doPlotMatches = False # don't get match catalogs from forcedPhotCcd
        # self.analysis.fluxColumn = "base_PsfFlux_instFlux"
        self.analysis.fluxColumn = "base_CircularApertureFlux_12_0_instFlux"
        self.analysisMatches.fluxColumn = "base_PsfFlux_instFlux"
        self.fluxToPlotList.remove("ext_photometryKron_KronFlux")
        if "base_PsfFlux" not in self.fluxToPlotList:
            self.fluxToPlotList.append("base_PsfFlux")  # Add PSF flux to default list for comparison scripts
        self.fluxToPlotList.append("ref_base_PsfFlux")
        self.fluxToPlotList.append("ref_base_CircularApertureFlux_12_0")
        self.flagsToAlias = {"calib_psf_used": "ref_calib_psf_used",
                             "calib_photometry_used": "ref_calib_photometry_used",
                             "calib_astrometry_used": "ref_calib_astrometry_used",
                             "base_ClassificationExtendedness_value":
                             "ref_base_ClassificationExtendedness_value",
                             "base_ClassificationExtendedness_flag":
                             "ref_base_ClassificationExtendedness_flag"}

    def validate(self):
        VisitAnalysisConfig.validate(self)

class ForcedVisitAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        if len(parsedCmd.id.refList) < 1:
            raise RuntimeWarning("refList from parsedCmd is empty...")
        kwargs["tract"] = parsedCmd.tract
        kwargs["subdir"] = parsedCmd.subdir
        visits = defaultdict(list)
        for ref in parsedCmd.id.refList:
            visits[ref.dataId["visit"]].append(ref)
        return [(visits[key], kwargs) for key in visits.keys()]


class ForcedVisitAnalysisTask(VisitAnalysisTask):
    _DefaultName = "forcedVisitAnalysis"
    ConfigClass = ForcedVisitAnalysisConfig
    RunnerClass = ForcedVisitAnalysisRunner
    AnalysisClass = CcdAnalysis

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "forced_src", help="data ID with raw CCD keys, "
                               "e.g. --id visit=12345 ccd=6^8..11", ContainerClass=PerTractCcdDataIdContainer)
        parser.add_argument("--tract", type=str, default=None,
                            help="Tract(s) to use (do one at a time for overlapping) e.g. 1^5^0")
        parser.add_argument("--subdir", type=str, default="",
                            help=("Subdirectory below plots/filter/tract-NNNN/forcedVisit-NNNN (useful "
                                  "for, e.g., subgrouping of CCDs.  Ignored if only one CCD is "
                                  "specified, in which case the subdir is set to ccd-NNN"))
        return parser

    def runDataRef(self, dataRefList, tract=None, subdir=""):
        self.log.info("dataRefList size: {:d}".format(len(dataRefList)))
        if tract is None:
            tractList = [0, ]
        else:
            tractList = [int(tractStr) for tractStr in tract.split('^')]
        dataRefListPerTract = [None]*len(tractList)
        for i, tract in enumerate(tractList):
            dataRefListPerTract[i] = [dataRef for dataRef in dataRefList if
                                      dataRef.dataId["tract"] == tract and dataRef.datasetExists("src")]
        commonZpDone = True
        for i, dataRefListTract in enumerate(dataRefListPerTract):
            if not dataRefListTract:
                self.log.info("No data found for tract: {:d}".format(tractList[i]))
                continue
            repoInfo = getRepoInfo(dataRefListTract[0],
                                   doApplyExternalPhotoCalib=self.config.doApplyExternalPhotoCalib,
                                   externalPhotoCalibName=self.config.externalPhotoCalibName,
                                   doApplyExternalSkyWcs=self.config.doApplyExternalSkyWcs,
                                   externalSkyWcsName=self.config.externalSkyWcsName)
            repoInfo.catDataset = "forced_src"
            self.log.info("dataId: {!s:s}".format(repoInfo.dataId))
            ccdListPerTract = getDataExistsRefList(dataRefListTract, repoInfo.catDataset)
            if not ccdListPerTract:
                raise RuntimeError("No datasets found for datasetType = {:s}".format(repoInfo.catDataset))

            subdir = "ccd-" + str(ccdListPerTract[0]) if len(ccdListPerTract) == 1 else subdir
            filenamer = Filenamer(repoInfo.butler, "plotForcedVisit", repoInfo.dataId, subdir=subdir)
            # Create list of alias mappings for differing schema naming conventions (if any)
            aliasDictList = [self.config.flagsToAlias, ]
            if repoInfo.hscRun and self.config.srcSchemaMap is not None:
                aliasDictList += [self.config.srcSchemaMap]
            # Always highlight points with x-axis flag set (for cases where
            # they do not get explicitly filtered out).
            highlightList = [
                (self.config.analysis.fluxColumn.replace("_instFlux", "_flag"), 0, "turquoise"), ]

            if any(doPlot for doPlot in [self.config.doPlotFootprintNpix, self.config.doPlotQuiver,
                                         self.config.doPlotMags, self.config.doPlotSizes,
                                         self.config.doPlotCentroids, self.config.doPlotStarGalaxy,
                                         self.config.doPlotSkyObjects, self.config.doWriteParquetTables]):
                commonZpCat, catalog = self.readCatalogs(dataRefListTract, repoInfo.catDataset, repoInfo,
                                                         aliasDictList=aliasDictList,
                                                         excludePrefixStr=self.config.excludePrefixStr)
                self.zpLabel = "forcedCcd"

                # Make sub-catalog of sky sources before flag culling as many of
                # these will have flags set due to measurement difficulties in
                # regions that are really blank sky.  Caveat: we don't want any
                # on the visit "edge", which could happen since these are from
                # the coadd sky object placements, so omit those.
                if self.config.doPlotSkyObjects:
                    if "merge_peak_sky" in catalog.schema:
                        skySrcCat = catalog[catalog["merge_peak_sky"] &
                                            ~catalog["base_PixelFlags_flag_edge"]].copy(deep=True)
                    else:
                        skySrcCat = None
                        self.log.warn("doPlotSkyObjects is True, but the \"merge_peak_sky\" "
                                      "column does not exist in catalog.schema.  Skipping "
                                      "skyObjects plot.")

                # Set boolean arrays indicating sources deemed unsuitable for qa analyses
                self.catLabel = "nChild = 0"
                bad = makeBadArray(catalog, flagList=["base_PixelFlags_flag_edge"],
                                   onlyReadStars=self.config.onlyReadStars)
                badCommonZp = makeBadArray(commonZpCat, flagList=["base_PixelFlags_flag_edge"],
                                           onlyReadStars=self.config.onlyReadStars)

                # purge the catalogs of flagged sources
                catalog = catalog[~bad].copy(deep=True)
                commonZpCat = commonZpCat[~badCommonZp].copy(deep=True)

                # Dict of all parameters common to plot* functions
                plotKwargs = dict(butler=repoInfo.butler, camera=repoInfo.camera, ccdList=ccdListPerTract,
                                  hscRun=repoInfo.hscRun, tractInfo=repoInfo.tractInfo)
                if self.config.doPlotSkyObjects and skySrcCat is not None:
                    self.plotSkyObjects(skySrcCat, filenamer(repoInfo.dataId, description="skySources",
                                                             style="hist"), repoInfo.dataId, **plotKwargs)
                if self.config.doPlotPsfFluxSnHists:
                    self.plotPsfFluxSnHists(commonZpCat,
                                            filenamer(repoInfo.dataId, description="base_PsfFlux_raw",
                                                      style="hist"),
                                        repoInfo.dataId, zpLabel="raw", **plotKwargs)
                    self.plotPsfFluxSnHists(catalog,
                                            filenamer(repoInfo.dataId, description="base_PsfFlux_cal",
                                                      style="hist"),
                                            repoInfo.dataId, zpLabel=self.zpLabel, **plotKwargs)
                plotKwargs.update(dict(zpLabel=self.zpLabel))
                if self.config.doPlotFootprintNpix:
                    self.plotFootprintHist(catalog,
                                           filenamer(repoInfo.dataId, description="footNpix", style="hist"),
                                           repoInfo.dataId, **plotKwargs)
                    self.plotFootprint(catalog, filenamer, repoInfo.dataId, plotRunStats=False,
                                       highlightList=highlightList + [("parent", 0, "yellow"), ],
                                       **plotKwargs)

                if self.config.doPlotQuiver:
                    self.plotQuiver(catalog,
                                    filenamer(repoInfo.dataId, description="ellipResids", style="quiver"),
                                    dataId=repoInfo.dataId, scale=2, **plotKwargs)

                plotKwargs.update(dict(highlightList=highlightList))
                # Create mag comparison plots using common ZP
                if self.config.doPlotMags and not commonZpDone:
                    zpLabel = "common (%s)" % self.config.analysis.commonZp
                    plotKwargs.update(dict(zpLabel=zpLabel))
                    self.plotMags(commonZpCat, filenamer, repoInfo.dataId,
                                  fluxToPlotList=["base_GaussianFlux", "base_CircularApertureFlux_12_0"],
                                  postFix="_commonZp", **plotKwargs)
                    commonZpDone = True
                # Now calibrate the source catalg to either the instrumental flux corresponding
                # to 0th magnitude, fluxMag0, from SFM or the external-calibration solution
                # (from jointcal, fgcm, or meas_mosaic) for remainder of plots.
                plotKwargs.update(dict(zpLabel=self.zpLabel))
                if self.config.doPlotMags:
                    self.plotMags(catalog, filenamer, repoInfo.dataId, **plotKwargs)
                if self.config.doPlotSizes:
                    if "base_SdssShape_psf_xx" in catalog.schema:
                        self.plotSizes(catalog, filenamer, repoInfo.dataId, **plotKwargs)
                    else:
                        self.log.warn("Cannot run plotSizes: base_SdssShape_psf_xx not in catalog.schema")
                if self.config.doPlotCentroids and self.haveFpCoords:
                    self.plotCentroidXY(catalog, filenamer, repoInfo.dataId, **plotKwargs)

            if self.config.doPlotMatches:
                matches = self.readSrcMatches(dataRefListTract, "forced_src", repoInfo,
                                              aliasDictList=aliasDictList)
                # Dict of all parameters common to plot* functions
                matchHighlightList = [("src_" + self.config.analysis.fluxColumn.replace("_instFlux", "_flag"),
                                       0, "turquoise"), ]
                plotKwargs = dict(butler=repoInfo.butler, camera=repoInfo.camera, ccdList=ccdListPerTract,
                                  hscRun=repoInfo.hscRun, zpLabel=self.zpLabel,
                                  highlightList=matchHighlightList)
                self.plotMatches(matches, repoInfo.filterName, filenamer, repoInfo.dataId, **plotKwargs)

                for cat in self.config.externalCatalogs:
                    if self.config.photoCatName not in cat:
                        with andCatalog(cat):
                            matches = self.matchCatalog(catalog, repoInfo.filterName,
                                                        self.config.externalCatalogs[cat])
                            self.plotMatches(matches, repoInfo.filterName, filenamer, repoInfo.dataId,
                                             matchRadius=self.matchRadius,
                                             matchRadiusUnitStr=self.matchRadiusUnitStr, **plotKwargs)


class CompareForcedVisitAnalysisConfig(CompareVisitAnalysisConfig, ForcedVisitAnalysisConfig):

    def setDefaults(self):
        CompareVisitAnalysisConfig.setDefaults(self)
        ForcedVisitAnalysisConfig.setDefaults(self)
        self.doApplyExternalPhotoCalib1 = False
        self.doApplyExternalPhotoCalib2 = False
        self.doApplyExternalSkyWcs1 = False
        self.doApplyExternalSkyWcs2 = False
        self.matchRadiusRaDec = 0.02  # These are matched forced catalogs

class CompareForcedVisitAnalysisTask(CompareVisitAnalysisTask):
    ConfigClass = CompareForcedVisitAnalysisConfig
    RunnerClass = CompareVisitAnalysisRunner
    _DefaultName = "compareForcedVisitAnalysis"

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_argument("--rerun2", required=True, help="Second rerun, for comparison")
        parser.add_id_argument("--id", "forced_src", help="data ID with raw CCD keys, "
                               "e.g. --id visit=12345 ccd=6^8..11", ContainerClass=PerTractCcdDataIdContainer)
        parser.add_argument("--tract", type=str, default=None,
                            help="Tract(s) to use (do one at a time for overlapping) e.g. 1^5^0")
        parser.add_argument("--subdir", type=str, default="",
                            help=("Subdirectory below plots/filter/tract-NNNN/forcedVisit-NNNN (useful "
                                  "for, e.g., subgrouping of CCDs.  Ignored if only one CCD is "
                                  "specified, in which case the subdir is set to ccd-NNN"))
        return parser

    def runDataRef(self, dataRefList1, dataRefList2, tract=None, subdir=""):
        if tract is None:
            tractList = [0, ]
        else:
            tractList = [int(tractStr) for tractStr in tract.split('^')]
        self.log.debug("tractList = {}".format(tractList))
        dataRefListPerTract1 = [None]*len(tractList)
        dataRefListPerTract2 = [None]*len(tractList)
        for i, tract in enumerate(tractList):
            dataRefListPerTract1[i] = [dataRef1 for dataRef1 in dataRefList1 if
                                       dataRef1.dataId["tract"] == tract]
            dataRefListPerTract2[i] = [dataRef2 for dataRef2 in dataRefList2 if
                                       dataRef2.dataId["tract"] == tract]
        if len(dataRefListPerTract1) != len(dataRefListPerTract2):
            raise TaskError("Lengths of comparison dataRefLists do not match!")
        commonZpDone = True

        i = -1
        for dataRefListTract1, dataRefListTract2 in zip(dataRefListPerTract1, dataRefListPerTract2):
            i += 1
            if not dataRefListTract1:
                self.log.info("No data found in --rerun for tract: {:d}".format(tractList[i]))
                continue
            if not dataRefListTract2:
                self.log.info("No data found in --rerun2 for tract: {:d}".format(tractList[i]))
                continue
            # Get a butler and dataId for each dataset.  Needed for feeding a butler and camera into the
            # plotting functions (for labelling the camera and plotting ccd outlines) in addition to
            # determining if the data were processed with the HSC stack.  We assume all processing in a
            # given rerun is self-consistent, so only need one valid dataId per comparison rerun.
            repoInfo1 = getRepoInfo(dataRefListTract1[0],
                                    doApplyExternalPhotoCalib=self.config.doApplyExternalPhotoCalib1,
                                    externalPhotoCalibName=self.config.externalPhotoCalibName1,
                                    doApplyExternalSkyWcs=self.config.doApplyExternalSkyWcs1,
                                    externalSkyWcsName=self.config.externalSkyWcsName1)
            repoInfo2 = getRepoInfo(dataRefListTract2[0],
                                    doApplyExternalPhotoCalib=self.config.doApplyExternalPhotoCalib2,
                                    externalPhotoCalibName=self.config.externalPhotoCalibName2,
                                    doApplyExternalSkyWcs=self.config.doApplyExternalSkyWcs2,
                                    externalSkyWcsName=self.config.externalSkyWcsName2)
            repoInfo1.catDataset = "forced_" + repoInfo1.catDataset
            repoInfo2.catDataset = "forced_" + repoInfo2.catDataset

            fullCameraCcdList1 = getCcdNameRefList(dataRefListTract1)

            ccdListPerTract1 = getDataExistsRefList(dataRefListTract1, repoInfo1.catDataset)
            ccdListPerTract2 = getDataExistsRefList(dataRefListTract2, repoInfo2.catDataset)
            if not ccdListPerTract1:
                raise RuntimeError(f"No datasets found for datasetType = {repoInfo1.catDataset}")
            if not ccdListPerTract2:
                raise RuntimeError(f"No datasets found for datasetType = {repoInfo2.catDataset}")

            if self.config.doApplyExternalPhotoCalib1:
                ccdPhotoCalibListPerTract1 = getDataExistsRefList(dataRefListTract1,
                                                                  repoInfo1.photoCalibDataset)
                if not ccdPhotoCalibListPerTract1:
                    self.log.fatal(f"No data found for {repoInfo1.photoCalibDataset} dataset...are you "
                                   "sure you ran the external calibration?  If not, run with "
                                   "--config doApplyExternalPhotoCalib1=False")
            if self.config.doApplyExternalPhotoCalib2:
                ccdPhotoCalibListPerTract2 = getDataExistsRefList(dataRefListTract2,
                                                                  repoInfo2.photoCalibDataset)
                if not ccdPhotoCalibListPerTract2:
                    self.log.fatal(f"No data found for {repoInfo2.photoCalibDataset} dataset...are you "
                                   "sure you ran the external calibration?  If not, run with "
                                   "--config doApplyExternalPhotoCalib2=False")

            ccdIntersectList = list(set(ccdListPerTract1).intersection(set(ccdListPerTract2)))
            self.log.info("tract: {:d}".format(repoInfo1.dataId["tract"]))
            self.log.info(f"ccdListPerTract1: \n{ccdListPerTract1}")
            self.log.info(f"ccdListPerTract2: \n{ccdListPerTract2}")
            self.log.info(f"ccdIntersectList: \n{ccdIntersectList}")
            if self.config.doApplyExternalPhotoCalib1:
                self.log.info(f"ccdPhotoCalibListPerTract1: \n{ccdPhotoCalibListPerTract1}")
            if self.config.doApplyExternalPhotoCalib2:
                self.log.info(f"ccdPhotoCalibListPerTract2: \n{ccdPhotoCalibListPerTract2}")

            doReadFootprints = None
            if self.config.doPlotFootprintNpix:
                doReadFootprints = "light"
            # Set some aliases for differing schema naming conventions
            aliasDictList = [self.config.flagsToAlias, ]
            if (repoInfo1.hscRun or repoInfo2.hscRun) and self.config.srcSchemaMap is not None:
                aliasDictList += [self.config.srcSchemaMap]
            commonZpCat1, catalog1, commonZpCat2, catalog2 = (
                self.readCatalogs(dataRefListTract1, dataRefListTract2, "forced_src", repoInfo1, repoInfo2,
                                  doReadFootprints=doReadFootprints, aliasDictList=aliasDictList,
                                  excludePrefixStr=self.config.excludePrefixStr))

            # Set boolean arrays indicating sources deemed unsuitable for qa analyses
            bad1 = makeBadArray(catalog1, flagList=["base_PixelFlags_flag_edge"],
                                onlyReadStars=self.config.onlyReadStars)
            bad2 = makeBadArray(catalog2, flagList=["base_PixelFlags_flag_edge"],
                                onlyReadStars=self.config.onlyReadStars)
            badCommonZp1 = makeBadArray(commonZpCat1, flagList=None, onlyReadStars=self.config.onlyReadStars)
            badCommonZp2 = makeBadArray(commonZpCat2, flagList=None, onlyReadStars=self.config.onlyReadStars)
            print(sum(bad1), sum(bad2))
            blendednessMax = 0.1
            blendednessStr = catalog1.schema.extract(self.config.excludePrefixStr + "base_Blendedness_abs")
            if not blendednessStr:
                blendednessStr = catalog1.schema.extract(
                    self.config.excludePrefixStr + "base_Blendedness_raw")
            if blendednessStr:
                blendednessStr = list(blendednessStr.keys())[0]
                # Exclude blended objects
                if blendednessStr in catalog1.schema:
                    bad1 |= catalog1[blendednessStr] > blendednessMax
                    badCommonZp1 |= catalog1[blendednessStr] > blendednessMax
                if blendednessStr in catalog2.schema:
                    bad2 |= catalog2[blendednessStr] > blendednessMax
                    badCommonZp2 |= catalog2[blendednessStr] > blendednessMax
            self.zpLabel = "forcedCcd\nblendedness <= " + str(blendednessMax)

            # purge the catalogs of flagged sources
            catalog1 = catalog1[~bad1].copy(deep=True)
            catalog2 = catalog2[~bad2].copy(deep=True)
            commonZpCat1 = commonZpCat1[~badCommonZp1].copy(deep=True)
            commonZpCat2 = commonZpCat2[~badCommonZp2].copy(deep=True)

            self.log.info("\nNumber of sources in catalogs: first = {0:d} and second = {1:d}".format(
                          len(catalog1), len(catalog2)))
            commonZpCat = self.matchCatalogs(commonZpCat1, commonZpCat2, matchRadius=self.matchRadius,
                                             matchControl=self.matchControl)
            catalog = self.matchCatalogs(catalog1, catalog2, matchRadius=self.matchRadius,
                                         matchControl=self.matchControl)
            # Set some aliases for differing schema naming conventions
            if aliasDictList:
                for cat in [commonZpCat, catalog]:
                    cat = setAliasMaps(cat, aliasDictList)

            self.log.info("Number of matches (maxDist = {0:.2f}{1:s}) = {2:d}".format(
                          self.matchRadius, self.matchRadiusUnitStr, len(catalog)))

            subdir = "ccd-" + str(ccdListPerTract1[0]) if len(ccdIntersectList) == 1 else subdir
            filenamer = Filenamer(repoInfo1.butler, "plotCompareForcedVisit", repoInfo1.dataId, subdir=subdir)
            hscRun = repoInfo1.hscRun if repoInfo1.hscRun else repoInfo2.hscRun

            # Dict of all parameters common to plot* functions
            tractInfo1 = repoInfo1.tractInfo if self.config.doApplyExternalPhotoCalib1 else None
            tractInfo2 = repoInfo2.tractInfo if self.config.doApplyExternalPhotoCalib2 else None
            tractInfo = tractInfo1 if (tractInfo1 or tractInfo2) else None
            # Always highlight points with x-axis flag set (for cases where
            # they do not get explicitly filtered out).
            highlightList = [(self.config.analysis.fluxColumn.replace("_instFlux", "_flag"), 0,
                              "turquoise"), ]
            plotKwargs1 = dict(butler=repoInfo1.butler, camera=repoInfo1.camera, hscRun=hscRun,
                               matchRadius=self.matchRadius, matchRadiusUnitStr=self.matchRadiusUnitStr,
                               zpLabel=self.zpLabel, tractInfo=tractInfo, highlightList=highlightList)

            if self.config.doPlotFootprintNpix:
                self.plotFootprint(catalog, filenamer, repoInfo1.dataId, ccdList=ccdIntersectList,
                                   **plotKwargs1)

            # Create mag comparison plots using common ZP
            if not commonZpDone:
                zpLabel = "common (" + str(self.config.analysis.commonZp) + ")"
                try:
                    zpLabel = zpLabel + " " + self.catLabel
                except Exception:
                    pass
                plotKwargs1.update(dict(zpLabel=zpLabel))
                self.plotMags(commonZpCat, filenamer, repoInfo1.dataId, ccdList=fullCameraCcdList1,
                              fluxToPlotList=["base_GaussianFlux", "base_CircularApertureFlux_12_0"],
                              postFix="_commonZp", **plotKwargs1)
                commonZpDone = True

            plotKwargs1.update(dict(zpLabel=self.zpLabel))
            if self.config.doPlotMags:
                self.plotMags(catalog, filenamer, repoInfo1.dataId, ccdList=ccdIntersectList, **plotKwargs1)
            if self.config.doPlotSizes:
                if ("first_base_SdssShape_psf_xx" in catalog.schema and
                        "second_base_SdssShape_psf_xx" in catalog.schema):
                    self.plotSizes(catalog, filenamer, repoInfo1.dataId, ccdList=ccdIntersectList,
                                   **plotKwargs1)
                else:
                    self.log.warn("Cannot run plotSizes: base_SdssShape_psf_xx not in catalog.schema")
            if self.config.doApCorrs:
                self.plotApCorrs(catalog, filenamer, repoInfo1.dataId, ccdList=ccdIntersectList,
                                 **plotKwargs1)
            if self.config.doPlotCentroids:
                self.plotCentroids(catalog, filenamer, repoInfo1.dataId, ccdList=ccdIntersectList,
                                   **plotKwargs1)
            if self.config.doPlotStarGalaxy:
                self.plotStarGal(catalog, filenamer, repoInfo1.dataId, ccdList=ccdIntersectList,
                                 **plotKwargs1)
