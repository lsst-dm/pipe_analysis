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
from .visitAnalysis import VisitAnalysisConfig, VisitAnalysisTask, CompareVisitAnalysisTask, CcdAnalysis
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
