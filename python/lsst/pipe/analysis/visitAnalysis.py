#!/usr/bin/env python

import os
import matplotlib
matplotlib.use("Agg")  # noqa 402
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all="ignore")  # noqa 402

from collections import defaultdict

from lsst.daf.persistence.butler import Butler
from lsst.pex.config import Field, ChoiceField
from lsst.pipe.base import ArgumentParser, TaskRunner, TaskError, Struct
from lsst.meas.base.forcedPhotCcd import PerTractCcdDataIdContainer
from lsst.afw.table.catalogMatches import matchesToCatalog
from .analysis import Analysis
from .coaddAnalysis import CoaddAnalysisConfig, CoaddAnalysisTask, CompareCoaddAnalysisTask
from .utils import (AngularDistance, concatenateCatalogs, addApertureFluxesHSC, addFpPoint,
                    addFootprintNPix, addRotPoint, makeBadArray, addIntFloatOrStrColumn,
                    calibrateSourceCatalogMosaic, calibrateSourceCatalogPhotoCalib,
                    calibrateSourceCatalog, backoutApCorr, matchNanojanskyToAB, andCatalog, writeParquet,
                    getRepoInfo, getCcdNameRefList, getDataExistsRefList, setAliasMaps,
                    addPreComputedColumns, savePlots, updateVerifyJob)
from .plotUtils import annotateAxes, labelVisit, labelCamera, plotText, getPlotInfo
from .fakesAnalysis import (addDegreePositions, matchCatalogs, addNearestNeighbor, fakesPositionCompare,
                            calcFakesAreaDepth, plotFakesAreaDepth, fakesMagnitudeCompare,
                            fakesMagnitudeNearestNeighbor, fakesMagnitudeBlendedness, fakesCompletenessPlot,
                            fakesMagnitudePositionError)

import lsst.afw.cameraGeom as cameraGeom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.geom as geom


class CcdAnalysis(Analysis):
    def plotAll(self, description, plotInfoDict, areaDict, log, enforcer=None, matchRadius=None,
                matchRadiusUnitStr=None, zpLabel=None, forcedStr=None, uberCalLabel=None, postFix="",
                plotRunStats=True, highlightList=None, haveFpCoords=None, doPrintMedian=False):
        stats = self.stats
        if self.config.doPlotCcdXy:
            yield from self.plotCcd(self.shortName, style="ccd" + postFix, stats=self.stats,
                                    matchRadius=matchRadius, matchRadiusUnitStr=matchRadiusUnitStr,
                                    zpLabel=zpLabel)
        if self.config.doPlotFP and haveFpCoords:
            yield from self.plotFocalPlane(self.shortname, plotInfoDict, style="fpa" + postFix,
                                           stats=stats, matchRadius=matchRadius,
                                           matchRadiusUnitStr=matchRadiusUnitStr, zpLabel=zpLabel)

        yield from Analysis.plotAll(self, description, plotInfoDict, areaDict, log, enforcer=enforcer,
                                    matchRadius=matchRadius,
                                    matchRadiusUnitStr=matchRadiusUnitStr, zpLabel=zpLabel, postFix=postFix,
                                    plotRunStats=plotRunStats, highlightList=highlightList,
                                    doPrintMedian=doPrintMedian)

    def plotFP(self, description, plotInfoDict, log, enforcer=None, matchRadius=None, matchRadiusUnitStr=None,
               zpLabel=None, forcedStr=None):
        yield from self.plotFocalPlane(self.shortName, plotInfoDict, style="fpa", stats=self.stats,
                                       matchRadius=matchRadius, matchRadiusUnitStr=matchRadiusUnitStr,
                                       zpLabel=zpLabel, forcedStr=forcedStr)

    def plotCcd(self, description, plotInfoDict, centroid="base_SdssCentroid", cmap=plt.cm.nipy_spectral,
                idBits=32, visitMultiplier=200, stats=None, matchRadius=None,
                matchRadiusUnitStr=None, zpLabel=None, doPrintMedian=False, style="ccd"):
        """Plot quantity as a function of CCD x,y"""
        xx = self.catalog[self.prefix + centroid + "_x"]
        yy = self.catalog[self.prefix + centroid + "_y"]
        ccd = (self.catalog[self.prefix + "id"] >> idBits) % visitMultiplier
        vMin, vMax = ccd.min(), ccd.max()
        if vMin == vMax:
            vMin, vMax = vMin - 2, vMax + 2
            self.log.info("Only one CCD ({0:d}) to analyze: setting vMin ({1:d}), vMax ({2:d})".format(
                          ccd.min(), vMin, vMax))
        good = (self.mag < self.config.magThreshold if self.config.magThreshold > 0 else
                np.ones(len(self.mag), dtype=bool))
        fig, axes = plt.subplots(2, 1)
        axes[0].axhline(0, linestyle="--", color="0.6")
        axes[1].axhline(0, linestyle="--", color="0.6")
        ptSize = None
        for name, data in self.data.items():
            if not data.plot:
                continue
            if len(data.mag) == 0:
                continue
            if not ptSize:
                ptSize = min(12, max(4, int(25/np.log10(len(data.mag)))))
            selection = data.selection & good
            quantity = data.quantity[good[data.selection]]
            kwargs = {"s": ptSize, "marker": "o", "lw": 0, "alpha": 0.5, "cmap": cmap,
                      "vmin": vMin, "vmax": vMax}
            axes[0].scatter(xx[selection], quantity, c=ccd[selection], **kwargs)
            axes[1].scatter(yy[selection], quantity, c=ccd[selection], **kwargs)

        axes[0].set_xlabel("x_ccd", labelpad=-1)
        axes[1].set_xlabel("y_ccd")
        fig.text(0.02, 0.5, self.quantityName, ha="center", va="center", rotation="vertical")
        if stats is not None:
            annotateAxes(description, plt, axes[0], stats, "star", self.config.magThreshold, x0=0.03,
                         yOff=0.07, hscRun=plotInfoDict["hscRun"], matchRadiusUnitStr=matchRadiusUnitStr,
                         unitScale=self.unitScale, doPrintMedian=doPrintMedian, matchRadius=matchRadius)
            annotateAxes(description, plt, axes[1], stats, "star", self.config.magThreshold, x0=0.03,
                         yOff=0.07, hscRun=plotInfoDict["hscRun"], matchRadiusUnitStr=matchRadiusUnitStr,
                         unitScale=self.unitScale, doPrintMedian=doPrintMedian, matchRadius=matchRadius)
        axes[0].set_xlim(-100, 2150)
        axes[1].set_xlim(-100, 4300)
        axes[0].set_ylim(self.qMin, self.qMax)
        axes[1].set_ylim(self.qMin, self.qMax)

        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vMin, vmax=vMax))
        mappable._A = []        # fake up the array of the scalar mappable. Urgh...
        fig.subplots_adjust(right=0.8)
        cax = fig.add_axes([0.83, 0.15, 0.04, 0.7])
        cb = fig.colorbar(mappable, cax=cax)
        cb.set_label("CCD index", rotation=270, labelpad=15)
        labelVisit(description, plt, axes[0], 0.5, 1.1)
        if zpLabel:
            plotText(zpLabel, plt, axes[0], 0.08, -0.11, prefix="zp: ", color="green")
        yield Struct(fig=fig, description=description, style=style, stats=stats)

    def plotFocalPlane(self, description, plotInfoDict, cmap=plt.cm.Spectral, stats=None, matchRadius=None,
                       matchRadiusUnitStr=None, zpLabel=None, forcedStr=None, fontSize=8, style="fpa"):
        """Plot quantity colormaped on the focal plane"""
        xFp = self.catalog[self.prefix + "base_FPPosition_x"]
        yFp = self.catalog[self.prefix + "base_FPPosition_y"]
        good = (self.mag < self.config.magThreshold if self.config.magThreshold > 0 else
                np.ones(len(self.mag), dtype=bool))
        if "galaxy" in self.data and "calib_psf_used" not in self.goodKeys:
            vMin, vMax = 0.5*self.qMin, 0.5*self.qMax
        else:
            vMin, vMax = self.qMin, self.qMax
        # Set limits to ccd pixel ranges when plotting the centroids (which are in pixel units)
        if description.find("Centroid") > -1:
            cmap = plt.cm.pink
            vMin = min(0, np.round(self.data["star"].quantity.min() - 10))
            vMax = np.round(self.data["star"].quantity.max() + 50, -2)
        fig, axes = plt.subplots(1, 1, subplot_kw=dict(facecolor="0.7"))
        axes.tick_params(which="both", direction="in", top=True, right=True, labelsize=fontSize)
        for name, data in self.data.items():
            if not data.plot:
                continue
            if len(data.mag) == 0:
                continue
            selection = data.selection & good
            axes.scatter(xFp[selection], yFp[selection], s=2, marker="o", lw=0,
                         c=data.quantity[good[data.selection]], cmap=cmap, vmin=vMin, vmax=vMax)
        axes.set_xlabel("x_fpa (pixels)")
        axes.set_ylabel("y_fpa (pixels)")

        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vMin, vmax=vMax))
        mappable._A = []        # fake up the array of the scalar mappable. Urgh...
        cb = plt.colorbar(mappable)
        cb.set_label(self.quantityName, rotation=270, labelpad=15)
        if plotInfoDict["hscRun"]:
            axes.set_title("HSC stack run: " + plotInfoDict["hscRun"], color="#800080")
        labelVisit(plotInfoDict, fig, axes, 0.5, 1.04)
        if plotInfoDict["camera"]:
            labelCamera(plotInfoDict, fig, axes, 0.5, 1.09)
        if zpLabel:
            plotText(zpLabel, fig, axes, 0.08, -0.1, prefix="zp: ", color="green")
        if forcedStr:
            plotText(forcedStr, fig, axes, 0.86, -0.1, prefix="cat: ", color="green")
        yield Struct(fig=fig, description=description, stats=stats, style=style)


class VisitAnalysisConfig(CoaddAnalysisConfig):
    doApplyExternalPhotoCalib = Field(dtype=bool, default=True,
                                      doc=("Whether to apply external photometric calibration (e.g. "
                                           "fgcmcal/jointcal/meas_mosaic) via an `lsst.afw.image.PhotoCalib` "
                                           "object?  If `True`, uses ``externalPhotoCalibName`` field to "
                                           "determine which calibration to load.  If `False`, the "
                                           "instrumental flux corresponding to 0th magnitude, fluxMag0, "
                                           "from SFM is applied."))
    externalPhotoCalibName = ChoiceField(dtype=str, default="jointcal",
                                         allowed={"jointcal": "Use jointcal_photoCalib",
                                                  "fgcm": "Use fgcm_photoCalib",
                                                  "fgcm_tract": "Use fgcm_tract_photoCalib"},
                                         doc=("Type of external `lsst.afw.image.PhotoCalib` if "
                                              "``doApplyExternalPhotoCalib`` is `True`."))
    doApplyExternalSkyWcs = Field(dtype=bool, default=True,
                                  doc=("Whether to apply external astrometric calibration via an "
                                       "`lsst.afw.geom.SkyWcs` object.  Uses ``externalSkyWcsName`` field "
                                       "to determine which calibration to load."))
    externalSkyWcsName = ChoiceField(dtype=str, default="jointcal",
                                     allowed={"jointcal": "Use jointcal_wcs"},
                                     doc=("Type of external `lsst.afw.geom.SkyWcs` if "
                                          "``doApplyExternalSkyWcs`` is `True`."))
    useMeasMosaic = Field(dtype=bool, default=False, doc="Use meas_mosaic's applyMosaicResultsExposure " +
                          "to apply meas_mosaic calibration results to catalog (i.e. as opposed to using " +
                          "the photoCalib object)?")

    hasFakes = Field(dtype=bool, default=False, doc="Include the analysis of the added fake sources?")

    inputFakesRaCol = Field(
        dtype=str,
        doc="RA column name used in the fake source catalog.",
        default="raJ2000"
    )

    inputFakesDecCol = Field(
        dtype=str,
        doc="Dec. column name used in the fake source catalog.",
        default="decJ2000"
    )

    catalogRaCol = Field(
        dtype=str,
        doc="RA column name used in the source catalog.",
        default="coord_ra",
    )

    catalogDecCol = Field(
        dtype=str,
        doc="Dec. column name used in the source catalog.",
        default="coord_dec",
    )

    def setDefaults(self):
        CoaddAnalysisConfig.setDefaults(self)
        self.analysis.fluxColumn = "base_PsfFlux_instFlux"
        self.analysisMatches.fluxColumn = "base_PsfFlux_instFlux"

    def validate(self):
        CoaddAnalysisConfig.validate(self)


class VisitAnalysisRunner(TaskRunner):
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


class VisitAnalysisTask(CoaddAnalysisTask):
    _DefaultName = "visitAnalysis"
    ConfigClass = VisitAnalysisConfig
    RunnerClass = VisitAnalysisRunner
    AnalysisClass = CcdAnalysis

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "src", help="data ID with raw CCD keys, "
                               "e.g. --id visit=12345 ccd=6^8..11", ContainerClass=PerTractCcdDataIdContainer)
        parser.add_argument("--tract", type=str, default=None,
                            help="Tract(s) to use (do one at a time for overlapping) e.g. 1^5^0")
        parser.add_argument("--subdir", type=str, default="",
                            help=("Subdirectory below plots/filter/tract-NNNN/visit-NNNN (useful "
                                  "for, e.g., subgrouping of CCDs.  Ignored if only one CCD is "
                                  "specified, in which case the subdir is set to ccd-NNN"))
        return parser

    def runDataRef(self, dataRefList, tract=None, subdir=""):
        plotList = []
        self.log.info("dataRefList size: {:d}".format(len(dataRefList)))
        if tract is None:
            tractList = [0, ]
        else:
            tractList = [int(tractStr) for tractStr in tract.split('^')]
        dataRefListPerTract = [None]*len(tractList)
        for i, tract in enumerate(tractList):
            dataRefListPerTract[i] = [dataRef for dataRef in dataRefList if
                                      dataRef.dataId["tract"] == tract and dataRef.datasetExists("src")]
        commonZpDone = False
        for i, dataRefListTract in enumerate(dataRefListPerTract):
            if not dataRefListTract:
                self.log.info("No data found for tract: {:d}".format(tractList[i]))
                continue
            repoInfo = getRepoInfo(dataRefListTract[0],
                                   doApplyExternalPhotoCalib=self.config.doApplyExternalPhotoCalib,
                                   externalPhotoCalibName=self.config.externalPhotoCalibName,
                                   doApplyExternalSkyWcs=self.config.doApplyExternalSkyWcs,
                                   externalSkyWcsName=self.config.externalSkyWcsName)
            self.log.info("dataId: {!s:s}".format(repoInfo.dataId))
            ccdListPerTract = getDataExistsRefList(dataRefListTract, repoInfo.catDataset)

            plotInfoDict = getPlotInfo(repoInfo)
            subdir = "ccd-" + str(ccdListPerTract[0]) if len(ccdListPerTract) == 1 else subdir
            repoInfo.dataId["subdir"] = "/" + subdir
            # Dict of all parameters common to plot* functions
            plotInfoDict.update(dict(plotType="plotVisit", subdir=subdir, ccdList=ccdListPerTract,
                                     hscRun=repoInfo.hscRun, tractInfo=repoInfo.tractInfo,
                                     dataId=repoInfo.dataId))

            if not ccdListPerTract:
                raise RuntimeError("No datasets found for datasetType = {:s}".format(repoInfo.catDataset))
            if self.config.doApplyExternalPhotoCalib:
                ccdPhotoCalibListPerTract = getDataExistsRefList(dataRefListTract, repoInfo.photoCalibDataset)
                if not ccdPhotoCalibListPerTract:
                    self.log.fatal(f"No data found for {repoInfo.photoCalibDataset} dataset...are you sure "
                                   "you ran the external photometric calibration?  If not, run with "
                                   "--config doApplyExternalPhotoCalib=False")
            if self.config.doApplyExternalSkyWcs:
                # Check for wcs for compatibility with old dataset naming
                ccdSkyWcsListPerTract = getDataExistsRefList(dataRefListTract, repoInfo.skyWcsDataset)
                if not ccdSkyWcsListPerTract:
                    ccdSkyWcsListPerTract = getDataExistsRefList(dataRefListTract, "wcs")
                    if ccdSkyWcsListPerTract:
                        repoInfo.skyWcsDataset = "wcs"
                        self.log.info("Old meas_mosaic dataset naming: wcs (new name is jointcal_wcs)")
                    else:
                        self.log.fatal(f"No data found for {repoInfo.skyWcsDataset} dataset...are you sure "
                                       "you ran the external astrometric calibration?  If not, run with "
                                       "--config doApplyExternalSkyWcs=False")
            self.log.info(f"Existing {repoInfo.catDataset} data for tract {tractList[i]}: "
                          f"ccdListPerTract = \n{ccdListPerTract}")
            if self.config.doApplyExternalPhotoCalib:
                self.log.info(f"Existing {repoInfo.photoCalibDataset} data for tract {tractList[i]}: "
                              f"ccdPhotoCalibListPerTract = \n{ccdPhotoCalibListPerTract}")
            if self.config.doApplyExternalSkyWcs:
                self.log.info(f"Existing {repoInfo.skyWcsDataset} data for tract {tractList[i]}: "
                              f"ccdSkyWcsListPerTract = \n{ccdSkyWcsListPerTract}")
            if self.config.doApplyExternalPhotoCalib and not ccdPhotoCalibListPerTract:
                raise RuntimeError(f"No {repoInfo.photoCalibDataset} datasets were found...are you sure "
                                   "you ran the specified external photometric calibration?  If no "
                                   "photometric external calibrations are to be applied, run with "
                                   "--config doApplyExternalPhotoCalib=False")
            if self.config.doApplyExternalSkyWcs and not ccdSkyWcsListPerTract:
                raise RuntimeError(f"No {repoInfo.skyWcsDataset} datasets were found...are you sure "
                                   "you ran the specified external astrometric calibration?  If no "
                                   "astrometric external calibrations are to be applied, run with "
                                   "--config doApplyExternalSkywcs=False")
            if self.config.doApplyExternalPhotoCalib:
                if set(ccdListPerTract) != set(ccdPhotoCalibListPerTract):
                    self.log.warn(f"Did not find {repoInfo.photoCalibDataset} external calibrations for "
                                  f"all dataIds that do have {repoInfo.catDataset} catalogs.")
            if self.config.doApplyExternalPhotoCalib:
                if set(ccdListPerTract) != set(ccdSkyWcsListPerTract):
                    self.log.warn(f"Did not find {repoInfo.skyWcsDataset} external calibrations for "
                                  f"all dataIds that do have {repoInfo.catDataset} catalogs.")

            # Create list of alias mappings for differing schema naming conventions (if any)
            aliasDictList = [self.config.flagsToAlias, ]
            if repoInfo.hscRun and self.config.srcSchemaMap is not None:
                aliasDictList += [self.config.srcSchemaMap]
            # Always highlight points with x-axis flag set (for cases where
            # they do not get explicitly filtered out).
            highlightList = [
                (self.config.analysis.fluxColumn.replace("_instFlux", "_flag"), 0, "turquoise"), ]

            # Dict of all parameters common to plot* functions
            plotInfoDict.update(dict(ccdList=ccdListPerTract, hscRun=repoInfo.hscRun,
                                     tractInfo=repoInfo.tractInfo, dataId=repoInfo.dataId))

            if any(doPlot for doPlot in
                   [self.config.doPlotPsfFluxSnHists, self.config.doPlotSkyObjects,
                    self.config.doPlotFootprintNpix, self.config.doPlotRhoStatistics,
                    self.config.doPlotQuiver, self.config.doPlotMags, self.config.doPlotStarGalaxy,
                    self.config.doPlotSizes, self.config.doPlotCentroids, self.config.doPlotRhoStatistics,
                    self.config.doWriteParquetTables]) and not self.config.plotMatchesOnly:
                if self.config.hasFakes:
                    inputFakes = repoInfo.butler.get("deepCoadd_fakeSourceCat", dataId=repoInfo.dataId)
                    inputFakes = inputFakes.toDataFrame()
                    datasetType = "fakes_src"
                else:
                    inputFakes = None
                    datasetType = "src"
                catStruct = self.readCatalogs(dataRefListTract, datasetType, repoInfo,
                                              aliasDictList=aliasDictList, fakeCat=inputFakes)
                commonZpCat = catStruct.commonZpCatalog
                catalog = catStruct.catalog
                areaDict = catStruct.areaDict
                # Make sub-catalog of sky sources before flag culling as many of
                # these will have flags set due to measurement difficulties in
                # regions that are really blank sky
                skySrcCat = None
                if self.config.doPlotSkyObjects:
                    if "sky_source" in catalog.schema:
                        skySrcCat = catalog[catalog["sky_source"]].copy(deep=True)
                    else:
                        self.log.warn("doPlotSkyObjects is True, but the \"sky_source\" "
                                      "column does not exist in catalog.schema.  Skipping "
                                      "skyObjects plot.")

                # Set boolean arrays indicating sources deemed unsuitable for qa analyses
                self.catLabel = "nChild = 0"
                bad = makeBadArray(catalog, flagList=self.config.analysis.flags,
                                   onlyReadStars=self.config.onlyReadStars)
                badCommonZp = makeBadArray(commonZpCat, flagList=self.config.analysis.flags,
                                           onlyReadStars=self.config.onlyReadStars)

                # Create and write parquet tables
                if self.config.doWriteParquetTables:
                    # Add pre-computed columns for parquet tables
                    catalog = addPreComputedColumns(catalog, fluxToPlotList=self.config.fluxToPlotList,
                                                    toMilli=self.config.toMilli)
                    commonZpCat = addPreComputedColumns(commonZpCat,
                                                        fluxToPlotList=self.config.fluxToPlotList,
                                                        toMilli=self.config.toMilli)
                    dataRef_catalog = repoInfo.butler.dataRef('analysisVisitTable', dataId=repoInfo.dataId)
                    writeParquet(dataRef_catalog, catalog, badArray=bad)
                    dataRef_commonZp = repoInfo.butler.dataRef('analysisVisitTable_commonZp',
                                                               dataId=repoInfo.dataId)
                    writeParquet(dataRef_commonZp, commonZpCat, badArray=badCommonZp)
                    if self.config.writeParquetOnly and not self.config.doPlotMatches:
                        self.log.info("Exiting after writing Parquet tables.  No plots generated.")
                        return

                # purge the catalogs of flagged sources
                catalog = catalog[~bad].copy(deep=True)
                commonZpCat = commonZpCat[~badCommonZp].copy(deep=True)

                if self.config.hasFakes:
                    processedFakes = catalog.asAstropy().to_pandas()
                    inputFakes = catStruct.fakeCat
                    inputFakes = addDegreePositions(inputFakes, self.config.inputFakesRaCol,
                                                    self.config.inputFakesDecCol)
                    processedFakes = addDegreePositions(processedFakes, self.config.catalogRaCol,
                                                        self.config.catalogDecCol)
                    processedFakes = addNearestNeighbor(processedFakes, self.config.catalogRaCol + "_deg",
                                                        self.config.catalogDecCol + "_deg")
                    inputFakesMatched, processedFakesMatched, inputFakes = matchCatalogs(
                        inputFakes, self.config.inputFakesRaCol + "_deg",
                        self.config.inputFakesDecCol + "_deg", processedFakes,
                        self.config.catalogRaCol + "_deg", self.config.catalogDecCol + "_deg")

                    areaDepthStruct = calcFakesAreaDepth(inputFakesMatched, processedFakesMatched, areaDict,
                                                         numSigmas=100.0)

                    plotList.append(plotFakesAreaDepth(inputFakesMatched, processedFakesMatched,
                                                       plotInfoDict, areaDict))
                    plotList.append(plotFakesAreaDepth(inputFakesMatched, processedFakesMatched,
                                                       plotInfoDict, areaDict, numSigmas=100.0))

                    plotInfoDict["magLim"] = areaDepthStruct.medMagsToLimit
                    plotList.append(fakesPositionCompare(inputFakesMatched, processedFakesMatched,
                                                         plotInfoDict))
                    plotList.append(fakesMagnitudeCompare(inputFakesMatched, processedFakesMatched,
                                                          plotInfoDict, verifyJob=self.verifyJob))
                    plotList.append(fakesMagnitudeCompare(inputFakesMatched, processedFakesMatched,
                                                          plotInfoDict, verifyJob=self.verifyJob,
                                                          magCol="base_CircularApertureFlux_12_0_mag"))
                    plotList.append(fakesMagnitudeNearestNeighbor(inputFakesMatched, processedFakesMatched,
                                                                  plotInfoDict))
                    plotList.append(fakesMagnitudeBlendedness(inputFakesMatched, processedFakesMatched,
                                                              plotInfoDict))
                    plotList.append(fakesCompletenessPlot(inputFakes, inputFakesMatched,
                                                          processedFakesMatched, plotInfoDict, areaDict))
                    plotList.append(fakesMagnitudePositionError(inputFakesMatched, processedFakesMatched,
                                                                plotInfoDict, areaDict))

                if self.config.doPlotSkyObjects and skySrcCat is not None:
                    plotList.append(self.plotSkyObjects(skySrcCat, "skySources", plotInfoDict, areaDict))
                if self.config.doPlotPsfFluxSnHists:
                    plotList.append(self.plotPsfFluxSnHists(commonZpCat, "base_PsfFlux_raw",
                                                            plotInfoDict, areaDict, zpLabel="raw"))
                    plotList.append(self.plotPsfFluxSnHists(catalog, "base_PsfFlux_cal",
                                                            plotInfoDict, areaDict, zpLabel=self.zpLabel))
                plotKwargs = dict(zpLabel=self.zpLabel)
                if self.config.doPlotFootprintNpix:
                    plotList.append(self.plotFootprintHist(catalog, "footNpix", plotInfoDict, **plotKwargs))
                    plotList.append(self.plotFootprint(catalog, plotInfoDict, areaDict, plotRunStats=False,
                                                       highlightList=[("parent", 0, "yellow"), ],
                                                       **plotKwargs))

                if self.config.doPlotQuiver:
                    plotList.append(self.plotQuiver(catalog, "ellipResids", plotInfoDict, areaDict, scale=2,
                                                    **plotKwargs))

                if self.config.doPlotRhoStatistics:
                    plotList.append(self.plotRhoStatistics(catalog, plotInfoDict,
                                                           **plotKwargs))

                plotKwargs.update(dict(highlightList=highlightList))
                # Create mag comparison plots using common ZP
                if self.config.doPlotMags and not commonZpDone:
                    zpLabel = "common (%s)" % self.config.analysis.commonZp
                    plotKwargs.update(dict(zpLabel=zpLabel))
                    plotList.append(self.plotMags(commonZpCat, plotInfoDict, areaDict,
                                                  fluxToPlotList=["base_GaussianFlux",
                                                                  "base_CircularApertureFlux_12_0"],
                                                  postFix="_commonZp", **plotKwargs))
                    commonZpDone = True
                # Now calibrate the source catalg to either the instrumental flux corresponding
                # to 0th magnitude, fluxMag0, from SFM or the external-calibration solution
                # (from jointcal, fgcm, or meas_mosaic) for remainder of plots.
                plotKwargs.update(dict(zpLabel=self.zpLabel))
                if self.config.doPlotMags:
                    plotList.append(self.plotMags(catalog, plotInfoDict, areaDict, **plotKwargs))
                if self.config.doPlotStarGalaxy:
                    if "ext_shapeHSM_HsmSourceMoments_xx" in catalog.schema:
                        plotList.append(self.plotStarGal(catalog, plotInfoDict, areaDict, **plotKwargs))
                    else:
                        self.log.warn("Cannot run plotStarGal: " +
                                      "ext_shapeHSM_HsmSourceMoments_xx not in catalog.schema")
                if self.config.doPlotSizes:
                    if "base_SdssShape_psf_xx" in catalog.schema:
                        plotList.append(self.plotSizes(catalog, plotInfoDict, areaDict, **plotKwargs))
                    else:
                        self.log.warn("Cannot run plotSizes: base_SdssShape_psf_xx not in catalog.schema")
                if self.config.doPlotCentroids and self.haveFpCoords:
                    plotList.append(self.plotCentroidXY(catalog, plotInfoDict, areaDict, **plotKwargs))

            matchAreaDict = {}
            if self.config.doPlotMatches:
                matches, matchAreaDict = self.readSrcMatches(dataRefListTract, "src", repoInfo,
                                                             aliasDictList=aliasDictList)
                if self.config.doWriteParquetTables:
                    matchesDataRef = repoInfo.butler.dataRef("analysisMatchFullRefVisitTable",
                                                             dataId=repoInfo.dataId)
                    writeParquet(matchesDataRef, matches, badArray=None, prefix="src_")
                if self.config.writeParquetOnly:
                    self.log.info("Exiting after writing Parquet tables.  No plots generated.")
                    return

                # Dict of all parameters common to plot* functions
                matchHighlightList = [("src_" + self.config.analysis.fluxColumn.replace("_instFlux", "_flag"),
                                       0, "turquoise"), ]
                plotList.append(self.plotMatches(matches, plotInfoDict, matchAreaDict, zpLabel=self.zpLabel,
                                highlightList=matchHighlightList))

            for cat in self.config.externalCatalogs:
                if self.config.photoCatName not in cat:
                    with andCatalog(cat):
                        matches = self.matchCatalog(catalog, plotInfoDict["filter"],
                                                    self.config.externalCatalogs[cat])
                        plotList.append(self.plotMatches(matches, plotInfoDict, matchAreaDict,
                                                         matchRadius=self.matchRadius,
                                                         matchRadiusUnitStr=self.matchRadiusUnitStr,
                                                         **plotKwargs))
        metaDict = {"tract": plotInfoDict["tract"], "visit": plotInfoDict["visit"],
                    "filter": plotInfoDict["filter"]}
        if plotInfoDict["cameraName"]:
            metaDict.update({"camera": plotInfoDict["cameraName"]})
        self.verifyJob = updateVerifyJob(self.verifyJob, metaDict=metaDict)
        verifyJobFilename = repoInfo.butler.get("visitAnalysis_verify_job_filename",
                                                dataId=repoInfo.dataId)[0]
        if plotList:
            savePlots(plotList, "plotVisit", repoInfo.dataId, repoInfo.butler, subdir=subdir)

        self.verifyJob.write(verifyJobFilename)

    def readCatalogs(self, dataRefList, dataset, repoInfo, aliasDictList=None, fakeCat=None,
                     raFakesCol="raJ2000", decFakesCol="decJ2000"):
        """Read in and concatenate catalogs of type dataset in lists of data references

        If self.config.doWriteParquetTables is True, before appending each catalog to a single
        list, an extra column indicating the ccd is added to the catalog.  This is useful for
        the subsequent interactive QA analysis.

        Also added to the catalog are columns with the focal plane coordinate (if not already
        present) and the number of pixels in the object's footprint.  Finally, the catalogs
        are calibrated according to the self.config.doApplyExternalPhotoCalib/SkyWcs config
        parameters:

        self.config.doApplyExternalPhotoCalib:
        - external photometric flux calibration (fgcmcal, jointcal, or meas_mosaic)
           if True
        - fluxMag0, the instrumental flux corresponding to 0th magnitude, from SFM
          if False

        self.config.doApplyExternalSkyWcs:
        - external astrometric calibration (jointcal or meas_mosaic) if True
        - no change to SFM astrometric calibration if False

        Parameters
        ----------
        dataRefList : `list` of `lsst.daf.persistence.butlerSubset.ButlerDataRef`
           A list of butler data references whose catalogs of dataset type are to be read in
        dataset : `str`
           Name of the catalog dataset to be read in
        repoInfo : `lsst.pipe.base.Struct`
           A struct containing relevant information about the repository under
           study.  Elements used here include the key name associated with a
           ccd and wether the processing was done with an HSC stack (now
           obsolete, but processing runs still exist)
        fakeCat : `pandas.core.frame.DataFrame`, optional
            Catalog of fake sources, used if config.hasFakes is `True` in which
            case a column (onCcd) is added with the ccd number if the fake
            source overlaps a ccd and np.nan if it does not.

        Raises
        ------
        `TaskError`
           If no data is read in for the dataRefList

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            A struct with attributes:
            ``commonZpCatalog``
                The concatenated common zeropoint calibrated catalog
                (`lsst.afw.table.SourceCatalog`s).
            ``catalog``
                The concatenated SFM or external calibration calibrated catalog
                (`lsst.afw.table.SourceCatalog`s).
            ``areaDict``
                Contains ccd keys that index the ccd corners in RA/Dec and
                the effective ccd area (i.e. neither the "BAD" nor "NO_DATA"
                mask bit is set) (`dict`).
            ``fakeCat``
                The updated catalog of fake sources or `None` if no ``fakeCat``
                provided (`pandas.core.frame.DataFrame`).
        """
        catList = []
        commonZpCatList = []
        areaDict = {}
        self.haveFpCoords = True
        for dataRef in dataRefList:
            if not dataRef.datasetExists(dataset):
                continue
            catalog = dataRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_HEAVY_FOOTPRINTS)
            # Set some aliases for differing schema naming conventions
            if aliasDictList:
                catalog = setAliasMaps(catalog, aliasDictList)

            # Add ccdId column (useful to have in Parquet tables for subsequent interactive analysis)
            catalog = addIntFloatOrStrColumn(catalog, dataRef.dataId[repoInfo.ccdKey], "ccdId",
                                             "Id of CCD on which source was detected")

            # getUri is less safe but enables us to use an efficient
            # ExposureFitsReader
            fname = repoInfo.butler.getUri("calexp", dataRef.dataId)
            reader = afwImage.ExposureFitsReader(fname)
            detector = reader.readDetector()
            if self.config.doApplyExternalSkyWcs:
                wcs = dataRef.get(repoInfo.skyWcsDataset)
            else:
                wcs = reader.readWcs()
            ccdCorners = wcs.pixelToSky(detector.getCorners(cameraGeom.PIXELS))
            areaDict["corners_" + str(dataRef.dataId[repoInfo.ccdKey])] = ccdCorners

            # Actual mask is needed because BAD pixels are more than
            # just the defects
            mask = reader.readMask()
            maskBad = mask.array & 2**mask.getMaskPlaneDict()["BAD"]
            maskNoData = mask.array & 2**mask.getMaskPlaneDict()["NO_DATA"]
            maskedPixels = maskBad + maskNoData
            numGoodPix = np.count_nonzero(maskedPixels == 0)
            pixScale = wcs.getPixelScale(detector.getCenter(cameraGeom.PIXELS)).asArcseconds()
            area = numGoodPix*pixScale**2
            areaDict[dataRef.dataId["ccd"]] = area

            if self.config.hasFakes:
                # Check which fake sources fall on the ccd
                cornerRas = [cx.asRadians() for (cx, cy) in ccdCorners]
                cornerDecs = [cy.asRadians() for (cx, cy) in ccdCorners]
                possOnCcd = np.where((fakeCat[raFakesCol].values > np.min(cornerRas)) &
                                     (fakeCat[raFakesCol].values < np.max(cornerRas)) &
                                     (fakeCat[decFakesCol].values > np.min(cornerDecs)) &
                                     (fakeCat[decFakesCol].values < np.max(cornerDecs)))[0]

                if "onCcd" not in fakeCat.columns:
                    fakeCat["onCcd"] = [np.nan]*len(fakeCat)

                validCcdPolygon = reader.readExposureInfo().getValidPolygon()
                onCcdList = []
                for rowId in possOnCcd:
                    skyCoord = geom.SpherePoint(fakeCat[raFakesCol].values[rowId],
                                                fakeCat[decFakesCol].values[rowId], geom.radians)
                    pixCoord = wcs.skyToPixel(skyCoord)
                    onCcd = validCcdPolygon.contains(pixCoord)
                    if onCcd:
                        onCcdList.append(rowId)
                fakeCat["onCcd"].iloc[np.array(onCcdList)] = dataRef.dataId[repoInfo.ccdKey]

            if self.config.doPlotCentroids or self.config.analysis.doPlotFP and self.haveFpCoords:
                # Compute Focal Plane coordinates for each source if not already there
                if "base_FPPosition_x" not in catalog.schema and "focalplane_x" not in catalog.schema:
                    det = repoInfo.butler.get("calexp_detector", dataRef.dataId)
                    catalog = addFpPoint(det, catalog)
                xFp = catalog["base_FPPosition_x"]
                if len(xFp[np.where(np.isfinite(xFp))]) <= 0:
                    self.haveFpCoords = False
            if self.config.doPlotFootprintNpix:
                catalog = addFootprintNPix(catalog)
            if repoInfo.hscRun and self.config.doAddAperFluxHsc:
                self.log.info("HSC run: adding aperture flux to schema...")
                catalog = addApertureFluxesHSC(catalog, prefix="")
            # Optionally backout aperture corrections
            if self.config.doBackoutApCorr:
                catalog = backoutApCorr(catalog)

            # Scale fluxes to common zeropoint to make basic comparison plots without calibrated ZP influence
            commonZpCat = catalog.copy(True)
            commonZpCat = calibrateSourceCatalog(commonZpCat, self.config.analysis.commonZp)
            commonZpCatList.append(commonZpCat)
            if self.config.doApplyExternalPhotoCalib:
                if repoInfo.hscRun:
                    if not dataRef.datasetExists("fcr_hsc_md") or not dataRef.datasetExists("wcs_hsc"):
                        continue
                else:
                    # Check for both jointcal_wcs and wcs for compatibility with old datasets
                    if not (dataRef.datasetExists(repoInfo.photoCalibDataset)
                            or dataRef.datasetExists("fcr_md")):
                        continue
            if self.config.doApplyExternalSkyWcs:
                if repoInfo.hscRun:
                    if not dataRef.datasetExists("fcr_hsc_md") or not dataRef.datasetExists("wcs_hsc"):
                        continue
                else:
                    # Check for both jointcal_wcs and wcs for compatibility with old datasets
                    if not (dataRef.datasetExists(repoInfo.skyWcsDataset)
                            or dataRef.datasetExists("wcs")):
                        continue
            fluxMag0 = None
            if not self.config.doApplyExternalPhotoCalib:
                photoCalib = repoInfo.butler.get("calexp_photoCalib", dataRef.dataId)
                fluxMag0 = photoCalib.getInstFluxAtZeroMagnitude()
            catalog = self.calibrateCatalogs(dataRef, catalog, fluxMag0, repoInfo)
            catList.append(catalog)

        if not catList:
            raise TaskError("No catalogs read: %s" % ([dataRef.dataId for dataRef in dataRefList]))

        return Struct(commonZpCatalog=concatenateCatalogs(commonZpCatList),
                      catalog=concatenateCatalogs(catList), areaDict=areaDict, fakeCat=fakeCat)

    def readSrcMatches(self, dataRefList, dataset, repoInfo, aliasDictList=None):
        catList = []
        dataIdSubList = []
        matchAreaDict = {}
        for dataRef in dataRefList:
            if not dataRef.datasetExists(dataset):
                continue
            if self.config.doApplyExternalPhotoCalib:
                if repoInfo.hscRun:
                    if not dataRef.datasetExists("fcr_hsc_md") or not dataRef.datasetExists("wcs_hsc"):
                        continue
                else:
                    # Check for both jointcal_wcs and wcs for compatibility with old datasets
                    if (not (dataRef.datasetExists(repoInfo.photoCalibDataset)
                             or dataRef.datasetExists("fcr_md"))):
                        continue
            if self.config.doApplyExternalSkyWcs:
                if repoInfo.hscRun:
                    if not dataRef.datasetExists("fcr_hsc_md") or not dataRef.datasetExists("wcs_hsc"):
                        continue
                else:
                    # Check for both jointcal_wcs and wcs for compatibility with old datasets
                    if not (dataRef.datasetExists(repoInfo.skyWcsDataset)
                            or dataRef.datasetExists("wcs")):
                        continue
            # Generate unnormalized match list (from normalized persisted one) with joinMatchListWithCatalog
            # (which requires a refObjLoader to be initialized).
            catalog = dataRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS)
            catalog = addIntFloatOrStrColumn(catalog, dataRef.dataId[repoInfo.ccdKey], "ccdId",
                                             "Id of CCD on which source was detected")
            # Set some aliases for differing schema naming conventions
            if aliasDictList:
                catalog = setAliasMaps(catalog, aliasDictList)
            fluxMag0 = None
            if not self.config.doApplyExternalPhotoCalib:
                photoCalib = repoInfo.butler.get("calexp_photoCalib", dataRef.dataId)
                fluxMag0 = photoCalib.getInstFluxAtZeroMagnitude()
            catalog = self.calibrateCatalogs(dataRef, catalog, fluxMag0, repoInfo)
            # getUri is less safe but enables us to use an efficient
            # ExposureFitsReader
            fname = repoInfo.butler.getUri("calexp", dataRef.dataId)
            reader = afwImage.ExposureFitsReader(fname)
            detector = reader.readDetector()
            if self.config.doApplyExternalSkyWcs:
                wcs = dataRef.get(repoInfo.skyWcsDataset)
            else:
                wcs = reader.readWcs()
            ccdCorners = wcs.pixelToSky(detector.getCorners(cameraGeom.PIXELS))
            matchAreaDict["corners_" + str(dataRef.dataId[repoInfo.ccdKey])] = ccdCorners

            packedMatches = repoInfo.butler.get(dataset + "Match", dataRef.dataId)
            # The reference object loader grows the bbox by the config parameter pixelMargin.  This
            # is set to 50 by default but is not reflected by the radius parameter set in the
            # metadata, so some matches may reside outside the circle searched within this radius
            # Thus, increase the radius set in the metadata fed into joinMatchListWithCatalog() to
            # accommodate.
            matchmeta = packedMatches.table.getMetadata()
            rad = matchmeta.getDouble("RADIUS")
            matchmeta.setDouble("RADIUS", rad*1.05, "field radius in degrees, approximate, padded")
            refObjLoader = self.config.refObjLoader.apply(butler=repoInfo.butler)
            matches = refObjLoader.joinMatchListWithCatalog(packedMatches, catalog)
            if not hasattr(matches[0].first, "schema"):
                raise RuntimeError("Unable to unpack matches.  "
                                   "Do you have the correct astrometry_net_data setup?")
            noMatches = False
            if len(matches) < 8:
                for m in matches:
                    if not hasattr(m.first, "get"):
                        matches = []
                        noMatches = True
                        break

            # LSST reads in a_net catalogs with flux in "nanojanskys", so must convert to AB
            if not noMatches:
                matches = matchNanojanskyToAB(matches)
                if repoInfo.hscRun and self.config.doAddAperFluxHsc:
                    addApertureFluxesHSC(matches, prefix="second_")

            if not matches:
                self.log.warn("No matches for {:s}".format(dataRef.dataId))
                continue

            matchMeta = repoInfo.butler.get(dataset, dataRef.dataId,
                                            flags=afwTable.SOURCE_IO_NO_FOOTPRINTS).getTable().getMetadata()
            catalog = matchesToCatalog(matches, matchMeta)
            if self.config.doApplyExternalSkyWcs:
                # Update "distance" between reference and source matches based
                # on external-calibration positions.
                angularDist = AngularDistance("ref_coord_ra", "src_coord_ra",
                                              "ref_coord_dec", "src_coord_dec")
                catalog["distance"] = angularDist(catalog)

            # Compute Focal Plane coordinates for each source if not already there
            if self.config.analysisMatches.doPlotFP:
                if "src_base_FPPosition_x" not in catalog.schema and "src_focalplane_x" not in catalog.schema:
                    det = repoInfo.butler.get("calexp_detector", dataRef.dataId)
                    catalog = addFpPoint(det, catalog, prefix="src_")
            # Optionally backout aperture corrections
            if self.config.doBackoutApCorr:
                catalog = backoutApCorr(catalog)
            # Need to set the alias map for the matched catalog sources
            if aliasDictList:
                catalog = setAliasMaps(catalog, aliasDictList, prefix="src_")
            # To avoid multiple counting when visit overlaps multiple tracts
            noTractId = dataRef.dataId.copy()
            noTractId.pop("tract")
            if noTractId not in dataIdSubList:
                catList.append(catalog)
            dataIdSubList.append(noTractId)

        if not catList:
            raise TaskError("No matches read: %s" % ([dataRef.dataId for dataRef in dataRefList]))

        return concatenateCatalogs(catList), matchAreaDict

    def calibrateCatalogs(self, dataRef, catalog, fluxMag0, repoInfo):
        """Determine and apply appropriate flux calibration to the catalog.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.butlerSubset.ButlerDataRef`
           If applying meas_mosaic calibrations, a dataRef is needed for call to
           meas_mosaic's applyMosaicResultsCatalog() in in utils'
           calibrateSourceCatalogMosaic().  It is also needed to distinguish
           between jointcal vs. meas_mosaic when calibrating through
           a `lsst.afw.image.PhotoCalib` object.
        catalog : `lsst.afw.table.source.source.SourceCatalog`
           The catalog to which the calibration is applied in place.
        fluxMag0 : `float`
           The instrumental flux corresponding to 0 magnitude from Single Frame
           Measurement for the catalog.
        repoInfo : `lsst.pipe.base.Struct`
           A struct containing relevant information about the repository under
           study.  Elements used here include the dataset names for any external
           calibrations to be applied.
        """
        self.zp = 0.0
        try:
            self.zpLabel = self.zpLabel
        except Exception:
            self.zpLabel = None

        if self.config.doApplyExternalPhotoCalib:
            if not self.config.useMeasMosaic:
                # i.e. the processing was post-photoCalib output generation
                # AND you want the photoCalib flux object used for the
                # calibration (as opposed to meas_mosaic's fcr object).
                if not self.zpLabel:
                    zpStr = ("MMphotoCalib" if dataRef.datasetExists("fcr_md")
                             else self.config.externalPhotoCalibName.upper())
                    self.log.info(f"Applying {zpStr} photoCalib calibration to catalog")
                    self.zpLabel = zpStr
                calibrated = calibrateSourceCatalogPhotoCalib(dataRef, catalog, repoInfo.photoCalibDataset,
                                                              zp=self.zp)
            else:
                # If here, the data were processed pre-photoCalib output
                # generation, so must use old method OR old method was
                # explicitly requested via useMeasMosaic.
                try:
                    import lsst.meas.mosaic  # noqa : F401
                except ImportError:
                    raise ValueError("Cannot apply calibrations because meas_mosaic could not "
                                     "be imported. \nEither setup meas_mosaic or run with "
                                     "--config doApplyExternalPhotoCalib=False doApplyExternalSkyWcs=False")
                if not self.zpLabel:
                    self.log.info("Applying meas_mosaic calibration to catalog")
                self.zpLabel = "MEAS_MOSAIC"
                calibrated = calibrateSourceCatalogMosaic(dataRef, catalog, zp=self.zp)
        else:
            # Scale fluxes to measured zeropoint
            self.zp = 2.5*np.log10(fluxMag0)
            if not self.zpLabel:
                self.log.info("Using 2.5*log10(fluxMag0) = {:.4f} from SFM for zeropoint".format(self.zp))
            self.zpLabel = "FLUXMAG0"
            calibrated = calibrateSourceCatalog(catalog, self.zp)

        if self.config.doApplyExternalSkyWcs:
            wcs = dataRef.get(repoInfo.skyWcsDataset)
            for record in calibrated:
                record.updateCoord(wcs)
            if "wcs" not in self.zpLabel:
                self.zpLabel += "\nwcs: " + self.config.externalSkyWcsName.upper()

        return calibrated


class CompareVisitAnalysisConfig(VisitAnalysisConfig):
    doApplyExternalPhotoCalib1 = Field(dtype=bool, default=True,
                                       doc=("Whether to apply external photometric calibration (e.g. "
                                            "fgcmcal/jointcal/meas_mosaic) via an "
                                            "`lsst.afw.image.PhotoCalib` object to input1.  If `True`, "
                                            "uses ``externalPhotoCalibName1`` field to determine which "
                                            "calibration to load.  If `False`, the instrumental flux "
                                            "corresponding to 0th magnitude, fluxMag0, from SFM is "
                                            "applied."))
    externalPhotoCalibName1 = ChoiceField(dtype=str, default="jointcal",
                                          allowed={"jointcal": "Use jointcal_photoCalib",
                                                   "fgcm": "Use fgcm_photoCalib",
                                                   "fgcm_tract": "Use fgcm_tract_photoCalib"},
                                          doc=("Type of external `lsst.afw.image.PhotoCalib` to apply to "
                                               "input1 if ``doApplyExternalPhotoCalib1`` is `True`."))
    doApplyExternalSkyWcs1 = Field(dtype=bool, default=True,
                                   doc=("Whether to apply external astrometric calibration via an "
                                        "`lsst.afw.geom.SkyWcs` object to input1.  Uses "
                                        "``externalSkyWcsName1`` field to determine which calibration "
                                        "to load."))
    externalSkyWcsName1 = ChoiceField(dtype=str, default="jointcal",
                                      allowed={"jointcal": "Use jointcal_wcs"},
                                      doc=("Type of external `lsst.afw.geom.SkyWcs` to apply to input1 if "
                                           "``doApplyExternalSkyWcs1`` is ``True``."))
    doApplyExternalPhotoCalib2 = Field(dtype=bool, default=True,
                                       doc=("Whether to apply external photometric calibration (e.g. "
                                            "fgcmcal/jointcal/meas_mosaic) via an "
                                            "`lsst.afw.image.PhotoCalib` object to input2.  If `True`, "
                                            "uses ``externalPhotoCalibName2`` field to determine which "
                                            "calibration to load.  If `False`, the instrumental flux "
                                            "corresponding to 0th magnitude, fluxMag0, from SFM is "
                                            "applied."))
    externalPhotoCalibName2 = ChoiceField(dtype=str, default="jointcal",
                                          allowed={"jointcal": "Use jointcal_photoCalib",
                                                   "fgcm": "Use fgcm_photoCalib",
                                                   "fgcm_tract": "Use fgcm_tract_photoCalib"},
                                          doc=("Type of external `lsst.afw.image.PhotoCalib` to apply to "
                                               "input2 if ``doApplyExternalPhotoCalib2`` is `True`."))
    doApplyExternalSkyWcs2 = Field(dtype=bool, default=True,
                                   doc=("Whether to apply external astrometric calibration via an "
                                        "`lsst.afw.geom.SkyWcs` object to input2.  Uses "
                                        "``externalSkyWcsName2`` field to determine which calibration "
                                        "to load."))
    externalSkyWcsName2 = ChoiceField(dtype=str, default="jointcal",
                                      allowed={"jointcal": "Use jointcal_wcs"},
                                      doc=("Type of external `lsst.afw.geom.SkyWcs` to apply to input2 if "
                                           "``doApplyExternalSkyWcs2`` is `True`."))
    useMeasMosaic1 = Field(dtype=bool, default=False, doc="Use meas_mosaic's applyMosaicResultsExposure " +
                           "to apply meas_mosaic calibration results to input1 (i.e. as opposed to using " +
                           "the photoCalib object)?")
    useMeasMosaic2 = Field(dtype=bool, default=False, doc="Use meas_mosaic's applyMosaicResultsExposure " +
                           "to apply meas_mosaic calibration results to input2 (i.e. as opposed to using " +
                           "the photoCalib object)?")

    def setDefaults(self):
        VisitAnalysisConfig.setDefaults(self)
        # If matching on Ra/Dec, use a tighter match radius for comparing runs:
        # they are calibrated and we want to avoid mis-matches
        self.matchRadiusRaDec = 0.2
        self.matchRadiusXy = 1.0e-5  # has to be bigger than absolute zero
        if "base_PsfFlux" not in self.fluxToPlotList:
            self.fluxToPlotList.append("base_PsfFlux")  # Add PSF flux to default list for comparison scripts

    def validate(self):
        super(CoaddAnalysisConfig, self).validate()
        if not self.doApplyExternalPhotoCalib and (self.doApplyExternalPhotoCalib1
                                                   and self.doApplyExternalPhotoCalib2):
            raise ValueError("doApplyExternalPhotoCalib is set to False, but doApplyExternalPhotoCalib1 and "
                             "doApplyExternalPhotoCalib2, the appropriate settings for the "
                             "compareVisitAnalysis.py scirpt, are both True, so external calibrations would "
                             "be applied.  Try running without setting doApplyExternalPhotoCalib (which is "
                             "only appropriate for the visitAnalysis.py script).")
        if (not self.doApplyExternalSkyWcs
                and (self.doApplyExternalSkyWcs1 and self.doApplyExternalSkyWcs2)):
            raise ValueError("doApplyExternalSkyWcs is set to False, but doApplyExternalSkyWcs1 and "
                             "doApplyExternalSkyWcs2, the appropriate settings for the "
                             "compareVisitAnalysis.py scirpt, are both True, so external calibrations would "
                             "be applied.  Try running without setting doApplyExternalSkyWcs (which is "
                             "only appropriate for the visitAnalysis.py script).")
        if not self.useMeasMosaic and (self.useMeasMosaic1 and self.useMeasMosaic2):
            raise ValueError("useMeasMosaic is set to False, but useMeasMosaic1 and useMeasMosaic2, "
                             "the appropriate settings for the compareVisitAnalysis.py scirpt, are both "
                             "True, so PhotoCalib-derived calibrations would be applied.  Try running "
                             "without setting useMeasMosaic (which is only appropriate for the "
                             "visitAnalysis.py script).")


class CompareVisitAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        rootDir = parsedCmd.input.split("rerun")[0] if len(parsedCmd.rerun) == 2 else parsedCmd.input
        kwargs["tract"] = parsedCmd.tract
        kwargs["subdir"] = parsedCmd.subdir
        # New butler requires identical RepositoryArgs and RepositoryCfg and mapperArgs={} is NOT
        # considered equivalent to mapperArgs={'calibRoot': None}, so only use if pasedCmd.calib
        # is not None
        butlerArgs = dict(root=os.path.join(rootDir, "rerun", parsedCmd.rerun2))
        if parsedCmd.calib is not None:
            butlerArgs["calibRoot"] = parsedCmd.calib
        butler2 = Butler(**butlerArgs)
        idParser = parsedCmd.id.__class__(parsedCmd.id.level)
        idParser.idList = parsedCmd.id.idList
        idParser.datasetType = parsedCmd.id.datasetType
        butler = parsedCmd.butler
        parsedCmd.butler = butler2
        idParser.makeDataRefList(parsedCmd)
        parsedCmd.butler = butler

        visits1 = defaultdict(list)
        visits2 = defaultdict(list)
        for ref1, ref2 in zip(parsedCmd.id.refList, idParser.refList):
            visits1[ref1.dataId["visit"]].append(ref1)
            visits2[ref2.dataId["visit"]].append(ref2)
        return [(refs1, dict(dataRefList2=refs2, **kwargs)) for
                refs1, refs2 in zip(visits1.values(), visits2.values())]


class CompareVisitAnalysisTask(CompareCoaddAnalysisTask):
    ConfigClass = CompareVisitAnalysisConfig
    RunnerClass = CompareVisitAnalysisRunner
    _DefaultName = "compareVisitAnalysis"

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_argument("--rerun2", required=True, help="Second rerun, for comparison")
        parser.add_id_argument("--id", "src", help="data ID with raw CCD keys, "
                               "e.g. --id visit=12345 ccd=6^8..11", ContainerClass=PerTractCcdDataIdContainer)
        parser.add_argument("--tract", type=str, default=None,
                            help="Tract(s) to use (do one at a time for overlapping) e.g. 1^5^0")
        parser.add_argument("--subdir", type=str, default="",
                            help=("Subdirectory below plots/filter/tract-NNNN/visit-NNNN (useful "
                                  "for, e.g., subgrouping of CCDs.  Ignored if only one CCD is "
                                  "specified, in which case the subdir is set to ccd-NNN"))
        return parser

    def runDataRef(self, dataRefList1, dataRefList2, tract=None, subdir=""):
        # This is for the commonZP plots (i.e. all ccds regardless of tract)
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
        commonZpDone = False

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
            if self.config.doApplyExternalSkyWcs1:
                # Check for wcs for compatibility with old dataset naming
                ccdSkyWcsListPerTract1 = getDataExistsRefList(dataRefListTract1, repoInfo1.skyWcsDataset)
                if not ccdSkyWcsListPerTract1:
                    ccdSkyWcsListPerTract1 = getDataExistsRefList(dataRefListTract1, "wcs")
                    if ccdSkyWcsListPerTract1:
                        repoInfo1.skyWcsDataset = "wcs"
                        self.log.info("Old meas_mosaic dataset naming: wcs (new name is jointcal_wcs)")
                    else:
                        self.log.fatal(f"No data found for {repoInfo1.wcsSkyDataset} dataset...are you "
                                       "sure you ran the external astrometric calibration?  If not, run "
                                       "with --config doApplyExternalSkyWcs1=False")
            if self.config.doApplyExternalSkyWcs2:
                # Check for wcs for compatibility with old dataset naming
                ccdSkyWcsListPerTract2 = getDataExistsRefList(dataRefListTract2, repoInfo2.skyWcsDataset)
                if not ccdSkyWcsListPerTract2:
                    ccdSkyWcsListPerTract2 = getDataExistsRefList(dataRefListTract2, "wcs")
                    if ccdSkyWcsListPerTract2:
                        repoInfo2.skyWcsDataset = "wcs"
                        self.log.info("Old meas_mosaic dataset naming: wcs (new name is jointcal_wcs)")
                    else:
                        self.log.fatal(f"No data found for {repoInfo2.wcsSkyDataset} dataset...are you "
                                       "sure you ran the external astrometric calibration?  If not, run "
                                       "with --config doApplyExternalSkyWcs2=False")

            ccdIntersectList = list(set(ccdListPerTract1).intersection(set(ccdListPerTract2)))
            self.log.info("tract: {:d}".format(repoInfo1.dataId["tract"]))
            self.log.info(f"ccdListPerTract1: \n{ccdListPerTract1}")
            self.log.info(f"ccdListPerTract2: \n{ccdListPerTract2}")
            self.log.info(f"ccdIntersectList: \n{ccdIntersectList}")
            if self.config.doApplyExternalPhotoCalib1:
                self.log.info(f"ccdPhotoCalibListPerTract1: \n{ccdPhotoCalibListPerTract1}")
            if self.config.doApplyExternalPhotoCalib2:
                self.log.info(f"ccdPhotoCalibListPerTract2: \n{ccdPhotoCalibListPerTract2}")
            if self.config.doApplyExternalSkyWcs1:
                self.log.info(f"ccdSkyWcsListPerTract1: \n{ccdSkyWcsListPerTract1}")
            if self.config.doApplyExternalSkyWcs2:
                self.log.info(f"ccdSkyWcsListPerTract2: \n{ccdSkyWcsListPerTract2}")

            doReadFootprints = None
            if self.config.doPlotFootprintNpix:
                doReadFootprints = "light"
            # Set some aliases for differing schema naming conventions
            aliasDictList = [self.config.flagsToAlias, ]
            if (repoInfo1.hscRun or repoInfo2.hscRun) and self.config.srcSchemaMap is not None:
                aliasDictList += [self.config.srcSchemaMap]
            commonZpCat1, catalog1, areaDict1, commonZpCat2, catalog2, areaDict2 = (
                self.readCatalogs(dataRefListTract1, dataRefListTract2, "src", repoInfo1, repoInfo2,
                                  doReadFootprints=doReadFootprints, aliasDictList=aliasDictList))

            # Set boolean arrays indicating sources deemed unsuitable for qa analyses
            self.catLabel = "nChild = 0"
            bad1 = makeBadArray(catalog1, flagList=self.config.analysis.flags,
                                onlyReadStars=self.config.onlyReadStars)
            bad2 = makeBadArray(catalog2, flagList=self.config.analysis.flags,
                                onlyReadStars=self.config.onlyReadStars)
            badCommonZp1 = makeBadArray(commonZpCat1, flagList=self.config.analysis.flags,
                                        onlyReadStars=self.config.onlyReadStars)
            badCommonZp2 = makeBadArray(commonZpCat2, flagList=self.config.analysis.flags,
                                        onlyReadStars=self.config.onlyReadStars)

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
            hscRun = repoInfo1.hscRun if repoInfo1.hscRun else repoInfo2.hscRun

            # Dict of all parameters common to plot* functions
            tractInfo1 = repoInfo1.tractInfo if self.config.doApplyExternalPhotoCalib1 else None
            # Always highlight points with x-axis flag set (for cases where
            # they do not get explicitly filtered out).
            highlightList = [(self.config.analysis.fluxColumn.replace("_instFlux", "_flag"), 0,
                              "turquoise"), ]
            plotKwargs1 = dict(matchRadius=self.matchRadius, matchRadiusUnitStr=self.matchRadiusUnitStr,
                               zpLabel=self.zpLabel, highlightList=highlightList)

            plotInfoDict = getPlotInfo(repoInfo1)
            plotInfoDict.update({"ccdList": ccdIntersectList, "allCcdList": fullCameraCcdList1,
                                 "plotType": "plotCompareVisit", "subdir": subdir,
                                 "hscRun1": repoInfo1.hscRun, "hscRun2": repoInfo2.hscRun,
                                 "hscRun": hscRun, "tractInfo": tractInfo1, "dataId": repoInfo1.dataId})
            plotList = []
            if self.config.doPlotFootprintNpix:
                plotList.append(self.plotFootprint(catalog, plotInfoDict, areaDict1, **plotKwargs1))

            # Create mag comparison plots using common ZP
            if not commonZpDone:
                zpLabel = "common (" + str(self.config.analysis.commonZp) + ")"
                try:
                    zpLabel = zpLabel + " " + self.catLabel
                except Exception:
                    pass
                plotKwargs1.update(dict(zpLabel=zpLabel))
                plotList.append(self.plotMags(commonZpCat, plotInfoDict, areaDict1,
                                              fluxToPlotList=["base_GaussianFlux",
                                                              "base_CircularApertureFlux_12_0"],
                                              postFix="_commonZp", **plotKwargs1))
                commonZpDone = True

            plotKwargs1.update(dict(zpLabel=self.zpLabel))
            if self.config.doPlotMags:
                plotList.append(self.plotMags(catalog, plotInfoDict, areaDict1, **plotKwargs1))
            if self.config.doPlotSizes:
                if ("first_base_SdssShape_psf_xx" in catalog.schema and
                        "second_base_SdssShape_psf_xx" in catalog.schema):
                    plotList.append(self.plotSizes(catalog, plotInfoDict, areaDict1, **plotKwargs1))
                else:
                    self.log.warn("Cannot run plotSizes: base_SdssShape_psf_xx not in catalog.schema")
            if self.config.doApCorrs:
                plotList.append(self.plotApCorrs(catalog, plotInfoDict, areaDict1, **plotKwargs1))
            if self.config.doPlotCentroids:
                plotList.append(self.plotCentroids(catalog, plotInfoDict, areaDict1, **plotKwargs1))
            if self.config.doPlotStarGalaxy:
                plotList.append(self.plotStarGal(catalog, plotInfoDict, areaDict1, **plotKwargs1))

            self.allStats, self.allStatsHigh = savePlots(plotList, "plotCompareVisit", repoInfo1.dataId,
                                                         repoInfo1.butler, subdir=subdir)

    def readCatalogs(self, dataRefList1, dataRefList2, dataset, repoInfo1, repoInfo2,
                     doReadFootprints=None, aliasDictList=None):
        """Read in and concatenate catalogs of type dataset in lists of data references

        Parameters
        ----------
        dataRefList1 : `list` of `lsst.daf.persistence.butlerSubset.ButlerDataRef`
           A list of butler data references whose catalogs of dataset type are to be read in
        dataRefList2 : `list` of `lsst.daf.persistence.butlerSubset.ButlerDataRef`
           A second list of butler data references whose catalogs of dataset type are to be read in and
           compared against the catalogs associated with dataRefList1
        dataset : `str`
           Name of the catalog dataset to be read in
        repoInfo1, repoInfo2 : `lsst.pipe.base.Struct`
           A struct containing relevant information about the repository under
           study.  Elements used here include the butler associated with the
           repository, the image metadata, and wether the processing was done
           with an HSC stack (now obsolete, but processing runs still exist)
        doReadFootprints : `NoneType` or `str`, optional
           A string dictating if and what type of Footprint to read in along with the catalog
           None (the default): do not read in Footprints
           light: read in regular Footprints (include SpanSet and list of peaks per Footprint)
           heavy: read in HeavyFootprints (include regular Footprint plus flux values per Footprint)

        Raises
        ------
        `TaskError`
           If no data is read in for either dataRefList

        Returns
        -------
        `list` of 4 concatenated `lsst.afw.table.source.source.SourceCatalog`
           The concatenated catalogs returned are (common ZP-calibrated of dataRefList1,
           sfm or external-calibrated of dataRefList1, common ZP-calibrated of dataRefList2,
           sfm or external-calibrated of dataRefList2)

        """
        catList1 = []
        commonZpCatList1 = []
        catList2 = []
        commonZpCatList2 = []
        self.zpLabel1 = None
        self.zpLabel2 = None
        areaDict1 = {}
        areaDict2 = {}

        info1 = [catList1, commonZpCatList1, dataRefList1, repoInfo1, self.config.doApplyExternalPhotoCalib1,
                 self.config.doApplyExternalSkyWcs1, self.config.useMeasMosaic1, areaDict1]
        info2 = [catList2, commonZpCatList2, dataRefList2, repoInfo2, self.config.doApplyExternalPhotoCalib2,
                 self.config.doApplyExternalSkyWcs2, self.config.useMeasMosaic2, areaDict2]

        for (iCat, catInfoList) in enumerate([info1, info2]):
            [catList, commonZpCatList, dataRefList, repoInfo, doApplyExternalPhotoCalib,
                doApplyExternalSkyWcs, useMeasMosaic, areaDict] = catInfoList

            for dataRef in dataRefList:
                if not dataRef.datasetExists(dataset):
                    continue
                if not doReadFootprints:
                    srcCat = dataRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS)
                elif doReadFootprints == "light":
                    srcCat = dataRef.get(dataset, immediate=True,
                                         flags=afwTable.SOURCE_IO_NO_HEAVY_FOOTPRINTS)
                elif doReadFootprints == "heavy":
                    srcCat = dataRef.get(dataset, immediate=True)

                # Set some aliases for differing src naming conventions
                if aliasDictList:
                    srcCat = setAliasMaps(srcCat, aliasDictList)

                if self.config.doBackoutApCorr:
                    srcCat = backoutApCorr(srcCat)

                fluxMag0 = None
                if not doApplyExternalPhotoCalib:
                    photoCalib = repoInfo.butler.get("calexp_photoCalib", dataRef.dataId)
                    fluxMag0 = photoCalib.getInstFluxAtZeroMagnitude()
                det = repoInfo.butler.get("calexp_detector", dataRef.dataId)
                nQuarter = det.getOrientation().getNQuarter()

                # getUri is less safe but enables us to use an efficient
                # ExposureFitsReader
                fname = repoInfo.butler.getUri("calexp", dataRef.dataId)
                reader = afwImage.ExposureFitsReader(fname)
                detector = reader.readDetector()
                if self.config.doApplyExternalSkyWcs:
                    wcs = dataRef.get(repoInfo.skyWcsDataset)
                else:
                    wcs = reader.readWcs()
                ccdCorners = wcs.pixelToSky(detector.getCorners(cameraGeom.PIXELS))
                areaDict["corners_" + str(dataRef.dataId["ccd"])] = ccdCorners

                # add footprint nPix column
                if self.config.doPlotFootprintNpix:
                    srcCat = addFootprintNPix(srcCat)
                # Add rotated point in LSST cat if comparing with HSC cat to compare centroid pixel positions
                if repoInfo.hscRun and not (repoInfo1.hscRun and repoInfo2.hscRun):
                    bbox = repoInfo.butler.get("calexp_bbox", dataRef.dataId)
                    srcCat = addRotPoint(srcCat, bbox.getWidth(), bbox.getHeight(), nQuarter)

                if repoInfo.hscRun and self.config.doAddAperFluxHsc:
                    self.log.info("HSC run: adding aperture flux to schema...")
                    srcCat = addApertureFluxesHSC(srcCat, prefix="")

                # Scale fluxes to common zeropoint to make basic comparison plots without calibration
                # influence
                commonZpCat = srcCat.copy(True)
                commonZpCat = calibrateSourceCatalog(commonZpCat, self.config.analysis.commonZp)
                commonZpCatList.append(commonZpCat)

                if self.config.doApplyExternalPhotoCalib:
                    if repoInfo.hscRun:
                        if not dataRef.datasetExists("fcr_hsc_md") or not dataRef.datasetExists("wcs_hsc"):
                            continue
                    else:
                        # Check for both jointcal_wcs and wcs for compatibility with old datasets
                        if (not (dataRef.datasetExists(repoInfo.photoCalibDataset)
                                 or dataRef.datasetExists("fcr_md"))):
                            continue
                if self.config.doApplyExternalSkyWcs:
                    if repoInfo.hscRun:
                        if not dataRef.datasetExists("fcr_hsc_md") or not dataRef.datasetExists("wcs_hsc"):
                            continue
                    else:
                        # Check for both jointcal_wcs and wcs for compatibility with old datasets
                        if not (dataRef.datasetExists(repoInfo.skyWcsDataset)
                                or dataRef.datasetExists("wcs")):
                            continue
                srcCat, zpLabel = self.calibrateCatalogs(dataRef, srcCat, fluxMag0, repoInfo,
                                                         doApplyExternalPhotoCalib, doApplyExternalSkyWcs,
                                                         useMeasMosaic, iCat)
                self.zpLabel1 = zpLabel if iCat == 0 and not self.zpLabel1 else self.zpLabel1
                self.zpLabel2 = zpLabel if iCat == 1 and not self.zpLabel2 else self.zpLabel2

                catList.append(srcCat)

        self.zpLabel = self.zpLabel1 + "\n zp: " + self.zpLabel2
        self.log.info("Applying {:} calibration to catalogs".format(self.zpLabel))
        if not catList1:
            raise TaskError("No catalogs read: %s" % ([dataRefList1[0].dataId for dataRef1 in dataRefList1]))
        if not catList2:
            raise TaskError("No catalogs read: %s" % ([dataRefList2[0].dataId for dataRef2 in dataRefList2]))

        return (concatenateCatalogs(commonZpCatList1), concatenateCatalogs(catList1), areaDict1,
                concatenateCatalogs(commonZpCatList2), concatenateCatalogs(catList2), areaDict2)

    def calibrateCatalogs(self, dataRef, catalog, fluxMag0, repoInfo, doApplyExternalPhotoCalib,
                          doApplyExternalSkyWcs, useMeasMosaic, iCat):
        """Determine and apply appropriate flux calibration to the catalog.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.butlerSubset.ButlerDataRef`
           A dataRef is needed for call to meas_mosaic's applyMosaicResultsCatalog() in
           utils' calibrateSourceCatalogMosaic()
        catalog : `lsst.afw.table.source.source.SourceCatalog`
           The catalog to which the calibration is applied in place
        fluxMag0 : `float`
           The instrumental flux corresponding to 0 magnitude from Single Frame
           Measurement for the catalog.
        repoInfo : `lsst.pipe.base.Struct`
           A struct containing relevant information about the repository under
           study.  Elements used here include the dataset names for any external
           calibrations to be applied.
        doApplyExternalPhotoCalib : `bool`
           If True: Apply the external photometric calibrations specified by
                    ``repoInfo.photoCalibDataset`` to the caltalog.
           If False: Apply the ``fluxMag0`` photometric calibration from single
                     frame processing to the catalog.
        doApplyExternalSkyWcs : `bool`
           If True: Apply the external astrometric calibrations specified by
                    ``repoInfo.skyWcsDataset`` the caltalog.
           If False: Retain the WCS from single frame processing.
        useMeasMosaic : `bool`
           Use meas_mosaic's applyMosaicResultsCatalog for the external
           calibration (even if photoCalib object exists).  For testing
           implementations.
        iCat : `int`
           Integer representing whether this is comparison catalog number 0 or 1

        Returns
        -------
        calibrated : `lsst.afw.table.source.source.SourceCatalog`
           The calibrated source catalog.
        zpLabel : `str`
           A label indicating the external calibration applied (currently
           either jointcal, fgcm, fgcm_tract, or meas_mosaic, but the latter
           is effectively retired).
        """
        self.zp = 0.0
        if doApplyExternalPhotoCalib:
            if not useMeasMosaic:
                # i.e. the processing was post-photoCalib output generation
                # AND you want the photoCalib flux object used for the
                # calibration (as opposed to meas_mosaic's fcr object).
                zpLabel = ("MMphotoCalib" if dataRef.datasetExists("fcr_md")
                           else repoInfo.photoCalibDataset.split("_")[0].upper())
                zpLabel += "_" + str(iCat + 1)
                calibrated = calibrateSourceCatalogPhotoCalib(dataRef, catalog, repoInfo.photoCalibDataset,
                                                              zp=self.zp)
            else:
                # If here, the data were processed pre-photoCalib output
                # generation, so must use old method OR old method was
                # explicitly requested via useMeasMosaic.
                try:
                    import lsst.meas.mosaic  # noqa : F401
                except ImportError:
                    raise ValueError("Cannot apply calibrations because meas_mosaic could not "
                                     "be imported. \nEither setup meas_mosaic or run with "
                                     "--config doApplyExternalPhotoCalib=False doApplyExternalSkyWcs=False")
                zpLabel = "MEAS_MOSAIC"
                calibrated = calibrateSourceCatalogMosaic(dataRef, catalog, zp=self.zp)
        else:
            # Scale fluxes to measured zeropoint
            self.zp = 2.5*np.log10(fluxMag0)
            zpLabel = "FLUXMAG0"
            calibrated = calibrateSourceCatalog(catalog, self.zp)

        if doApplyExternalSkyWcs:
            wcs = dataRef.get(repoInfo.skyWcsDataset)
            afwTable.updateSourceCoords(wcs, catalog)
            if "wcs" not in zpLabel:
                zpLabel += " wcs: " + repoInfo.skyWcsDataset.split("_")[0].upper() + "_" + str(iCat + 1)

        return calibrated, zpLabel
