#!/usr/bin/env python

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all="ignore")

from collections import defaultdict

from lsst.daf.persistence.butler import Butler
from lsst.pipe.base import ArgumentParser, TaskRunner, TaskError
from lsst.meas.base.forcedPhotCcd import PerTractCcdDataIdContainer
from lsst.afw.table.catalogMatches import matchesToCatalog
from lsst.meas.extensions.astrometryNet import LoadAstrometryNetObjectsTask
from .analysis import Analysis
from .coaddAnalysis import (CoaddAnalysisConfig, CoaddAnalysisTask, CompareCoaddAnalysisConfig,
                            CompareCoaddAnalysisTask)
from .utils import *
from .plotUtils import *

import lsst.afw.table as afwTable


class CcdAnalysis(Analysis):
    def plotAll(self, dataId, filenamer, log, enforcer=None, butler=None, camera=None, ccdList=None,
                tractInfo=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None, postFix="",
                plotRunStats=True, highlightList=None):
        stats = self.stats
        if self.config.doPlotCcdXy:
            self.plotCcd(filenamer(dataId, description=self.shortName, style="ccd" + postFix), stats=stats,
                         hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel)
        if self.config.doPlotFP:
            self.plotFocalPlane(filenamer(dataId, description=self.shortName, style="fpa" + postFix),
                                stats=stats, camera=camera, ccdList=ccdList, hscRun=hscRun,
                                matchRadius=matchRadius, zpLabel=zpLabel)

        return Analysis.plotAll(self, dataId, filenamer, log, enforcer=enforcer, butler=butler, camera=camera,
                                ccdList=ccdList, hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel,
                                postFix=postFix, plotRunStats=plotRunStats, highlightList=highlightList)

    def plotFP(self, dataId, filenamer, log, enforcer=None, camera=None, ccdList=None, hscRun=None,
               matchRadius=None, zpLabel=None):
        self.plotFocalPlane(filenamer(dataId, description=self.shortName, style="fpa"), stats=self.stats,
                            camera=camera, ccdList=ccdList, hscRun=hscRun, matchRadius=matchRadius,
                            zpLabel=zpLabel)

    def plotCcd(self, filename, centroid="base_SdssCentroid", cmap=plt.cm.nipy_spectral, idBits=32,
                visitMultiplier=200, stats=None, hscRun=None, matchRadius=None, zpLabel=None):
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
        for name, data in self.data.iteritems():
            if not data.plot:
                continue
            if len(data.mag) == 0:
                continue
            if ptSize is None:
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
            annotateAxes(filename, plt, axes[0], stats, "star", self.config.magThreshold, x0=0.03, yOff=0.07,
                         hscRun=hscRun, matchRadius=matchRadius, unitScale=self.unitScale)
            annotateAxes(filename, plt, axes[1], stats, "star", self.config.magThreshold, x0=0.03, yOff=0.07,
                         hscRun=hscRun, matchRadius=matchRadius, unitScale=self.unitScale)
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
        labelVisit(filename, plt, axes[0], 0.5, 1.1)
        if zpLabel is not None:
            labelZp(zpLabel, plt, axes[0], 0.08, -0.11, color="green")
        fig.savefig(filename)
        plt.close(fig)

    def plotFocalPlane(self, filename, cmap=plt.cm.Spectral, stats=None, camera=None, ccdList=None,
                       hscRun=None, matchRadius=None, zpLabel=None, fontSize=8):
        """Plot quantity colormaped on the focal plane"""
        xFp = self.catalog[self.prefix + "base_FPPosition_x"]
        yFp = self.catalog[self.prefix + "base_FPPosition_y"]
        good = (self.mag < self.config.magThreshold if self.config.magThreshold > 0 else
                np.ones(len(self.mag), dtype=bool))
        if self.data.has_key("galaxy") and "calib_psfUsed" not in self.goodKeys:
            vMin, vMax = 0.5*self.qMin, 0.5*self.qMax
        else:
            vMin, vMax = self.qMin, self.qMax
        # Set limits to ccd pixel ranges when plotting the centroids (which are in pixel units)
        if filename.find("Centroid") > -1:
            cmap = plt.cm.pink
            vMin = min(0, np.round(self.data["star"].quantity.min() - 10))
            vMax = np.round(self.data["star"].quantity.max() + 50, -2)
        fig, axes = plt.subplots(1, 1, subplot_kw=dict(facecolor="0.7"))
        axes.tick_params(which="both", direction="in", top="on", right="on", labelsize=fontSize)
        for name, data in self.data.iteritems():
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
        if hscRun is not None:
            axes.set_title("HSC stack run: " + hscRun, color="#800080")
        labelVisit(filename, plt, axes, 0.5, 1.04)
        if camera is not None:
            labelCamera(camera, plt, axes, 0.5, 1.09)
        if zpLabel is not None:
            labelZp(zpLabel, plt, axes, 0.08, -0.1, color="green")
        fig.savefig(filename)
        plt.close(fig)


class VisitAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        if len(parsedCmd.id.refList) < 1:
            raise RuntimeWarning("refList from parsedCmd is empty...")
        kwargs["tract"] = parsedCmd.tract
        visits = defaultdict(list)
        for ref in parsedCmd.id.refList:
            visits[ref.dataId["visit"]].append(ref)
        return [(refs, kwargs) for refs in visits.itervalues()]


class VisitAnalysisTask(CoaddAnalysisTask):
    _DefaultName = "visitAnalysis"
    ConfigClass = CoaddAnalysisConfig
    RunnerClass = VisitAnalysisRunner
    AnalysisClass = CcdAnalysis

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "src", help="data ID with raw CCD keys, "
                               "e.g. --id visit=12345 ccd=6^8..11", ContainerClass=PerTractCcdDataIdContainer)
        parser.add_argument("--tract", type=str, default=None,
                            help="Tract(s) to use (do one at a time for overlapping) e.g. 1^5^0")
        return parser

    def run(self, dataRefList, tract=None):
        self.log.info("dataRefList size: {:d}".format(len(dataRefList)))
        ccdList = [dataRef.dataId["ccd"] for dataRef in dataRefList]
        # cull multiple entries
        ccdList = list(set(ccdList))
        if tract is None:
            tractList = [0, ]
        else:
            tractList = [int(tractStr) for tractStr in tract.split('^')]
        dataRefListPerTract = [None]*len(tractList)
        for i, tract in enumerate(tractList):
            dataRefListPerTract[i] = [dataRef for dataRef in dataRefList if dataRef.dataId["tract"] == tract]
        commonZpDone = False
        for i, dataRefListTract in enumerate(dataRefListPerTract):
            if len(dataRefListTract) == 0:
                self.log.info("No data found for tract: {:d}".format(tractList[i]))
                continue
            butler = dataRefListTract[0].getButler()
            camera = butler.get("camera")
            dataId = dataRefListTract[0].dataId
            self.log.info("dataId: {:s}".format(dataId))
            # Check metadata to see if stack used was HSC
            metadata = butler.get("calexp_md", dataRefListTract[0].dataId)
            hscRun = checkHscStack(metadata)
            dataset = "src"
            if self.config.doApplyUberCal:
                if hscRun is not None:
                    dataset = "wcs_hsc_md"
                else:
                    dataset = "wcs_md"
            ccdListPerTract = [dataRef.dataId["ccd"] for dataRef in dataRefListTract if
                               dataRef.datasetExists(dataset)]
            if len(ccdListPerTract) == 0:
                if self.config.doApplyUberCal:
                    self.log.fatal("No dataset found...are you sure you ran meas_mosaic? "
                                   "If not, run with --config doApplyUberCal=False")
                raise RuntimeError("No datasets found for datasetType = {:s}".format(dataset))
            filterName = dataId["filter"]
            filenamer = Filenamer(butler, "plotVisit", dataRefListTract[0].dataId)
            if any(doPlot for doPlot in [self.config.doPlotFootprintNpix, self.config.doPlotQuiver,
                              self.config.doPlotMags, self.config.doPlotSizes, self.config.doPlotCentroids,
                              self.config.doPlotStarGalaxy]):
                commonZpCat, catalog = self.readCatalogs(dataRefListTract, "src", hscRun=hscRun)
                if hscRun and self.config.doAddAperFluxHsc:
                    self.log.info("HSC run: adding aperture flux to schema...")
                    catalog = addApertureFluxesHSC(catalog, prefix="")

            try:
                self.zpLabel = self.zpLabel + " " + self.catLabel
            except:
                pass

            self.unitScale = 1.0
            if self.config.toMilli:
                self.unitScale = 1000.0

            if self.config.doPlotFootprintNpix:
                catalog = addFootprintNPix(catalog)
                self.plotFootprintHist(catalog, filenamer(dataId, description="footNpix", style="hist"),
                                       dataId, butler=butler, camera=camera, ccdList=ccdListPerTract,
                                       hscRun=hscRun, zpLabel=self.zpLabel)
                self.plotFootprint(catalog, filenamer, dataId, butler=butler, camera=camera,
                                   ccdList=ccdListPerTract, hscRun=hscRun, zpLabel=self.zpLabel,
                                   plotRunStats=False, highlightList=[("parent", 0, "yellow"), ])

            if self.config.doPlotQuiver:
                self.plotQuiver(catalog, filenamer(dataId, description="ellipResids", style="quiver"),
                                dataId=dataId, butler=butler, camera=camera, ccdList=ccdListPerTract,
                                hscRun=hscRun, zpLabel=self.zpLabel, scale=2)

            # Create mag comparison plots using common ZP
            if self.config.doPlotMags and not commonZpDone:
                zpLabel = "common (" + str(self.config.analysis.commonZp) + ")"
                try:
                    zpLabel = zpLabel + " " + self.catLabel
                except:
                    pass
                self.plotMags(commonZpCat, filenamer, dataId, butler=butler, camera=camera, ccdList=ccdList,
                              hscRun=hscRun, zpLabel=zpLabel,
                              fluxToPlotList=["base_GaussianFlux", "base_CircularApertureFlux_12_0"],
                              postFix="_commonZp")
                commonZpDone = True
            # Now source catalog calibrated to either FLUXMAG0 or meas_mosaic result for remainder of plots
            if self.config.doPlotSizes:
                if "base_SdssShape_psf_xx" in catalog.schema:
                    self.plotSizes(catalog, filenamer, dataId, butler=butler, camera=camera,
                                   ccdList=ccdListPerTract, hscRun=hscRun, zpLabel=self.zpLabel)
                else:
                    self.log.warn("Cannot run plotSizes: base_SdssShape_psf_xx not in catalog.schema")
            if self.config.doPlotMags:
                self.plotMags(catalog, filenamer, dataId, butler=butler, camera=camera,
                              ccdList=ccdListPerTract, hscRun=hscRun, zpLabel=self.zpLabel)
            if self.config.doPlotCentroids:
                self.plotCentroidXY(catalog, filenamer, dataId, butler=butler, camera=camera,
                                    ccdList=ccdListPerTract, hscRun=hscRun, zpLabel=self.zpLabel)
            if self.config.doPlotStarGalaxy:
                if "ext_shapeHSM_HsmSourceMoments_xx" in catalog.schema:
                    self.plotStarGal(catalog, filenamer, dataId, butler=butler, camera=camera,
                                     ccdList=ccdListPerTract, hscRun=hscRun, zpLabel=self.zpLabel)
                else:
                    self.log.warn("Cannot run plotStarGal: " +
                                  "ext_shapeHSM_HsmSourceMoments_xx not in catalog.schema")
            if self.config.doPlotMatches:
                matches = self.readSrcMatches(dataRefListTract, "src")
                self.plotMatches(matches, filterName, filenamer, dataId, butler=butler, camera=camera,
                                 ccdList=ccdListPerTract, hscRun=hscRun, zpLabel=self.zpLabel)

            for cat in self.config.externalCatalogs:
                if self.config.photoCatName not in cat:
                    with andCatalog(cat):
                        matches = self.matchCatalog(catalog, filterName, self.config.externalCatalogs[cat])
                        self.plotMatches(matches, filterName, filenamer, dataId, butler=butler,
                                         camera=camera, ccdList=ccdListPerTract, hscRun=hscRun,
                                         matchRadius=self.config.matchRadius, zpLabel=self.zpLabel)

    def readCatalogs(self, dataRefList, dataset, hscRun=None):
        catList = []
        commonZpCatList = []
        for dataRef in dataRefList:
            if not dataRef.datasetExists(dataset):
                continue
            catalog = dataRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_HEAVY_FOOTPRINTS)
            # Set an alias map for differing src naming conventions of different stacks (if any)
            if hscRun and self.config.srcSchemaMap:
                aliasMap = catalog.schema.getAliasMap()
                for lsstName, otherName in self.config.srcSchemaMap.iteritems():
                    aliasMap.set(lsstName, otherName)
            # purge the catalogs of flagged sources
            bad = np.zeros(len(catalog), dtype=bool)
            bad |= catalog["deblend_nChild"] > 0
            self.catLabel = "nChild = 0"
            for flag in self.config.analysis.flags:
                if flag in catalog.schema:
                    bad |= catalog[flag]
            catalog = catalog[~bad].copy(deep=True)

            butler = dataRef.getButler()
            metadata = butler.get("calexp_md", dataRef.dataId)

            # Compute Focal Plane coordinates for each source if not already there
            if "base_FPPosition_x" not in catalog.schema and "focalplane_x" not in catalog.schema:
                exp = butler.get("calexp", dataRef.dataId)
                det = exp.getDetector()
                catalog = addFpPoint(det, catalog)
            # Optionally backout aperture corrections
            if self.config.doBackoutApCorr:
                catalog = backoutApCorr(catalog)

            # Scale fluxes to common zeropoint to make basic comparison plots without calibrated ZP influence
            commonZpCat = catalog.copy(True)
            commonZpCat = calibrateSourceCatalog(commonZpCat, self.config.analysis.commonZp)
            commonZpCatList.append(commonZpCat)
            if self.config.doApplyUberCal:
                if hscRun is not None:
                    if not dataRef.datasetExists("wcs_hsc_md") or not dataRef.datasetExists("fcr_hsc_md"):
                        continue
                else:
                    if not dataRef.datasetExists("wcs_md") or not dataRef.datasetExists("fcr_md"):
                        continue
            catalog = self.calibrateCatalogs(dataRef, catalog, metadata)
            catList.append(catalog)

        if len(catList) == 0:
            raise TaskError("No catalogs read: %s" % ([dataRef.dataId for dataRef in dataRefList]))

        return concatenateCatalogs(commonZpCatList), concatenateCatalogs(catList)

    def readSrcMatches(self, dataRefList, dataset):
        catList = []
        dataIdSubList = []
        for dataRef in dataRefList:
            if not dataRef.datasetExists(dataset):
                continue
            butler = dataRef.getButler()
            metadata = butler.get("calexp_md", dataRef.dataId)
            hscRun = checkHscStack(metadata)
            if self.config.doApplyUberCal:
                if hscRun is not None:
                    if not dataRef.datasetExists("wcs_hsc_md") or not dataRef.datasetExists("fcr_hsc_md"):
                        continue
                else:
                    if not dataRef.datasetExists("wcs_md") or not dataRef.datasetExists("fcr_md"):
                        continue
            # Generate unnormalized match list (from normalized persisted one) with joinMatchListWithCatalog
            # (which requires a refObjLoader to be initialized).
            catalog = dataRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS)
            # Set an alias map for differing src naming conventions of different stacks (if any)
            if hscRun is not None and self.config.srcSchemaMap:
                # for cat in [commonZpCat, catalog]:
                aliasMap = catalog.schema.getAliasMap()
                for lsstName, otherName in self.config.srcSchemaMap.iteritems():
                    aliasMap.set(lsstName, otherName)
            catalog = self.calibrateCatalogs(dataRef, catalog, metadata)
            packedMatches = butler.get(dataset + "Match", dataRef.dataId)
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
            noMatches = False
            if len(matches) < 8:
                for m in matches:
                    if not hasattr(m.first, "get"):
                        matches = []
                        noMatches = True
                        break

            # LSST reads in a_net catalogs with flux in "janskys", so must convert back to DN
            if not noMatches:
                matches = matchJanskyToDn(matches)
                if checkHscStack(metadata) is not None and self.config.doAddAperFluxHsc:
                    addApertureFluxesHSC(matches, prefix="second_")

            if len(matches) == 0:
                self.log.warn("No matches for {:s}".format(dataRef.dataId))
                continue

            zp = -2.5*np.log10(metadata.get("FLUXMAG0"))
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
            # Need to set the aliap map for the matched catalog sources
            if self.config.srcSchemaMap is not None and checkHscStack(metadata) is not None:
                aliasMap = catalog.schema.getAliasMap()
                for lsstName, otherName in self.config.srcSchemaMap.iteritems():
                    aliasMap.set("src_" + lsstName, "src_" + otherName)
            # To avoid multiple counting when visit overlaps multiple tracts
            if (dataRef.dataId['visit'], dataRef.dataId['ccd']) not in dataIdSubList:
                catList.append(catalog)
            dataIdSubList.append((dataRef.dataId["visit"], dataRef.dataId["ccd"]))

        if len(catList) == 0:
            raise TaskError("No matches read: %s" % ([dataRef.dataId for dataRef in dataRefList]))

        return concatenateCatalogs(catList)

    def calibrateCatalogs(self, dataRef, catalog, metadata):
        self.zp = 0.0
        try:
            self.zpLabel = self.zpLabel
        except:
            self.zpLabel = None
        if self.config.doApplyUberCal:
            calibrated = calibrateSourceCatalogMosaic(dataRef, catalog, zp=self.zp)
            if self.zpLabel is None:
                self.log.info("Applying meas_mosaic calibration to catalog")
            self.zpLabel = "MEAS_MOSAIC"
        else:
            # Scale fluxes to measured zeropoint
            self.zp = 2.5*np.log10(metadata.get("FLUXMAG0"))
            if self.zpLabel is None:
                self.log.info("Using 2.5*log10(FLUXMAG0) = {:.4f} from FITS header for zeropoint".format(
                        self.zp))
            self.zpLabel = "FLUXMAG0"
            calibrated = calibrateSourceCatalog(catalog, self.zp)

        return calibrated

class CompareVisitAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        parentDir = parsedCmd.input
        kwargs["tract"] = parsedCmd.tract
        while os.path.exists(os.path.join(parentDir, "_parent")):
            parentDir = os.path.realpath(os.path.join(parentDir, "_parent"))
        butler2 = Butler(root=os.path.join(parentDir, "rerun", parsedCmd.rerun2), calibRoot=parsedCmd.calib)
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
                refs1, refs2 in zip(visits1.itervalues(), visits2.itervalues())]


class CompareVisitAnalysisTask(CompareCoaddAnalysisTask):
    ConfigClass = CompareCoaddAnalysisConfig
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
        return parser

    def run(self, dataRefList1, dataRefList2, tract=None):
        # This is for the commonZP plots (i.e. all ccds regardless of tract)
        fullCcdList = list(set(
                [dataRef1.dataId["ccd"] for dataRef1 in dataRefList1 if dataRef1.datasetExists("src")]))

        if tract is None:
            tractList = [0, ]
        else:
            tractList = [int(tractStr) for tractStr in tract.split('^')]
        self.log.debug("tractList = {:s}".format(tractList))
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
        dataset = "src"
        if self.config.doApplyUberCal:
            if hscRun is not None:
                dataset = "wcs_hsc_md"
            else:
                dataset = "wcs_md"

        i = -1
        for dataRefListTract1, dataRefListTract2 in zip(dataRefListPerTract1, dataRefListPerTract2):
            i += 1
            if len(dataRefListTract1) == 0:
                self.log.info("No data found in --rerun for tract: {:d}".format(tractList[i]))
                continue
            if len(dataRefListTract2) == 0:
                self.log.info("No data found in --rerun2 for tract: {:d}".format(tractList[i]))
                continue
            ccdListPerTract1 = [dataRef1.dataId["ccd"] for dataRef1 in dataRefListTract1 if
                                dataRef1.datasetExists(dataset)]
            if len(ccdListPerTract1) == 0:
                if self.config.doApplyUberCal:
                    self.log.fatal("No dataset found...are you sure you ran meas_mosaic?")
                raise RuntimeError("No datasets found for datasetType = {:s}".format(dataset))
            self.log.info("tract: {:d} ".format(dataRef1.dataId["tract"]))
            self.log.info("ccdListPerTract1: {:s} ".format(ccdListPerTract1))
            dataId1 = dataRefListTract1[0].dataId
            butler1 = dataRefListTract1[0].getButler()
            metadata1 = butler1.get("calexp_md", dataId1)
            camera1 = butler1.get("camera")
            filenamer = Filenamer(dataRefListTract1[0].getButler(), "plotCompareVisit", dataId1)
            butler2 = dataRefListTract2[0].getButler()
            metadata2 = butler2.get("calexp_md", dataRefListTract2[0].dataId)
            # Check metadata to see if stack used was HSC
            hscRun1 = checkHscStack(metadata1)
            hscRun2 = checkHscStack(metadata2)
            doReadFootprints = None
            if self.config.doPlotFootprintNpix:
                doReadFootprints = "light"
            commonZpCat1, catalog1, commonZpCat2, catalog2 = (
                self.readCatalogs(dataRefListTract1, dataRefListTract2, "src", hscRun1=hscRun1,
                                  hscRun2=hscRun2, doReadFootprints=doReadFootprints))

            try:
                self.zpLabel = self.zpLabel + " " + self.catLabel
            except:
                pass

            if hscRun2 and self.config.doAddAperFluxHsc:
                self.log.info("HSC run: adding aperture flux to schema...")
                catalog2 = addApertureFluxesHSC(catalog2, prefix="")

            if hscRun1 and self.config.doAddAperFluxHsc:
                self.log.info("HSC run: adding aperture flux to schema...")
                catalog1 = addApertureFluxesHSC(catalog1, prefix="")

            self.log.info("\nNumber of sources in catalogs: first = {0:d} and second = {1:d}".format(
                    len(catalog1), len(catalog2)))
            commonZpCat = self.matchCatalogs(commonZpCat1, commonZpCat2)
            catalog = self.matchCatalogs(catalog1, catalog2)

            if self.config.doBackoutApCorr:
                commonZpCat = backoutApCorr(commonZpCat)
                catalog = backoutApCorr(catalog)

            self.log.info("Number of matches (maxDist = {0:.2f} arcsec) = {1:d}".format(
                    self.config.matchRadius, len(catalog)))


            if self.config.doPlotFootprintNpix:
                self.plotFootprint(catalog, filenamer, dataId1, butler=butler1, camera=camera1,
                                   ccdList=ccdListPerTract1, hscRun=hscRun2,
                                   matchRadius=self.config.matchRadius, zpLabel=self.zpLabel)

            # Create mag comparison plots using common ZP
            if not commonZpDone:
                zpLabel = "common (" + str(self.config.analysis.commonZp) + ")"
                try:
                    zpLabel = zpLabel + " " + self.catLabel
                except:
                    pass

                self.plotMags(commonZpCat, filenamer, dataId1, butler=butler1, camera=camera1,
                              ccdList=fullCcdList, hscRun=hscRun2, matchRadius=self.config.matchRadius,
                              zpLabel=zpLabel,
                              fluxToPlotList=["base_GaussianFlux", "base_CircularApertureFlux_12_0"],
                              postFix="_commonZp")
                commonZpDone = True

            if self.config.doPlotMags:
                self.plotMags(catalog, filenamer, dataId1, butler=butler1, camera=camera1,
                              ccdList=ccdListPerTract1,
                              hscRun=hscRun2, matchRadius=self.config.matchRadius, zpLabel=self.zpLabel)
            if self.config.doPlotSizes:
                if ("first_base_SdssShape_psf_xx" in catalog.schema and
                    "second_base_SdssShape_psf_xx" in catalog.schema):
                    self.plotSizes(catalog, filenamer, dataId1, butler=butler1, camera=camera1,
                                   ccdList=ccdListPerTract1,
                                   hscRun=hscRun2, matchRadius=self.config.matchRadius, zpLabel=self.zpLabel)
                else:
                    self.log.warn("Cannot run plotSizes: base_SdssShape_psf_xx not in catalog.schema")
            if self.config.doApCorrs:
                self.plotApCorrs(catalog, filenamer, dataId1, butler=butler1, camera=camera1,
                                 ccdList=ccdListPerTract1,
                                 hscRun=hscRun2, matchRadius=self.config.matchRadius, zpLabel=self.zpLabel)
            if self.config.doPlotCentroids:
                self.plotCentroids(catalog, filenamer, dataId1, butler=butler1, camera=camera1,
                                   ccdList=ccdListPerTract1, hscRun1=hscRun1,
                                   hscRun2=hscRun2, matchRadius=self.config.matchRadius, zpLabel=self.zpLabel)

    def readCatalogs(self, dataRefList1, dataRefList2, dataset, hscRun1=None, hscRun2=None,
                     doReadFootprints=None):
        catList1 = []
        commonZpCatList1 = []
        catList2 = []
        commonZpCatList2 = []
        for dataRef1, dataRef2 in zip(dataRefList1, dataRefList2):
            if not dataRef1.datasetExists(dataset) or not dataRef2.datasetExists(dataset):
                continue
            if doReadFootprints == None:
                srcCat1 = dataRef1.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS)
                srcCat2 = dataRef2.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS)
            elif doReadFootprints == "light":
                srcCat1 = dataRef1.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_HEAVY_FOOTPRINTS)
                srcCat2 = dataRef2.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_HEAVY_FOOTPRINTS)
            elif doReadFootprints == "heavy":
                srcCat1 = dataRef1.get(dataset, immediate=True)
                srcCat2 = dataRef2.get(dataset, immediate=True)
            # Set an alias map for differing src naming conventions of different stacks (if any)
            for cat, hscRun in ([srcCat1, hscRun1], [srcCat2, hscRun2]):
                if self.config.srcSchemaMap and hscRun:
                    aliasMap = cat.schema.getAliasMap()
                    for lsstName, otherName in self.config.srcSchemaMap.iteritems():
                        aliasMap.set(lsstName, otherName)
            srcCat1 = srcCat1[srcCat1["deblend_nChild"] == 0].copy(True) # Exclude non-deblended objects
            srcCat2 = srcCat2[srcCat2["deblend_nChild"] == 0].copy(True) # Exclude non-deblended objects
            self.catLabel = "nChild = 0"
            butler1 = dataRef1.getButler()
            butler2 = dataRef2.getButler()
            metadata1 = butler1.get("calexp_md", dataRef1.dataId)
            metadata2 = butler2.get("calexp_md", dataRef2.dataId)
            calexp1 = butler1.get("calexp", dataRef1.dataId)
            calexp2 = butler2.get("calexp", dataRef2.dataId)
            nQuarter = calexp1.getDetector().getOrientation().getNQuarter()
            # add footprint nPix column
            if self.config.doPlotFootprintNpix:
                srcCat1 = addFootprintNPix(srcCat1)
                srcCat2 = addFootprintNPix(srcCat2)
            # Add rotated point in LSST cat if comparing with HSC cat to compare centroid pixel positions
            if hscRun2 is not None and hscRun1 is None:
                srcCat1 = addRotPoint(srcCat1, calexp1.getWidth(), calexp1.getHeight(), nQuarter)
            if hscRun1 is not None and hscRun2 is None:
                srcCat2 = addRotPoint(srcCat2, calexp2.getWidth(), calexp2.getHeight(), nQuarter)

            # Scale fluxes to common zeropoint to make basic comparison plots without calibrated ZP influence
            commonZpCat1 = srcCat1.copy(True)
            commonZpCat1 = calibrateSourceCatalog(commonZpCat1, self.config.analysis.commonZp)
            commonZpCatList1.append(commonZpCat1)
            commonZpCat2 = srcCat2.copy(True)
            commonZpCat2 = calibrateSourceCatalog(commonZpCat2, self.config.analysis.commonZp)
            commonZpCatList2.append(commonZpCat2)
            if self.config.doApplyUberCal:
                if hscRun is not None:
                    if not dataRef1.datasetExists("wcs_hsc_md") or not dataRef1.datasetExists("fcr_hsc_md"):
                        continue
                    if not dataRef2.datasetExists("wcs_hsc_md") or not dataRef2.datasetExists("fcr_hsc_md"):
                        continue
                else:
                    if not dataRef1.datasetExists("wcs_md") or not dataRef1.datasetExists("fcr_md"):
                        continue
                    if not dataRef2.datasetExists("wcs_md") or not dataRef2.datasetExists("fcr_md"):
                        continue
            srcCat1 = self.calibrateCatalogs(dataRef1, srcCat1, metadata1)
            catList1.append(srcCat1)
            srcCat2 = self.calibrateCatalogs(dataRef2, srcCat2, metadata2)
            catList2.append(srcCat2)

        if len(catList1) == 0:
            raise TaskError("No catalogs read: %s" % ([dataRefList1[0].dataId for dataRef1 in dataRefList1]))
        return (concatenateCatalogs(commonZpCatList1), concatenateCatalogs(catList1),
                concatenateCatalogs(commonZpCatList2), concatenateCatalogs(catList2))

    def calibrateCatalogs(self, dataRef, catalog, metadata):
        self.zp = 0.0
        try:
            self.zpLabel = self.zpLabel
        except:
            self.zpLabel = None
        if self.config.doApplyUberCal:
            calibrated = calibrateSourceCatalogMosaic(dataRef, catalog, zp=self.zp)
            self.zpLabel = "MEAS_MOSAIC"
            if self.zpLabel is None:
                self.log.info("Applying meas_mosaic calibration to catalog")
        else:
            # Scale fluxes to measured zeropoint
            self.zp = 2.5*np.log10(metadata.get("FLUXMAG0"))
            if self.zpLabel is None:
                self.log.info("Using 2.5*log10(FLUXMAG0) = {:.4f} from FITS header for zeropoint".format(
                    self.zp))
            self.zpLabel = "FLUXMAG0"
            calibrated = calibrateSourceCatalog(catalog, self.zp)

        return calibrated

    def plotMags(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None, hscRun=None,
                 matchRadius=None, zpLabel=None, fluxToPlotList=None, postFix="", highlightList=None):
        unitStr = "mag"
        if self.config.toMilli:
            unitStr = "mmag"
        if fluxToPlotList is None:
            fluxToPlotList = self.config.fluxToPlotList
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.02*self.unitScale}})
        for col in fluxToPlotList:
            if "first_" + col + "_flux" in catalog.schema and "second_" + col + "_flux" in catalog.schema:
                shortName = "diff_" + col + postFix
                self.log.info("shortName = {:s}".format(shortName))
                Analysis(catalog, MagDiffCompare(col + "_flux", unitScale=self.unitScale),
                         "Run Comparison: %s mag diff (%s)" % (fluxToPlotString(col), unitStr), shortName,
                         self.config.analysis, prefix="first_", qMin=-0.05, qMax=0.05, flags=[col + "_flag"],
                         errFunc=MagDiffErr(col + "_flux"), labeller=OverlapsStarGalaxyLabeller(),
                         unitScale=self.unitScale,
                         ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                   camera=camera, ccdList=ccdList, hscRun=hscRun, matchRadius=matchRadius,
                                   zpLabel=zpLabel)

    def plotCentroids(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None,
                      tractInfo=None, patchList=None, hscRun1=None, hscRun2=None, matchRadius=None,
                      zpLabel=None, flagsCat=None, highlightList=None):
        distEnforcer = None
        shortName = "diff_x"
        self.log.info("shortName = {:s}".format(shortName))
        centroidStr1, centroidStr2 = "base_SdssCentroid", "base_SdssCentroid"
        if (hscRun1 is not None) != (hscRun2 is not None):
            if hscRun1 is None:
                centroidStr1 = "base_SdssCentroid_Rot"
            if hscRun2 is None:
                centroidStr2 = "base_SdssCentroid_Rot"
        Analysis(catalog, CentroidDiff("x", centroid1=centroidStr1, centroid2=centroidStr2),
                 "Run Comparison: x offset (arcsec)", shortName, self.config.analysis, prefix="first_",
                 qMin=-0.08, qMax=0.08, errFunc=None, labeller=OverlapsStarGalaxyLabeller(),
                 ).plotAll(dataId, filenamer, self.log, enforcer=distEnforcer, butler=butler, camera=camera,
                           ccdList=ccdList, tractInfo=tractInfo, patchList=patchList,
                           hscRun=(hscRun1 or hscRun2), matchRadius=matchRadius, zpLabel=zpLabel)
        shortName = "diff_y"
        self.log.info("shortName = {:s}".format(shortName))
        Analysis(catalog, CentroidDiff("y", centroid1=centroidStr1, centroid2=centroidStr2),
                 "Run Comparison: y offset (arcsec)", shortName, self.config.analysis, prefix="first_",
                 qMin=-0.08, qMax=0.08, errFunc=None, labeller=OverlapsStarGalaxyLabeller(),
                 ).plotAll(dataId, filenamer, self.log, enforcer=distEnforcer, butler=butler, camera=camera,
                           ccdList=ccdList, tractInfo=tractInfo, patchList=patchList,
                           hscRun=(hscRun1 or hscRun2), matchRadius=matchRadius, zpLabel=zpLabel)

    def plotSizes(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None, hscRun=None,
                 matchRadius=None, zpLabel=None):
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.02*self.unitScale}})
        for col in ["base_PsfFlux"]:
            if "first_" + col + "_flux" in catalog.schema and "second_" + col + "_flux" in catalog.schema:
                shortName = "trace_"
                self.log.info("shortName = {:s}".format(shortName))
                Analysis(catalog, sdssTraceSizeCompare(), "SdssShape Trace Radius Diff (%)", shortName,
                         self.config.analysis, flags=[col + "_flag"], prefix="first_",
                         goodKeys=["calib_psfUsed"], qMin=-0.5, qMax=1.5,
                         labeller=OverlapsStarGalaxyLabeller(),
                         ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                   camera=camera, ccdList=ccdList, hscRun=hscRun,
                                   matchRadius=matchRadius, zpLabel=zpLabel)
                shortName = "psfTrace_"
                self.log.info("shortName = {:s}".format(shortName))
                Analysis(catalog, sdssPsfTraceSizeCompare(), "SdssShape PSF Trace Radius Diff (%)", shortName,
                         self.config.analysis, flags=[col + "_flag"], prefix="first_",
                         goodKeys=["calib_psfUsed"], qMin=-1.1, qMax=1.1,
                         labeller=OverlapsStarGalaxyLabeller(),
                         ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                   camera=camera, ccdList=ccdList, hscRun=hscRun,
                                   matchRadius=matchRadius, zpLabel=zpLabel)
                shortName = "sdssXx_"
                self.log.info("shortName = {:s}".format(shortName))
                Analysis(catalog, sdssXxCompare(), "SdssShape xx Moment Diff (%)", shortName,
                         self.config.analysis, flags=[col + "_flag"], prefix="first_",
                         goodKeys=["calib_psfUsed"], qMin=-0.5, qMax=1.5,
                         labeller=OverlapsStarGalaxyLabeller(),
                         ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                   camera=camera, ccdList=ccdList, hscRun=hscRun,
                                   matchRadius=matchRadius, zpLabel=zpLabel)
                shortName = "sdssYy_"
                self.log.info("shortName = {:s}".format(shortName))
                Analysis(catalog, sdssYyCompare(), "SdssShape yy Moment Diff (%)", shortName,
                         self.config.analysis, flags=[col + "_flag"], prefix="first_",
                         goodKeys=["calib_psfUsed"], qMin=-0.5, qMax=1.5,
                         labeller=OverlapsStarGalaxyLabeller(),
                         ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                   camera=camera, ccdList=ccdList, hscRun=hscRun,
                                   matchRadius=matchRadius, zpLabel=zpLabel)

                shortName = "hsmTrace_"
                self.log.info("shortName = {:s}".format(shortName))
                Analysis(catalog, hsmTraceSizeCompare(), "HSM Trace Radius Diff (%)", shortName,
                         self.config.analysis, flags=[col + "_flag"], prefix="first_",
                         goodKeys=["calib_psfUsed"], qMin=-0.5, qMax=1.5,
                         labeller=OverlapsStarGalaxyLabeller(),
                         ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                   camera=camera, ccdList=ccdList, hscRun=hscRun,
                                   matchRadius=matchRadius, zpLabel=zpLabel)
                shortName = "hsmPsfTrace_"
                self.log.info("shortName = {:s}".format(shortName))
                Analysis(catalog, hsmPsfTraceSizeCompare(), "HSM PSF Trace Radius Diff (%)", shortName,
                         self.config.analysis, flags=[col + "_flag"], prefix="first_",
                         goodKeys=["calib_psfUsed"], qMin=-1.1, qMax=1.1,
                         labeller=OverlapsStarGalaxyLabeller(),
                         ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                   camera=camera, ccdList=ccdList, hscRun=hscRun,
                                   matchRadius=matchRadius, zpLabel=zpLabel)

    def plotApCorrs(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None, hscRun=None,
                    matchRadius=None, zpLabel=None, fluxToPlotList=None):
        if fluxToPlotList is None:
            fluxToPlotList = self.config.fluxToPlotList
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.02*self.unitScale}})
        for col in fluxToPlotList:
            if "first_" + col + "_apCorr" in catalog.schema and "second_" + col + "_apCorr" in catalog.schema:
                shortName = "diff_" + col + "_apCorr"
                self.log.info("shortName = {:s}".format(shortName))
                Analysis(catalog, ApCorrDiffCompare(col + "_apCorr"),
                         "Run Comparison: %s apCorr diff" % fluxToPlotString(col),
                         shortName, self.config.analysis,
                         prefix="first_", qMin=-0.025, qMax=0.025, flags=[col + "_flag_apCorr"],
                         errFunc=ApCorrDiffErr(col + "_apCorr"), labeller=OverlapsStarGalaxyLabeller(),
                         ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                   camera=camera, ccdList=ccdList, hscRun=hscRun, matchRadius=matchRadius,
                                   zpLabel=None)
