#!/usr/bin/env python

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all="ignore")
from eups import Eups
eups = Eups()

from collections import defaultdict

from lsst.daf.persistence.butler import Butler
from lsst.pipe.base import ArgumentParser, TaskRunner, TaskError
from lsst.meas.base.forcedPhotCcd import PerTractCcdDataIdContainer
from lsst.afw.table.catalogMatches import matchesToCatalog
from lsst.meas.astrom import LoadAstrometryNetObjectsTask
from .analysis import Analysis
from .coaddAnalysis import (CoaddAnalysisConfig, CoaddAnalysisTask, CompareCoaddAnalysisConfig,
                            CompareCoaddAnalysisTask)
from .utils import *
from .plotUtils import *

import lsst.afw.table as afwTable


class CcdAnalysis(Analysis):
    def plotAll(self, dataId, filenamer, log, enforcer=None, forcedMean=None, butler=None, camera=None,
                ccdList=None, skymap=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None):
        stats = self.stats(forcedMean=forcedMean)
        self.plotCcd(filenamer(dataId, description=self.shortName, style="ccd"), stats=stats,
                     hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel)
        if self.config.doPlotFP:
            self.plotFocalPlane(filenamer(dataId, description=self.shortName, style="fpa"), stats=stats,
                                camera=camera, ccdList=ccdList, hscRun=hscRun, matchRadius=matchRadius,
                                zpLabel=zpLabel)

        return Analysis.plotAll(self, dataId, filenamer, log, enforcer=enforcer, forcedMean=forcedMean,
                                butler=butler, camera=camera, ccdList=ccdList, hscRun=hscRun,
                                matchRadius=matchRadius, zpLabel=zpLabel)

    def plotFP(self, dataId, filenamer, log, enforcer=None, forcedMean=None, camera=None, ccdList=None,
               hscRun=None, matchRadius=None, zpLabel=None):
        stats = self.stats(forcedMean=forcedMean)
        self.plotFocalPlane(filenamer(dataId, description=self.shortName, style="fpa"), stats=stats,
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
            print "Only one CCD (%d) to analyze: setting vMin (%d), vMax (%d)" % (ccd.min(), vMin, vMax)
        good = (self.mag < self.config.magThreshold if self.config.magThreshold > 0 else
                np.ones(len(self.mag), dtype=bool))
        fig, axes = plt.subplots(2, 1)
        axes[0].axhline(0, linestyle="--", color="0.6")
        axes[1].axhline(0, linestyle="--", color="0.6")
        for name, data in self.data.iteritems():
            if not data.plot:
                continue
            if len(data.mag) == 0:
                continue
            selection = data.selection & good
            quantity = data.quantity[good[data.selection]]
            kwargs = {"s": 4, "marker": "o", "lw": 0, "alpha": 0.5, "cmap": cmap, "vmin": vMin, "vmax": vMax}
            axes[0].scatter(xx[selection], quantity, c=ccd[selection], **kwargs)
            axes[1].scatter(yy[selection], quantity, c=ccd[selection], **kwargs)

        axes[0].set_xlabel("x_ccd", labelpad=-1)
        axes[1].set_xlabel("y_ccd")
        fig.text(0.02, 0.5, self.quantityName, ha="center", va="center", rotation="vertical")
        if stats is not None:
            annotateAxes(plt, axes[0], stats, "star", self.config.magThreshold, x0=0.03, yOff=0.07,
                         hscRun=hscRun, matchRadius=matchRadius)
            annotateAxes(plt, axes[1], stats, "star", self.config.magThreshold, x0=0.03, yOff=0.07,
                         hscRun=hscRun, matchRadius=matchRadius)
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
                       hscRun=None, matchRadius=None, zpLabel=None):
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
        fig, axes = plt.subplots(1, 1, subplot_kw=dict(axisbg="0.7"))
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
        labelVisit(filename, plt, axes, 0.5, 1.07)
        if zpLabel is not None:
            labelZp(zpLabel, plt, axes, 0.08, -0.11, color="green")
        fig.savefig(filename)
        plt.close(fig)


class VisitAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
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
        return parser

    def run(self, dataRefList):
        self.log.info("dataRefList size: %d" % len(dataRefList))
        ccdList = [dataRef.dataId["ccd"] for dataRef in dataRefList]
        butler = dataRefList[0].getButler()
        camera = butler.get("camera")
        dataId = dataRefList[0].dataId
        self.log.info("dataId: %s" % (dataId,))
        filterName = dataId["filter"]
        filenamer = Filenamer(butler, "plotVisit", dataRefList[0].dataId)
        if (self.config.doPlotMags or self.config.doPlotSizes or self.config.doPlotStarGalaxy or
            self.config.doPlotOverlaps or cosmos or self.config.externalCatalogs):
            catalog = self.readCatalogs(dataRefList, "src")
        calexp = None
        if (self.config.doPlotSizes):
            calexp = butler.get("calexp", dataId)
        # Check metadata to see if stack used was HSC
        metadata = butler.get("calexp_md", dataRefList[0].dataId)
        # Set an alias map for differing src naming conventions of different stacks (if any)
        hscRun = checkHscStack(metadata)
        if hscRun is not None and self.config.doAddAperFluxHsc:
            print "HSC run: adding aperture flux to schema..."
            catalog = addApertureFluxesHSC(catalog, prefix="")
        if hscRun is not None and self.config.srcSchemaMap is not None:
            aliasMap = catalog.schema.getAliasMap()
            for lsstName, otherName in self.config.srcSchemaMap.iteritems():
                aliasMap.set(lsstName, otherName)
        if self.config.doPlotSizes:
            if "base_SdssShape_psf_xx" in catalog.schema:
                self.plotSizes(catalog, filenamer, dataId, butler=butler, camera=camera, ccdList=ccdList,
                               hscRun=hscRun, zpLabel=self.zpLabel)
            else:
                self.log.warn("Cannot run plotSizes: base_SdssShape_psf_xx not in catalog.schema")
        if self.config.doPlotMags:
            self.plotMags(catalog, filenamer, dataId, butler=butler, camera=camera, ccdList=ccdList,
                          hscRun=hscRun, zpLabel=self.zpLabel)
        if self.config.doPlotCentroids:
            self.plotCentroidXY(catalog, filenamer, dataId, butler=butler, camera=camera, ccdList=ccdList,
                                hscRun=hscRun, zpLabel=self.zpLabel)
        if self.config.doPlotStarGalaxy:
            if "ext_shapeHSM_HsmSourceMoments_xx" in catalog.schema:
                self.plotStarGal(catalog, filenamer, dataId, butler=butler, camera=camera, ccdList=ccdList,
                                 hscRun=hscRun, zpLabel=self.zpLabel)
            else:
                self.log.warn("Cannot run plotStarGal: ext_shapeHSM_HsmSourceMoments_xx not in catalog.schema")
        if self.config.doPlotMatches:
            matches = self.readSrcMatches(dataRefList, "src")
            self.plotMatches(matches, filterName, filenamer, dataId, butler=butler, camera=camera,
                             ccdList=ccdList, hscRun=hscRun, matchRadius=self.config.matchRadius,
                             zpLabel=self.zpLabel)

        for cat in self.config.externalCatalogs:
            if self.config.photoCatName not in cat:
                with andCatalog(cat):
                    matches = self.matchCatalog(catalog, filterName, self.config.externalCatalogs[cat])
                    self.plotMatches(matches, filterName, filenamer, dataId, cat, butler=butler,
                                     camera=camera, ccdList=ccdList, hscRun=hscRun,
                                     matchRadius=self.config.matchRadius, zpLabel=self.zpLabel)

    def readCatalogs(self, dataRefList, dataset):
        catList = []
        for dataRef in dataRefList:
            if not dataRef.datasetExists(dataset):
                continue
            catalog = dataRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS)
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

            calibrated = self.calibrateCatalogs(dataRef, catalog, metadata)
            catList.append(calibrated)

        if len(catList) == 0:
            raise TaskError("No catalogs read: %s" % ([dataRef.dataId for dataRef in dataRefList]))

        return concatenateCatalogs(catList)

    def readSrcMatches(self, dataRefList, dataset):
        catList = []
        for dataRef in dataRefList:
            if not dataRef.datasetExists(dataset):
                continue
            butler = dataRef.getButler()
            metadata = butler.get("calexp_md", dataRef.dataId)
            # Generate unnormalized match list (from normalized persisted one) with joinMatchListWithCatalog
            # (which requires a refObjLoader to be initialized).
            catalog = dataRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS)
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
            refObjLoader = LoadAstrometryNetObjectsTask(self.config.refObjLoaderConfig)
            matches = refObjLoader.joinMatchListWithCatalog(packedMatches, catalog)
            # LSST reads in a_net catalogs with flux in "janskys", so must convert back to DN
            matches = matchJanskyToDn(matches)
            if checkHscStack(metadata) is not None and self.config.doAddAperFluxHsc:
                addApertureFluxesHSC(matches, prefix="second_")

            if len(matches) == 0:
                self.log.warn("No matches for %s" % (dataRef.dataId,))
                continue

            # Set the aliap map for the matches sources (i.e. the .second attribute schema for each match)
            if self.config.srcSchemaMap is not None and checkHscStack(metadata) is not None:
                for mm in matches:
                    aliasMap = mm.second.schema.getAliasMap()
                    for lsstName, otherName in self.config.srcSchemaMap.iteritems():
                        aliasMap.set(lsstName, otherName)

            schema = matches[0].second.schema
            src = afwTable.SourceCatalog(schema)
            src.reserve(len(catalog))
            for mm in matches:
                src.append(mm.second)
            matches[0].second.table.defineCentroid("base_SdssCentroid")
            src.table.defineCentroid("base_SdssCentroid")

            for mm, ss in zip(matches, src):
                mm.second = ss

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
            catList.append(catalog)

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
            if self.config.doApplyCalexpZp:
                # Scale fluxes to measured zeropoint
                self.zp = 2.5*np.log10(metadata.get("FLUXMAG0"))
                if self.zpLabel is None:
                    self.log.info("Using 2.5*log10(FLUXMAG0) = %.4f from FITS header for zeropoint" % self.zp)
                self.zpLabel = "FLUXMAG0"
            else:
                # Scale fluxes to common zeropoint
                self.zp = 33.0
                if self.zpLabel is None:
                    self.log.info("Using common value of %.4f for zeropoint" % (self.zp))
                self.zpLabel = "common (" + str(self.zp) + ")"
            calibrated = calibrateSourceCatalog(catalog, self.zp)
        return calibrated


class CompareVisitAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        parentDir = parsedCmd.input
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
    _DefaultName = "compareVisitAnalysis"
    ConfigClass = CompareCoaddAnalysisConfig
    RunnerClass = CompareVisitAnalysisRunner

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_argument("--rerun2", required=True, help="Second rerun, for comparison")
        parser.add_id_argument("--id", "src", help="data ID with raw CCD keys, "
                               "e.g. --id visit=12345 ccd=6^8..11", ContainerClass=PerTractCcdDataIdContainer)
        return parser

    def run(self, dataRefList1, dataRefList2):
        dataId = dataRefList1[0].dataId
        ccdList1 = [dataRef.dataId["ccd"] for dataRef in dataRefList1]
        butler1 = dataRefList1[0].getButler()
        metadata1 = butler1.get("calexp_md", dataRefList1[0].dataId)
        camera1 = butler1.get("camera")
        filenamer = Filenamer(dataRefList1[0].getButler(), "plotCompareVisit", dataId)
        butler2 = dataRefList2[0].getButler()
        metadata2 = butler2.get("calexp_md", dataRefList2[0].dataId)
        # Check metadata to see if stack used was HSC
        hscRun = checkHscStack(metadata2)
        hscRun1 = checkHscStack(metadata1)
        # If comparing LSST vs HSC run, need to rotate the LSST x, y coordinates for rotated CCDs
        catalog1 = self.readCatalogs(dataRefList1, "src", hscRun=hscRun, hscRun1=hscRun1)
        catalog2 = self.readCatalogs(dataRefList2, "src")

        if hscRun is not None and self.config.doAddAperFluxHsc:
            print "HSC run: adding aperture flux to schema..."
            catalog2 = addApertureFluxesHSC(catalog2, prefix="")

        if hscRun1 is not None and self.config.doAddAperFluxHsc:
            print "HSC run: adding aperture flux to schema..."
            catalog1 = addApertureFluxesHSC(catalog1, prefix="")

        self.log.info("\nNumber of sources in catalogs: first = {0:d} and second = {1:d}".format(
                len(catalog1), len(catalog2)))
        catalog = self.matchCatalogs(catalog1, catalog2)
        self.log.info("Number of matches (maxDist = {0:.2f} arcsec) = {1:d}".format(
                self.config.matchRadius, len(catalog)))

        # Set an alias map for differing src naming conventions of different stacks (if any)
        if self.config.srcSchemaMap is not None and hscRun is not None:
            aliasMap = catalog.schema.getAliasMap()
            for lsstName, otherName in self.config.srcSchemaMap.iteritems():
                aliasMap.set("second_" + lsstName, "second_" + otherName)
        if self.config.srcSchemaMap is not None and hscRun1 is not None:
            aliasMap = catalog.schema.getAliasMap()
            for lsstName, otherName in self.config.srcSchemaMap.iteritems():
                aliasMap.set("first_" + lsstName, "first_" + otherName)

        if self.config.doBackoutApCorr:
            catalog = backoutApCorr(catalog)

        if self.config.doPlotMags:
            self.plotMags(catalog, filenamer, dataId, butler=butler1, camera=camera1, ccdList=ccdList1,
                          hscRun=hscRun, matchRadius=self.config.matchRadius, zpLabel=self.zpLabel)
        if self.config.doPlotSizes:
            if "base_SdssShape_psf_xx" in catalog.schema:
                self.plotSizes(catalog, filenamer, dataId, butler=butler1, camera=camera1, ccdList=ccdList1,
                               hscRun=hscRun, matchRadius=self.config.matchRadius, zpLabel=self.zpLabel)
            else:
                self.log.warn("Cannot run plotSizes: base_SdssShape_psf_xx not in catalog.schema")
        if self.config.doApCorrs:
            self.plotApCorrs(catalog, filenamer, dataId, butler=butler1, camera=camera1, ccdList=ccdList1,
                             hscRun=hscRun, matchRadius=self.config.matchRadius, zpLabel=self.zpLabel)
        if self.config.doPlotCentroids:
            self.plotCentroids(catalog, filenamer, dataId, butler=butler1, camera=camera1, ccdList=ccdList1,
                               hscRun=hscRun, matchRadius=self.config.matchRadius, zpLabel=self.zpLabel)

    def readCatalogs(self, dataRefList, dataset, hscRun=None, hscRun1=None):
        catList = []
        for dataRef in dataRefList:
            if not dataRef.datasetExists(dataset):
                continue
            srcCat = dataRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS)
            butler = dataRef.getButler()
            metadata = butler.get("calexp_md", dataRef.dataId)
            calexp = butler.get("calexp", dataRef.dataId)
            nQuarter = calexp.getDetector().getOrientation().getNQuarter()
            calibrated = self.calibrateCatalogs(dataRef, srcCat, metadata)
            if hscRun is not None and hscRun1 is None:
                if nQuarter%4 != 0:
                    calibrated = rotatePixelCoords(calibrated, calexp.getWidth(), calexp.getHeight(), nQuarter)
            catList.append(calibrated)

        if len(catList) == 0:
            raise TaskError("No catalogs read: %s" % ([dataRefList[0].dataId for dataRef in dataRefList]))
        return concatenateCatalogs(catList)

    def plotMags(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None, hscRun=None,
                 matchRadius=None, zpLabel=None, fluxToPlotList=None, postFix=""):
        if fluxToPlotList is None:
            fluxToPlotList = self.config.fluxToPlotList
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.02}})
        for col in fluxToPlotList:
            # ["base_CircularApertureFlux_12_0"]:
            if "first_" + col + "_flux" in catalog.schema and "second_" + col + "_flux" in catalog.schema:
                if "CircularAperture" in col:
                    zpLabel = None
                Analysis(catalog, MagDiffCompare(col + "_flux"),
                         "Run Comparison: Mag difference (%s)" % col, "diff_" + col, self.config.analysis,
                         prefix="first_", qMin=-0.05, qMax=0.05, flags=[col + "_flag"],
                         errFunc=MagDiffErr(col + "_flux"), labeller=OverlapsStarGalaxyLabeller(),
                         ).plotAll(dataId, filenamer, self.log, enforcer, butler=butler, camera=camera,
                                   ccdList=ccdList, hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel)

    def plotSizes(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None, hscRun=None,
                 matchRadius=None, zpLabel=None):
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.02}})
        for col in ["base_PsfFlux"]:
            if "first_" + col + "_flux" in catalog.schema and "second_" + col + "_flux" in catalog.schema:
                Analysis(catalog, psfSdssTraceSizeDiff(),
                         "SdssShape Trace Radius Diff (psfUsed - PSF model)/(PSF model)", "trace_",
                         self.config.analysis, flags=[col + "_flag"], prefix="first_",
                         goodKeys=["calib_psfUsed"], qMin=-0.04, qMax=0.04,
                         labeller=OverlapsStarGalaxyLabeller(),
                         ).plotAll(dataId, filenamer, self.log, enforcer, butler=butler,
                                   camera=camera, ccdList=ccdList, hscRun=hscRun,
                                   matchRadius=matchRadius, zpLabel=zpLabel)
                Analysis(catalog, psfHsmTraceSizeDiff(),
                         "HSM Trace Radius Diff (psfUsed - PSF model)/(PSF model)", "hsmTrace_",
                         self.config.analysis, flags=[col + "_flag"], prefix="first_",
                         goodKeys=["calib_psfUsed"], qMin=-0.04, qMax=0.04,
                         labeller=OverlapsStarGalaxyLabeller(),
                         ).plotAll(dataId, filenamer, self.log, enforcer, butler=butler,
                                   camera=camera, ccdList=ccdList, hscRun=hscRun,
                                   matchRadius=matchRadius, zpLabel=zpLabel)

    def plotApCorrs(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None, hscRun=None,
                    matchRadius=None, zpLabel=None, fluxToPlotList=None):
        if fluxToPlotList is None:
            fluxToPlotList = self.config.fluxToPlotList
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.02}})
        for col in fluxToPlotList:
            if "first_" + col + "_apCorr" in catalog.schema and "second_" + col + "_apCorr" in catalog.schema:
                Analysis(catalog, ApCorrDiffCompare(col + "_apCorr"),
                         "Run Comparison: apCorr difference (%s)" % col, "diff_" + col + "_apCorr",
                         self.config.analysis,
                         prefix="first_", qMin=-0.025, qMax=0.025, flags=[col + "_flag_apCorr"],
                         errFunc=ApCorrDiffErr(col + "_apCorr"), labeller=OverlapsStarGalaxyLabeller(),
                         ).plotAll(dataId, filenamer, self.log, enforcer, butler=butler, camera=camera,
                                   ccdList=ccdList, hscRun=hscRun, matchRadius=matchRadius, zpLabel=None)
