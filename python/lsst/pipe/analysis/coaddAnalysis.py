#!/usr/bin/env python

import os
import numpy as np
np.seterr(all="ignore")
from eups import Eups
eups = Eups()
import functools

from collections import defaultdict

from lsst.daf.persistence.butler import Butler
from lsst.pex.config import Config, Field, ConfigField, ListField, DictField, ConfigDictField
from lsst.pipe.base import CmdLineTask, ArgumentParser, TaskRunner, TaskError
from lsst.coadd.utils import TractDataIdContainer
from lsst.afw.table.catalogMatches import matchesToCatalog
from lsst.meas.astrom import AstrometryConfig, LoadAstrometryNetObjectsTask, LoadAstrometryNetObjectsConfig
from lsst.pipe.tasks.colorterms import ColortermLibrary
from .analysis import AnalysisConfig, Analysis
from .utils import *
from .plotUtils import *

import lsst.afw.coord as afwCoord
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable

__all__ = ["CoaddAnalysisConfig", "CoaddAnalysisRunner", "CoaddAnalysisTask", "CompareCoaddAnalysisConfig",
           "CompareCoaddAnalysisRunner", "CompareCoaddAnalysisTask"]


class CoaddAnalysisConfig(Config):
    coaddName = Field(dtype=str, default="deep", doc="Name for coadd")
    matchRadius = Field(dtype=float, default=0.5, doc="Matching radius (arcseconds)")
    colorterms = ConfigField(dtype=ColortermLibrary, doc="Library of color terms")
    photoCatName = Field(dtype=str, default="sdss", doc="Name of photometric reference catalog; "
                         "used to select a color term dict in colorterms.""Name for coadd")
    analysis = ConfigField(dtype=AnalysisConfig, doc="Analysis plotting options")
    analysisMatches = ConfigField(dtype=AnalysisConfig, doc="Analysis plotting options for matches")
    matchesMaxDistance = Field(dtype=float, default=0.15, doc="Maximum plotting distance for matches")
    externalCatalogs = ConfigDictField(keytype=str, itemtype=AstrometryConfig, default={},
                                       doc="Additional external catalogs for matching")
    refObjLoaderConfig = ConfigField(dtype=LoadAstrometryNetObjectsConfig,
                                     doc="Configuration for reference object loader")
    doPlotMags = Field(dtype=bool, default=True, doc="Plot magnitudes?")
    doPlotSizes = Field(dtype=bool, default=True, doc="Plot PSF sizes?")
    doPlotCentroids = Field(dtype=bool, default=True, doc="Plot centroids?")
    doBackoutApCorr = Field(dtype=bool, default=False, doc="Backout aperture corrections?")
    doAddAperFluxHsc = Field(dtype=bool, default=False,
                             doc="Add a field containing 12 pix circular aperture flux to HSC table?")
    doPlotStarGalaxy = Field(dtype=bool, default=True, doc="Plot star/galaxy?")
    doPlotOverlaps = Field(dtype=bool, default=True, doc="Plot overlaps?")
    doPlotMatches = Field(dtype=bool, default=True, doc="Plot matches?")
    doPlotCompareUnforced = Field(dtype=bool, default=True, doc="Plot difference between forced and unforced?")
    onlyReadStars = Field(dtype=bool, default=False, doc="Only read stars (to save memory)?")
    srcSchemaMap = DictField(keytype=str, itemtype=str, default=None, optional=True,
                             doc="Mapping between different stack (e.g. HSC vs. LSST) schema names")
    fluxToPlotList = ListField(dtype=str, default=["base_GaussianFlux", ],
                               doc="List of fluxes to plot: mag(flux)-mag(base_PsfFlux) vs mag(base_PsfFlux)")
    # "ext_photometryKron_KronFlux", "modelfit_Cmodel", "slot_CalibFlux"]:
    doApplyUberCal = Field(dtype=bool, default=True, doc="Apply meas_mosaic ubercal results to input?")
    doApplyCalexpZp = Field(dtype=bool, default=True,
                            doc="Apply FLUXMAG0 zeropoint to sources? Ignored if doApplyUberCal is True")

    def saveToStream(self, outfile, root="root"):
        """Required for loading colorterms from a Config outside the 'lsst' namespace"""
        print >> outfile, "import lsst.meas.photocal.colorterms"
        return Config.saveToStream(self, outfile, root)

    def setDefaults(self):
        Config.setDefaults(self)
        # self.externalCatalogs = {"sdss-dr9-fink-v5b": astrom}
        self.analysisMatches.magThreshold = 21.0  # External catalogs like PS1 & SDSS used smaller telescopes


class CoaddAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        kwargs["cosmos"] = parsedCmd.cosmos

        # Partition all inputs by tract,filter
        FilterRefsDict = functools.partial(defaultdict, list)  # Dict for filter-->dataRefs
        tractFilterRefs = defaultdict(FilterRefsDict)  # tract-->filter-->dataRefs
        for patchRef in sum(parsedCmd.id.refList, []):
            if patchRef.datasetExists("deepCoadd_meas"):
                tract = patchRef.dataId["tract"]
                filterName = patchRef.dataId["filter"]
                tractFilterRefs[tract][filterName].append(patchRef)

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

    def run(self, patchRefList, cosmos=None):
        dataId = patchRefList[0].dataId
        patchList = [dataRef.dataId["patch"] for dataRef in patchRefList]
        butler = patchRefList[0].getButler()
        skymap = butler.get("deepCoadd_skyMap", {"tract": dataRef.dataId["tract"]})

        filterName = dataId["filter"]
        filenamer = Filenamer(patchRefList[0].getButler(), self.outputDataset, patchRefList[0].dataId)
        if (self.config.doPlotMags or self.config.doPlotStarGalaxy or self.config.doPlotOverlaps or
            self.config.doPlotCompareUnforced or cosmos or self.config.externalCatalogs):
            ### catalog = catalog[catalog["deblend_nChild"] == 0].copy(True) # Don't care about blended objects
            forced = self.readCatalogs(patchRefList, "deepCoadd_forced_src")
            forced = self.calibrateCatalogs(forced)
            unforced = self.readCatalogs(patchRefList, "deepCoadd_meas")
            unforced = self.calibrateCatalogs(unforced)
            # catalog = joinCatalogs(meas, forced, prefix1="meas_", prefix2="forced_")

        # Check metadata to see if stack used was HSC
        metadata = butler.get("deepCoadd_md", patchRefList[0].dataId)
        # Set an alias map for differing src naming conventions of different stacks (if any)
        hscRun = checkHscStack(metadata)
        if hscRun is not None and self.config.srcSchemaMap is not None:
            aliasMap = forced.schema.getAliasMap()
            for lsstName, otherName in self.config.srcSchemaMap.iteritems():
                aliasMap.set(lsstName, otherName)

        if self.config.doPlotMags:
            self.plotMags(forced, filenamer, dataId, skymap=skymap, patchList=patchList, hscRun=hscRun,
                          zpLabel=self.zpLabel)
        if self.config.doPlotStarGalaxy:
            if "ext_shapeHSM_HsmSourceMoments_xx" in unforced.schema:
                self.plotStarGal(unforced, filenamer, dataId, skymap=skymap, patchList=patchList,
                                 hscRun=hscRun, zpLabel=self.zpLabel)
            else:
                self.log.warn("Cannot run plotStarGal: ext_shapeHSM_HsmSourceMoments_xx not in forced.schema")
        if cosmos:
            self.plotCosmos(forced, filenamer, cosmos, dataId)
        if self.config.doPlotCompareUnforced:
            self.plotCompareUnforced(forced, unforced, filenamer, dataId, skymap=skymap, patchList=patchList,
                                     hscRun=hscRun, zpLabel=self.zpLabel)
        if self.config.doPlotOverlaps:
            overlaps = self.overlaps(forced)
            self.plotOverlaps(overlaps, filenamer, dataId, skymap=skymap, patchList=patchList, hscRun=hscRun,
                              zpLabel=self.zpLabel)
        if self.config.doPlotMatches:
            matches = self.readSrcMatches(patchRefList, "deepCoadd_forced_src")
            self.plotMatches(matches, filterName, filenamer, dataId, skymap=skymap, patchList=patchList,
                             hscRun=hscRun, matchRadius=self.config.matchRadius, zpLabel=self.zpLabel)

        for cat in self.config.externalCatalogs:
            with andCatalog(cat):
                matches = self.matchCatalog(forced, filterName, self.config.externalCatalogs[cat])
                self.plotMatches(matches, filterName, filenamer, dataId, cat)

    def readCatalogs(self, patchRefList, dataset):
        catList = [patchRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS) for
                   patchRef in patchRefList if patchRef.datasetExists(dataset)]
        if len(catList) == 0:
            raise TaskError("No catalogs read: %s" % ([patchRef.dataId for patchRef in patchRefList]))
        if self.config.onlyReadStars and "base_ClassificationExtendedness_value" in catList[0].schema:
            catList = [cat[cat["base_ClassificationExtendedness_value"] < 0.5].copy(True) for cat in catList]
        return concatenateCatalogs(catList)

    def readSrcMatches(self, dataRefList, dataset):
        catList = []
        for dataRef in dataRefList:
            print "dataRef, dataset: ", dataRef.dataId, dataset
            if not dataRef.datasetExists(dataset):
                print "Dataset does not exist: ", dataRef.dataId, dataset
                continue
            butler = dataRef.getButler()
            if dataset.startswith("deepCoadd_"):
                metadata = butler.get("deepCoadd_md", dataRef.dataId)
            else:
                metadata = butler.get("calexp_md", dataRef.dataId)
            # Generate unnormalized match list (from normalized persisted one) with joinMatchListWithCatalog
            # (which requires a refObjLoader to be initialized).
            catalog = dataRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS)
            catalog = self.calibrateCatalogs(catalog)
            if dataset.startswith("deepCoadd_"):
                packedMatches = butler.get("deepCoadd_src" + "Match", dataRef.dataId)
            else:
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
            centroidStr = "base_SdssCentroid"
            if centroidStr not in schema:
                centroidStr = "base_TransformedCentroid"
            matches[0].second.table.defineCentroid(centroidStr)
            src.table.defineCentroid(centroidStr)

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

    def calibrateCatalogs(self, catalog):
        self.zpLabel = "common (" + str(self.config.analysis.zp) + ")"
        calibrated = calibrateCoaddSourceCatalog(catalog, self.config.analysis.zp)
        return calibrated

    def plotMags(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None, skymap=None,
                 patchList=None, hscRun=None, matchRadius=None, zpLabel=None):
        enforcer = Enforcer(requireLess={"star": {"stdev": 0.02}})
        for col in self.config.fluxToPlotList:
        # ["base_GaussianFlux", ]: # "ext_photometryKron_KronFlux", "modelfit_Cmodel", "slot_CalibFlux"]:
            if col + "_flux" in catalog.schema:
                self.AnalysisClass(catalog, MagDiff(col + "_flux", "base_PsfFlux_flux"), "Mag(%s) - PSFMag"
                                   % col, "mag_" + col, self.config.analysis, flags=[col + "_flag"],
                                   labeller=StarGalaxyLabeller(),
                                   ).plotAll(dataId, filenamer, self.log, enforcer, butler=butler,
                                             camera=camera, ccdList=ccdList, skymap=skymap,
                                             patchList=patchList, hscRun=hscRun,
                                             matchRadius=matchRadius, zpLabel=zpLabel)

    def plotSizes(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None, skymap=None,
                  patchList=None, hscRun=None, matchRadius=None, zpLabel=None):
        enforcer = None
        for col in ["base_PsfFlux", ]:
            if col + "_flux" in catalog.schema:
                self.AnalysisClass(catalog, psfSdssTraceSizeDiff(),
                                   "SdssShape Trace (psfUsed - PSFmodel)/PSFmodel", "trace_",
                                   self.config.analysis, flags=[col + "_flag"],
                                   goodKeys=["calib_psfUsed"], qMin=-0.04, qMax=0.04,
                                   labeller=StarGalaxyLabeller(),
                                   ).plotAll(dataId, filenamer, self.log, enforcer, butler=butler,
                                             camera=camera, ccdList=ccdList, skymap=skymap,
                                             patchList=patchList, hscRun=hscRun,
                                             matchRadius=matchRadius, zpLabel=zpLabel)
                self.AnalysisClass(catalog, psfHsmTraceSizeDiff(),
                                   "HSM Trace (psfUsed - PSFmodel)/PSFmodel", "hsmTrace_",
                                   self.config.analysis, flags=[col + "_flag"],
                                   goodKeys=["calib_psfUsed"], qMin=-0.04, qMax=0.04,
                                   labeller=StarGalaxyLabeller(),
                                   ).plotAll(dataId, filenamer, self.log, enforcer, butler=butler,
                                             camera=camera, ccdList=ccdList, skymap=skymap,
                                             patchList=patchList, hscRun=hscRun,
                                             matchRadius=matchRadius, zpLabel=zpLabel)

    def plotCentroidXY(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None, skymap=None,
                       patchList=None, hscRun=None, matchRadius=None, zpLabel=None):
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.02}})
        for col in ["base_SdssCentroid_x", "base_SdssCentroid_y"]:
            if col in catalog.schema:
                self.AnalysisClass(catalog, catalog[col], "(%s)" % col, col, self.config.analysis,
                                   flags=["base_SdssCentroid_flag", "base_TransformedCentroid_flag"],
                                   labeller=StarGalaxyLabeller(),
                                   ).plotFP(dataId, filenamer, self.log, enforcer,
                                            camera=camera, ccdList=ccdList, hscRun=hscRun,
                                            matchRadius=matchRadius, zpLabel=zpLabel)

    def plotStarGal(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None, skymap=None,
                    patchList=None, hscRun=None, matchRadius=None, zpLabel=None):
        enforcer = None
        self.AnalysisClass(catalog, deconvMomStarGal, "pStar", "pStar", self.config.analysis,
                           qMin=-0.1, qMax=1.39, labeller=StarGalaxyLabeller()
                           ).plotAll(dataId, filenamer, self.log, enforcer, butler=butler, camera=camera,
                                     ccdList=ccdList, skymap=skymap, patchList=patchList, hscRun=hscRun,
                                     matchRadius=matchRadius, zpLabel=zpLabel)
        self.AnalysisClass(catalog, deconvMom, "Deconvolved moments (unforced)", "deconvMom",
                           self.config.analysis, qMin=-1.0, qMax=3.0, labeller=StarGalaxyLabeller()
                           ).plotAll(dataId, filenamer, self.log,
                                     Enforcer(requireLess={"star": {"stdev": 0.2}}), butler=butler,
                                     camera=camera, ccdList=ccdList, skymap=skymap, patchList=patchList,
                                     hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel)

    def plotCompareUnforced(self, forced, unforced, filenamer, dataId, butler=None, camera=None, ccdList=None,
                            skymap=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None):
        enforcer = None
        catalog = joinMatches(afwTable.matchRaDec(forced, unforced,
                                                  self.config.matchRadius*afwGeom.arcseconds),
                              "forced_", "unforced_")
        catalog.writeFits(dataId["filter"] + ".fits")
        for col in self.config.fluxToPlotList:
            # ["base_PsfFlux", "base_GaussianFlux", "slot_CalibFlux", "ext_photometryKron_KronFlux",
            # "modelfit_Cmodel", "modelfit_Cmodel_exp_flux", "modelfit_Cmodel_dev_flux"]:
            if "forced_" + col in catalog.schema:
                self.AnalysisClass(catalog, MagDiff("forced_" + col, "unforced_" + col),
                                   "Forced - Unforced mag difference (%s)" % col, "forced_" + col,
                                   self.config.analysis, prefix="unforced_", flags=[col + "_flags"],
                                   labeller=OverlapsStarGalaxyLabeller("forced_", "unforced_"),
                                   ).plotAll(dataId, filenamer, self.log, enforcer, butler=butler,
                                             camera=camera, ccdList=ccdList, skymap=skymap,
                                             patchList=patchList, hscRun=hscRun,
                                             matchRadius=matchRadius, zpLabel=zpLabel)

    def overlaps(self, catalog):
        matches = afwTable.matchRaDec(catalog, self.config.matchRadius*afwGeom.arcseconds, False)
        return joinMatches(matches, "first_", "second_")

    def plotOverlaps(self, overlaps, filenamer, dataId, butler=None, camera=None, ccdList=None,
                     skymap=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None):
        magEnforcer = Enforcer(requireLess={"star": {"stdev": 0.003}})
        for col in self.config.fluxToPlotList:
            # ["base_PsfFlux", "base_GaussianFlux", "ext_photometryKron_KronFlux", "modelfit_Cmodel"]:
            if "first_" + col + "_flux" in overlaps.schema:
                self.AnalysisClass(overlaps, MagDiff("first_" + col + "_flux", "second_" + col + "_flux"),
                                   "Overlap mag difference (%s)" % col, "overlap_" + col,
                                   self.config.analysis,
                                   prefix="first_", flags=[col + "_flag"],
                                   labeller=OverlapsStarGalaxyLabeller(),
                                   ).plotAll(dataId, filenamer, self.log, magEnforcer, butler=butler,
                                             camera=camera, ccdList=ccdList, skymap=skymap,
                                             patchList=patchList, hscRun=hscRun, zpLabel=zpLabel)

        distEnforcer = Enforcer(requireLess={"star": {"stdev": 0.005}})
        self.AnalysisClass(overlaps, lambda cat: cat["distance"]*(1.0*afwGeom.radians).asArcseconds(),
                           "Distance (arcsec)", "overlap_distance", self.config.analysis, prefix="first_",
                           qMin=0.0, qMax=0.15, labeller=OverlapsStarGalaxyLabeller(),
                           ).plotAll(dataId, filenamer, self.log, distEnforcer, forcedMean=0.0,
                                     butler=butler, camera=camera, ccdList=ccdList, skymap=skymap,
                                     patchList=patchList, hscRun=hscRun, zpLabel=zpLabel)

    def plotMatches(self, matches, filterName, filenamer, dataId, description="matches", butler=None,
                    camera=None, ccdList=None, skymap=None, patchList=None, hscRun=None, matchRadius=None,
                    zpLabel=None):
        ct = self.config.colorterms.getColorterm(filterName, self.config.photoCatName)
        if "src_calib_psfUsed" in matches.schema:
            self.AnalysisClass(matches, MagDiffMatches("base_PsfFlux_flux", ct, zp=0.0),
                               "MagPsf(unforced) - ref (calib_psfUsed)",
                               description + "_mag_calib_psfUsed", self.config.analysisMatches, prefix="src_",
                               goodKeys=["calib_psfUsed"], qMin=-0.05, qMax=0.05,
                               labeller=MatchesStarGalaxyLabeller(),
                               ).plotAll(dataId, filenamer, self.log,
                                         Enforcer(requireLess={"star": {"stdev": 0.030}}), butler=butler,
                                         camera=camera, ccdList=ccdList, skymap=skymap, patchList=patchList,
                                         hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel)

        self.AnalysisClass(matches, MagDiffMatches("base_PsfFlux_flux", ct, zp=0.0), "MagPsf(unforced) - ref",
                           description + "_mag", self.config.analysisMatches, prefix="src_",
                           qMin=-0.05, qMax=0.05, labeller=MatchesStarGalaxyLabeller(),
                           ).plotAll(dataId, filenamer, self.log,
                                     Enforcer(requireLess={"star": {"stdev": 0.030}}), butler=butler,
                                     camera=camera, ccdList=ccdList, skymap=skymap, patchList=patchList,
                                     hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel)
        self.AnalysisClass(matches, lambda cat: cat["distance"]*(1.0*afwGeom.radians).asArcseconds(),
                           "Distance (arcsec)", description + "_distance", self.config.analysisMatches,
                           prefix="src_", qMin=-0.02*self.config.matchesMaxDistance,
                           qMax=self.config.matchesMaxDistance, labeller=MatchesStarGalaxyLabeller()
                           ).plotAll(dataId, filenamer, self.log,
                                     Enforcer(requireLess={"star": {"stdev": 0.050}}), forcedMean=0.0,
                                     butler=butler, camera=camera, ccdList=ccdList, skymap=skymap,
                                     patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                     zpLabel=zpLabel)
        self.AnalysisClass(matches, AstrometryDiff("src_coord_ra", "ref_coord_ra", "ref_coord_dec"),
                           "dRA*cos(Dec) (arcsec)", description + "_ra", self.config.analysisMatches,
                           prefix="src_", qMin=-self.config.matchesMaxDistance,
                           qMax=self.config.matchesMaxDistance, labeller=MatchesStarGalaxyLabeller(),
                           ).plotAll(dataId, filenamer, self.log,
                                     Enforcer(requireLess={"star": {"stdev": 0.050}}),
                                     butler=butler, camera=camera, ccdList=ccdList, skymap=skymap,
                                     patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                     zpLabel=zpLabel)
        self.AnalysisClass(matches, AstrometryDiff("src_coord_dec", "ref_coord_dec"),
                           "dDec (arcsec)", description + "_dec", self.config.analysisMatches, prefix="src_",
                           qMin=-self.config.matchesMaxDistance, qMax=self.config.matchesMaxDistance,
                           labeller=MatchesStarGalaxyLabeller(),
                           ).plotAll(dataId, filenamer, self.log,
                                     Enforcer(requireLess={"star": {"stdev": 0.050}}),
                                     butler=butler, camera=camera, ccdList=ccdList, skymap=skymap,
                                     patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                     zpLabel=zpLabel)

    def plotCosmos(self, catalog, filenamer, cosmos, dataId):
        labeller = CosmosLabeller(cosmos, self.config.matchRadius*afwGeom.arcseconds)
        self.AnalysisClass(catalog, deconvMom, "Deconvolved moments", "cosmos", self.config.analysis,
                           qMin=-1.0, qMax=6.0, labeller=labeller,
                           ).plotAll(dataId, filenamer, self.log,
                                     Enforcer(requireLess={"star": {"stdev": 0.2}}))

    def matchCatalog(self, catalog, filterName, astrometryConfig):
        refObjLoader = LoadAstrometryNetObjectsTask(self.config.refObjLoaderConfig)
        average = sum((afwGeom.Extent3D(src.getCoord().getVector()) for src in catalog),
                      afwGeom.Extent3D(0, 0, 0))/len(catalog)
        center = afwCoord.IcrsCoord(afwGeom.Point3D(average))
        radius = max(center.angularSeparation(src.getCoord()) for src in catalog)
        filterName = afwImage.Filter(afwImage.Filter(filterName).getId()).getName()  # Get primary name
        refs = refObjLoader.loadSkyCircle(center, radius, filterName).refCat
        matches = afwTable.matchRaDec(refs, catalog, self.config.matchRadius*afwGeom.arcseconds)
        matches = matchJanskyToDn(matches)
        return joinMatches(matches, "ref_", "src_")

    def _getConfigName(self):
        return None
    def _getMetadataName(self):
        return None
    def _getEupsVersionsName(self):
        return None


class CompareCoaddAnalysisConfig(Config):
    coaddName = Field(dtype=str, default="deep", doc="Name for coadd")
    matchRadius = Field(dtype=float, default=0.2, doc="Matching radius (arcseconds)")
    analysis = ConfigField(dtype=AnalysisConfig, doc="Analysis plotting options")
    doPlotMags = Field(dtype=bool, default=True, doc="Plot magnitudes?")
    doPlotSizes = Field(dtype=bool, default=False, doc="Plot PSF sizes?")
    doPlotCentroids = Field(dtype=bool, default=True, doc="Plot centroids?")
    doApCorrs = Field(dtype=bool, default=True, doc="Plot aperture corrections?")
    doBackoutApCorr = Field(dtype=bool, default=False, doc="Backout aperture corrections?")
    sysErrMags = Field(dtype=float, default=0.015, doc="Systematic error in magnitudes")
    sysErrCentroids = Field(dtype=float, default=0.15, doc="Systematic error in centroids (pixels)")
    srcSchemaMap = DictField(keytype=str, itemtype=str, default=None, optional=True,
                             doc="Mapping between different stack (e.g. HSC vs. LSST) schema names")
    doAddAperFluxHsc = Field(dtype=bool, default=False,
                             doc="Add a field containing 12 pix circular aperture flux to HSC table?")
    fluxToPlotList = ListField(dtype=str, default=["base_PsfFlux", "base_GaussianFlux"],
                               doc="List of fluxes to plot: mag(flux)-mag(base_PsfFlux) vs mag(base_PsfFlux)")
    # "ext_photometryKron_KronFlux", "modelfit_Cmodel", "slot_CalibFlux"]:
    doApplyUberCal = Field(dtype=bool, default=True, doc="Apply meas_mosaic ubercal results to input?")
    doApplyCalexpZp = Field(dtype=bool, default=True,
                            doc="Apply FLUXMAG0 zeropoint to sources? Ignored if doApplyUberCal is True")


class CompareCoaddAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        parentDir = parsedCmd.input
        while os.path.exists(os.path.join(parentDir, "_parent")):
            parentDir = os.path.realpath(os.path.join(parentDir, "_parent"))
        butler2 = Butler(root=os.path.join(parentDir, "rerun", parsedCmd.rerun2), calibRoot=parsedCmd.calib)
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

    def run(self, patchRefList1, patchRefList2):
        dataId = patchRefList1[0].dataId
        filenamer = Filenamer(patchRefList1[0].getButler(), "plotCompare", patchRefList1[0].dataId)
        catalog1 = self.readCatalogs(patchRefList1, self.config.coaddName + "Coadd_forced_src")
        catalog2 = self.readCatalogs(patchRefList2, self.config.coaddName + "Coadd_forced_src")
        catalog = self.matchCatalogs(catalog1, catalog2)
        if self.config.doPlotMags:
            self.plotMags(catalog, filenamer, dataId)
        if self.config.doPlotCentroids:
            self.plotCentroids(catalog, filenamer, dataId)

    def readCatalogs(self, patchRefList, dataset):
        catList = [patchRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS) for
                   patchRef in patchRefList if patchRef.datasetExists(dataset)]
        if len(catList) == 0:
            raise TaskError("No catalogs read: %s" % ([patchRefList[0].dataId for dataRef in patchRefList]))
        return concatenateCatalogs(catList)

    def matchCatalogs(self, catalog1, catalog2):
        matches = afwTable.matchRaDec(catalog1, catalog2, self.config.matchRadius*afwGeom.arcseconds)
        if len(matches) == 0:
            raise TaskError("No matches found")
        return joinMatches(matches, "first_", "second_")

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

    def plotCentroids(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None, hscRun=None,
                      matchRadius=None, zpLabel=None):
        distEnforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.005}})
        Analysis(catalog, CentroidDiff("x"), "Run Comparison: x offset (arcsec)", "diff_x",
                 self.config.analysis, prefix="first_", qMin=-0.3, qMax=0.3, errFunc=CentroidDiffErr("x"),
                 labeller=OverlapsStarGalaxyLabeller(),
                 ).plotAll(dataId, filenamer, self.log, distEnforcer, butler=butler, camera=camera,
                           ccdList=ccdList, hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel)
        Analysis(catalog, CentroidDiff("y"), "Run Comparison: y offset (arcsec)", "diff_y",
                 self.config.analysis, prefix="first_", qMin=-0.1, qMax=0.1, errFunc=CentroidDiffErr("y"),
                 labeller=OverlapsStarGalaxyLabeller(),
                 ).plotAll(dataId, filenamer, self.log, distEnforcer, butler=butler, camera=camera,
                           ccdList=ccdList, hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel)

    def _getConfigName(self):
        return None
    def _getMetadataName(self):
        return None
    def _getEupsVersionsName(self):
        return None
