#!/usr/bin/env python

import os
import numpy as np
np.seterr(all="ignore")
import functools

from collections import defaultdict

from lsst.daf.persistence.butler import Butler
from lsst.pex.config import (Config, Field, ConfigField, ListField, DictField, ConfigDictField,
                             ConfigurableField)
from lsst.pipe.base import CmdLineTask, ArgumentParser, TaskRunner, TaskError
from lsst.coadd.utils import TractDataIdContainer
from lsst.afw.table.catalogMatches import matchesToCatalog
from lsst.meas.astrom import AstrometryConfig, LoadAstrometryNetObjectsTask, LoadAstrometryNetObjectsConfig
from lsst.pipe.tasks.colorterms import ColortermLibrary

from lsst.meas.algorithms import LoadIndexedReferenceObjectsTask

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
    matchOverlapRadius = Field(dtype=float, default=0.5, doc="Matching radius for overlaps (arcseconds)")
    colorterms = ConfigField(dtype=ColortermLibrary, doc="Library of color terms")
    photoCatName = Field(dtype=str, default="ps1", doc="Name of photometric reference catalog; "
                         "used to select a color term dict in colorterms.")
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
    doPlotCompareUnforced = Field(dtype=bool, default=True, doc="Plot difference between forced and unforced?")
    doPlotQuiver = Field(dtype=bool, default=True, doc="Plot ellipticity residuals quiver plot?")
    doPlotFootprintNpix = Field(dtype=bool, default=True, doc="Plot histogram of footprint nPix?")
    onlyReadStars = Field(dtype=bool, default=False, doc="Only read stars (to save memory)?")
    srcSchemaMap = DictField(keytype=str, itemtype=str, default=None, optional=True,
                             doc="Mapping between different stack (e.g. HSC vs. LSST) schema names")
    fluxToPlotList = ListField(dtype=str, default=["base_GaussianFlux", "ext_photometryKron_KronFlux",
                                                   "modelfit_CModel"],
                               doc="List of fluxes to plot: mag(flux)-mag(base_PsfFlux) vs mag(base_PsfFlux)")
    doApplyUberCal = Field(dtype=bool, default=True, doc="Apply meas_mosaic ubercal results to input?" +
                           " FLUXMAG0 zeropoint is applied if doApplyUberCal is False")

    def saveToStream(self, outfile, root="root"):
        """Required for loading colorterms from a Config outside the 'lsst' namespace"""
        print >> outfile, "import lsst.meas.photocal.colorterms"
        return Config.saveToStream(self, outfile, root)

    def setDefaults(self):
        Config.setDefaults(self)
        # self.externalCatalogs = {"sdss-dr9-fink-v5b": astrom}
        self.analysisMatches.magThreshold = 21.0  # External catalogs like PS1 & SDSS used smaller telescopes
        self.refObjLoader.ref_dataset_name = "ps1_pv3_3pi_20170110"
        self.colorterms.load(os.path.join(os.environ["OBS_SUBARU_DIR"], "config", "hsc", "colorterms.py"))


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
        # find index and dataId for first dataset in list that exists
        dataId = None
        for indexExists, patchRef in enumerate(patchRefList):
            if patchRef.datasetExists("deepCoadd_forced_src"):
                dataId = patchRef.dataId
                break
        if dataId is None:
            raise TaskError("No data exists in patRefList: %s" %
                            ([patchRef.dataId for patchRef in patchRefList]))
        dataId = patchRefList[indexExists].dataId
        patchList = [dataRef.dataId["patch"] for dataRef in patchRefList]
        butler = patchRefList[indexExists].getButler()
        # Check metadata to see if stack used was HSC
        forcedMd = butler.get("deepCoadd_forced_src", patchRefList[indexExists].dataId).getMetadata()
        hscRun = checkHscStack(forcedMd)
        if hscRun:
            coadd = butler.get("deepCoadd_calexp_hsc", dataId)
        else:
            coadd = butler.get("deepCoadd_calexp", dataId)
        wcs = coadd.getWcs()
        skymap = butler.get("deepCoadd_skyMap")
        tractInfo = skymap[dataRef.dataId["tract"]]
        filterName = dataId["filter"]
        filenamer = Filenamer(patchRefList[indexExists].getButler(), self.outputDataset,
                              patchRefList[indexExists].dataId)
        if (self.config.doPlotMags or self.config.doPlotStarGalaxy or self.config.doPlotOverlaps or
            self.config.doPlotCompareUnforced or cosmos or self.config.externalCatalogs):
            forced = self.readCatalogs(patchRefList, "deepCoadd_forced_src", indexExists)
            forced = self.calibrateCatalogs(forced, wcs=wcs)

            # XXX
            wcs = skymap[dataRef.dataId["tract"]].getWcs()
            for src in forced:
                src.updateCoord(wcs)

            unforced = self.readCatalogs(patchRefList, "deepCoadd_meas", indexExists)
            unforced = self.calibrateCatalogs(unforced, wcs=wcs)
            # catalog = joinCatalogs(meas, forced, prefix1="meas_", prefix2="forced_")

        # Set an alias map for differing src naming conventions of different stacks (if any)
        if hscRun is not None and self.config.srcSchemaMap is not None:
            for aliasMap in [forced.schema.getAliasMap(), unforced.schema.getAliasMap()]:
                for lsstName, otherName in self.config.srcSchemaMap.iteritems():
                    aliasMap.set(lsstName, otherName)

        # copy over some fields from unforced to forced catalog
        flagsToCopy = ["base_SdssShape_flag", "base_SdssShape_xx", "base_SdssShape_yy", "base_SdssShape_xy",
                       "base_SdssShape_psf_xx", "base_SdssShape_psf_yy", "base_SdssShape_psf_xy",
                       "ext_shapeHSM_HsmSourceMoments_xx", "ext_shapeHSM_HsmSourceMoments_yy",
                       "ext_shapeHSM_HsmSourceMoments_xy", "ext_shapeHSM_HsmPsfMoments_xx",
                       "ext_shapeHSM_HsmPsfMoments_yy", "ext_shapeHSM_HsmPsfMoments_xy", "deblend_nChild",
                       "calib_psfUsed","calib_psfCandidate"]
                       #, "calib_psfReserved"]
        for flag in set(list(self.config.analysis.flags) + flagsToCopy):
            if flag not in forced.schema:
                if hscRun and flag == "slot_Centroid_flag":
                    continue
                forced = addColumnToSchema(unforced, forced, flag)

        if self.config.doPlotFootprintNpix:
            forced = addFootprintNPix(forced, fromCat=unforced)

        # purge the catalogs of flagged sources
        for flag in self.config.analysis.flags:
            forced = forced[~unforced[flag]].copy(deep=True)
            unforced = unforced[~unforced[flag]].copy(deep=True)

        forced = forced[unforced["deblend_nChild"] == 0].copy(deep=True) # Exclude non-deblended
        unforced = unforced[unforced["deblend_nChild"] == 0].copy(deep=True) # Exclude non-deblended
        self.catLabel = "nChild = 0"
        self.zpLabel = self.zpLabel + " " + self.catLabel
        print "len(forced) = ", len(forced), "  len(unforced) = ",len(unforced)

        flagsCat = unforced

        if self.config.doPlotFootprintNpix:
            self.plotFootprintHist(forced, filenamer(dataId, description="footNpix", style="hist"),
                                   dataId, butler=butler, tractInfo=tractInfo, patchList=patchList,
                                   hscRun=hscRun, zpLabel=self.zpLabel, flagsCat=flagsCat)
            self.plotFootprint(forced, filenamer, dataId, butler=butler, tractInfo=tractInfo,
                               patchList=patchList, hscRun=hscRun, zpLabel=self.zpLabel, flagsCat=flagsCat)

        if self.config.doPlotQuiver:
            self.plotQuiver(forced, filenamer(dataId, description="ellipResids", style="quiver"),
                            dataId=dataId, butler=butler, tractInfo=tractInfo, patchList=patchList,
                            hscRun=hscRun, zpLabel=self.zpLabel, scale=2)

        if self.config.doPlotMags:
            self.plotMags(forced, filenamer, dataId, tractInfo=tractInfo, patchList=patchList, hscRun=hscRun,
                          zpLabel=self.zpLabel, flagsCat=flagsCat)
        if self.config.doPlotStarGalaxy:
            if "ext_shapeHSM_HsmSourceMoments_xx" in unforced.schema:
                self.plotStarGal(unforced, filenamer, dataId, tractInfo=tractInfo, patchList=patchList,
                                 hscRun=hscRun, zpLabel=self.zpLabel)
            else:
                self.log.warn("Cannot run plotStarGal: ext_shapeHSM_HsmSourceMoments_xx not in forced.schema")
        if self.config.doPlotSizes:
            if "base_SdssShape_psf_xx" in forced.schema:
                self.plotSizes(forced, filenamer, dataId, butler=butler, tractInfo=tractInfo,
                               patchList=patchList, hscRun=hscRun, zpLabel=self.zpLabel)
            else:
                self.log.warn("Cannot run plotSizes: base_SdssShape_psf_xx not in catalog.schema")
        if cosmos:
            self.plotCosmos(forced, filenamer, cosmos, dataId)
        if self.config.doPlotCompareUnforced:
            self.plotCompareUnforced(forced, unforced, filenamer, dataId, tractInfo=tractInfo,
                                     patchList=patchList, hscRun=hscRun, matchRadius=self.config.matchRadius,
                                     zpLabel=self.zpLabel)
        if self.config.doPlotOverlaps:
            overlaps = self.overlaps(forced, unforced)
            self.log.info("Number of overlap objects matched = {:d}".format(len(overlaps)))
            if len(overlaps) > 5:
                self.plotOverlaps(overlaps, filenamer, dataId, tractInfo=tractInfo, patchList=patchList,
                                  hscRun=hscRun, matchRadius=self.config.matchOverlapRadius,
                                  zpLabel=self.zpLabel)
        if self.config.doPlotMatches:
            matches = self.readSrcMatches(patchRefList, "deepCoadd_forced_src", hscRun=hscRun, wcs=wcs)
            self.plotMatches(matches, filterName, filenamer, dataId, tractInfo=tractInfo, patchList=patchList,
                             hscRun=hscRun, matchRadius=self.config.matchRadius, zpLabel=self.zpLabel)

        for cat in self.config.externalCatalogs:
            with andCatalog(cat):
                matches = self.matchCatalog(forced, filterName, self.config.externalCatalogs[cat])
                self.plotMatches(matches, filterName, filenamer, dataId, cat)

    def readCatalogs(self, patchRefList, dataset, index=0):
        catList = [patchRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_HEAVY_FOOTPRINTS) for
                   patchRef in patchRefList if patchRef.datasetExists(dataset)]
        if len(catList) == 0:
            raise TaskError("No catalogs read: %s" % ([patchRef.dataId for patchRef in patchRefList]))
        if self.config.onlyReadStars and "base_ClassificationExtendedness_value" in catList[index].schema:
            catList = [cat[cat["base_ClassificationExtendedness_value"] < 0.5].copy(deep=True)
                       for cat in catList]
        return concatenateCatalogs(catList)

    def readSrcMatches(self, dataRefList, dataset, hscRun=None, wcs=None):
        catList = []
        for dataRef in dataRefList:
            if not dataRef.datasetExists(dataset):
                self.log.info("Dataset does not exist: {0:r}, {1:s}".format(dataRef.dataId, dataset))
                continue
            butler = dataRef.getButler()

            # Generate unnormalized match list (from normalized persisted one) with joinMatchListWithCatalog
            # (which requires a refObjLoader to be initialized).
            catalog = dataRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS)
            catalog = self.calibrateCatalogs(catalog, wcs=wcs)


            # XXX RA,Dec not set in forced measurement
            if dataset == "deepCoadd_forced_src":
                skymap = dataRefList[0].get("deepCoadd_skyMap")
                wcs = skymap[dataRefList[0].dataId["tract"]].getWcs()
                for src in catalog:
                    src.updateCoord(wcs)


            if dataset.startswith("deepCoadd_"):
                packedMatches = butler.get("deepCoadd_measMatch", dataRef.dataId)
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
            refObjLoader = self.config.refObjLoader.apply(butler=butler)
            matches = refObjLoader.joinMatchListWithCatalog(packedMatches, catalog)
            # LSST reads in a_net catalogs with flux in "janskys", so must convert back to DN
            matches = matchJanskyToDn(matches)
            if hscRun and self.config.doAddAperFluxHsc:
                addApertureFluxesHSC(matches, prefix="second_")

            if len(matches) == 0:
                self.log.warn("No matches for %s" % (dataRef.dataId,))
                continue

            # Set the alias map for the matches sources (i.e. the .second attribute schema for each match)
            if self.config.srcSchemaMap and hscRun:
                for mm in matches:
                    aliasMap = mm.second.schema.getAliasMap()
                    for lsstName, otherName in self.config.srcSchemaMap.iteritems():
                        aliasMap.set(lsstName, otherName)

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
            if self.config.srcSchemaMap and hscRun:
                aliasMap = catalog.schema.getAliasMap()
                for lsstName, otherName in self.config.srcSchemaMap.iteritems():
                    aliasMap.set("src_" + lsstName, "src_" + otherName)
            catList.append(catalog)

        if len(catList) == 0:
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
        calibrated = calibrateCoaddSourceCatalog(catalog, self.config.analysis.coaddZp)
        return calibrated

    def plotMags(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None, tractInfo=None,
                 patchList=None, hscRun=None, matchRadius=None, zpLabel=None, fluxToPlotList=None,
                 postFix="", flagsCat=None):
        if fluxToPlotList is None:
            fluxToPlotList = self.config.fluxToPlotList
        enforcer = Enforcer(requireLess={"star": {"stdev": 0.02}})
        for col in fluxToPlotList:
            if col + "_flux" in catalog.schema:
                shortName = "mag_" + col + postFix
                self.log.info("shortName = {:s}".format(shortName))
                self.AnalysisClass(catalog, MagDiff(col + "_flux", "base_PsfFlux_flux"), "Mag(%s) - PSFMag" %
                                   fluxToPlotString(col), shortName, self.config.analysis,
                                   flags=[col + "_flag"], labeller=StarGalaxyLabeller(),
                                   flagsCat=flagsCat,
                                   ).plotAll(dataId, filenamer, self.log, enforcer, butler=butler,
                                             camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                             patchList=patchList, hscRun=hscRun,
                                             matchRadius=matchRadius, zpLabel=zpLabel)

    def plotSizes(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None, tractInfo=None,
                  patchList=None, hscRun=None, matchRadius=None, zpLabel=None, flagsCat=None):
        enforcer = None
        for col in ["base_PsfFlux", ]:
            if col + "_flux" in catalog.schema:
                shortName = "trace_"
                # set limits dynamically...can be very different visit-to-visit due to seeing differences
                psfUsed = catalog[catalog["calib_psfUsed"]].copy(deep=True)
                sdssTrace = sdssTraceSize()
                sdssTrace = sdssTrace(psfUsed)
                sdssTrace = sdssTrace[np.where(np.isfinite(sdssTrace))]
                traceMean = np.around(sdssTrace.mean(), 1)
                traceStd = np.around(4.5*sdssTrace.std(), 1)
                qMin = traceMean - traceStd
                qMax = traceMean + traceStd
                self.log.info("shortName = {:s}".format(shortName))
                self.AnalysisClass(catalog, sdssTraceSize(),
                                   "SdssShape Trace: $\sqrt{0.5*(I_{xx}+I_{yy})}$ (pixels)", shortName,
                                   self.config.analysis, flags=[col + "_flag"],
                                   goodKeys=["calib_psfUsed"], qMin=qMin, qMax=qMax,
                                   labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                   ).plotAll(dataId, filenamer, self.log, enforcer, butler=butler,
                                             camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                             patchList=patchList, hscRun=hscRun,
                                             matchRadius=matchRadius, zpLabel=zpLabel)
                shortName = "hsmTrace_"
                self.log.info("shortName = {:s}".format(shortName))
                self.AnalysisClass(catalog, hsmTraceSize(),
                                   "HSM Trace: $\sqrt{0.5*(I_{xx}+I_{yy})}$ (pixels)", shortName,
                                   self.config.analysis, flags=[col + "_flag"],
                                   goodKeys=["calib_psfUsed"], qMin=qMin, qMax=qMax,
                                   labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                   ).plotAll(dataId, filenamer, self.log, enforcer, butler=butler,
                                             camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                             patchList=patchList, hscRun=hscRun,
                                             matchRadius=matchRadius, zpLabel=zpLabel)
            if col + "_flux" in catalog.schema:
                shortName = "psfTraceDiff_"
                self.log.info("shortName = {:s}".format(shortName))
                self.AnalysisClass(catalog, psfSdssTraceSizeDiff(),
                                   "SdssShape Trace % diff (psfUsed - PSFmodel)", shortName,
                                   self.config.analysis, flags=[col + "_flag"],
                                   goodKeys=["calib_psfUsed"], qMin=-3.0, qMax=3.0,
                                   labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                   ).plotAll(dataId, filenamer, self.log, enforcer, butler=butler,
                                             camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                             patchList=patchList, hscRun=hscRun,
                                             matchRadius=matchRadius, zpLabel=zpLabel)
                shortName = "psfHsmTraceDiff_"
                self.log.info("shortName = {:s}".format(shortName))
                self.AnalysisClass(catalog, psfHsmTraceSizeDiff(),
                                   "HSM Trace % diff (psfUsed - PSFmodel)", shortName,
                                   self.config.analysis, flags=[col + "_flag"],
                                   goodKeys=["calib_psfUsed"], qMin=-3.0, qMax=3.0,
                                   labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                   ).plotAll(dataId, filenamer, self.log, enforcer, butler=butler,
                                             camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                             patchList=patchList, hscRun=hscRun,
                                             matchRadius=matchRadius, zpLabel=zpLabel)
                shortName = "e1Resids_"
                self.log.info("shortName = {:s}".format(shortName))
                self.AnalysisClass(catalog, e1ResidsSdss(),
                                   "SdssShape e1 residuals (psfUsed - PSFmodel)", shortName,
                                   self.config.analysis, flags=[col + "_flag"],
                                   goodKeys=["calib_psfUsed"], qMin=-0.05, qMax=0.05,
                                   labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                   ).plotAll(dataId, filenamer, self.log, enforcer, butler=butler,
                                             camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                             patchList=patchList, hscRun=hscRun,
                                             matchRadius=matchRadius, zpLabel=zpLabel)
                shortName = "e1ResidsHsm_"
                self.log.info("shortName = {:s}".format(shortName))
                self.AnalysisClass(catalog, e1ResidsHsm(),
                                   "HSM e1 residuals (psfUsed - PSFmodel)", shortName,
                                   self.config.analysis, flags=[col + "_flag"],
                                   goodKeys=["calib_psfUsed"], qMin=-0.05, qMax=0.05,
                                   labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                   ).plotAll(dataId, filenamer, self.log, enforcer, butler=butler,
                                             camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                             patchList=patchList, hscRun=hscRun,
                                             matchRadius=matchRadius, zpLabel=zpLabel)

    def plotCentroidXY(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None,
                       tractInfo=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None,
                       flagsCat=None):
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.02}})
        for col in ["base_SdssCentroid_x", "base_SdssCentroid_y"]:
            if col in catalog.schema:
                shortName = col
                self.log.info("shortName = {:s}".format(shortName))
                self.AnalysisClass(catalog, catalog[col], "(%s)" % col, shortName, self.config.analysis,
                                   labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                                   ).plotFP(dataId, filenamer, self.log, enforcer,
                                            camera=camera, ccdList=ccdList, hscRun=hscRun,
                                            matchRadius=matchRadius, zpLabel=zpLabel)

    def plotFootprint(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None,
                      tractInfo=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None,
                      postFix="", flagsCat=None, plotRunStats=False, highlightList=None):
        enforcer = None
        shortName = "footNpix_calib_psfUsed"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(catalog, catalog["base_Footprint_nPix"], "%s" % shortName, shortName,
                           self.config.analysis, flags=["base_Footprint_nPix_flag"],
                           goodKeys=["calib_psfUsed"],
                           qMin=-100, qMax=2000, labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                           ).plotAll(dataId, filenamer, self.log, enforcer, butler=butler, camera=camera,
                                     ccdList=ccdList, tractInfo=tractInfo, patchList=patchList,
                                     hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel,
                                     plotRunStats=plotRunStats, highlightList=highlightList)
        shortName = "footNpix"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(catalog, catalog["base_Footprint_nPix"], "%s" % shortName, shortName,
                           self.config.analysis, flags=["base_Footprint_nPix_flag"],
                           qMin=0, qMax=3000, labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                           ).plotAll(dataId, filenamer, self.log, enforcer, butler=butler, camera=camera,
                                     ccdList=ccdList, tractInfo=tractInfo, patchList=patchList,
                                     hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel,
                                     plotRunStats=plotRunStats)

    def plotFootprintHist(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None,
                          tractInfo=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None,
                          postFix="", flagsCat=None):
        stats = None #self.config.analysis.stats()
        shortName = "footNpix"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(catalog, catalog["base_Footprint_nPix"], "%s" % shortName, shortName,
                           self.config.analysis, flags=["base_Footprint_nPix_flag"],
                           qMin=0, qMax=3000, labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                           ).plotHistogram(filenamer, stats=stats, hscRun=hscRun, matchRadius=matchRadius,
                                           zpLabel=zpLabel, filterStr=dataId['filter'])

    def plotStarGal(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None, tractInfo=None,
                    patchList=None, hscRun=None, matchRadius=None, zpLabel=None, flagsCat=None):
        enforcer = None
        shortName = "pStar"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(catalog, deconvMomStarGal, "pStar", shortName, self.config.analysis,
                           qMin=-0.1, qMax=1.39, labeller=StarGalaxyLabeller(), flagsCat=flagsCat,
                           ).plotAll(dataId, filenamer, self.log, enforcer, butler=butler, camera=camera,
                                     ccdList=ccdList, tractInfo=tractInfo, patchList=patchList, hscRun=hscRun,
                                     matchRadius=matchRadius, zpLabel=zpLabel)
        shortName = "deconvMom"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(catalog, deconvMom, "Deconvolved moments (unforced)", shortName,
                           self.config.analysis, qMin=-1.0, qMax=3.0, labeller=StarGalaxyLabeller(),
                           flagsCat=flagsCat,
                           ).plotAll(dataId, filenamer, self.log,
                                     Enforcer(requireLess={"star": {"stdev": 0.2}}), butler=butler,
                                     camera=camera, ccdList=ccdList, tractInfo=tractInfo, patchList=patchList,
                                     hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel)

    def plotCompareUnforced(self, forced, unforced, filenamer, dataId, butler=None, camera=None, ccdList=None,
                            tractInfo=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None,
                            fluxToPlotList=None):
        if fluxToPlotList is None:
            fluxToPlotList = self.config.fluxToPlotList
        enforcer = None
        catalog = joinMatches(afwTable.matchRaDec(forced, unforced, matchRadius*afwGeom.arcseconds),
                              "forced_", "unforced_")
        for col in fluxToPlotList:
            shortName = "compareUnforced_" + col
            self.log.info("shortName = {:s}".format(shortName))
            if "forced_" + col + "_flux" in catalog.schema:
                self.AnalysisClass(catalog, MagDiff("forced_" + col + "_flux", "unforced_" + col + "_flux"),
                                   "Forced - Unforced mag (%s)" % fluxToPlotString(col),
                                   shortName, self.config.analysis, prefix="forced_",
                                   flags=[col + "_flag"],
                                   labeller=OverlapsStarGalaxyLabeller("forced_", "unforced_"),
                                   ).plotAll(dataId, filenamer, self.log, enforcer, butler=butler,
                                             camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                             patchList=patchList, hscRun=hscRun,
                                             matchRadius=matchRadius, zpLabel=zpLabel)

    def isBad(self, source):
        """Return True if any of config.badFlags are set for this source."""
        for flag in self.config.analysis.flags:
            if source.get(flag):
                return True
        return False

    def overlaps(self, forcedCat, flagsCat):
        # Don't include parents of blended objects
        noParentsForcedCat = forcedCat[flagsCat["deblend_nChild"] == 0].copy(deep=True)
        matches = afwTable.matchRaDec(noParentsForcedCat, self.config.matchOverlapRadius*afwGeom.arcseconds)
        if len(matches) == 0:
            self.log.info("Did not find any overlapping matches")
        return joinMatches(matches, "first_", "second_")

    def plotOverlaps(self, overlaps, filenamer, dataId, butler=None, camera=None, ccdList=None,
                     tractInfo=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None,
                     fluxToPlotList=None, flagsCat=None):
        if fluxToPlotList is None:
            fluxToPlotList = self.config.fluxToPlotList
        magEnforcer = Enforcer(requireLess={"star": {"stdev": 0.003}})
        for col in fluxToPlotList:
            shortName = "overlap_" + col
            self.log.info("shortName = {:s}".format(shortName))
            if "first_" + col + "_flux" in overlaps.schema:
                self.AnalysisClass(overlaps, MagDiff("first_" + col + "_flux", "second_" + col + "_flux"),
                                   "Overlap mag difference (%s)" % fluxToPlotString(col), shortName,
                                   self.config.analysis, prefix="first_", flags=[col + "_flag"],
                                   labeller=OverlapsStarGalaxyLabeller(), magThreshold=23
                                   ).plotAll(dataId, filenamer, self.log, magEnforcer, butler=butler,
                                             camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                             patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                             zpLabel=zpLabel)

        distEnforcer = Enforcer(requireLess={"star": {"stdev": 0.005}})
        shortName = "overlap_distance"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(overlaps, lambda cat: cat["distance"]*(1.0*afwGeom.radians).asArcseconds(),
                           "Distance (arcsec)", shortName, self.config.analysis, prefix="first_",
                           qMin=-0.01, qMax=0.11, labeller=OverlapsStarGalaxyLabeller(), flagsCat=flagsCat,
                           ).plotAll(dataId, filenamer, self.log, distEnforcer, forcedMean=0.0,
                                     butler=butler, camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                     patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                     zpLabel=zpLabel)

    def plotMatches(self, matches, filterName, filenamer, dataId, description="matches", butler=None,
                    camera=None, ccdList=None, tractInfo=None, patchList=None, hscRun=None, matchRadius=None,
                    zpLabel=None, flagsCat=None):
        enforcer = None # Enforcer(requireLess={"star": {"stdev": 0.030}}),

        ct = self.config.colorterms.getColorterm(filterName, self.config.photoCatName)
        if "src_calib_psfUsed" in matches.schema:
            shortName = description + "_mag_calib_psfUsed"
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(matches, MagDiffMatches("base_PsfFlux_flux", ct, zp=0.0),
                               "MagPsf(unforced) - ref (calib_psfUsed)", shortName,
                               self.config.analysisMatches, prefix="src_", goodKeys=["calib_psfUsed"],
                               qMin=-0.15, qMax=0.15, labeller=MatchesStarGalaxyLabeller(), flagsCat=flagsCat,
                               ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                         camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                         patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                         zpLabel=zpLabel)

        shortName = description + "_mag"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(matches, MagDiffMatches("base_PsfFlux_flux", ct, zp=0.0), "MagPsf(unforced) - ref",
                           shortName, self.config.analysisMatches, prefix="src_",
                           qMin=-0.15, qMax=5.15, labeller=MatchesStarGalaxyLabeller(),
                           ).plotAll(dataId, filenamer, self.log, enforcer=enforcer, butler=butler,
                                     camera=camera, ccdList=ccdList, tractInfo=tractInfo, patchList=patchList,
                                     hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel)
        shortName = description + "_distance"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(matches, lambda cat: cat["distance"]*(1.0*afwGeom.radians).asArcseconds(),
                           "Distance (arcsec)", shortName, self.config.analysisMatches,
                           prefix="src_", qMin=-0.05*self.config.matchesMaxDistance,
                           qMax=self.config.matchesMaxDistance, labeller=MatchesStarGalaxyLabeller(),
                           flagsCat=flagsCat,
                           ).plotAll(dataId, filenamer, self.log,
                                     Enforcer(requireLess={"star": {"stdev": 0.050}}), forcedMean=0.0,
                                     butler=butler, camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                     patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                     zpLabel=zpLabel)
        shortName = description + "_ra"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(matches, AstrometryDiff("src_coord_ra", "ref_coord_ra", "ref_coord_dec"),
                           "dRA*cos(Dec) (arcsec)", shortName, self.config.analysisMatches,
                           prefix="src_", qMin=-self.config.matchesMaxDistance,
                           qMax=self.config.matchesMaxDistance, labeller=MatchesStarGalaxyLabeller(),
                           flagsCat=flagsCat,
                           ).plotAll(dataId, filenamer, self.log,
                                     Enforcer(requireLess={"star": {"stdev": 0.050}}),
                                     butler=butler, camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                     patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                     zpLabel=zpLabel)
        shortName = description + "_dec"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(matches, AstrometryDiff("src_coord_dec", "ref_coord_dec"),
                           "dDec (arcsec)", shortName, self.config.analysisMatches, prefix="src_",
                           qMin=-self.config.matchesMaxDistance, qMax=self.config.matchesMaxDistance,
                           labeller=MatchesStarGalaxyLabeller(), flagsCat=flagsCat,
                           ).plotAll(dataId, filenamer, self.log,
                                     Enforcer(requireLess={"star": {"stdev": 0.050}}),
                                     butler=butler, camera=camera, ccdList=ccdList, tractInfo=tractInfo,
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


    def plotQuiver(self, catalog, filenamer, dataId=None, butler=None, camera=None, ccdList=None,
                   tractInfo=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None,
                   postFix="", flagsCat=None, scale=1):
        stats = None #self.config.analysis.stats()
        shortName = "quiver"
        self.log.info("shortName = {:s}".format(shortName))
        self.AnalysisClass(catalog, None, "%s" % shortName, shortName,
                           self.config.analysis, labeller=None,
                           ).plotQuiver(catalog, filenamer, stats=stats, dataId=dataId, butler=butler,
                                        camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                        patchList=patchList,hscRun=hscRun, zpLabel=zpLabel, scale=scale)

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
        # find index and dataId for first dataset in list that exists
        for indexExists1, patchRef1 in enumerate(patchRefList1):
            if patchRef1.datasetExists("deepCoadd_forced_src"):
                dataId = patchRef1.dataId
                break
        patchList1 = [dataRef1.dataId["patch"] for dataRef1 in patchRefList1 if
                      dataRef1.datasetExists("deepCoadd_forced_src")]
        filenamer = Filenamer(patchRefList1[indexExists1].getButler(), "plotCompareCoadd",
                              patchRefList1[indexExists1].dataId)

        # Check metadata to see if stack used was HSC
        butler1 = patchRefList1[indexExists1].getButler()
        forcedMd1 = butler1.get("deepCoadd_forced_src", patchRefList1[indexExists1].dataId).getMetadata()
        hscRun1 = checkHscStack(forcedMd1)
        skymap1 = butler1.get("deepCoadd_skyMap")
        if hscRun1:
            coadd1 = butler1.get("deepCoadd_calexp_hsc", dataId)
        else:
            coadd1 = butler1.get("deepCoadd_calexp", dataId)
        wcs1 = coadd1.getWcs()
        tractInfo1 = skymap1[dataRef1.dataId["tract"]]
        butler2 = patchRefList2[indexExists1].getButler()
        forcedMd2 = butler2.get("deepCoadd_forced_src", patchRefList2[indexExists1].dataId).getMetadata()
        hscRun2 = checkHscStack(forcedMd2)
        if hscRun2:
            coadd2 = butler2.get("deepCoadd_calexp_hsc", dataId)
        else:
            coadd2 = butler2.get("deepCoadd_calexp", dataId)
        wcs2 = coadd2.getWcs()
        forced1 = self.readCatalogs(patchRefList1, self.config.coaddName + "Coadd_forced_src")
        forced1 = self.calibrateCatalogs(forced1, wcs=wcs1)
        forced2 = self.readCatalogs(patchRefList2, self.config.coaddName + "Coadd_forced_src")
        forced2 = self.calibrateCatalogs(forced2, wcs=wcs2)
        unforced1 = self.readCatalogs(patchRefList1, self.config.coaddName + "Coadd_meas")
        unforced1 = self.calibrateCatalogs(unforced1, wcs=wcs1)
        unforced2 = self.readCatalogs(patchRefList2, self.config.coaddName + "Coadd_meas")
        unforced2 = self.calibrateCatalogs(unforced2, wcs=wcs2)

        # Set an alias map for differing src naming conventions of different stacks (if any)
        if self.config.srcSchemaMap:
            for hscRun, catalog in zip([hscRun1, hscRun2, hscRun1, hscRun2],
                                       [forced1, forced2, unforced1, unforced2]):
                if hscRun:
                    aliasMap = catalog.schema.getAliasMap()
                    for lsstName, otherName in self.config.srcSchemaMap.iteritems():
                        aliasMap.set(lsstName, otherName)
                else:
                    if "base_SdssCentroid_x" not in catalog.schema:
                        if "base_TransformedCentroid_x" in catalog.schema:
                            # Need this for LSST cat since base_SdssCentroid doesn't exist in forced schema
                            # but still don't have Sigmas...
                            aliasMap = catalog.schema.getAliasMap()
                            aliasMap.set("base_SdssCentroid", "base_TransformedCentroid")
                            aliasMap.set("base_SdssCentroid_x", "base_TransformedCentroid_x")
                            aliasMap.set("base_SdssCentroid_y", "base_TransformedCentroid_y")
                            aliasMap.set("base_SdssCentroid_flag", "base_TransformedCentroid_flag")
                        else:
                            self.log.warn("Could not find base_SdssCentroid (or equivalent) flags")

        # purge the catalogs of flagged sources
        for flag in self.config.analysis.flags:
            forced1 = forced1[~unforced1[flag]].copy(deep=True)
            unforced1 = unforced1[~unforced1[flag]].copy(deep=True)
            forced2 = forced2[~unforced2[flag]].copy(deep=True)
            unforced2 = unforced2[~unforced2[flag]].copy(deep=True)

        forced = self.matchCatalogs(forced1, forced2)
        unforced = self.matchCatalogs(unforced1, unforced2)

        self.log.info("\nNumber of sources in catalogs: first = {0:d} and second = {1:d}".format(
                len(forced1), len(forced2)))

        if self.config.doPlotMags:
            self.plotMags(forced, filenamer, dataId, tractInfo=tractInfo1, patchList=patchList1,
                          hscRun=hscRun2, matchRadius=self.config.matchRadius, zpLabel=self.zpLabel)
        if self.config.doPlotCentroids:
            self.plotCentroids(forced, filenamer, dataId, tractInfo=tractInfo1, patchList=patchList1,
                          hscRun=hscRun2, matchRadius=self.config.matchRadius, zpLabel=self.zpLabel)

    def readCatalogs(self, patchRefList, dataset, index=0):
        catList = [patchRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS) for
                   patchRef in patchRefList if patchRef.datasetExists(dataset)]
        if len(catList) == 0:
            raise TaskError("No catalogs read: %s" % ([patchRef.dataId for patchRef in patchRefList]))
        return concatenateCatalogs(catList)

    def matchCatalogs(self, catalog1, catalog2):
        matches = afwTable.matchRaDec(catalog1, catalog2, self.config.matchRadius*afwGeom.arcseconds)
        if len(matches) == 0:
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
                 patchList=None, hscRun=None, matchRadius=None, zpLabel=None, fluxToPlotList=None,
                 postFix="", flagsCat=None, highlightList=None):
        if fluxToPlotList is None:
            fluxToPlotList = self.config.fluxToPlotList
        enforcer = None  # Enforcer(requireLess={"star": {"stdev": 0.02}})
        for col in fluxToPlotList:
            if "first_" + col + "_flux" in catalog.schema and "second_" + col + "_flux" in catalog.schema:
                shortName = "diff_" + col
                self.log.info("shortName = {:s}".format(shortName))
                Analysis(catalog, MagDiffCompare(col + "_flux"), "Run Comparison: Mag difference (%s)" %
                         fluxToPlotString(col), shortName, self.config.analysis,
                         prefix="first_", qMin=-0.05, qMax=0.05, flags=[col + "_flag"],
                         errFunc=MagDiffErr(col + "_flux"), labeller=OverlapsStarGalaxyLabeller(),
                         flagsCat=flagsCat,
                         ).plotAll(dataId, filenamer, self.log, enforcer, butler=butler, camera=camera,
                                   ccdList=ccdList, tractInfo=tractInfo, patchList=patchList, hscRun=hscRun,
                                   matchRadius=matchRadius, zpLabel=zpLabel, postFix=postFix)

    def plotCentroids(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None,
                      tractInfo=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None,
                      flagsCat=None, highlightList=None):
        distEnforcer = None
        shortName = "diff_x"
        self.log.info("shortName = {:s}".format(shortName))
        Analysis(catalog, CentroidDiff("x", centroid1="base_SdssCentroid"),
                 "Run Comparison: x offset (arcsec)", shortName, self.config.analysis, prefix="first_",
                 qMin=-0.3, qMax=0.3, errFunc=None, labeller=OverlapsStarGalaxyLabeller(),
                 ).plotAll(dataId, filenamer, self.log, distEnforcer, butler=butler, camera=camera,
                           ccdList=ccdList, tractInfo=tractInfo, patchList=patchList, hscRun=hscRun,
                           matchRadius=matchRadius, zpLabel=zpLabel)
        shortName = "diff_y"
        self.log.info("shortName = {:s}".format(shortName))
        Analysis(catalog, CentroidDiff("y", centroid1="base_SdssCentroid"),
                 "Run Comparison: y offset (arcsec)", shortName, self.config.analysis, prefix="first_",
                 qMin=-0.1, qMax=0.1, errFunc=None, labeller=OverlapsStarGalaxyLabeller(),
                 ).plotAll(dataId, filenamer, self.log, distEnforcer, butler=butler, camera=camera,
                           ccdList=ccdList, tractInfo=tractInfo, patchList=patchList, hscRun=hscRun,
                           matchRadius=matchRadius, zpLabel=zpLabel)

    def plotFootprint(self, catalog, filenamer, dataId, butler=None, camera=None, ccdList=None,
                      tractInfo=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None,
                      postFix="", flagsCat=None, highlightList=None):
        enforcer = None
        shortName = "diff_footNpix"
        col = "base_Footprint_nPix"
        self.log.info("shortName = {:s}".format(shortName))
        Analysis(catalog, FootNpixDiffCompare(col), "Run Comparison: Footprint nPix difference", shortName,
                 self.config.analysis, prefix="first_", qMin=-250, qMax=250, flags=[col + "_flag"],
                 labeller=OverlapsStarGalaxyLabeller(), flagsCat=flagsCat,
                 ).plotAll(dataId, filenamer, self.log, enforcer, butler=butler, camera=camera,
                           ccdList=ccdList, tractInfo=tractInfo, patchList=patchList, hscRun=hscRun,
                           matchRadius=matchRadius, zpLabel=zpLabel, postFix=postFix)
        shortName = "diff_footNpix_calib_psfUsed"
        self.log.info("shortName = {:s}".format(shortName))
        Analysis(catalog, FootNpixDiffCompare(col), "Run Comparison: Footprint nPix diff (psfUsed)", shortName,
                 self.config.analysis, prefix="first_", goodKeys=["calib_psfUsed"], qMin=-150, qMax=150,
                 flags=[col + "_flag"], labeller=OverlapsStarGalaxyLabeller(), flagsCat=flagsCat,
                 ).plotAll(dataId, filenamer, self.log, enforcer, butler=butler, camera=camera,
                           ccdList=ccdList, tractInfo=tractInfo, patchList=patchList, hscRun=hscRun,
                           matchRadius=matchRadius, zpLabel=zpLabel, postFix=postFix,
                           highlightList=highlightList)
    def _getConfigName(self):
        return None
    def _getMetadataName(self):
        return None
    def _getEupsVersionsName(self):
        return None
