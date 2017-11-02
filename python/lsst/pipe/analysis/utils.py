from __future__ import print_function

import os
import re

import numpy as np
try:
    import fastparquet
except ImportError:
    fastparquet = None
    import logging
    logging.warning('fastparquet package not available.  Parquet files will not be written.')

from contextlib import contextmanager

from lsst.daf.persistence.safeFileIo import safeMakeDir
from lsst.pipe.base import Struct, TaskError

import lsst.afw.cameraGeom as cameraGeom
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable

try:
    from lsst.meas.mosaic.updateExposure import applyMosaicResultsCatalog
except ImportError:
    applyMosaicResultsCatalog = None

__all__ = ["Filenamer", "Data", "Stats", "Enforcer", "MagDiff", "MagDiffMatches", "MagDiffCompare",
           "ApCorrDiffCompare", "AstrometryDiff", "sdssTraceSize", "hsmTraceSize",
           "psfSdssTraceSizeDiff", "psfHsmTraceSizeDiff", "sdssTraceSizeCompare", "sdssXxCompare",
           "sdssYyCompare", "hsmTraceSizeCompare", "hsmMomentsXxCompare", "hsmMomentsYyCompare",
           "sdssPsfTraceSizeCompare", "hsmPsfTraceSizeCompare", "e1ResidsSdss", "e2ResidsSdss",
           "e1ResidsHsm", "e2ResidsHsm", "FootNpixDiffCompare", "MagDiffErr", "ApCorrDiffErr", "CentroidDiff",
           "CentroidDiffErr", "deconvMom", "deconvMomStarGal", "concatenateCatalogs", "joinMatches",
           "checkIdLists", "checkPatchOverlap", "joinCatalogs", "getFluxKeys", "addColumnsToSchema",
           "addApertureFluxesHSC", "addFpPoint", "addFootprintNPix", "addRotPoint",
           "calibrateSourceCatalogMosaic", "calibrateSourceCatalog", "calibrateCoaddSourceCatalog",
           "backoutApCorr", "matchJanskyToDn", "checkHscStack", "fluxToPlotString", "andCatalog",
           "writeParquet", "getRepoInfo", "findCcdKey", "getCcdNameRefList", "getDataExistsRefList"]

def writeParquet(table, path):
    """
    Write an afwTable into Parquet format

    Parameters
    ----------
    table : `lsst.afw.table.source.source.SourceCatalog`
        Table to be written to parquet
    path : `str`
        Path to which to write.  Must end in ".parq".

    Returns
    -------
    None

    Notes
    -----
    This function first converts the afwTable to an astropy table,
    then to a pandas DataFrame, which is then written to parquet
    format using the fastparquet library.  If fastparquet is not
    available, then it will do nothing.

    """
    if fastparquet is None:
        return

    if not path.endswith('.parq'):
        raise ValueError('Please provide a filename ending in .parq.')
    df = table.asAstropy().to_pandas()
    df = df.set_index('id', drop=True)
    fastparquet.write(path, df)

class Filenamer(object):
    """Callable that provides a filename given a style"""
    def __init__(self, butler, dataset, dataId={}):
        self.butler = butler
        self.dataset = dataset
        self.dataId = dataId
    def __call__(self, dataId, **kwargs):
        filename = self.butler.get(self.dataset + "_filename", self.dataId, **kwargs)[0]
        # When trying to write to a different rerun (or output), if the given dataset exists in the _parent
        # rerun (or input) directory, _parent is added to the filename, and thus the output files
        # will actually oversrite those in the _parent rerun (or input) directory (which is bad if
        # your intention is to write to a different output dir!).  So, here we check for the presence
        # of _parent in the filename and strip it out if present.
        if "_parent/" in filename:
            print("Note: stripping _parent from filename: ", filename)
            filename = filename.replace("_parent/", "")
        safeMakeDir(os.path.dirname(filename))
        return filename

class Data(Struct):
    def __init__(self, catalog, quantity, mag, selection, color, error=None, plot=True):
        Struct.__init__(self, catalog=catalog[selection].copy(deep=True), quantity=quantity[selection],
                        mag=mag[selection], selection=selection, color=color, plot=plot,
                        error=error[selection] if error is not None else None)

class Stats(Struct):
    def __init__(self, dataUsed, num, total, mean, stdev, forcedMean, median, clip):
        Struct.__init__(self, dataUsed=dataUsed, num=num, total=total, mean=mean, stdev=stdev,
                        forcedMean=forcedMean, median=median, clip=clip)
    def __repr__(self):
        return "Stats(mean={0.mean:.4f}; stdev={0.stdev:.4f}; num={0.num:d}; total={0.total:d}; " \
            "median={0.median:.4f}; clip={0.clip:.4f}; forcedMean={0.forcedMean:})".format(self)


class Enforcer(object):
    """Functor for enforcing limits on statistics"""
    def __init__(self, requireGreater={}, requireLess={}, doRaise=False):
        self.requireGreater = requireGreater
        self.requireLess = requireLess
        self.doRaise = doRaise
    def __call__(self, stats, dataId, log, description):
        for label in self.requireGreater:
            for ss in self.requireGreater[label]:
                value = getattr(stats[label], ss)
                if value <= self.requireGreater[label][ss]:
                    text = ("%s %s = %.2f exceeds minimum limit of %.2f: %s" %
                            (description, ss, value, self.requireGreater[label][ss], dataId))
                    log.warn(text)
                    if self.doRaise:
                        raise AssertionError(text)
        for label in self.requireLess:
            for ss in self.requireLess[label]:
                value = getattr(stats[label], ss)
                if value >= self.requireLess[label][ss]:
                    text = ("%s %s = %.2f exceeds maximum limit of %.2f: %s" %
                            (description, ss, value, self.requireLess[label][ss], dataId))
                    log.warn(text)
                    if self.doRaise:
                        raise AssertionError(text)

class MagDiff(object):
    """Functor to calculate magnitude difference"""
    def __init__(self, col1, col2, unitScale=1.0):
        self.col1 = col1
        self.col2 = col2
        self.unitScale = unitScale
    def __call__(self, catalog):
        return -2.5*np.log10(catalog[self.col1]/catalog[self.col2])*self.unitScale

class MagDiffMatches(object):
    """Functor to calculate magnitude difference for match catalog"""
    def __init__(self, column, colorterm, zp=27.0, unitScale=1.0):
        self.column = column
        self.colorterm = colorterm
        self.zp = zp
        self.unitScale = unitScale
    def __call__(self, catalog):
        ref1 = -2.5*np.log10(catalog["ref_" + self.colorterm.primary + "_flux"])
        ref2 = -2.5*np.log10(catalog["ref_" + self.colorterm.secondary + "_flux"])
        ref = self.colorterm.transformMags(ref1, ref2)
        src = self.zp - 2.5*np.log10(catalog["src_" + self.column])
        return (src - ref)*self.unitScale

class MagDiffCompare(object):
    """Functor to calculate magnitude difference between two entries in comparison catalogs
    """
    def __init__(self, column, unitScale=1.0):
        self.column = column
        self.unitScale = unitScale
    def __call__(self, catalog):
        src1 = -2.5*np.log10(catalog["first_" + self.column])
        src2 = -2.5*np.log10(catalog["second_" + self.column])
        return (src1 - src2)*self.unitScale


class ApCorrDiffCompare(object):
    """Functor to calculate magnitude difference between two entries in comparison catalogs
    """
    def __init__(self, column, unitScale=1.0):
        self.column = column
        self.unitScale = unitScale
    def __call__(self, catalog):
        apCorr1 = catalog["first_" + self.column]
        apCorr2 = catalog["second_" + self.column]
        return -2.5*np.log10(apCorr1/apCorr2)*self.unitScale


class AstrometryDiff(object):
    """Functor to calculate difference between astrometry"""
    def __init__(self, first, second, declination1=None, declination2=None, unitScale=1.0):
        self.first = first
        self.second = second
        self.declination1 = declination1
        self.declination2 = declination2
        self.unitScale = unitScale
    def __call__(self, catalog):
        first = catalog[self.first]
        second = catalog[self.second]
        cosDec1 = np.cos(catalog[self.declination1]) if self.declination1 is not None else 1.0
        cosDec2 = np.cos(catalog[self.declination2]) if self.declination2 is not None else 1.0
        return (first*cosDec1 - second*cosDec2)*(1.0*afwGeom.radians).asArcseconds()*self.unitScale

class sdssTraceSize(object):
    """Functor to calculate SDSS trace radius size for sources"""
    def __call__(self, catalog):
        srcSize = np.sqrt(0.5*(catalog["base_SdssShape_xx"] + catalog["base_SdssShape_yy"]))
        return np.array(srcSize)


class psfSdssTraceSizeDiff(object):
    """Functor to calculate SDSS trace radius size difference (%) between object and psf model"""
    def __call__(self, catalog):
        srcSize = np.sqrt(0.5*(catalog["base_SdssShape_xx"] + catalog["base_SdssShape_yy"]))
        psfSize = np.sqrt(0.5*(catalog["base_SdssShape_psf_xx"] + catalog["base_SdssShape_psf_yy"]))
        sizeDiff = 100*(srcSize - psfSize)/(0.5*(srcSize + psfSize))
        return np.array(sizeDiff)


class hsmTraceSize(object):
    """Functor to calculate HSM trace radius size for sources"""
    def __call__(self, catalog):
        srcSize = np.sqrt(0.5*(catalog["ext_shapeHSM_HsmSourceMoments_xx"] +
                               catalog["ext_shapeHSM_HsmSourceMoments_yy"]))
        return np.array(srcSize)


class psfHsmTraceSizeDiff(object):
    """Functor to calculate HSM trace radius size difference (%) between object and psf model"""
    def __call__(self, catalog):
        srcSize = np.sqrt(0.5*(catalog["ext_shapeHSM_HsmSourceMoments_xx"] +
                               catalog["ext_shapeHSM_HsmSourceMoments_yy"]))
        psfSize = np.sqrt(0.5*(catalog["ext_shapeHSM_HsmPsfMoments_xx"] +
                               catalog["ext_shapeHSM_HsmPsfMoments_yy"]))
        sizeDiff = 100*(srcSize - psfSize)/(0.5*(srcSize + psfSize))
        return np.array(sizeDiff)


class sdssTraceSizeCompare(object):
    """Functor to calculate SDSS trace radius size difference (%) between objects in matched catalog"""
    def __call__(self, catalog):
        srcSize1 = np.sqrt(0.5*(catalog["first_base_SdssShape_xx"] + catalog["first_base_SdssShape_yy"]))
        srcSize2 = np.sqrt(0.5*(catalog["second_base_SdssShape_xx"] + catalog["second_base_SdssShape_yy"]))
        sizeDiff = 100.0*(srcSize1 - srcSize2)/(0.5*(srcSize1 + srcSize2))
        return np.array(sizeDiff)


class sdssXxCompare(object):
    """Functor to calculate SDSS trace radius size difference (%) between objects in matched catalog"""
    def __call__(self, catalog):
        srcXx1 = catalog["first_base_SdssShape_xx"]
        srcXx2 = catalog["second_base_SdssShape_xx"]
        xxDiff = 100.0*(srcXx1 - srcXx2)/(0.5*(srcXx1 + srcXx2))
        return np.array(xxDiff)


class sdssYyCompare(object):
    """Functor to calculate HSM trace radius size difference (%) between objects in matched catalog"""
    def __call__(self, catalog):
        srcYy1 = catalog["first_base_SdssShape_yy"]
        srcYy2 = catalog["second_base_SdssShape_yy"]
        yyDiff = 100.0*(srcYy1 - srcYy2)/(0.5*(srcYy1 + srcYy2))
        return np.array(yyDiff)


class hsmTraceSizeCompare(object):
    """Functor to calculate HSM trace radius size difference (%) between objects in matched catalog"""
    def __call__(self, catalog):
        srcSize1 = np.sqrt(0.5*(catalog["first_ext_shapeHSM_HsmSourceMoments_xx"] +
                                catalog["first_ext_shapeHSM_HsmSourceMoments_yy"]))
        srcSize2 = np.sqrt(0.5*(catalog["second_ext_shapeHSM_HsmSourceMoments_xx"] +
                                catalog["second_ext_shapeHSM_HsmSourceMoments_yy"]))
        sizeDiff = 100.0*(srcSize1 - srcSize2)/(0.5*(srcSize1 + srcSize2))
        return np.array(sizeDiff)


class hsmMomentsXxCompare(object):
    """Functor to calculate HSM trace radius size difference (%) between objects in matched catalog"""
    def __call__(self, catalog):
        srcXx1 = catalog["first_ext_shapeHSM_HsmSourceMoments_xx"]
        srcXx2 = catalog["second_ext_shapeHSM_HsmSourceMoments_xx"]
        xxDiff = 100.0*(srcXx1 - srcXx2)/(0.5*(srcXx1 + srcXx2))
        return np.array(xxDiff)


class hsmMomentsYyCompare(object):
    """Functor to calculate HSM trace radius size difference (%) between objects in matched catalog"""
    def __call__(self, catalog):
        srcYy1 = catalog["first_ext_shapeHSM_HsmSourceMoments_yy"]
        srcYy2 = catalog["second_ext_shapeHSM_HsmSourceMoments_yy"]
        yyDiff = 100.0*(srcYy1 - srcYy2)/(0.5*(srcYy1 + srcYy2))
        return np.array(yyDiff)


class sdssPsfTraceSizeCompare(object):
    """Functor to calculate SDSS PSF trace radius size difference (%) between objects in matched catalog"""
    def __call__(self, catalog):
        psfSize1 = np.sqrt(0.5*(catalog["first_base_SdssShape_psf_xx"] +
                                catalog["first_base_SdssShape_psf_yy"]))
        psfSize2 = np.sqrt(0.5*(catalog["second_base_SdssShape_psf_xx"] +
                                catalog["second_base_SdssShape_psf_yy"]))
        sizeDiff = 100.0*(psfSize1 - psfSize2)/(0.5*(psfSize1 + psfSize2))
        return np.array(sizeDiff)


class hsmPsfTraceSizeCompare(object):
    """Functor to calculate HSM PSF trace radius size difference (%) between objects in matched catalog"""
    def __call__(self, catalog):
        psfSize1 = np.sqrt(0.5*(catalog["first_ext_shapeHSM_HsmPsfMoments_xx"] +
                                catalog["first_ext_shapeHSM_HsmPsfMoments_yy"]))
        psfSize2 = np.sqrt(0.5*(catalog["second_ext_shapeHSM_HsmPsfMoments_xx"] +
                                catalog["second_ext_shapeHSM_HsmPsfMoments_yy"]))
        sizeDiff = 100.0*(psfSize1 - psfSize2)/(0.5*(psfSize1 + psfSize2))
        return np.array(sizeDiff)


class e1ResidsSdss(object):
    """Functor to calculate SDSS e1 ellipticity residuals for a given object and psf model"""
    def __init__(self, unitScale=1.0):
        self.unitScale = unitScale
    def __call__(self, catalog):
        srcE1 = ((catalog["base_SdssShape_xx"] - catalog["base_SdssShape_yy"])/
                 (catalog["base_SdssShape_xx"] + catalog["base_SdssShape_yy"]))
        psfE1 = ((catalog["base_SdssShape_psf_xx"] - catalog["base_SdssShape_psf_yy"])/
                 (catalog["base_SdssShape_psf_xx"] + catalog["base_SdssShape_psf_yy"]))
        e1Resids = srcE1 - psfE1
        return np.array(e1Resids)*self.unitScale

class e2ResidsSdss(object):
    """Functor to calculate SDSS e2 ellipticity residuals for a given object and psf model"""
    def __init__(self, unitScale=1.0):
        self.unitScale = unitScale
    def __call__(self, catalog):
        srcE2 = (2.0*catalog["base_SdssShape_xy"]/
                 (catalog["base_SdssShape_xx"] + catalog["base_SdssShape_yy"]))
        psfE2 = (2.0*catalog["base_SdssShape_psf_xy"]/
                 (catalog["base_SdssShape_psf_xx"] + catalog["base_SdssShape_psf_yy"]))
        e2Resids = srcE2 - psfE2
        return np.array(e2Resids)*self.unitScale

class e1ResidsHsm(object):
    """Functor to calculate HSM e1 ellipticity residuals for a given object and psf model"""
    def __init__(self, unitScale=1.0):
        self.unitScale = unitScale
    def __call__(self, catalog):
        srcE1 = ((catalog["ext_shapeHSM_HsmSourceMoments_xx"] - catalog["ext_shapeHSM_HsmSourceMoments_yy"])/
                 (catalog["ext_shapeHSM_HsmSourceMoments_xx"] + catalog["ext_shapeHSM_HsmSourceMoments_yy"]))
        psfE1 = ((catalog["ext_shapeHSM_HsmPsfMoments_xx"] - catalog["ext_shapeHSM_HsmPsfMoments_yy"])/
                 (catalog["ext_shapeHSM_HsmPsfMoments_xx"] + catalog["ext_shapeHSM_HsmPsfMoments_yy"]))
        e1Resids = srcE1 - psfE1
        return np.array(e1Resids)*self.unitScale

class e2ResidsHsm(object):
    """Functor to calculate HSM e1 ellipticity residuals for a given object and psf model"""
    def __init__(self, unitScale=1.0):
        self.unitScale = unitScale
    def __call__(self, catalog):
        srcE2 = (2.0*catalog["ext_shapeHSM_HsmSourceMoments_xy"]/
                 (catalog["ext_shapeHSM_HsmSourceMoments_xx"] + catalog["ext_shapeHSM_HsmSourceMoments_yy"]))
        psfE2 = (2.0*catalog["ext_shapeHSM_HsmPsfMoments_xy"]/
                 (catalog["ext_shapeHSM_HsmPsfMoments_xx"] + catalog["ext_shapeHSM_HsmPsfMoments_yy"]))
        e2Resids = srcE2 - psfE2
        return np.array(e2Resids)*self.unitScale


class FootNpixDiffCompare(object):
    """Functor to calculate footprint nPix difference between two entries in comparison catalogs
    """
    def __init__(self, column):
        self.column = column
    def __call__(self, catalog):
        nPix1 = catalog["first_" + self.column]
        nPix2 = catalog["second_" + self.column]
        return nPix1 - nPix2


class MagDiffErr(object):
    """Functor to calculate magnitude difference error"""
    def __init__(self, column, unitScale=1.0):
        zp = 27.0 # Exact value is not important, since we're differencing the magnitudes
        self.column = column
        self.calib = afwImage.Calib()
        self.calib.setFluxMag0(10.0**(0.4*zp))
        self.calib.setThrowOnNegativeFlux(False)
        self.unitScale = unitScale
    def __call__(self, catalog):
        mag1, err1 = self.calib.getMagnitude(catalog["first_" + self.column],
                                             catalog["first_" + self.column + "Sigma"])
        mag2, err2 = self.calib.getMagnitude(catalog["second_" + self.column],
                                             catalog["second_" + self.column + "Sigma"])
        return np.sqrt(err1**2 + err2**2)*self.unitScale

class ApCorrDiffErr(object):
    """Functor to calculate magnitude difference error"""
    def __init__(self, column, unitScale=1.0):
        self.column = column
        self.unitScale = unitScale
    def __call__(self, catalog):
        err1 = catalog["first_" + self.column + "Sigma"]
        err2 = catalog["second_" + self.column + "Sigma"]
        return np.sqrt(err1**2 + err2**2)*self.unitScale

class CentroidDiff(object):
    """Functor to calculate difference in astrometry"""
    def __init__(self, component, first="first_", second="second_", centroid1="base_SdssCentroid",
                 centroid2="base_SdssCentroid", unitScale=1.0):
        self.component = component
        self.first = first
        self.second = second
        self.centroid1 = centroid1
        self.centroid2 = centroid2
        self.unitScale = unitScale

    def __call__(self, catalog):
        first = self.first + self.centroid1 + "_" + self.component
        second = self.second + self.centroid2 + "_" + self.component
        return (catalog[first] - catalog[second])*self.unitScale

class CentroidDiffErr(CentroidDiff):
    """Functor to calculate difference error for astrometry"""
    def __call__(self, catalog):
        firstx = self.first + self.centroid + "_xSigma"
        firsty = self.first + self.centroid + "_ySigma"
        secondx = self.second + self.centroid + "_xSigma"
        secondy = self.second + self.centroid + "_ySigma"

        subkeys1 = [catalog.schema[firstx].asKey(), catalog.schema[firsty].asKey()]
        subkeys2 = [catalog.schema[secondx].asKey(), catalog.schema[secondy].asKey()]
        menu = {"x": 0, "y": 1}

        return np.hypot(catalog[subkeys1[menu[self.component]]],
                        catalog[subkeys2[menu[self.component]]])*self.unitScale


def deconvMom(catalog):
    """Calculate deconvolved moments"""
    if "ext_shapeHSM_HsmSourceMoments_xx" in catalog.schema:
        hsm = catalog["ext_shapeHSM_HsmSourceMoments_xx"] + catalog["ext_shapeHSM_HsmSourceMoments_yy"]
    else:
        hsm = np.ones(len(catalog))*np.nan
    sdss = catalog["base_SdssShape_xx"] + catalog["base_SdssShape_yy"]
    if "ext_shapeHSM_HsmPsfMoments_xx" in catalog.schema:
        psfXxName = "ext_shapeHSM_HsmPsfMoments_xx"
        psfYyName = "ext_shapeHSM_HsmPsfMoments_yy"
    elif "base_SdssShape_psf_xx" in catalog.schema:
        psfXxName = "base_SdssShape_psf_xx"
        psfYyName = "base_SdssShape_psf_yy"
    else:
        raise RuntimeError("No psf shape parameter found in catalog")
    psf = catalog[psfXxName] + catalog[psfYyName]
    return np.where(np.isfinite(hsm), hsm, sdss) - psf

def deconvMomStarGal(catalog):
    """Calculate P(star) from deconvolved moments"""
    rTrace = deconvMom(catalog)
    snr = catalog["base_PsfFlux_flux"]/catalog["base_PsfFlux_fluxSigma"]
    poly = (-4.2759879274 + 0.0713088756641*snr + 0.16352932561*rTrace - 4.54656639596e-05*snr*snr -
            0.0482134274008*snr*rTrace + 4.41366874902e-13*rTrace*rTrace + 7.58973714641e-09*snr*snr*snr +
            1.51008430135e-05*snr*snr*rTrace + 4.38493363998e-14*snr*rTrace*rTrace +
            1.83899834142e-20*rTrace*rTrace*rTrace)
    return 1.0/(1.0 + np.exp(-poly))

def concatenateCatalogs(catalogList):
    assert len(catalogList) > 0, "No catalogs to concatenate"
    template = catalogList[0]
    catalog = type(template)(template.schema)
    catalog.reserve(sum(len(cat) for cat in catalogList))
    for cat in catalogList:
        catalog.extend(cat, True)
    return catalog

def joinMatches(matches, first="first_", second="second_"):
    if len(matches)==0:
        return []

    mapperList = afwTable.SchemaMapper.join([matches[0].first.schema,
                                                matches[0].second.schema],
                                            [first, second])
    firstAliases = matches[0].first.schema.getAliasMap()
    secondAliases = matches[0].second.schema.getAliasMap()
    schema = mapperList[0].getOutputSchema()
    distanceKey = schema.addField("distance", type="Angle", doc="Distance between %s and %s" % (first, second))
    catalog = afwTable.BaseCatalog(schema)
    aliases = catalog.schema.getAliasMap()
    catalog.reserve(len(matches))
    for mm in matches:
        row = catalog.addNew()
        row.assign(mm.first, mapperList[0])
        row.assign(mm.second, mapperList[1])
        row.set(distanceKey, mm.distance*afwGeom.radians)
    # make sure aliases get persisted to match catalog
    for k, v in firstAliases.items():
        aliases.set(first + k, first + v)
    for k, v in secondAliases.items():
        aliases.set(second + k, second + v)
    return catalog

def checkIdLists(catalog1, catalog2, prefix=""):
    # Check to see if two catalogs have an identical list of objects by id
    idStrList = ["", ""]
    for i, cat in enumerate((catalog1, catalog2)):
        if "id" in cat.schema:
            idStrList[i] = "id"
        elif "objectId" in cat.schema:
            idStrList[i] = "objectId"
        elif prefix + "id" in cat.schema:
            idStrList[i] = prefix + "id"
        elif prefix + "objectId" in cat.schema:
            idStrList[i] = prefix + "objectId"
        else:
            raise RuntimeError("Cannot identify object id field (tried id, objectId, " + prefix + "id, and " +
                               prefix + "objectId)")

    return np.all(catalog1[idStrList[0]] == catalog2[idStrList[1]])

def checkPatchOverlap(patchList, tractInfo):
    # Given a list of patch dataIds along with the associated tractInfo, check if any of the patches overlap
    for i, patch0 in enumerate(patchList):
        overlappingPatches = False
        patchIndex = [int(val) for val in patch0.split(",")]
        patchInfo = tractInfo.getPatchInfo(patchIndex)
        patchBBox0 = patchInfo.getOuterBBox()
        for j, patch1 in enumerate(patchList):
            if patch1 != patch0 and j > i:
                patchIndex = [int(val) for val in patch1.split(",")]
                patchInfo = tractInfo.getPatchInfo(patchIndex)
                patchBBox1 = patchInfo.getOuterBBox()
                if patchBBox0.overlaps(patchBBox1):
                    overlappingPatches = True
                    break
        if overlappingPatches:
            break
    return overlappingPatches

def joinCatalogs(catalog1, catalog2, prefix1="cat1_", prefix2="cat2_"):
    # Make sure catalogs entries are all associated with the same object

    if not checkIdLists(catalog1, catalog2):
        raise RuntimeError("Catalogs with different sets of objects cannot be joined")

    mapperList = afwTable.SchemaMapper.join([catalog1[0].schema, catalog2[0].schema],
                                            [prefix1, prefix2])
    schema = mapperList[0].getOutputSchema()
    catalog = afwTable.BaseCatalog(schema)
    catalog.reserve(len(catalog1))
    for s1, s2 in zip(catalog1, catalog2):
        row = catalog.addNew()
        row.assign(s1, mapperList[0])
        row.assign(s2, mapperList[1])
    return catalog

def getFluxKeys(schema):
    """Retrieve the flux and flux error keys from a schema
    Both are returned as dicts indexed on the flux name (e.g. "flux.psf" or "cmodel.flux").
    """
    schemaKeys = dict((s.field.getName(), s.key) for s in schema)
    fluxKeys = dict((name, key) for name, key in schemaKeys.items() if
                    re.search(r"^(\w+_flux)$", name) and key.getTypeString() != "Flag")
    errKeys = dict((name + "Sigma", schemaKeys[name + "Sigma"]) for name in fluxKeys.keys() if
                   name + "Sigma" in schemaKeys)
    # Also check for any in HSC format
    fluxKeysHSC = dict((name, key) for name, key in schemaKeys.items() if
                       (re.search(r"^(flux\_\w+|\w+\_flux)$", name) or
                        re.search(r"^(\w+flux\_\w+|\w+\_flux)$", name))
                       and not re.search(r"^(\w+\_apcorr)$", name) and name + "_err" in schemaKeys)
    errKeysHSC = dict((name + "_err", schemaKeys[name + "_err"]) for name in fluxKeysHSC.keys() if
                       name + "_err" in schemaKeys)
    if len(fluxKeysHSC) > 0:
        fluxKeys.update(fluxKeysHSC)
        errKeys.update(errKeysHSC)
    if len(fluxKeys) == 0:
        raise RuntimeError("No flux keys found")
    return fluxKeys, errKeys

def addColumnsToSchema(fromCat, toCat, colNameList, prefix=""):
    """Copy columns from fromCat to new version of toCat"""
    fromMapper = afwTable.SchemaMapper(fromCat.schema)
    fromMapper.addMinimalSchema(toCat.schema, False)
    toMapper = afwTable.SchemaMapper(toCat.schema)
    toMapper.addMinimalSchema(toCat.schema)
    schema = fromMapper.editOutputSchema()
    for col in colNameList:
        colName = prefix + col
        fromKey = fromCat.schema.find(colName).getKey()
        fromField = fromCat.schema.find(colName).getField()
        schema.addField(fromField)
        toField = schema.find(colName).getField()
        fromMapper.addMapping(fromKey, toField, doReplace=True)

    newCatalog = afwTable.SourceCatalog(schema)
    newCatalog.reserve(len(toCat))

    newCatalog.extend(toCat, toMapper)
    for srcFrom, srcTo in zip(fromCat, newCatalog):
        srcTo.assign(srcFrom, fromMapper)

    aliases = newCatalog.schema.getAliasMap()
    for k, v in toCat.schema.getAliasMap().items():
        aliases.set(k, v)

    return newCatalog

def addApertureFluxesHSC(catalog, prefix=""):
    mapper = afwTable.SchemaMapper(catalog[0].schema)
    mapper.addMinimalSchema(catalog[0].schema)
    schema = mapper.getOutputSchema()
    apName = prefix + "base_CircularApertureFlux"
    apRadii = ["3_0", "4_5", "6_0", "9_0", "12_0", "17_0", "25_0", "35_0", "50_0", "70_0"]

    # for ia in range(len(apRadii)):
    # Just to 12 pixels for now...takes a long time...
    for ia in (4,):
        apFluxKey = schema.addField(apName + "_" + apRadii[ia] + "_flux", type="D",
                                    doc="flux within " + apRadii[ia].replace("_", ".")
                                    + "-pixel aperture", units="count")
        apFluxSigmaKey = schema.addField(apName + "_" + apRadii[ia] + "_fluxSigma", type="D",
                                         doc="1-sigma flux uncertainty")
    apFlagKey = schema.addField(apName + "_flag", type="Flag", doc="general failure flag")

    newCatalog = afwTable.SourceCatalog(schema)
    newCatalog.reserve(len(catalog))

    for source in catalog:
        row = newCatalog.addNew()
        row.assign(source, mapper)
        # for ia in range(len(apRadii)):
        for ia in (4,):
            row.set(apFluxKey, source[prefix+"flux_aperture"][ia])
            row.set(apFluxSigmaKey, source[prefix+"flux_aperture_err"][ia])
        row.set(apFlagKey, source[prefix + "flux_aperture_flag"])

    return newCatalog

def addFpPoint(det, catalog, prefix=""):
    # Compute Focal Plane coordinates for SdssCentroid of each source and add to schema
    mapper = afwTable.SchemaMapper(catalog[0].schema)
    mapper.addMinimalSchema(catalog[0].schema)
    schema = mapper.getOutputSchema()
    fpName = prefix + "base_FPPosition"
    fpxKey = schema.addField(fpName + "_x", type="D", doc="Position on the focal plane (in FP pixels)")
    fpyKey = schema.addField(fpName + "_y", type="D", doc="Position on the focal plane (in FP pixels)")
    fpFlag = schema.addField(fpName + "_flag", type="Flag", doc="Set to True for any fatal failure")

    newCatalog = afwTable.SourceCatalog(schema)
    newCatalog.reserve(len(catalog))
    xCentroidKey = catalog.schema[prefix + "base_SdssCentroid_x"].asKey()
    yCentroidKey = catalog.schema[prefix + "base_SdssCentroid_y"].asKey()
    for source in catalog:
        row = newCatalog.addNew()
        row.assign(source, mapper)
        try:
            center = afwGeom.Point2D(source[xCentroidKey], source[yCentroidKey])
            posInPix = det.makeCameraPoint(center, cameraGeom.PIXELS)
            fpPoint = det.transform(posInPix, cameraGeom.FOCAL_PLANE).getPoint()
        except:
            fpPoint = afwGeom.Point2D(np.nan, np.nan)
            row.set(fpFlag, True)
        row.set(fpxKey, fpPoint[0])
        row.set(fpyKey, fpPoint[1])

    aliases = newCatalog.schema.getAliasMap()
    for k, v in catalog[0].schema.getAliasMap().items():
        aliases.set(k, v)

    return newCatalog

def addFootprintNPix(catalog, fromCat=None, prefix=""):
    # Retrieve the number of pixels in an sources footprint and add to schema
    mapper = afwTable.SchemaMapper(catalog[0].schema)
    mapper.addMinimalSchema(catalog[0].schema)
    schema = mapper.getOutputSchema()
    fpName = prefix + "base_Footprint_nPix"
    fpKey = schema.addField(fpName, type="I", doc="Number of pixels in Footprint")
    fpFlag = schema.addField(fpName + "_flag", type="Flag", doc="Set to True for any fatal failure")
    newCatalog = afwTable.SourceCatalog(schema)
    newCatalog.reserve(len(catalog))
    if fromCat:
        if len(fromCat) != len(catalog):
            raise TaskError("Lengths of fromCat and catalog for getting footprint Npixs do not agree")
    if fromCat is None:
        fromCat = catalog
    for srcFrom, srcTo in zip(fromCat, catalog):
        row = newCatalog.addNew()
        row.assign(srcTo, mapper)
        try:
            footNpix = srcFrom.getFootprint().getArea()
        except:
            raise
            footNpix = 0 # used to be np.nan, but didn't work.
            row.set(fpFlag, True)
        row.set(fpKey, footNpix)

    aliases = newCatalog.schema.getAliasMap()
    for k, v in catalog[0].schema.getAliasMap().items():
        aliases.set(k, v)

    return newCatalog

def rotatePixelCoord(s, width, height, nQuarter):
    """Rotate single (x, y) pixel coordinate such that LLC of detector in FP is (0, 0)
    """
    xKey = s.schema.find("slot_Centroid_x").key
    yKey = s.schema.find("slot_Centroid_y").key
    x0 = s[xKey]
    y0 = s[yKey]
    if nQuarter == 1:
        s.set(xKey, height - y0 - 1.0)
        s.set(yKey, x0)
    if nQuarter == 2:
        s.set(xKey, width - x0 - 1.0)
        s.set(yKey, height - y0 - 1.0)
    if nQuarter == 3:
        s.set(xKey, y0)
        s.set(yKey, width - x0 - 1.0)
    return s

def addRotPoint(catalog, width, height, nQuarter, prefix=""):
    # Compute rotated CCD pixel coords for comparing LSST vs HSC run centroids
    mapper = afwTable.SchemaMapper(catalog[0].schema)
    mapper.addMinimalSchema(catalog[0].schema)
    schema = mapper.getOutputSchema()
    rotName = prefix + "base_SdssCentroid_Rot"
    rotxKey = schema.addField(rotName + "_x", type="D", doc="Centroid x (in rotated pixels)")
    rotyKey = schema.addField(rotName + "_y", type="D", doc="Centroid y (in rotated pixels)")
    rotFlag = schema.addField(rotName + "_flag", type="Flag", doc="Set to True for any fatal failure")

    newCatalog = afwTable.SourceCatalog(schema)
    newCatalog.reserve(len(catalog))
    for source in catalog:
        row = newCatalog.addNew()
        row.assign(source, mapper)
        try:
            rotPoint = rotatePixelCoord(source, width, height, nQuarter).getCentroid()
        except:
            rotPoint = afwGeom.Point2D(np.nan, np.nan)
            row.set(rotFlag, True)
        row.set(rotxKey, rotPoint[0])
        row.set(rotyKey, rotPoint[1])

    aliases = newCatalog.schema.getAliasMap()
    for k, v in catalog[0].schema.getAliasMap().items():
        aliases.set(k, v)
    return newCatalog

def calibrateSourceCatalogMosaic(dataRef, catalog, fluxKeys=None, errKeys=None, zp=27.0):
    """Calibrate catalog with meas_mosaic results

    Requires a SourceCatalog input.
    """
    result = applyMosaicResultsCatalog(dataRef, catalog, True)
    catalog = result.catalog
    ffp = result.ffp
    # Convert to constant zero point, as for the coadds
    factor = ffp.calib.getFluxMag0()[0]/10.0**(0.4*zp)

    if fluxKeys is None:
        fluxKeys, errKeys = getFluxKeys(catalog.schema)
    for key in list(fluxKeys.values()) + list(errKeys.values()):
        if len(catalog[key].shape) > 1:
            continue
        catalog[key] /= factor
    return catalog

def calibrateSourceCatalog(catalog, zp):
    """Calibrate catalog in the case of no meas_mosaic results using FLUXMAG0 as zp

    Requires a SourceCatalog and zeropoint as input.
    """
    # Convert to constant zero point, as for the coadds
    fluxKeys, errKeys = getFluxKeys(catalog.schema)
    factor = 10.0**(0.4*zp)
    for name, key in list(fluxKeys.items()) + list(errKeys.items()):
        catalog[key] /= factor
    return catalog

def calibrateCoaddSourceCatalog(catalog, zp):
    """Calibrate coadd catalog

    Requires a SourceCatalog and zeropoint as input.
    """
    # Convert to constant zero point, as for the coadds
    fluxKeys, errKeys = getFluxKeys(catalog.schema)
    factor = 10.0**(0.4*zp)
    for name, key in list(fluxKeys.items()) + list(errKeys.items()):
        catalog[key] /= factor
    return catalog

def backoutApCorr(catalog):
    """Back out the aperture correction to all fluxes
    """
    ii = 0
    for k in catalog.schema.getNames():
        if "_flux" in k and k[:-5] + "_apCorr" in catalog.schema.getNames() and "_apCorr" not in k:
            if ii == 0:
                print("Backing out apcorr for:", k)
                ii += 1
            catalog[k] /= catalog[k[:-5] + "_apCorr"]
    return catalog

def matchJanskyToDn(matches):
    # LSST reads in a_net catalogs with flux in "janskys", so must convert back to DN
    JANSKYS_PER_AB_FLUX = 3631.0
    schema = matches[0].first.schema
    keys = [schema[kk].asKey() for kk in schema.getNames() if "_flux" in kk]

    for m in matches:
        for k in keys:
            m.first[k] /= JANSKYS_PER_AB_FLUX
    return matches

def checkHscStack(metadata):
    """Check to see if data were processed with the HSC stack
    """
    try:
        hscPipe = metadata.get("HSCPIPE_VERSION")
    except:
        hscPipe = None
    return hscPipe

def fluxToPlotString(fluxToPlot):
    """Return a more succint string for fluxes for label plotting
    """
    fluxStrMap = {"base_PsfFlux_flux": "PSF",
                  "base_GaussianFlux": "Gaussian",
                  "ext_photometryKron_KronFlux": "Kron",
                  "modelfit_CModel": "CModel",
                  "base_CircularApertureFlux_12_0": "CircAper 12pix"}
    if fluxToPlot in fluxStrMap:
        return fluxStrMap[fluxToPlot]
    else:
        print("WARNING: " + fluxToPlot + " not in fluxStrMap")
        return fluxToPlot


_eups = None
def getEups():
    """Return a EUPS handle

    We instantiate this once only, because instantiation is expensive.
    """
    global _eups
    from eups import Eups  #noqa Nothing else depends on eups, so prevent it from importing unless needed
    if not _eups:
        _eups = Eups()
    return _eups


@contextmanager
def andCatalog(version):
    eups = getEups()
    current = eups.findSetupVersion("astrometry_net_data")[0]
    eups.setup("astrometry_net_data", version, noRecursion=True)
    try:
        yield
    finally:
        eups.setup("astrometry_net_data", current, noRecursion=True)

def getRepoInfo(dataRef, coaddName=None, coaddDataset=None, doApplyUberCal=False):
    """Obtain the relevant repository information for the given dataRef

    Parameters
    ----------
    dataRef : `lsst.daf.persistence.butlerSubset.ButlerDataRef`
       The data reference for which the relevant repository information is to be retrieved
    coaddName : `str`, optional
       The base name of the coadd (e.g. deep or goodSeeing) if dataRef is for coadd level processing
    doApplyUberCal : `bool`, optional
      If True: Set the appropriate dataset type for the uber calibration from meas_mosaic
      If False (the default): Set the dataset type to the source catalog from single frame processing
    """
    butler = dataRef.getButler()
    camera = butler.get("camera")
    dataId = dataRef.dataId
    filterName = dataId["filter"]
    isCoadd = True if "patch" in dataId else False
    ccdKey = None if isCoadd else findCcdKey(dataId)
    # Check metadata to see if stack used was HSC
    metaStr = coaddName + coaddDataset if coaddName is not None else "calexp_md"
    metadata = butler.get(metaStr, dataId)
    hscRun = checkHscStack(metadata)
    dataset = "src"
    if doApplyUberCal:
        dataset = "wcs_hsc_md" if hscRun is not None else "wcs_md"
    skymap =  butler.get(coaddName + "Coadd_skyMap") if coaddName is not None else None
    wcs = None
    tractInfo = None
    if isCoadd:
        coaddDatatype = "Coadd_calexp_hsc" if hscRun else "Coadd_calexp"
        coadd = butler.get(coaddName + coaddDatatype, dataId)
        wcs = coadd.getWcs()
        tractInfo = skymap[dataId["tract"]]
    return Struct(
        butler = butler,
        camera = camera,
        dataId = dataId,
        filterName = filterName,
        ccdKey = ccdKey,
        metadata = metadata,
        hscRun = hscRun,
        dataset = dataset,
        skymap = skymap,
        wcs = wcs,
        tractInfo = tractInfo,
    )

def findCcdKey(dataId):
    """Determine the convention for identifying a "ccd" for the current camera

    Parameters
    ----------
    dataId : `instance` of `lsst.daf.persistence.DataId`

    Raises
    ------
    `RuntimeError`
       If "ccd" key could not be identified from the current hardwired list

    Returns
    -------
    ccdKey : `str`
       The string associated with the "ccd" key.
    """
    ccdKey = None
    ccdKeyList = ["ccd", "sensor", "camcol"]
    for ss in ccdKeyList:
        if ss in dataId:
            ccdKey = ss
            break
    if ccdKey is None:
        raise RuntimeError("Could not identify ccd key for dataId: %s: \nNot in list of known keys: %s" %
                           (dataId, ccdKeyList))
    return ccdKey

def getCcdNameRefList(dataRefList):
    ccdNameRefList = None
    ccdKey = findCcdKey(dataRefList[0].dataId)
    if "raft" in dataRefList[0].dataId:
        ccdNameRefList = [re.sub("[,]", "", str(dataRef.dataId["raft"]) + str(dataRef.dataId[ccdKey])) for
                          dataRef in dataRefList]
    else:
        ccdNameRefList = [dataRef.dataId[ccdKey] for dataRef in dataRefList]
    # cull multiple entries
    ccdNameRefList = list(set(ccdNameRefList))

    if ccdNameRefList is None:
        raise RuntimeError("Failed to create ccdNameRefList")
    return ccdNameRefList

def getDataExistsRefList(dataRefList, dataset):
    dataExistsRefList = None
    ccdKey = findCcdKey(dataRefList[0].dataId)
    if "raft" in dataRefList[0].dataId:
        dataExistsRefList = [re.sub("[,]", "", str(dataRef.dataId["raft"]) + str(dataRef.dataId[ccdKey])) for
                             dataRef in dataRefList if dataRef.datasetExists(dataset)]
    else:
        dataExistsRefList = [dataRef.dataId[ccdKey] for dataRef in dataRefList if
                             dataRef.datasetExists(dataset)]
    # cull multiple entries
    dataExistsRefList = list(set(dataExistsRefList))

    if dataExistsRefList is None:
        raise RuntimeError("dataExistsRef list is empty")
    return dataExistsRefList
