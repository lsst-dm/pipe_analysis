import os
import re

import numpy as np

from contextlib import contextmanager

from lsst.daf.persistence.safeFileIo import safeMakeDir
from lsst.meas.mosaic.updateExposure import applyMosaicResultsCatalog
from lsst.pipe.base import Struct

import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable

__all__ = ["Filenamer", "Data", "Stats", "Enforcer", "MagDiff", "MagDiffMatches", "MagDiffCompare",
           "ApCorrDiffCompare", "AstrometryDiff", "sdssTraceSize", "hsmTraceSize",
           "psfSdssTraceSizeDiff", "psfHsmTraceSizeDiff",
           "sdssTraceSizeCompare", "sdssXxCompare", "sdssYyCompare", "hsmTraceSizeCompare",
           "hsmMomentsXxCompare",  "hsmMomentsYyCompare", "sdssPsfTraceSizeCompare",
           "hsmPsfTraceSizeCompare", "e1ResidsSdss", "e2ResidsSdss", "e1ResidsHsm", "e2ResidsHsm",
           "FootNpixDiffCompare", "MagDiffErr", "ApCorrDiffErr", "CentroidDiff",
           "CentroidDiffErr", "deconvMom", "deconvMomStarGal", "concatenateCatalogs", "joinMatches",
           "checkIdLists", "joinCatalogs", "getFluxKeys", "addApertureFluxesHSC", "addFpPoint",
           "addFootprintNPix", "addRotPoint", "calibrateSourceCatalogMosaic", "calibrateSourceCatalog",
           "calibrateCoaddSourceCatalog", "backoutApCorr", "matchJanskyToDn", "checkHscStack",
           "fluxToPlotString", "andCatalog"]

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
            print "Note: stripping _parent from filename: ", filename
            filename = filename.replace("_parent/", "")
        safeMakeDir(os.path.dirname(filename))
        return filename

class Data(Struct):
    def __init__(self, catalog, quantity, mag, selection, color, error=None, plot=True):
        Struct.__init__(self, catalog=catalog[selection], quantity=quantity[selection], mag=mag[selection],
                        selection=selection, color=color, plot=plot,
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
                    text = ("%s %s = %f exceeds minimum limit of %f: %s" %
                            (description, ss, value, self.requireGreater[label][ss], dataId))
                    log.warn(text)
                    if self.doRaise:
                        raise AssertionError(text)
        for label in self.requireLess:
            for ss in self.requireLess[label]:
                value = getattr(stats[label], ss)
                if value >= self.requireLess[label][ss]:
                    text = ("%s %s = %f exceeds maximum limit of %f: %s" %
                            (description, ss, value, self.requireLess[label][ss], dataId))
                    log.warn(text)
                    if self.doRaise:
                        raise AssertionError(text)

class MagDiff(object):
    """Functor to calculate magnitude difference"""
    def __init__(self, col1, col2):
        self.col1 = col1
        self.col2 = col2
    def __call__(self, catalog):
        return -2.5*np.log10(catalog[self.col1]/catalog[self.col2])

class MagDiffMatches(object):
    """Functor to calculate magnitude difference for match catalog"""
    def __init__(self, column, colorterm, zp=27.0):
        self.column = column
        self.colorterm = colorterm
        self.zp = zp
    def __call__(self, catalog):
        ref1 = -2.5*np.log10(catalog.get("ref_" + self.colorterm.primary + "_flux"))
        ref2 = -2.5*np.log10(catalog.get("ref_" + self.colorterm.secondary + "_flux"))
        ref = self.colorterm.transformMags(ref1, ref2)
        src = self.zp - 2.5*np.log10(catalog.get("src_" + self.column))
        return src - ref

class MagDiffCompare(object):
    """Functor to calculate magnitude difference between two entries in comparison catalogs
    """
    def __init__(self, column):
        self.column = column
    def __call__(self, catalog):
        src1 = -2.5*np.log10(catalog["first_" + self.column])
        src2 = -2.5*np.log10(catalog["second_" + self.column])
        return src1 - src2

class ApCorrDiffCompare(object):
    """Functor to calculate magnitude difference between two entries in comparison catalogs
    """
    def __init__(self, column):
        self.column = column
    def __call__(self, catalog):
        apCorr1 = catalog["first_" + self.column]
        apCorr2 = catalog["second_" + self.column]
        return -2.5*np.log10(apCorr1/apCorr2)

class AstrometryDiff(object):
    """Functor to calculate difference between astrometry"""
    def __init__(self, first, second, declination=None):
        self.first = first
        self.second = second
        self.declination = declination
    def __call__(self, catalog):
        first = catalog[self.first]
        second = catalog[self.second]
        cosDec = np.cos(catalog[self.declination]) if self.declination is not None else 1.0
        return (first - second)*cosDec*(1.0*afwGeom.radians).asArcseconds()


class sdssTraceSize(object):
    """Functor to calculate SDSS trace radius size for sources"""
    def __call__(self, catalog):
        srcSize = np.sqrt(0.5*(catalog["base_SdssShape_xx"] + catalog["base_SdssShape_yy"]))
        return np.array(srcSize)


class psfSdssTraceSizeDiff(object):
    """Functor to calculate SDSS trace radius size difference between object and psf model"""
    def __call__(self, catalog):
        srcSize = np.sqrt(0.5*(catalog["base_SdssShape_xx"] + catalog["base_SdssShape_yy"]))
        psfSize = np.sqrt(0.5*(catalog["base_SdssShape_psf_xx"] + catalog["base_SdssShape_psf_yy"]))
        sizeDiff = (srcSize - psfSize)/psfSize
        return np.array(sizeDiff)


class hsmTraceSize(object):
    """Functor to calculate HSM trace radius size for sources"""
    def __call__(self, catalog):
        srcSize = np.sqrt(0.5*(catalog["ext_shapeHSM_HsmSourceMoments_xx"] +
                               catalog["ext_shapeHSM_HsmSourceMoments_yy"]))
        return np.array(srcSize)


class psfHsmTraceSizeDiff(object):
    """Functor to calculate HSM trace radius size difference between object and psf model"""
    def __call__(self, catalog):
        srcSize = np.sqrt(0.5*(catalog["ext_shapeHSM_HsmSourceMoments_xx"] +
                               catalog["ext_shapeHSM_HsmSourceMoments_yy"]))
        psfSize = np.sqrt(0.5*(catalog["ext_shapeHSM_HsmPsfMoments_xx"] +
                               catalog["ext_shapeHSM_HsmPsfMoments_yy"]))
        sizeDiff = (srcSize - psfSize)/psfSize
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
    def __call__(self, catalog):
        srcE1 = ((catalog["base_SdssShape_xx"] - catalog["base_SdssShape_yy"])/
                 (catalog["base_SdssShape_xx"] + catalog["base_SdssShape_yy"]))
        psfE1 = ((catalog["base_SdssShape_psf_xx"] - catalog["base_SdssShape_psf_yy"])/
                 (catalog["base_SdssShape_psf_xx"] + catalog["base_SdssShape_psf_yy"]))
        e1Resids = srcE1 - psfE1
        return np.array(e1Resids)

class e2ResidsSdss(object):
    """Functor to calculate SDSS e2 ellipticity residuals for a given object and psf model"""
    def __call__(self, catalog):
        srcE2 = (2.0*catalog["base_SdssShape_xy"]/
                 (catalog["base_SdssShape_xx"] + catalog["base_SdssShape_yy"]))
        psfE2 = (2.0*catalog["base_SdssShape_psf_xy"]/
                 (catalog["base_SdssShape_psf_xx"] + catalog["base_SdssShape_psf_yy"]))
        e2Resids = srcE2 - psfE2
        return np.array(e2Resids)

class e1ResidsHsm(object):
    """Functor to calculate HSM e1 ellipticity residuals for a given object and psf model"""
    def __call__(self, catalog):
        srcE1 = ((catalog["ext_shapeHSM_HsmSourceMoments_xx"] - catalog["ext_shapeHSM_HsmSourceMoments_yy"])/
                 (catalog["ext_shapeHSM_HsmSourceMoments_xx"] + catalog["ext_shapeHSM_HsmSourceMoments_yy"]))
        psfE1 = ((catalog["ext_shapeHSM_HsmPsfMoments_xx"] - catalog["ext_shapeHSM_HsmPsfMoments_yy"])/
                 (catalog["ext_shapeHSM_HsmPsfMoments_xx"] + catalog["ext_shapeHSM_HsmPsfMoments_yy"]))
        e1Resids = srcE1 - psfE1
        return np.array(e1Resids)

class e2ResidsHsm(object):
    """Functor to calculate HSM e1 ellipticity residuals for a given object and psf model"""
    def __call__(self, catalog):
        srcE2 = (2.0*catalog["ext_shapeHSM_HsmSourceMoments_xy"]/
                 (catalog["ext_shapeHSM_HsmSourceMoments_xx"] + catalog["ext_shapeHSM_HsmSourceMoments_yy"]))
        psfE2 = (2.0*catalog["ext_shapeHSM_HsmPsfMoments_xy"]/
                 (catalog["ext_shapeHSM_HsmPsfMoments_xx"] + catalog["ext_shapeHSM_HsmPsfMoments_yy"]))
        e2Resids = srcE2 - psfE2
        return np.array(e2Resids)


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
    def __init__(self, column):
        zp = 27.0 # Exact value is not important, since we're differencing the magnitudes
        self.column = column
        self.calib = afwImage.Calib()
        self.calib.setFluxMag0(10.0**(0.4*zp))
        self.calib.setThrowOnNegativeFlux(False)
    def __call__(self, catalog):
        mag1, err1 = self.calib.getMagnitude(catalog["first_" + self.column],
                                             catalog["first_" + self.column + "Sigma"])
        mag2, err2 = self.calib.getMagnitude(catalog["second_" + self.column],
                                             catalog["second_" + self.column + "Sigma"])
        return np.sqrt(err1**2 + err2**2)

class ApCorrDiffErr(object):
    """Functor to calculate magnitude difference error"""
    def __init__(self, column):
        self.column = column
    def __call__(self, catalog):
        err1 = catalog["first_" + self.column + "Sigma"]
        err2 = catalog["second_" + self.column + "Sigma"]
        return np.sqrt(err1**2 + err2**2)

class CentroidDiff(object):
    """Functor to calculate difference in astrometry"""
    def __init__(self, component, first="first_", second="second_", centroid1="base_SdssCentroid",
                 centroid2="base_SdssCentroid"):
        self.component = component
        self.first = first
        self.second = second
        self.centroid1 = centroid1
        self.centroid2 = centroid2

    def __call__(self, catalog):
        first = self.first + self.centroid1 + "_" + self.component
        second = self.second + self.centroid2 + "_" + self.component
        return catalog[first] - catalog[second]

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

        return np.hypot(catalog[subkeys1[menu[self.component]]], catalog[subkeys2[menu[self.component]]])


def deconvMom(catalog):
    """Calculate deconvolved moments"""
    if "ext_shapeHSM_HsmSourceMoments" in catalog.schema:
        hsm = catalog["ext_shapeHSM_HsmSourceMoments_xx"] + catalog["ext_shapeHSM_HsmSourceMoments_yy"]
    else:
        hsm = np.ones(len(catalog))*np.nan
    sdss = catalog["base_SdssShape_xx"] + catalog["base_SdssShape_yy"]
    if "ext_shapeHSM_HsmPsfMoments_xx" in catalog.schema:
        psf = catalog["ext_shapeHSM_HsmPsfMoments_xx"] + catalog["ext_shapeHSM_HsmPsfMoments_yy"]
    else:
        # LSST does not have shape.sdss.psf.  Could instead add base_PsfShape to catalog using
        # exposure.getPsf().computeShape(s.getCentroid()).getIxx()
        raise RuntimeError("No psf shape parameter found in catalog")
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
    mapperList = afwTable.SchemaMapper.join(afwTable.SchemaVector([matches[0].first.schema,
                                                                   matches[0].second.schema]),
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

def joinCatalogs(catalog1, catalog2, prefix1="cat1_", prefix2="cat2_"):
    # Make sure catalogs entries are all associated with the same object

    if not checkIdLists(catalog1, catalog2):
        raise RuntimeError("Catalogs with different sets of objects cannot be joined")

    mapperList = afwTable.SchemaMapper.join(afwTable.SchemaVector([catalog1[0].schema, catalog2[0].schema]),
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
    errKeys = dict((name, schemaKeys[name + "Sigma"]) for name in fluxKeys.keys() if
                   name + "Sigma" in schemaKeys)
    # Also check for any in HSC format
    fluxKeysHSC = dict((name, key) for name, key in schemaKeys.items() if
                       (re.search(r"^(flux\_\w+|\w+\_flux)$", name) or
                        re.search(r"^(\w+flux\_\w+|\w+\_flux)$", name))
                       and not re.search(r"^(\w+\_apcorr)$", name) and name + "_err" in schemaKeys)
    errKeysHSC = dict((name, schemaKeys[name + "_err"]) for name in fluxKeysHSC.keys() if
                       name + "_err" in schemaKeys)
    if len(fluxKeysHSC) > 0:
        fluxKeys.update(fluxKeysHSC)
        errKeys.update(errKeysHSC)
    if len(fluxKeys) == 0:
        raise RuntimeError("No flux keys found")
    return fluxKeys, errKeys

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
        row.set(apFlagKey, source.get(prefix+"flux_aperture_flag"))

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
    for source in catalog:
        row = newCatalog.addNew()
        row.assign(source, mapper)
        try:
            center = afwGeom.Point2D(source[prefix + "base_SdssCentroid_x"],
                                     source[prefix + "base_SdssCentroid_y"])
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

def addFootprintNPix(det, catalog, prefix=""):
    # Retrieve the number of pixels in an sources footprint and add to schema
    mapper = afwTable.SchemaMapper(catalog[0].schema)
    mapper.addMinimalSchema(catalog[0].schema)
    schema = mapper.getOutputSchema()
    fpName = prefix + "base_Footprint_nPix"
    fpKey = schema.addField(fpName, type="I", doc="Number of pixels in Footprint")
    fpFlag = schema.addField(fpName + "_flag", type="Flag", doc="Set to True for any fatal failure")
    newCatalog = afwTable.SourceCatalog(schema)
    newCatalog.reserve(len(catalog))
    for source in catalog:
        row = newCatalog.addNew()
        row.assign(source, mapper)
        try:
            footNpix = source.getFootprint().getNpix()
        except:
            footNpix = np.nan
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
    x0 = s.get(xKey)
    y0 = s.get(yKey)
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
    for key in fluxKeys.values() + errKeys.values():
        if len(catalog[key].shape) > 1:
            continue
        catalog[key][:] /= factor

    return catalog

def calibrateSourceCatalog(catalog, zp):
    """Calibrate catalog in the case of no meas_mosaic results using FLUXMAG0 as zp

    Requires a SourceCatalog and zeropoint as input.
    """
    # Convert to constant zero point, as for the coadds
    fluxKeys, errKeys = getFluxKeys(catalog.schema)
    for name, key in fluxKeys.items() + errKeys.items():
        factor = 10.0**(0.4*zp)
        if re.search(r"perture", name):
            factor = 10.0**(0.4*33.0)
        for src in catalog:
            src[key] /= factor
    return catalog

def calibrateCoaddSourceCatalog(catalog, zp):
    """Calibrate coadd catalog

    Requires a SourceCatalog and zeropoint as input.
    """
    # Convert to constant zero point, as for the coadds
    fluxKeys, errKeys = getFluxKeys(catalog.schema)
    for name, key in fluxKeys.items() + errKeys.items():
        factor = 10.0**(0.4*zp)
        for src in catalog:
            src[key] /= factor
    return catalog

def backoutApCorr(catalog):
    """Back out the aperture correction to all fluxes
    """
    ii = 0
    for src in catalog:
        for k in src.schema.getNames():
            if "_flux" in k and k[:-5] + "_apCorr" in src.schema.getNames() and "_apCorr" not in k:
                if ii == 0:
                    print "Backing out apcorr for:", k
                    ii += 1
                src[k] /= src[k[:-5] + "_apCorr"]
    return catalog

def matchJanskyToDn(matches):
    # LSST reads in a_net catalogs with flux in "janskys", so must convert back to DN
    JANSKYS_PER_AB_FLUX = 3631.0
    for m in matches:
        for k in m.first.schema.getNames():
            if "_flux" in k:
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
        print "WARNING: " + fluxToPlot + " not in fluxStrMap"
        return fluxToPlot


@contextmanager
def andCatalog(version):
    current = eups.findSetupVersion("astrometry_net_data")[0]
    eups.setup("astrometry_net_data", version, noRecursion=True)
    try:
        yield
    finally:
        eups.setup("astrometry_net_data", current, noRecursion=True)
