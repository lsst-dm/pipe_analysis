import os
import re

import numpy as np

from contextlib import contextmanager

from lsst.daf.persistence.safeFileIo import safeMakeDir
from lsst.pipe.base import Struct

import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable

__all__ = ["Filenamer", "Data", "Stats", "Enforcer", "MagDiff", "MagDiffMatches", "MagDiffCompare",
           "ApCorrDiffCompare", "AstrometryDiff", "psfSdssTraceSizeDiff", "psfHsmTraceSizeDiff", "MagDiffErr",
           "ApCorrDiffErr", "CentroidDiff", "CentroidDiffErr", "deconvMom", "deconvMomStarGal",
           "concatenateCatalogs", "joinMatches", "joinCatalogs", "getFluxKeys", "addApertureFluxesHSC",
           "addFpPoint", "calibrateSourceCatalogMosaic", "calibrateSourceCatalog",
           "calibrateCoaddSourceCatalog", "backoutApCorr", "matchJanskyToDn", "checkHscStack",
           "andCatalog"]

class Filenamer(object):
    """Callable that provides a filename given a style"""
    def __init__(self, butler, dataset, dataId={}):
        self.butler = butler
        self.dataset = dataset
        self.dataId = dataId
    def __call__(self, dataId, **kwargs):
        filename = self.butler.get(self.dataset + "_filename", self.dataId, **kwargs)[0]
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

class psfSdssTraceSizeDiff(object):
    """Functor to calculate trace radius size difference between object and psf model"""
    def __call__(self, catalog):
        srcSize = np.sqrt(0.5*(catalog["base_SdssShape_xx"] + catalog["base_SdssShape_yy"]))
        psfSize = np.sqrt(0.5*(catalog["base_SdssShape_psf_xx"] + catalog["base_SdssShape_psf_yy"]))
        sizeDiff = (srcSize - psfSize)/psfSize
        return np.array(sizeDiff)

class psfHsmTraceSizeDiff(object):
    """Functor to calculate trace radius size difference between object and psf model"""
    def __call__(self, catalog):
        srcSize = np.sqrt(0.5*(catalog["ext_shapeHSM_HsmSourceMoments_xx"] +
                               catalog["ext_shapeHSM_HsmSourceMoments_yy"]))
        psfSize = np.sqrt(0.5*(catalog["ext_shapeHSM_HsmPsfMoments_xx"] +
                               catalog["ext_shapeHSM_HsmPsfMoments_yy"]))
        sizeDiff = (srcSize - psfSize)/psfSize
        return np.array(sizeDiff)

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
    def __init__(self, component, first="first_", second="second_", centroid="base_SdssCentroid"):
        self.component = component
        self.first = first
        self.second = second
        self.centroid = centroid

    def __call__(self, catalog):
        first = self.first + self.centroid + "_" + self.component
        second = self.second + self.centroid + "_" + self.component
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
        raise TaskError("No psf shape parameter found in catalog")
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
    schema = mapperList[0].getOutputSchema()
    distanceKey = schema.addField("distance", type="Angle", doc="Distance between %s and %s" % (first, second))
    catalog = afwTable.BaseCatalog(schema)
    catalog.reserve(len(matches))
    for mm in matches:
        row = catalog.addNew()
        row.assign(mm.first, mapperList[0])
        row.assign(mm.second, mapperList[1])
        row.set(distanceKey, mm.distance*afwGeom.radians)
    return catalog

def joinCatalogs(catalog1, catalog2, prefix1="cat1_", prefix2="cat2_"):
    # Make sure catalogs entries are all associated with the same object
    idStrList = ["", ""]
    for i, cat in enumerate((catalog1, catalog2)):
        if "id" in cat.schema:
            idStrList[i] = "id"
        elif "objectId" in cat.schema:
            idStrList[i] = "objectId"
        else:
            raise RuntimeError("Cannot identify object id field (tried id and objectId)")

    if not np.all(catalog1[idStrList[0]] == catalog2[idStrList[1]]):
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
    if len(fluxKeys) == 0: # The schema is likely the HSC format
        fluxKeys = dict((name, key) for name, key in schemaKeys.items() if
                        re.search(r"^(flux\_\w+|\w+\_flux)$", name)
                        and not re.search(r"^(\w+\_apcorr)$", name) and name + "_err" in schemaKeys)
        errKeys = dict((name, schemaKeys[name + "_err"]) for name in fluxKeys.keys() if
                       name + "_err" in schemaKeys)
    if len(fluxKeys) == 0:
        raise TaskError("No flux keys found")
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

    return newCatalog

def calibrateSourceCatalogMosaic(dataRef, catalog, zp=27.0):
    """Calibrate catalog with meas_mosaic results

    Requires a SourceCatalog input.
    """
    result = applyMosaicResultsCatalog(dataRef, catalog, True)
    catalog = result.catalog
    ffp = result.ffp
    # Convert to constant zero point, as for the coadds
    factor = ffp.calib.getFluxMag0()[0]/10.0**(0.4*zp)

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

@contextmanager
def andCatalog(version):
    current = eups.findSetupVersion("astrometry_net_data")[0]
    eups.setup("astrometry_net_data", version, noRecursion=True)
    try:
        yield
    finally:
        eups.setup("astrometry_net_data", current, noRecursion=True)
