from __future__ import print_function

import os
import re

import astropy.coordinates as coord
import astropy.units as units
import numpy as np
import pandas as pd
import scipy.odr as scipyOdr
import scipy.optimize as scipyOptimize
import scipy.stats as scipyStats

from contextlib import contextmanager

from lsst.daf.persistence.safeFileIo import safeMakeDir
from lsst.pipe.base import Struct, TaskError

import lsst.afw.cameraGeom as cameraGeom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.pex.config as pexConfig
import lsst.verify as verify

try:
    from lsst.meas.mosaic.updateExposure import applyMosaicResultsCatalog
except ImportError:
    applyMosaicResultsCatalog = None

__all__ = ["Filenamer", "Data", "Stats", "Enforcer", "MagDiff", "MagDiffMatches", "MagDiffCompare",
           "AstrometryDiff", "AngularDistance", "TraceSize", "PsfTraceSizeDiff", "TraceSizeCompare",
           "PercentDiff", "E1Resids", "E2Resids", "E1ResidsHsmRegauss", "E2ResidsHsmRegauss",
           "FootNpixDiffCompare", "MagDiffErr", "MagDiffCompareErr", "ApCorrDiffErr",
           "CentroidDiff", "CentroidDiffErr", "deconvMom", "deconvMomStarGal",
           "concatenateCatalogs", "joinMatches", "matchAndJoinCatalogs", "checkIdLists", "checkPatchOverlap",
           "joinCatalogs", "getFluxKeys", "addColumnsToSchema", "addApertureFluxesHSC", "addFpPoint",
           "addFootprintNPix", "addRotPoint", "makeBadArray", "addFlag", "addIntFloatOrStrColumn",
           "calibrateSourceCatalogMosaic", "calibrateSourceCatalogPhotoCalib",
           "calibrateSourceCatalog", "calibrateCoaddSourceCatalog",
           "backoutApCorr", "matchNanojanskyToAB", "checkHscStack", "fluxToPlotString", "andCatalog",
           "writeParquet", "getRepoInfo", "findCcdKey", "getCcdNameRefList", "getDataExistsRefList",
           "orthogonalRegression", "distanceSquaredToPoly", "p1CoeffsFromP2x0y0", "p2p1CoeffsFromLinearFit",
           "lineFromP2Coeffs", "linesFromP2P1Coeffs", "makeEqnStr", "catColors", "setAliasMaps",
           "addAliasColumns", "addPreComputedColumns", "addMetricMeasurement", "updateVerifyJob",
           "computeMeanOfFrac", "calcQuartileClippedStats", "getSchema"]


NANOJANSKYS_PER_AB_FLUX = (0*units.ABmag).to_value(units.nJy)

def writeParquet(dataRef, table, badArray=None):
    """Write an afwTable to a desired ParquetTable butler dataset

    Parameters
    ----------
    dataRef : `lsst.daf.persistence.butlerSubset.ButlerDataRef`
        Reference to butler dataset.
    table : `lsst.afw.table.SourceCatalog`
       Table to be written to parquet.
    badArray : `numpy.ndarray`, optional
       Boolean array with same length as catalog whose values indicate whether the source was deemed
       inappropriate for qa analyses (`None` by default).

    Returns
    -------
    None

    Notes
    -----
    This function first converts the afwTable to an astropy table,
    then to a pandas DataFrame, which is then written to parquet
    format using the butler.  If qa_explorer is not
    available, then it will do nothing.
    """

    try:
        from lsst.pipe.tasks.parquetTable import ParquetTable
    except ImportError:
        import logging
        logging.warning('Parquet files will not be written (qa_explorer is not setup).')
        return

    if badArray is not None:
        # Add flag indicating source "badness" for qa analyses for the benefit of the Parquet files
        # being written to disk for subsequent interactive QA analysis.
        table = addFlag(table, badArray, "qaBad_flag", "Set to True for any source deemed bad for qa")
    df = table.asAstropy().to_pandas()
    df = df.set_index('id', drop=True)

    dataRef.put(ParquetTable(dataFrame=df))

class Filenamer(object):
    """Callable that provides a filename given a style"""
    def __init__(self, butler, dataset, dataId={}, subdir=""):
        self.butler = butler
        self.dataset = dataset
        self.dataId = dataId
        self.subdir = subdir

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
        if self.subdir:
            lastSlashInd = filename.rfind("/")
            filename = filename[:lastSlashInd] + "/" + self.subdir + "/" + filename[lastSlashInd + 1:]
        safeMakeDir(os.path.dirname(filename))
        return filename


class Data(Struct):
    def __init__(self, catalog, quantity, mag, signalToNoise, selection, color, error=None, plot=True):
        Struct.__init__(self, catalog=catalog[selection].copy(deep=True), quantity=quantity[selection],
                        mag=mag[selection], signalToNoise=signalToNoise[selection], selection=selection,
                        color=color, plot=plot, error=error[selection] if error is not None else None)


class Stats(Struct):
    def __init__(self, dataUsed, num, total, mean, stdev, forcedMean, median, clip, thresholdType,
                 thresholdValue):
        Struct.__init__(self, dataUsed=dataUsed, num=num, total=total, mean=mean, stdev=stdev,
                        forcedMean=forcedMean, median=median, clip=clip, thresholdType=thresholdType,
                        thresholdValue=thresholdValue)

    def __repr__(self):
        return ("Stats(mean={0.mean:.4f}; stdev={0.stdev:.4f}; num={0.num:d}; total={0.total:d}; "
                "median={0.median:.4f}; clip={0.clip:.4f}; forcedMean={0.forcedMean:}; "
                "thresholdType={0.thresholdType:s}; thresholdValue={0.thresholdValue:})".format(self))


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

    def __call__(self, catalog1, catalog2=None):
        catalog2 = catalog2 if catalog2 is not None else catalog1
        return -2.5*np.log10(catalog1[self.col1]/catalog2[self.col2])*self.unitScale


class MagDiffErr(object):
    """Functor to calculate magnitude difference error"""
    def __init__(self, col1, col2, unitScale=1.0):
        self.col1 = col1
        self.col2 = col2
        self.unitScale = unitScale

    def __call__(self, catalog):
        err1 = 2.5*np.log10(np.e)*(catalog[self.col1 + "Err"]/catalog[self.col1])
        err2 = 2.5*np.log10(np.e)*(catalog[self.col2 + "Err"]/catalog[self.col2])
        return np.sqrt(err1**2 + err2**2)*self.unitScale


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

    Note that the column entries are in flux units and converted to mags here.
    """
    def __init__(self, column, unitScale=1.0):
        self.column = column
        self.unitScale = unitScale

    def __call__(self, catalog):
        src1 = -2.5*np.log10(catalog["first_" + self.column])
        src2 = -2.5*np.log10(catalog["second_" + self.column])
        return (src1 - src2)*self.unitScale


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
        return (first*cosDec1 - second*cosDec2)*(1.0*geom.radians).asArcseconds()*self.unitScale

class AngularDistance(object):
    """Functor to calculate the Haversine angular distance between two points

    The Haversine formula, which determines the great-circle distance between
    two points on a sphere given their longitudes (ra) and latitudes (dec), is
    given by:

    distance =
    2arcsin(sqrt(sin**2((dec2-dec1)/2) + cos(del1)cos(del2)sin**2((ra1-ra2)/2)))

    Parameters
    ----------
    raStr1 : `str`
       The name of the column for the ra (in radians) of the first point
    decStr1 : `str`
       The name of the column for the dec (in radians) of the first point
    raStr2 : `str`
       The name of the column for the ra (in radians) of the second point
    decStr1 : `str`
       The name of the column for the dec (in radians) of the second point
    catalog : `lsst.afw.table.SourceCatalog`
       The source catalog under consideration containing columns representing
       the (ra, dec) coordinates for each object with names given by
       ``raStr1``, ``decStr1``, ``raStr2``, and ``decStr2``.

    Returns
    -------
    angularDistance : `numpy.ndarray`
       An array containing the Haversine angular distance (in radians) between
       the points:
       (``catalog``[``ra1Str``], ``catalog``[``dec1Str``]) and
       (``catalog``[``ra2Str``], ``catalog``[``dec2Str``]).
    """
    def __init__(self, raStr1, raStr2, decStr1, decStr2):
        self.raStr1 = raStr1
        self.raStr2 = raStr2
        self.decStr1 = decStr1
        self.decStr2 = decStr2

    def __call__(self, catalog):
        ra1 = catalog[self.raStr1]
        ra2 = catalog[self.raStr2]
        deltaRa = ra1 - ra2
        dec1 = catalog[self.decStr1]
        dec2 = catalog[self.decStr2]
        deltaDec = dec1 - dec2
        haverDeltaRa = np.sin(deltaRa/2.00)
        haverDeltaDec = np.sin(deltaDec/2.00)
        haverAlpha = np.sqrt(np.square(haverDeltaDec) + np.cos(dec1)*np.cos(dec2)*np.square(haverDeltaRa))
        angularDistance = 2.0*np.arcsin(haverAlpha)
        return angularDistance


class TraceSize(object):
    """Functor to calculate trace radius size for sources"""
    def __init__(self, column):
        self.column = column

    def __call__(self, catalog):
        srcSize = np.sqrt(0.5*(catalog[self.column + "_xx"] + catalog[self.column + "_yy"]))
        return np.array(srcSize)


class PsfTraceSizeDiff(object):
    """Functor to calculate trace radius size difference (%) between object and psf model"""
    def __init__(self, column, psfColumn):
        self.column = column
        self.psfColumn = psfColumn

    def __call__(self, catalog):
        srcSize = np.sqrt(0.5*(catalog[self.column + "_xx"] + catalog[self.column + "_yy"]))
        psfSize = np.sqrt(0.5*(catalog[self.psfColumn + "_xx"] + catalog[self.psfColumn + "_yy"]))
        sizeDiff = 100*(srcSize - psfSize)/(0.5*(srcSize + psfSize))
        return np.array(sizeDiff)


class TraceSizeCompare(object):
    """Functor to calculate trace radius size difference (%) between objects in matched catalog"""
    def __init__(self, column):
        self.column = column

    def __call__(self, catalog):
        srcSize1 = np.sqrt(0.5*(catalog["first_" + self.column + "_xx"] +
                                catalog["first_" + self.column + "_yy"]))
        srcSize2 = np.sqrt(0.5*(catalog["second_" + self.column + "_xx"] +
                                catalog["second_" + self.column + "_yy"]))
        sizeDiff = 100.0*(srcSize1 - srcSize2)/(0.5*(srcSize1 + srcSize2))
        return np.array(sizeDiff)


class PercentDiff(object):
    """Functor to calculate the percent difference between a given column entry in matched catalog"""
    def __init__(self, column):
        self.column = column

    def __call__(self, catalog):
        value1 = catalog["first_" + self.column]
        value2 = catalog["second_" + self.column]
        percentDiff = 100.0*(value1 - value2)/(0.5*(value1 + value2))
        return np.array(percentDiff)


class E1Resids(object):
    """Functor to calculate e1 ellipticity residuals for a given object and psf model"""
    def __init__(self, column, psfColumn, unitScale=1.0):
        self.column = column
        self.psfColumn = psfColumn
        self.unitScale = unitScale

    def __call__(self, catalog):
        srcE1 = ((catalog[self.column + "_xx"] - catalog[self.column + "_yy"])/
                 (catalog[self.column + "_xx"] + catalog[self.column + "_yy"]))
        psfE1 = ((catalog[self.psfColumn + "_xx"] - catalog[self.psfColumn + "_yy"])/
                 (catalog[self.psfColumn + "_xx"] + catalog[self.psfColumn + "_yy"]))
        e1Resids = srcE1 - psfE1
        return np.array(e1Resids)*self.unitScale


class E2Resids(object):
    """Functor to calculate e2 ellipticity residuals for a given object and psf model"""
    def __init__(self, column, psfColumn, unitScale=1.0):
        self.column = column
        self.psfColumn = psfColumn
        self.unitScale = unitScale

    def __call__(self, catalog):
        srcE2 = (2.0*catalog[self.column + "_xy"]/
                 (catalog[self.column + "_xx"] + catalog[self.column + "_yy"]))
        psfE2 = (2.0*catalog[self.psfColumn + "_xy"]/
                 (catalog[self.psfColumn + "_xx"] + catalog[self.psfColumn + "_yy"]))
        e2Resids = srcE2 - psfE2
        return np.array(e2Resids)*self.unitScale


class E1ResidsHsmRegauss(object):
    """Functor to calculate HSM e1 ellipticity residuals for a given object and psf model"""
    def __init__(self, unitScale=1.0):
        self.unitScale = unitScale

    def __call__(self, catalog):
        srcE1 = catalog["ext_shapeHSM_HsmShapeRegauss_e1"]
        psfE1 = ((catalog["ext_shapeHSM_HsmPsfMoments_xx"] - catalog["ext_shapeHSM_HsmPsfMoments_yy"])/
                 (catalog["ext_shapeHSM_HsmPsfMoments_xx"] + catalog["ext_shapeHSM_HsmPsfMoments_yy"]))
        e1Resids = srcE1 - psfE1
        return np.array(e1Resids)*self.unitScale


class E2ResidsHsmRegauss(object):
    """Functor to calculate HSM e1 ellipticity residuals for a given object and psf model"""
    def __init__(self, unitScale=1.0):
        self.unitScale = unitScale

    def __call__(self, catalog):
        srcE2 = catalog["ext_shapeHSM_HsmShapeRegauss_e2"]
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


class MagDiffCompareErr(object):
    """Functor to calculate magnitude difference error"""
    def __init__(self, column, unitScale=1.0):
        self.column = column
        self.unitScale = unitScale

    def __call__(self, catalog):
        err1 = 2.5*np.log10(np.e)*(catalog["first_" + self.column + "Err"]/catalog["first_" + self.column])
        err2 = 2.5*np.log10(np.e)*(catalog["second_" + self.column + "Err"]/catalog["second_" + self.column])
        return np.sqrt(err1**2 + err2**2)*self.unitScale


class ApCorrDiffErr(object):
    """Functor to calculate magnitude difference error"""
    def __init__(self, column, unitScale=1.0):
        self.column = column
        self.unitScale = unitScale

    def __call__(self, catalog):
        err1 = catalog["first_" + self.column + "Err"]
        err2 = catalog["second_" + self.column + "Err"]
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
        schema = getSchema(catalog)
        firstx = self.first + self.centroid + "_xErr"
        firsty = self.first + self.centroid + "_yErr"
        secondx = self.second + self.centroid + "_xErr"
        secondy = self.second + self.centroid + "_yErr"

        subkeys1 = [schema[firstx].asKey(), schema[firsty].asKey()]
        subkeys2 = [schema[secondx].asKey(), schema[secondy].asKey()]
        menu = {"x": 0, "y": 1}

        return np.hypot(catalog[subkeys1[menu[self.component]]],
                        catalog[subkeys2[menu[self.component]]])*self.unitScale


def deconvMom(catalog):
    """Calculate deconvolved moments"""
    schema = getSchema(catalog)
    if "ext_shapeHSM_HsmSourceMoments_xx" in schema:
        hsm = catalog["ext_shapeHSM_HsmSourceMoments_xx"] + catalog["ext_shapeHSM_HsmSourceMoments_yy"]
    else:
        hsm = np.ones(len(catalog))*np.nan
    sdss = catalog["base_SdssShape_xx"] + catalog["base_SdssShape_yy"]
    if "ext_shapeHSM_HsmPsfMoments_xx" in schema:
        psfXxName = "ext_shapeHSM_HsmPsfMoments_xx"
        psfYyName = "ext_shapeHSM_HsmPsfMoments_yy"
    elif "base_SdssShape_psf_xx" in schema:
        psfXxName = "base_SdssShape_psf_xx"
        psfYyName = "base_SdssShape_psf_yy"
    else:
        raise RuntimeError("No psf shape parameter found in catalog")
    psf = catalog[psfXxName] + catalog[psfYyName]
    return np.where(np.isfinite(hsm), hsm, sdss) - psf


def deconvMomStarGal(catalog):
    """Calculate P(star) from deconvolved moments"""
    rTrace = deconvMom(catalog)
    snr = catalog["base_PsfFlux_instFlux"]/catalog["base_PsfFlux_instFluxErr"]
    poly = (-4.2759879274 + 0.0713088756641*snr + 0.16352932561*rTrace - 4.54656639596e-05*snr*snr -
            0.0482134274008*snr*rTrace + 4.41366874902e-13*rTrace*rTrace + 7.58973714641e-09*snr*snr*snr +
            1.51008430135e-05*snr*snr*rTrace + 4.38493363998e-14*snr*rTrace*rTrace +
            1.83899834142e-20*rTrace*rTrace*rTrace)
    return 1.0/(1.0 + np.exp(-poly))


def concatenateCatalogs(catalogList):
    assert len(catalogList) > 0, "No catalogs to concatenate"
    template = catalogList[0]
    schema = getSchema(template)
    catalog = type(template)(schema)
    catalog.reserve(sum(len(cat) for cat in catalogList))
    for cat in catalogList:
        catalog.extend(cat, True)
    return catalog


def joinMatches(matches, first="first_", second="second_"):
    if not matches:
        return []

    mapperList = afwTable.SchemaMapper.join([matches[0].first.schema, matches[0].second.schema],
                                            [first, second])
    firstAliases = matches[0].first.schema.getAliasMap()
    secondAliases = matches[0].second.schema.getAliasMap()
    schema = mapperList[0].getOutputSchema()
    distanceKey = schema.addField("distance", type="Angle",
                                  doc="Distance between {0:s} and {1:s}".format(first, second))
    catalog = afwTable.BaseCatalog(schema)
    aliases = catalog.schema.getAliasMap()
    catalog.reserve(len(matches))
    for mm in matches:
        row = catalog.addNew()
        row.assign(mm.first, mapperList[0])
        row.assign(mm.second, mapperList[1])
        row.set(distanceKey, mm.distance*geom.radians)
    # make sure aliases get persisted to match catalog
    for k, v in firstAliases.items():
        aliases.set(first + k, first + v)
    for k, v in secondAliases.items():
        aliases.set(second + k, second + v)
    return catalog


def matchAndJoinCatalogs(catalog1, catalog2, matchRadius, raColStr="coord_ra", decColStr="coord_dec",
                         unit=units.rad, prefix1="first_", prefix2="second_", nthNeighbor=1, log=None,
                         matchXy=False, camera1=None, camera2=None):
    """Match two catalogs by Ra/Dec using astropy and join the results.

    Parameters
    ----------
    catalog1, catalog2 : `pandas.core.frame.DataFrame`
        The two source catalogs under on which to do the matching.
    matchRadius : `float`
        The match radius within which to consider two objects a match
        in units of arcsec if ``matchXy`` is `False` else in pixels
        (which will get converted to arcsec prior to matching).
    raColStr, decColStr : `str`, optional
        The string names for the ra and dec columns in the catalogs.
    unit : `astropy.units.core.IrreducibleUnit`
    prefix1, prefix2 : `str`, optional
        The prefix strings to prepend to the two catalogs upon joining them.
    nthNeighbor : `int`
        Which closest neighbor to search for in astropy's match_coordinates_sky
        function.  As per the astropy documentation, this is typically 1 as is
        appropriate for matching one set of coordinates to another. Another use
        case is 2, for matching a coordinate catalog against itself (i.e. when
        ``catalog1`` and ``catalog2`` are actually the same catalog, as in our
        overlaps identification case).  A value of 1 would inappropriate in
        that case because each point will find itself as the closest match).
        However, this is not robust agaist other zero distance matches (i.e.
        if any object other than the source itself has identical coordinates,
        the returned index may still be that of "self" as it will just choose
        between the two, and the choice may be that of "self").
        TODO: file an issue with astropy to robustly preclude self-matches.
    log : `lsst.log.Log`
        Logger object for logging messages.
    camera1, camera2 : `lsst.afw.cameraGeom.Camera`
        The cameras associated with ``catalog1`` and ``catalog2``.
    matchXy : `bool`, optional
        Whether to perform the matching in "x/y" coordinates (these
        are converted to pseudo-arcsec coordinates to make use of
        astropy's match_coordinates_sky function.

    Raises
    ------
    `RuntimeError`
        If ``matchXy`` is `True` but either ``camera1`` or ``camera2`` was not
        provided.

    Returns
    -------
    matches : `pandas.core.frame.DataFrame`
        The matched and joined catalog.  The parameters associated with
        ``catalog1`` and ``catalog2`` are prefixed with ``prefix1`` and
        ``prefix2``, respectively.
    """
    if matchXy:
        if camera1 is None or camera2 is None:
            raise RuntimeError(f"matchXy is True, but at least one of the two cameras was not provided: "
                               f"camera1 = {camera1}, camera2 = {camera2}")
        # The astropy matching requires "sky" coordinates, so convert to rough "arcsec" units
        pixelSize1 = camera1[0].getPixelSize()[0]  # rough arcsec/pixel (assumes square pixels)
        pixelSize2 = camera2[0].getPixelSize()[0]  # rough arcsec/pixel (assumes square pixels)
        matchRadius *= pixelSize1  # convert from pixel to arcsec
        skyCoords1 = coord.SkyCoord(catalog1["slot_Centroid_x"]*pixelSize1,
                                    catalog1["slot_Centroid_y"]*pixelSize1,
                                    unit=units.arcsec)
        skyCoords2 = coord.SkyCoord(catalog2["slot_Centroid_x"]*pixelSize2,
                                    catalog2["slot_Centroid_y"]*pixelSize2,
                                    unit=units.arcsec)
    else:
        skyCoords1 = coord.SkyCoord(catalog1[raColStr], catalog1[decColStr], unit=unit)
        skyCoords2 = coord.SkyCoord(catalog2[raColStr], catalog2[decColStr], unit=unit)
    inds, dists, _ = coord.match_coordinates_sky(skyCoords1, skyCoords2, nthneighbor=nthNeighbor)
    if nthNeighbor > 1:
        selfMatches = [i == ind for i, ind in enumerate(inds)]
        if sum(selfMatches) > 0 and log is not None:
            log.warn("There were {} objects self-matched by "
                     "astropy.coordinates.match_coordinates_sky()").format(sum(selfMatches))
    matchedIds = dists < matchRadius*units.arcsec
    matchedIndices = inds[matchedIds]
    matchedDistances = dists[matchedIds]
    matchFirst = catalog1[matchedIds].copy(deep=True)
    matchSecond = catalog2.iloc[matchedIndices].copy(deep=True)
    matchFirst.rename(columns=lambda x: prefix1 + x, inplace=True)
    matchSecond.rename(columns=lambda x: prefix2 + x, inplace=True)
    matchFirst.index = pd.RangeIndex(len(matchFirst.index))
    matchSecond.index = pd.RangeIndex(len(matchSecond.index))
    matches = pd.concat([matchFirst, matchSecond], axis=1)
    matches["distance"] = matchedDistances.rad
    return matches


def checkIdLists(catalog1, catalog2, prefix=""):
    # Check to see if two catalogs have an identical list of objects by id
    schema1 = getSchema(catalog1)
    schema2 = getSchema(catalog2)
    idStrList = ["", ""]
    for i, schema in enumerate([schema1, schema2]):
        if "id" in schema:
            idStrList[i] = "id"
        elif "objectId" in schema:
            idStrList[i] = "objectId"
        elif prefix + "id" in schema:
            idStrList[i] = prefix + "id"
        elif prefix + "objectId" in schema:
            idStrList[i] = prefix + "objectId"
        else:
            raise RuntimeError("Cannot identify object id field (tried id, objectId, " +
                               prefix + "id, and " + prefix + "objectId)")
    identicalIds = np.all(catalog1[idStrList[0]] == catalog2[idStrList[1]])
    return identicalIds


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
                xCen0, xCen1 = patchBBox0.getCenterX(), patchBBox1.getCenterX()
                yCen0, yCen1 = patchBBox0.getCenterY(), patchBBox1.getCenterY()
                if xCen0 == xCen1 or yCen0 == yCen1:  # omit patches that only overlap at corners
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

    Both are returned as dicts indexed on the flux name (e.g. "base_PsfFlux_instFlux" or
    "modelfit_CModel_instFlux").
    """

    fluxTypeStr = "_instFlux"
    fluxSchemaItems = schema.extract("*" + fluxTypeStr)
    # Do not include any flag fields (as determined by their type).  Also exclude
    # slot fields, as these would effectively duplicate whatever they point to.
    fluxKeys = dict((name, schemaItem.key) for name, schemaItem in list(fluxSchemaItems.items()) if
                    schemaItem.field.getTypeString() != "Flag" and
                    not name.startswith("slot"))
    errSchemaItems = schema.extract("*" + fluxTypeStr + "Err")
    errKeys = dict((name, schemaItem.key) for name, schemaItem in list(errSchemaItems.items()) if
                   name[:-len("Err")] in fluxKeys)

    # Also check for any in HSC format
    schemaKeys = dict((s.field.getName(), s.key) for s in schema)
    fluxKeysHSC = dict((name, key) for name, key in schemaKeys.items() if
                       (re.search(r"^(flux\_\w+|\w+\_flux)$", name) or
                        re.search(r"^(\w+flux\_\w+|\w+\_flux)$", name)) and not
                       re.search(r"^(\w+\_apcorr)$", name) and name + "_err" in schemaKeys)
    errKeysHSC = dict((name + "_err", schemaKeys[name + "_err"]) for name in fluxKeysHSC.keys() if
                      name + "_err" in schemaKeys)
    if fluxKeysHSC:
        fluxKeys.update(fluxKeysHSC)
        errKeys.update(errKeysHSC)

    if not fluxKeys:
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
        apFluxKey = schema.addField(apName + "_" + apRadii[ia] + "_instFlux", type="D",
                                    doc="flux within " + apRadii[ia].replace("_", ".") + "-pixel aperture",
                                    units="count")
        apFluxErrKey = schema.addField(apName + "_" + apRadii[ia] + "_instFluxErr", type="D",
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
            row.set(apFluxErrKey, source[prefix+"flux_aperture_err"][ia])
        row.set(apFlagKey, source[prefix + "flux_aperture_flag"])

    return newCatalog


def addFpPoint(det, catalog, prefix=""):
    # Compute Focal Plane coordinates for SdssCentroid of each source and add to schema
    mapper = afwTable.SchemaMapper(catalog[0].schema, shareAliasMap=True)
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
            center = geom.Point2D(source[xCentroidKey], source[yCentroidKey])
            pixelsToFocalPlane = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
            fpPoint = pixelsToFocalPlane.applyForward(center)
        except Exception:
            fpPoint = geom.Point2D(np.nan, np.nan)
            row.set(fpFlag, True)
        row.set(fpxKey, fpPoint[0])
        row.set(fpyKey, fpPoint[1])
    return newCatalog


def addFootprintNPix(catalog, fromCat=None, prefix=""):
    # Retrieve the number of pixels in an sources footprint and add to schema
    mapper = afwTable.SchemaMapper(catalog[0].schema, shareAliasMap=True)
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
        except Exception:
            raise
            footNpix = 0  # used to be np.nan, but didn't work.
            row.set(fpFlag, True)
        row.set(fpKey, footNpix)
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
    mapper = afwTable.SchemaMapper(catalog[0].schema, shareAliasMap=True)
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
        except Exception:
            rotPoint = geom.Point2D(np.nan, np.nan)
            row.set(rotFlag, True)
        row.set(rotxKey, rotPoint[0])
        row.set(rotyKey, rotPoint[1])

    return newCatalog


def makeBadArray(catalog, flagList=[], onlyReadStars=False, patchInnerOnly=True, tractInnerOnly=False):
    """Create a boolean array indicating sources deemed unsuitable for qa analyses

    Sets value to True for unisolated objects (deblend_nChild > 0), "sky" objects (merge_peak_sky),
    and any of the flags listed in self.config.analysis.flags.  If onlyReadStars is True, sets boolean
    as True for all galaxies classified as extended (base_ClassificationExtendedness_value > 0.5).  If
    patchInnerOnly is True (the default), sets the bad boolean array value to True for any sources
    for which detect_isPatchInner is False (to avoid duplicates in overlapping patches).  If
    tractInnerOnly is True, sets the bad boolean value to True for any sources for which
    detect_isTractInner is False (to avoid duplicates in overlapping patches).  Note, however, that
    the default for tractInnerOnly is False as we are currently only running these scripts at the
    per-tract level, so there are no tract duplicates (and omitting the "outer" ones would just leave
    an empty band around the tract edges).

    Parameters
    ----------
    catalog : `lsst.afw.table.SourceCatalog`
       The source catalog under consideration.
    flagList : `list`
       The list of flags for which, if any is set for a given source, set bad entry to `True` for
       that source.
    onlyReadStars : `bool`, optional
       Boolean indicating if you want to select objects classified as stars only (based on
       base_ClassificationExtendedness_value > 0.5, `False` by default).
    patchInnerOnly : `bool`, optional
       Whether to select only sources for which detect_isPatchInner is `True` (`True` by default).
    tractInnerOnly : `bool`, optional
       Whether to select only sources for which detect_isTractInner is `True` (`False` by default).
       Note that these scripts currently only ever run at the per-tract level, so we do not need
       to filter out sources for which detect_isTractInner is `False` as, with only one tract, there
       are no duplicated tract inner/outer sources.

    Returns
    -------
    badArray : `numpy.ndarray`
       Boolean array with same length as catalog whose values indicate whether the source was deemed
       inappropriate for qa analyses.
    """
    schema = getSchema(catalog)
    bad = np.zeros(len(catalog), dtype=bool)
    if isinstance(catalog, pd.DataFrame):
        if "detect_isPatchInner" in schema and patchInnerOnly:
            bad |= ~catalog["detect_isPatchInner"].values
        if "detect_isTractInner" in schema and tractInnerOnly:
            bad |= ~catalog["detect_isTractInner"].values
        bad |= catalog["deblend_nChild"].values > 0  # Exclude non-deblended (i.e. parents)
        if "merge_peak_sky" in schema:  # Exclude "sky" objects (currently only inserted in coadds)
            bad |= catalog["merge_peak_sky"].values
        for flag in flagList:
            bad |= catalog[flag].values
        if onlyReadStars and "base_ClassificationExtendedness_value" in schema:
            bad |= catalog["base_ClassificationExtendedness_value"].values > 0.5
    else:
        if "detect_isPatchInner" in schema and patchInnerOnly:
            bad |= ~catalog["detect_isPatchInner"]
        if "detect_isTractInner" in schema and tractInnerOnly:
            bad |= ~catalog["detect_isTractInner"]
        bad |= catalog["deblend_nChild"] > 0  # Exclude non-deblended (i.e. parents)
        if "merge_peak_sky" in schema:  # Exclude "sky" objects (currently only inserted in coadds)
            bad |= catalog["merge_peak_sky"]
        for flag in flagList:
            bad |= catalog[flag]
        if onlyReadStars and "base_ClassificationExtendedness_value" in schema:
            bad |= catalog["base_ClassificationExtendedness_value"] > 0.5
    return bad


def addFlag(catalog, badArray, flagName, doc="General failure flag"):
    """Add a flag for any sources deemed not appropriate for qa analyses

    Parameters
    ----------
    catalog : `lsst.afw.table.SourceCatalog`
       Source catalog to which the flag will be added.
    badArray : `numpy.ndarray`
       Boolean array with same length as catalog whose values indicate whether the flag flagName
       should be set for a given oject.
    flagName : `str`
       Name of flag to be set
    doc : `str`, optional
       Docstring for ``flagName``

    Raises
    ------
    `RuntimeError`
       If lengths of ``catalog`` and ``badArray`` are not equal.

    Returns
    -------
    newCatalog : `lsst.afw.table.SourceCatalog`
       Source catalog with ``flagName`` column added.
    """
    if len(catalog) != len(badArray):
        raise RuntimeError('Lengths of catalog and bad objects array do not match.')

    mapper = afwTable.SchemaMapper(catalog[0].schema, shareAliasMap=True)
    mapper.addMinimalSchema(catalog[0].schema)
    schema = mapper.getOutputSchema()
    badFlag = schema.addField(flagName, type="Flag", doc=doc)
    newCatalog = afwTable.SourceCatalog(schema)
    newCatalog.reserve(len(catalog))
    newCatalog.extend(catalog, mapper)

    for i, row in enumerate(newCatalog):
        row.set(badFlag, bool(badArray[i]))
    return newCatalog


def addIntFloatOrStrColumn(catalog, values, fieldName, fieldDoc, fieldUnits=""):
    """Add a column of values with name fieldName and doc fieldDoc to the catalog schema

    Parameters
    ----------
    catalog : `lsst.afw.table.SourceCatalog`
       Source catalog to which the column will be added.
    values : `list`, `numpy.ndarray`, or scalar of type `int`, `float`, or `str`
       The list of values to be added.  This list must have the same length as ``catalog`` or
       length 1 (to add a column with the same value for all objects).
    fieldName : `str`
       Name of the field to be added to the schema.
    fieldDoc : `str`
       Documentation string for the field to be added to the schema.
    fieldUnits : `str`, optional
       Units of the column to be added.

    Raises
    ------
    `RuntimeError`
       If type of all ``values`` is not one of `int`, `float`, or `str`.
    `RuntimeError`
       If length of ``values`` list is neither 1 nor equal to the ``catalog`` length.

    Returns
    -------
    newCatalog : `lsst.afw.table.SourceCatalog`
       Source catalog with ``fieldName`` column added.
    """
    if not isinstance(values, (list, np.ndarray)):
        if type(values) in (int, float, str):
            values = [values, ]
        else:
            raise RuntimeError(("Have only accommodated int, float, or str types.  Type provided was : "
                                "{}.  (Note, if you want to add a boolean flag column, use the addFlag "
                                "function.)").format(type(values)))
    if len(values) not in (len(catalog), 1):
        raise RuntimeError(("Length of values list must be either 1 or equal to the catalog length "
                            "({0:d}).  Length of values list provided was: {1:d}").
                           format(len(catalog), len(values)))

    size = None
    mapper = afwTable.SchemaMapper(catalog[0].schema, shareAliasMap=True)
    mapper.addMinimalSchema(catalog[0].schema)
    schema = mapper.getOutputSchema()

    if all(type(value) is int for value in values):
        fieldType = "I"
    elif all(type(value) is np.longlong for value in values):
        fieldType = "L"
    elif all(isinstance(value, float) for value in values):
        fieldType = "D"
    elif all(type(value) is str for value in values):
        fieldType = str
        size = len(max(values, key=len))
    else:
        raise RuntimeError(("Have only accommodated int, np.longlong, float, or str types.  Type provided "
                            "for the first element was: {} (and note that all values in the list must "
                            "have the same type.  Also note, if you want to add a boolean flag column, "
                            "use the addFlag function.)").format(type(values[0])))

    fieldKey = schema.addField(fieldName, type=fieldType, size=size, doc=fieldDoc, units=fieldUnits)

    newCatalog = afwTable.SourceCatalog(schema)
    newCatalog.reserve(len(catalog))

    newCatalog.extend(catalog, mapper)
    if len(values) == 1:
        for row in newCatalog:
            row.set(fieldKey, values[0])
    else:
        for i, row in enumerate(newCatalog):
            row.set(fieldKey, values[i])
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
    for fluxName, fluxKey in list(fluxKeys.items()) + list(errKeys.items()):
        if len(catalog[fluxKey].shape) > 1:
            continue
        catalog[fluxKey] /= factor
    return catalog


def calibrateSourceCatalogPhotoCalib(dataRef, catalog, photoCalibDataset, fluxKeys=None, zp=27.0):
    """Calibrate catalog with PhotoCalib results.

    The suite of external photometric calibrations in existence include:
    fgcm, fgcm_tract, jointcal, and meas_mosaic (but the latter has been
    effectively retired).

    Parameters
    ----------
    dataRef : `lsst.daf.persistence.butlerSubset.ButlerDataRef`
       The data reference for which the relevant datasets are to be retrieved.
    catalog : `lsst.afw.table.SourceCatalog` or `pandas.core.frame.DataFrame`
       The source catalog to which the calibrations will be applied.
    photoCalibDataset : `str`:
       The name of the photoCalib dataset to be applied (e.g.
       "jointcal_photoCalib" or "fgcm_tract_photoCalib").
    fluxKeys : `dict`, optional
       A `dict` of the flux keys to which the photometric calibration will
       be applied.  If not provided, the getFluxKeys function will be used
       to set it.
    zp : `float`, optional
       A constant zero point magnitude to which to scale all the fluxes in
       ``fluxKeys``.

    Returns
    -------
    catalog : `lsst.afw.table.SourceCatalog` or `pandas.core.frame.DataFrame`
       The calibrated catalog.

    Notes
    -----
    Adds columns with magnitudes and their associated errors for all flux
    fields returned by the getFluxKeys() method to the returned catalog.
    These columns are named <flux column>_mag and <flux column>_magErr,
    respectively.
    """
    photoCalib = dataRef.get(photoCalibDataset)
    schema = getSchema(catalog)
    # Scale to AB and convert to constant zero point, as for the coadds
    factor = NANOJANSKYS_PER_AB_FLUX/10.0**(0.4*zp)
    if fluxKeys is None:
        if isinstance(catalog, pd.DataFrame):
            fluxKeys = {flux: flux for flux in catalog.columns if flux.endswith("_instFlux")}
            errKeys = {flux + "Err": flux + "Err" for (flux, flux) in fluxKeys.items()}
        else:
            fluxKeys, errKeys = getFluxKeys(schema)

    magColsToAdd = []
    for fluxName, fluxKey in list(fluxKeys.items()):
        if len(catalog[fluxKey].shape) > 1:
            continue
        # photoCalib.instFluxToNanojansky() requires an error for each flux
        fluxErrKey = errKeys[fluxName + "Err"] if fluxName + "Err" in errKeys else None
        baseName = fluxName.replace("_instFlux", "")
        if fluxErrKey:
            if "Flux" in baseName:
                magsAndErrArray = photoCalib.instFluxToMagnitude(catalog, baseName)
                magColsToAdd.append((magsAndErrArray, baseName))
            calibratedFluxAndErrArray = photoCalib.instFluxToNanojansky(catalog, baseName)
            catalog[fluxKey] = calibratedFluxAndErrArray[:, 0]
            catalog[fluxErrKey] = calibratedFluxAndErrArray[:, 1]
        else:
            # photoCalib requires an error for each flux, but some don't
            # have one in the schema (currently only certain deblender
            # fields, e.g. deblend_psf_instFlux), so we compute the flux
            # correction factor from any slot flux (it only depends on
            # position, so any slot with a successful measurement will do)
            # and apply that to any flux entries that do not have errors.
            for fluxSlotName in catalog.schema.extract("slot*instFlux"):
                photoCalibFactor = None
                for src in catalog:
                    if np.isfinite(src[fluxSlotName]):
                        baseSlotName = fluxSlotName.replace("_instFlux", "")
                        photoCalibFactor = (photoCalib.instFluxToNanojansky(src, baseSlotName).value /
                                            src[fluxSlotName])
                        break
                if photoCalibFactor:
                    catalog[fluxKey] *= photoCalibFactor
                    break
        catalog[fluxKey] /= factor
        if fluxErrKey:
            catalog[fluxErrKey] /= factor

    for values, colName in magColsToAdd:
        fieldName = colName + "_mag"
        fieldDoc = "Magnitude calculated from " + colName
        catalog = addIntFloatOrStrColumn(catalog, values[:, 0], fieldName, fieldDoc, fieldUnits="mag")
        fieldName = colName + "_magErr"
        fieldDoc = "Magnitude error calculated from " + colName
        catalog = addIntFloatOrStrColumn(catalog, values[:, 1], fieldName, fieldDoc, fieldUnits="mag")

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

    Requires a SourceCatalog or pandas Dataframe and zeropoint as input.
    """
    # Convert to constant zero point, as for the coadds
    if isinstance(catalog, pd.DataFrame):
        fluxKeys = {flux: flux for flux in catalog.columns if flux.endswith("_instFlux")}
        errKeys = {flux + "Err": flux + "Err" for (flux, flux) in fluxKeys.items()}
    else:
        fluxKeys, errKeys = getFluxKeys(catalog.schema)

    factor = 10.0**(0.4*zp)
    for name, key in list(fluxKeys.items()) + list(errKeys.items()):
        catalog[key] /= factor
    return catalog


def backoutApCorr(catalog):
    """Back out the aperture correction to all fluxes
    """
    ii = 0
    fluxStr = "_instFlux"
    apCorrStr = "_apCorr"
    if isinstance(catalog, pd.DataFrame):
        keys = {flux: flux for flux in catalog.columns if (
            flux.endswith(fluxStr) or flux.endswith(apCorrStr))}
    else:
        keys = catalog.schema.getNames()
    for k in keys:
        if fluxStr in k and k[:-len(fluxStr)] + apCorrStr in keys and apCorrStr not in k:
            if ii == 0:
                print("Backing out aperture corrections to fluxes")
                ii += 1
            catalog[k] /= catalog[k[:-len(fluxStr)] + apCorrStr]
    return catalog


def matchNanojanskyToAB(matches):
    # LSST reads in catalogs with flux in "nanojanskys", so must convert to AB.
    # Using astropy units for conversion for consistency with PhotoCalib.
    if isinstance(matches, pd.DataFrame):
        schema = getSchema(matches)
        keys = [kk for kk in schema if kk.startswith("ref_") and "_flux" in kk]
        for k in keys:
            matches[k] = matches[k].apply(lambda x: x/NANOJANSKYS_PER_AB_FLUX)
    else:
        schema = matches[0].first.schema
        keys = [schema[kk].asKey() for kk in schema.getNames() if "_flux" in kk]
        for m in matches:
            for k in keys:
                m.first[k] /= NANOJANSKYS_PER_AB_FLUX
    return matches


def checkHscStack(metadata):
    """Check to see if data were processed with the HSC stack
    """
    try:
        hscPipe = metadata.getScalar("HSCPIPE_VERSION")
    except Exception:
        hscPipe = None
    return hscPipe


def fluxToPlotString(fluxToPlot):
    """Return a more succint string for fluxes for label plotting
    """
    fluxStrMap = {"base_PsfFlux_instFlux": "PSF",
                  "base_PsfFlux_flux": "PSF",
                  "base_PsfFlux": "PSF",
                  "base_GaussianFlux": "Gaussian",
                  "ext_photometryKron_KronFlux": "Kron",
                  "base_InputCount_value": "InputCount",
                  "modelfit_CModel_instFlux": "CModel",
                  "modelfit_CModel_flux": "CModel",
                  "modelfit_CModel": "CModel",
                  "base_CircularApertureFlux_12_0_instFlux": "CircApRad12pix",
                  "base_CircularApertureFlux_12_0": "CircApRad12pix",
                  "base_CircularApertureFlux_9_0_instFlux": "CircApRad9pix",
                  "base_CircularApertureFlux_9_0": "CircApRad9pix",
                  "base_CircularApertureFlux_25_0_instFlux": "CircApRad25pix",
                  "base_CircularApertureFlux_25_0": "CircApRad25pix"}
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
    from eups import Eups  # noqa Nothing else depends on eups, so prevent it from importing unless needed
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


def getRepoInfo(dataRef, coaddName=None, coaddDataset=None, doApplyExternalPhotoCalib=False,
                externalPhotoCalibName="jointcal", doApplyExternalSkyWcs=False,
                externalSkyWcsName="jointcal"):
    """Obtain the relevant repository information for the given dataRef

    Parameters
    ----------
    dataRef : `lsst.daf.persistence.butlerSubset.ButlerDataRef`
       The data reference for which the relevant repository information
       is to be retrieved.
    coaddName : `str`, optional
       The base name of the coadd (e.g. deep or goodSeeing) if
       ``dataRef`` is for coadd level processing (`None` by default).
    coaddDataset : `str`, optional
       The name of the coadd dataset (e.g. Coadd_forced_src or
       Coadd_meas) if ``dataRef`` is for coadd level processing
       (`None` by default).
    doApplyUberCal : `bool`, optional
       If `True`: Set the appropriate dataset type for the uber
       calibration from meas_mosaic.
       If `False` (the default): Set the dataset type to the source
       catalog from single frame processing.

    Raises
    ------
    `RuntimeError`
       If one of ``coaddName`` or ``coaddDataset`` is specified but
       the other is not.

    Returns
    -------
    result : `lsst.pipe.base.Struct`
       Result struct with components:

       - ``butler`` : the butler associated with ``dataRef``
         (`lsst.daf.persistence.Butler`).
       - ``camera`` : the camera associated with ``butler``
         (`lsst.afw.cameraGeom.Camera`).
       - ``dataId`` : the dataId associated with ``dataRef``
         (`lsst.daf.persistence.DataId`).
       - ``filterName`` : the name of the filter associated with ``dataRef``
         (`str`).
       - ``genericFilterName`` : a generic form for ``filterName`` (`str`).
       - ``ccdKey`` : the ccd key associated with ``dataId`` (`str` or `None`).
       - ``metadata`` : the metadata associated with ``butler`` and ``dataId``
         (`lsst.daf.base.propertyContainer.PropertyList`).
       - ``hscRun`` : string representing "HSCPIPE_VERSION" fits header if
         the data associated with ``dataRef``'s ``dataset`` were processed with
         the (now obsolete, but old reruns still exist) "HSC stack", None
         otherwise (`str` or `None`).
       - ``dataset`` : the dataset name ("src" if ``dataRef`` is visit level,
         coaddName + coaddDataset if ``dataRef`` is a coadd (`str`)
       - ``skyMap`` : the sky map associated with ``dataRef`` if it is a
         coadd (`lsst.skymap.SkyMap` or `None`).
       - ``wcs`` : the wcs of the coadd image associated with ``dataRef``
         -- only needed as a workaround for some old coadd catalogs that were
         persisted with all nan for ra dec (`lsst.afw.geom.SkyWcs` or `None`).
       - ``tractInfo`` : the tract information associated with ``dataRef`` if
         it is a coadd (`lsst.skymap.tractInfo.ExplicitTractInfo` or `None`).
    """
    if coaddName and not coaddDataset or not coaddName and coaddDataset:
        raise RuntimeError("If one of coaddName or coaddDataset is specified, the other must be as well.")

    butler = dataRef.getButler()
    camera = butler.get("camera")
    dataId = dataRef.dataId
    isCoadd = True if "patch" in dataId else False
    try:
        filterName = dataId["filter"]
    except KeyError:
        if isCoadd:
            filterName = butler.get(coaddName + "Coadd_calexp_filter", dataId)
        else:
            filterName = butler.get("calexp_filter", dataId)
        dataId['filter'] = filterName
    genericFilterName = afwImage.Filter(afwImage.Filter(filterName).getId()).getName()
    ccdKey = None if isCoadd else findCcdKey(dataId)
    # Check metadata to see if stack used was HSC
    metaStr = coaddName + coaddDataset + "_md" if coaddName else "calexp_md"
    metadata = butler.get(metaStr, dataId)
    hscRun = checkHscStack(metadata)
    catDataset = "src"
    skymap = butler.get(coaddName + "Coadd_skyMap") if coaddName else None
    wcs = None
    tractInfo = None
    if isCoadd:
        coaddImageName = "Coadd_calexp_hsc" if hscRun else "Coadd_calexp"  # To get the coadd's WCS
        wcs = butler.get(coaddName + coaddImageName + "_wcs", dataId)
        catDataset = coaddName + coaddDataset
        tractInfo = skymap[dataId["tract"]]
    photoCalibDataset = None
    if doApplyExternalPhotoCalib:
        photoCalibDataset = "fcr_hsc" if hscRun else externalPhotoCalibName + "_photoCalib"
    skyWcsDataset = None
    if doApplyExternalSkyWcs:
        skyWcsDataset = "wcs_hsc" if hscRun else externalSkyWcsName + "_wcs"
        skymap = skymap if skymap else butler.get("deepCoadd_skyMap")
        try:
            tractInfo = skymap[dataId["tract"]]
        except:
            tractInfo = None
    return Struct(
        butler=butler,
        camera=camera,
        dataId=dataId,
        filterName=filterName,
        genericFilterName=genericFilterName,
        ccdKey=ccdKey,
        metadata=metadata,
        hscRun=hscRun,
        catDataset=catDataset,
        photoCalibDataset=photoCalibDataset,
        skyWcsDataset=skyWcsDataset,
        skymap=skymap,
        wcs=wcs,
        tractInfo=tractInfo,
    )


def findCcdKey(dataId):
    """Determine the convention for identifying a "ccd" for the current camera

    Parameters
    ----------
    dataId : `instance` of `lsst.daf.persistence.DataId`

    Raises
    ------
    `RuntimeError`
       If "ccd" key could not be identified from the current hardwired list.

    Returns
    -------
    ccdKey : `str`
       The string associated with the "ccd" key.
    """
    ccdKey = None
    ccdKeyList = ["ccd", "sensor", "camcol", "detector", "ccdnum"]
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


def fLinear(p, x):
    return p[0] + p[1]*x


def fQuadratic(p, x):
    return p[0] + p[1]*x + p[2]*x**2


def fCubic(p, x):
    return p[0] + p[1]*x + p[2]*x**2 + p[3]*x**3


def orthogonalRegression(x, y, order, initialGuess=None):
    """Perform an Orthogonal Distance Regression on the given data

    Parameters
    ----------
    x, y : `array`
       Arrays of x and y data to fit
    order : `int`, optional
       Order of the polynomial to fit
    initialGuess : `list` of `float`, optional
       List of the polynomial coefficients (highest power first) of an initial guess to feed to
       the ODR fit.  If no initialGuess is provided, a simple linear fit is performed and used
       as the guess (`None` by default).

    Returns
    -------
    result : `list` of `float`
       List of the fit coefficients (highest power first to mimic `numpy.polyfit` return).
    """
    if initialGuess is None:
        linReg = scipyStats.linregress(x, y)
        initialGuess = [linReg[0], linReg[1]]
        for i in range(order - 1):  # initialGuess here is linear, so need to pad array to match order
            initialGuess.insert(0, 0.0)
    if order == 1:
        odrModel = scipyOdr.Model(fLinear)
    elif order == 2:
        odrModel = scipyOdr.Model(fQuadratic)
    elif order == 3:
        odrModel = scipyOdr.Model(fCubic)
    else:
        raise RuntimeError("Order must be between 1 and 3 (value requested, {:}, not accommodated)".
                           format(order))
    odrData = scipyOdr.Data(x, y)
    orthDist = scipyOdr.ODR(odrData, odrModel, beta0=initialGuess)
    orthRegFit = orthDist.run()

    return list(reversed(orthRegFit.beta))


def distanceSquaredToPoly(x1, y1, x2, poly):
    """Calculate the square of the distance between point (x1, y1) and poly at x2

    Parameters
    ----------
    x1, y1 : `float`
       Point from which to calculate the square of the distance to the
       polynomial ``poly``.
    x2 : `float`
       Position on x axis from which to calculate the square of the distance
       between (``x1``, ``y1``) and ``poly`` (the position of the tangent of
       the polynomial curve closest to point (``x1``, ``y1``)).
    poly : `numpy.lib.polynomial.poly1d`
       Numpy polynomial fit from which to calculate the square of the distance
       to (``x1``, ``y1``) at ``x2``.

    Returns
    -------
    result : `float`
       Square of the distance between (``x1``, ``y1``) and ``poly`` at ``x2``
    """
    return (x2 - x1)**2 + (poly(x2) - y1)**2


def p1CoeffsFromP2x0y0(p2Coeffs, x0, y0):
    """Compute Ivezic P1 coefficients using the P2 coeffs and origin (x0, y0)

    Reference: Ivezic et al. 2004 (2004AN....325..583I)

    theta = arctan(mP1), where mP1 is the slope of the equivalent straight
                         line (the P1 line) from the P2 coeffs in the (x, y)
                         coordinate system and x = c1 - c2, y = c2 - c3
    P1 = cos(theta)*c1 + ((sin(theta) - cos(theta))*c2 - sin(theta)*c3 + deltaP1
    P1 = 0 at x0, y0 ==> deltaP1 = -cos(theta)*x0 - sin(theta)*y0

    Parameters
    ----------
    p2Coeffs : `list` of `float`
       List of the four P2 coefficients from which, along with the origin point
       (``x0``, ``y0``), to compute/derive the associated P1 coefficients.
    x0, y0 : `float`
       Coordinates at which to set P1 = 0 (i.e. the P1/P2 axis origin).

    Returns
    -------
    p1Coeffs: `list` of `float`
       The four P1 coefficients.
    """
    mP1 = p2Coeffs[0]/p2Coeffs[2]
    cosTheta = np.cos(np.arctan(mP1))
    sinTheta = np.sin(np.arctan(mP1))
    deltaP1 = -cosTheta*x0 - sinTheta*y0
    p1Coeffs = [cosTheta, sinTheta - cosTheta, -sinTheta, deltaP1]

    return p1Coeffs


def p2p1CoeffsFromLinearFit(m, b, x0, y0):
    """Derive the Ivezic et al. 2004 P2 and P1 equations based on linear fit

    Where the linear fit is to the given region in color-color space.
    Reference: Ivezic et al. 2004 (2004AN....325..583I)

    For y = m*x + b fit, where x = c1 - c2 and y = c2 - c3,
    P2 = (-m*c1 + (m + 1)*c2 - c3 - b)/sqrt(m**2 + 1)
    P2norm = P2/sqrt[(m**2 + (m + 1)**2 + 1**2)/(m**2 + 1)]

    P1 = cos(theta)*x + sin(theta)*y + deltaP1, theta = arctan(m)
    P1 = cos(theta)*(c1 - c2) + sin(theta)*(c2 - c3) + deltaP1
    P1 = cos(theta)*c1 + ((sin(theta) - cos(theta))*c2 - sin(theta)*c3 + deltaP1
    P1 = 0 at x0, y0 ==> deltaP1 = -cos(theta)*x0 - sin(theta)*y0

    Parameters
    ----------
    m : `float`
       Slope of line to convert.
    b : `float`
       Intercept of line to convert.
    x0, y0 : `float`
       Coordinates at which to set P1 = 0.

    Returns
    -------
    result : `lsst.pipe.base.Struct`
       Result struct with components:

       - ``p2Coeffs`` : four P2 equation coefficents (`list` of `float`).
       - ``p1Coeffs`` : four P1 equation coefficents (`list` of `float`).
    """
    # Compute Ivezic P2 coefficients using the linear fit slope and intercept
    scaleFact = np.sqrt(m**2 + 1.0)
    p2Coeffs = [-m/scaleFact, (m + 1.0)/scaleFact, -1.0/scaleFact, -b/scaleFact]
    p2Norm = 0.0
    for coeff in p2Coeffs[:-1]:  # Omit the constant normalization term
        p2Norm += coeff**2
    p2Norm = np.sqrt(p2Norm)
    p2Coeffs /= p2Norm

    # Compute Ivezic P1 coefficients equation using the linear fit slope and
    # point (x0, y0) as the origin
    p1Coeffs = p1CoeffsFromP2x0y0(p2Coeffs, x0, y0)

    return Struct(
        p2Coeffs=p2Coeffs,
        p1Coeffs=p1Coeffs,
    )


def lineFromP2Coeffs(p2Coeffs):
    """Compute P1 line in color-color space for given set P2 coefficients

    Reference: Ivezic et al. 2004 (2004AN....325..583I)

    Parameters
    ----------
    p2Coeffs : `list` of `float`
       List of the four P2 coefficients.

    Returns
    -------
    result : `lsst.pipe.base.Struct`
       Result struct with components:

       - ``mP1`` : associated slope for P1 in color-color coordinates (`float`).
       - ``bP1`` : associated intercept for P1 in color-color coordinates
                   (`float`).
    """
    mP1 = p2Coeffs[0]/p2Coeffs[2]
    bP1 = -p2Coeffs[3]*np.sqrt(mP1**2 + (mP1 + 1.0)**2 + 1.0)
    return Struct(
        mP1=mP1,
        bP1=bP1,
    )


def linesFromP2P1Coeffs(p2Coeffs, p1Coeffs):
    """Derive P1/P2 axes in color-color space based on the P2 and P1 coeffs

    Reference: Ivezic et al. 2004 (2004AN....325..583I)

    Parameters
    ----------
    p2Coeffs : `list` of `float`
       List of the four P2 coefficients.
    p1Coeffs : `list` of `float`
       List of the four P1 coefficients.

    Returns
    -------
    result : `lsst.pipe.base.Struct`
       Result struct with components:

       - ``mP2``, ``mP1`` : associated slopes for P2 and P1 in color-color
                            coordinates (`float`).
       - ``bP2``, ``bP1`` : associated intercepts for P2 and P1 in color-color
                            coordinates (`float`).
       - ``x0``, ``y0`` : x and y coordinates of the P2/P1 axes origin in
                          color-color coordinates (`float`).
    """
    p1Line = lineFromP2Coeffs(p2Coeffs)
    mP1 = p1Line.mP1
    bP1 = p1Line.bP1

    cosTheta = np.cos(np.arctan(mP1))
    sinTheta = np.sin(np.arctan(mP1))

    def func2(x):
        y = [cosTheta*x[0] + sinTheta*x[1] + p1Coeffs[3], mP1*x[0] - x[1] + bP1]
        return y

    x0y0 = scipyOptimize.fsolve(func2, [1, 1])
    mP2 = -1.0/mP1
    bP2 = x0y0[1] - mP2*x0y0[0]
    return Struct(
        mP2=mP2,
        bP2=bP2,
        mP1=mP1,
        bP1=bP1,
        x0=x0y0[0],
        y0=x0y0[1],
    )


def makeEqnStr(varName, coeffList, exponentList):
    """Make a string-formatted equation

    Parameters
    ----------
    varName : `str`
       Name of the equation to be stringified.
    coeffList : `list` of `float`
       List of equation coefficients (matched to exponenets in ``exponentList`` list).
    exponentList : `list` of `str`
       List of equation exponents (matched to coefficients in ``coeffList`` list).

    Raises
    ------
    `RuntimeError`
       If lengths of ``coeffList`` and ``exponentList`` are not equal.

    Returns
    -------
    eqnStr : `str`
       The stringified equation of the form:
       varName = coeffList[0]exponent[0] + ... + coeffList[n-1]exponent[n-1].
    """
    if len(coeffList) != len(exponentList):
        raise RuntimeError("Lengths of coeffList ({0:d}) and exponentList ({1:d}) are not equal".
                           format(len(coeffList), len(exponentList)))

    eqnStr = varName + " = "
    for i, (coeff, band) in enumerate(zip(coeffList, exponentList)):
        coeffStr = "{:.3f}".format(abs(coeff)) + band
        plusMinus = " $-$ " if coeff < 0.0 else " + "
        if i == 0:
            eqnStr += plusMinus.strip(" ") + coeffStr
        else:
            eqnStr += plusMinus + coeffStr

    return eqnStr


def catColors(c1, c2, magsCat, goodArray=None):
    """Compute color for a set of filters given a catalog of magnitudes by filter

    Parameters
    ----------
    c1, c2 : `str`
       String representation of the filters from which to compute the color.
    magsCat : `dict` of `numpy.ndarray`
       Dict of arrays of magnitude values.  Dict keys are the string representation of the filters.
    goodArray : `numpy.ndarray`, optional
       Boolean array with same length as the magsCat arrays whose values indicate whether the
       source was deemed "good" for intended use.  If `None`, all entries are considered "good"
       (`None` by default).

    Raises
    ------
    `RuntimeError`
       If lengths of ``goodArray`` and ``magsCat`` arrays are not equal.

    Returns
    -------
    `numpy.ndarray` of "good" colors (magnitude differeces).
    """
    if goodArray is None:
        goodArray = np.ones(len(magsCat[c1]), dtype=bool)

    if len(goodArray) != len(magsCat[c1]):
        raise RuntimeError("Lengths of goodArray ({0:d}) and magsCat ({1:d}) are not equal".
                           format(len(goodArray), len(magsCat[c1])))

    return (magsCat[c1] - magsCat[c2])[goodArray]


def setAliasMaps(catalog, aliasDictList, prefix=""):
    """Set an alias map for differing schema naming conventions

    Parameters
    ----------
    catalog : `lsst.afw.table.SourceCatalog`
       The source catalog to which the mapping will be added.
    aliasDictList : `dict` of `str` or `list` of `dict` of `str`
       A `list` of `dict` or single `dict` representing the alias mappings to
       be added to ``catalog``'s schema with the key representing the new
       name to be mapped to the value which represents the old name.  Note
       that the mapping will only be added if the old name exists in
       ``catalog``'s schema.

    prefix : `str`, optional
       This `str` will be prepended to the alias names (used, e.g., in matched
       catalogs for which "src_" and "ref_" prefixes have been added to all
       schema names).  Both the old and new names have ``prefix`` associated
       with them (default is an empty string).

    Raises
    ------
    `RuntimeError`
       If not all elements in ``aliasDictList`` are instances of type `dict` or
       `lsst.pex.config.dictField.Dict`.

    Returns
    -------
    catalog : `lsst.afw.table.SourceCatalog`
       The source catalog with the alias mappings added to the schema.
    """
    if isinstance(aliasDictList, dict):
        aliasDictList = [aliasDictList, ]
    if not all(isinstance(aliasDict, (dict, pexConfig.dictField.Dict)) for aliasDict in aliasDictList):
        raise RuntimeError("All elements in aliasDictList must be instances of type dict")
    aliasMap = catalog.schema.getAliasMap()
    for aliasDict in aliasDictList:
        for newName, oldName in aliasDict.items():
            if prefix + oldName in catalog.schema:
                aliasMap.set(prefix + newName, prefix + oldName)
    return catalog


def addAliasColumns(catalog, aliasDictList, prefix=""):
    """Copy columns from an alias map for differing schema naming conventions.

    Parameters
    ----------
    catalog : `lsst.afw.table.SourceCatalog`
        The source catalog to which the mapping columns will be added.
    aliasDictList : `dict` of `str` or `list` of `dict` of `str`
        A `list` of `dict` or single `dict` representing the column "mappings"
        to be added to ``catalog``'s schema, i.e. a new column with the "new"
        name will by added to the catalog which is simply a duplicate of the
        "old" name column.  Note that the new column will only be added if
        the column with the old name exists in ``catalog``'s schema.
    prefix : `str`, optional
        This `str` will be prepended to the alias names (used, e.g., in matched
        catalogs for which "src_" and "ref_" prefixes have been added to all
        schema names).  Both the old and new names have ``prefix`` associated
        with them (default is an empty string).

    Raises
    ------
    `RuntimeError`
        If not all elements in ``aliasDictList`` are instances of type `dict`
        or `lsst.pex.config.dictField.Dict`.

    Returns
    -------
    catalog : `lsst.afw.table.SourceCatalog`
        The source catalog with the "alias" columns added to the schema.
    """
    if isinstance(aliasDictList, dict):
        aliasDictList = [aliasDictList, ]
    if not all(isinstance(aliasDict, (dict, pexConfig.dictField.Dict)) for aliasDict in aliasDictList):
        raise RuntimeError("All elements in aliasDictList must be instances of type dict")
    for aliasDict in aliasDictList:
        for newName, oldName in aliasDict.items():
            if oldName in catalog.schema and newName not in catalog.schema:
                fieldDoc = catalog.schema[oldName].asField().getDoc()
                fieldUnits = catalog.schema[oldName].asField().getUnits()
                fieldType = catalog.schema[oldName].asField().getTypeString()
                if fieldType == "Flag":
                    catalog = addFlag(catalog, catalog[oldName], newName, doc=fieldDoc)
                else:
                    catalog = addIntFloatOrStrColumn(catalog, catalog[oldName], newName, fieldDoc,
                                                     fieldUnits=fieldUnits)
    return catalog


def addPreComputedColumns(catalog, fluxToPlotList, toMilli=False, unforcedCat=None):
    """Add column entries for a set of pre-computed values

    This is for the parquet tables to facilitate the interactive
    drilldown analyses.

    Parameters
    ----------
    catalog : `lsst.afw.table.SourceCatalog`
       The source catalog to which the columns will be added.
    fluxToPlotList : `list`
       List of flux field names to make mag(fluxName) - mag(PsfFlux) columns.
    toMilli : `bool`, optional
       Whether to use units of "milli" (e.g. mmag, mas).
    unforcedCat : `lsst.afw.table.SourceCatalog`, optional
       If `catalog` is a coadd forced catalog, this is its associated unforced
       catalog for direct comparison of forced vs. unforced parameters.

    Returns
    -------
    catalog : `lsst.afw.table.SourceCatalog`
       The source catalog with the columns of pre-computed values added.
    """
    unitScale = 1000.0 if toMilli else 1.0
    fieldUnits = " (mmag)" if toMilli else " (mag)"
    for col in fluxToPlotList:
        colStr = fluxToPlotString(col)
        if col + "_instFlux" in catalog.schema:
            compCol = "base_PsfFlux"
            compColStr = fluxToPlotString(compCol)
            fieldName = colStr + "-" + compColStr + "_magDiff_" + fieldUnits.strip(" ()")
            fieldDoc = "Magnitude difference: " + colStr + "-" + compCol + fieldUnits
            parameterFunc = MagDiff(col + "_instFlux", compCol + "_instFlux", unitScale=unitScale)
            magDiff = parameterFunc(catalog)
            parameterFunc = MagDiffErr(col + "_instFlux", compCol + "_instFlux", unitScale=unitScale)
            magDiffErr = parameterFunc(catalog)
            catalog = addIntFloatOrStrColumn(catalog, magDiff, fieldName, fieldDoc,
                                             fieldUnits=fieldUnits.strip(" ()"))
            fieldErrName = colStr + "-" + compColStr + "_magDiffErr_" + fieldUnits.strip(" ()")
            fieldErrDoc = "Error of: " + fieldDoc
            catalog = addIntFloatOrStrColumn(catalog, magDiffErr, fieldErrName, fieldErrDoc,
                                             fieldUnits=fieldUnits.strip(" ()"))


    for compareCol, psfCompareCol, compareStr in [["base_SdssShape", "base_SdssShape_psf", "Sdss"],
                                                  ["ext_shapeHSM_HsmSourceMoments",
                                                   "ext_shapeHSM_HsmPsfMoments", "Hsm"]]:
        if compareCol + "_xx" in catalog.schema:
            # Source Trace
            fieldUnits = " (pixel)"
            fieldName = "trace" + compareStr + "_" + fieldUnits.strip(" ()")
            fieldDoc = fieldName + " = sqrt(0.5*(Ixx + Iyy))" + fieldUnits
            parameterFunc = TraceSize(compareCol)
            traceSize = parameterFunc(catalog)
            catalog = addIntFloatOrStrColumn(catalog, traceSize, fieldName, fieldDoc,
                                             fieldUnits=fieldUnits.strip(" ()"))
            fieldName = "trace" + compareStr + "_fwhm_" + fieldUnits.strip(" ()")
            fieldDoc = fieldName + " = 2.0*np.sqrt(2.0*np.log(2.0))*Trace" + fieldUnits
            fwhmSize = 2.0*np.sqrt(2.0*np.log(2.0))*traceSize
            catalog = addIntFloatOrStrColumn(catalog, fwhmSize, fieldName, fieldDoc,
                                             fieldUnits=fieldUnits.strip(" ()"))

            # Source Trace - PSF model Trace
            fieldUnits = " (%)"
            fieldName =  "psfTrace" + compareStr + "Diff_percent"
            fieldDoc = fieldName + " = srcTrace" + compareStr + " - psfModelTrace" + compareStr + fieldUnits
            parameterFunc = PsfTraceSizeDiff(compareCol, psfCompareCol)
            psfTraceSizeDiff = parameterFunc(catalog)
            catalog = addIntFloatOrStrColumn(catalog, psfTraceSizeDiff, fieldName, fieldDoc,
                                             fieldUnits=fieldUnits.strip(" ()"))

            # Source - PSF model E1/E2 Residuals
            fieldUnits = " (milli)" if toMilli else ""
            fieldName = "e1Resids" + compareStr + "_" + fieldUnits.strip(" ()")
            fieldDoc = fieldName + " = src(e1) - psfModel(e1), e1 = (Ixx - Iyy)/(Ixx + Iyy)" + fieldUnits
            parameterFunc = E1Resids(compareCol, psfCompareCol, unitScale)
            e1Resids = parameterFunc(catalog)
            catalog = addIntFloatOrStrColumn(catalog, e1Resids, fieldName, fieldDoc)
            fieldName = "e2Resids" + compareStr + "_" + fieldUnits.strip(" ()")
            fieldDoc = fieldName + " = src(e2) - psfModel(e2), e2 = 2*Ixy/(Ixx + Iyy)" + fieldUnits
            parameterFunc = E2Resids(compareCol, psfCompareCol, unitScale)
            e2Resids = parameterFunc(catalog)
            catalog = addIntFloatOrStrColumn(catalog, e2Resids, fieldName, fieldDoc)


    # HSM Regauss E1/E2 resids
    fieldUnits = " (milli)" if toMilli else ""
    if "ext_shapeHSM_HsmShapeRegauss_e1" in catalog.schema:
        fieldName = "e1ResidsHsmRegauss_" + fieldUnits.strip(" ()")
        fieldDoc = fieldName + " = src(e1) - hsmPsfMoments(e1), e1 = (Ixx - Iyy)/(Ixx + Iyy)" + fieldUnits
        parameterFunc = E1ResidsHsmRegauss(unitScale=unitScale)
        e1ResidsHsmRegauss = parameterFunc(catalog)
        catalog = addIntFloatOrStrColumn(catalog, e1ResidsHsmRegauss, fieldName, fieldDoc)
    if "ext_shapeHSM_HsmShapeRegauss_e2" in catalog.schema:
        fieldName = "e2ResidsHsmRegauss_" + fieldUnits.strip(" ()")
        fieldDoc = fieldName + " = src(e2) - hsmPsfMoments(e2), e2 = (Ixx - Iyy)/(Ixx + Iyy)" + fieldUnits
        parameterFunc = E2ResidsHsmRegauss(unitScale=unitScale)
        e2ResidsHsmRegauss = parameterFunc(catalog)
        catalog = addIntFloatOrStrColumn(catalog, e2ResidsHsmRegauss, fieldName, fieldDoc)

    if "base_SdssShape_xx" in catalog.schema:
        fieldName = "deconvMoments"
        fieldDoc = "Deconvolved moments"
        deconvMoments = deconvMom(catalog)
        catalog = addIntFloatOrStrColumn(catalog, deconvMoments, fieldName, fieldDoc)

    if unforcedCat:
        fieldUnits = " (mmag)" if toMilli else " (mag)"
        for col in fluxToPlotList:
            colStr = fluxToPlotString(col)
            fieldName = "compareUnforced_" + colStr + "_magDiff_" + fieldUnits.strip(" ()")
            fieldDoc = "Compare forced - unforced" + colStr + " (mmag)"
            parameterFunc = MagDiff(col + "_instFlux", col + "_instFlux", unitScale=unitScale)
            magDiff = parameterFunc(catalog, unforcedCat)
            catalog = addIntFloatOrStrColumn(catalog, magDiff, fieldName, fieldDoc,
                                             fieldUnits=fieldUnits.strip(" ()"))

    return catalog


def addMetricMeasurement(job, metricName, metricValue, measExtrasDictList=None):
    """Add a measurement to a lsst.verify.Job instance

    TODO: this functionality will likely be moved to MeasurementSet in
          lsst.verify per DM-12655.

    Parameters
    ----------
    job : `lsst.verify.Job`
       The verify Job to add the measurement to.
    metricName : `str`
       The name of the metric to be added.
    metricValue : `float`, `int`, `bool`, `str`
       The value of the metric to be added.
    measExtrasDictList : `list` of `dict`, optional
       A dict of key value pairs of any "extras" to be added to the
       metric measurement.  All of the following keys must be provided for
       each `dict` in the `list`: "name", "value", "label", "description".
       The "label" is meant to be a `str` suitable for plot axis labelling.

    Returns
    -------
    job : `lsst.verify.Job`
       The updated job with the new measurement added.
    """
    meas = verify.Measurement(job.metrics[metricName], metricValue)
    if measExtrasDictList:
        for extra in measExtrasDictList:
            meas.extras[extra["name"]] = verify.Datum(extra["value"], label=extra["label"],
                                                      description=extra["description"])
    job.measurements.insert(meas)
    return job


def updateVerifyJob(job, metaDict=None, specsList=None):
    """Update an lsst.verify.Job with metadata and specifications

    Parameters
    ----------
    job : `lsst.verify.Job`
       The verify Job to add the measurement to.
    metaDict : `dict`, optional
       A dict of key value pairs of any metadata to be added to the
       verify `job`.
    specsList : `list` of `lsst.verify.Specification`, optional
       A `list` of valid `lsst.verify.Specifications`s to be added
       verify `job`.

    Returns
    -------
    job : `lsst.verify.Job`
       The updated job with the new metadata and specifications added.
    """
    if metaDict:
        for metaName, metaValue in metaDict.items():
            job.meta.update({metaName: metaValue})
    if specsList:
        for spec in specsList:
            job.specs.update(spec)
    return job


def computeMeanOfFrac(valueArray, tailStr="upper", fraction=0.1, floorFactor=1):
    """Compute the rounded mean of the upper/lower fraction of the input array

    In other words, sort ``valueArray`` by value and compute the mean values of
    the highest[lowest] ``fraction`` of points for ``tailStr`` = upper[lower]
    and round this mean to a number of significant digits given by ``floorFactor``.
    e.g.
     ``floorFactor`` = 0.001, round to nearest thousandth (657.14727 -> 657.147)
     ``floorFactor`` = 0.01,  round to nearest hundredth (657.14727 -> 657.15)
     ``floorFactor`` = 0.1,   round to nearest tenth     (657.14727 -> 657.1)
     ``floorFactor`` = 1,     round to nearest integer   (657.14727 -> 657.0)
     ``floorFactor`` = 10,    round to nearest ten       (657.14727 -> 660.0)
     ``floorFactor`` = 100,   round to nearest hundred   (657.14727 -> 700.0)

    Parameters
    ----------
    valueArray : `numpy.ndarray`
       The array of values from which to compute the rounded mean.
    taiStr : `str`, optional
       Whether to compute the mean of the upper or lower ``fraction`` of points
       in ``valueArray`` ("upper" by default).
    fraction : `float`, optional
       The fraction of the upper or lower tail of the sorted values of
       ``valueArray`` to use in the calculation (0.1, i.e. 10% by default).
    floorFactor : `float`, optional
       Factor of 10 representing the number of significant digits to round to.
       See above for examples (1.0 by default).

    Raises
    ------
    `RuntimeError`
       If ``tailStr`` is not either \"upper\" or \"lower\".

    Returns
    -------
    meanOfFrac : `float`
       The mean of the upper/lower ``fraction`` of the values in
       ``valueArray``.
    """
    valueArray = valueArray.array if hasattr(valueArray, "array") else valueArray
    pad = 0.49
    ptFrac = max(2, int(fraction*len(valueArray)))
    if tailStr == "upper":
        meanOfFrac = np.floor(
            valueArray[valueArray.argsort()[-ptFrac:]].mean()/floorFactor + pad)*floorFactor
    elif tailStr == "lower":
        meanOfFrac = np.floor(
            valueArray[valueArray.argsort()[0:ptFrac]].mean()/floorFactor - pad)*floorFactor
    else:
        raise RuntimeError("tailStr must be either \"upper\" or \"lower\" (" + tailStr +
                           "was provided")

    return meanOfFrac

def calcQuartileClippedStats(dataArray, nSigmaToClip=3.0):
    """Calculate the quartile-based clipped statistics of a data array.

    The difference between quartiles[2] and quartiles[0] is the interquartile
    distance.  0.74*interquartileDistance is an estimate of standard deviation
    so, in the case that ``dataArray`` has an approximately Gaussian
    distribution, this is equivalent to nSigma clipping.

    Parameters
    ----------
    dataArray : `list` or `numpy.ndarray` of `float`
        List or array containing the values for which the quartile-based
        clipped statistics are to be calculated.
    nSigmaToClip : `float`, optional
        Number of \"sigma\" outside of which to clip data when computing the
        statistics.

    Returns
    -------
    result : `lsst.pipe.base.Struct`
        The quartile-based clipped statistics with ``nSigmaToClip`` clipping.
        Atributes are:

        ``median``
            The median of the full ``dataArray`` (`float`).
        ``mean``
            The quartile-based clipped mean (`float`).
        ``stdDev``
            The quartile-based clipped standard deviation (`float`).
        ``rms``
            The quartile-based clipped root-mean-squared (`float`).
        ``clipValue``
            The value outside of which to clip the data before computing the
            statistics (`float`).
        ``goodArray``
            A boolean array indicating which data points in ``dataArray`` were
            used in the calculation of the statistics, where `False` indicates
            a clipped datapoint (`numpy.ndarray` of `bool`).
    """
    quartiles = np.percentile(dataArray, [25, 50, 75])
    assert len(quartiles) == 3
    median = quartiles[1]
    interQuartileDistance = quartiles[2] - quartiles[0]
    clipValue = nSigmaToClip*0.74*interQuartileDistance
    good = np.logical_not(np.abs(dataArray - median) > clipValue)
    quartileClippedMean = dataArray[good].mean()
    quartileClippedStdDev = dataArray[good].std()
    quartileClippedRms = np.sqrt(np.mean(dataArray[good]**2))

    return Struct(
        median=median,
        mean=quartileClippedMean,
        stdDev=quartileClippedStdDev,
        rms=quartileClippedRms,
        clipValue=clipValue,
        goodArray=good,
    )

def getSchema(catalog):
    """Helper function to determine "schema" of catalog.

    This will be the list of columns if the catlaog type is a pandas DataFrame,
    or the schema object if it is an `lsst.afw.table.SourceCatalog`.
    """
    if isinstance(catalog, pd.DataFrame):
        schema = catalog.columns
    elif isinstance(catalog, pd.Series):
        schema = [catalog.name, ]
    else:
        schema = catalog.schema
    return schema
