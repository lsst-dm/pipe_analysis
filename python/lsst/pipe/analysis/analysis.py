#!/usr/bin/env python

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, AutoMinorLocator
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
from lsst.meas.base.forcedPhotCcd import PerTractCcdDataIdContainer
from lsst.afw.table.catalogMatches import matchesToCatalog, matchesFromCatalog
from lsst.meas.astrom import AstrometryConfig, LoadAstrometryNetObjectsTask, LoadAstrometryNetObjectsConfig
from lsst.pipe.tasks.colorterms import ColortermLibrary
from lsst.meas.mosaic.updateExposure import applyMosaicResultsCatalog, applyMosaicResultsExposure
from .utils import *
from .plotUtils import *

import lsst.afw.cameraGeom as cameraGeom
import lsst.afw.coord as afwCoord
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable

colorList = ["blue", "red", "green", "black", "yellow", "cyan", "magenta", ]

class AnalysisConfig(Config):
    flags = ListField(dtype=str, doc="Flags of objects to ignore",
                      default=["base_SdssCentroid_flag", "base_PixelFlags_flag_saturatedCenter",
                               "base_PixelFlags_flag_interpolatedCenter", "base_PsfFlux_flag"])
    clip = Field(dtype=float, default=4.0, doc="Rejection threshold (stdev)")
    magThreshold = Field(dtype=float, default=21.0, doc="Magnitude threshold to apply")
    magPlotMin = Field(dtype=float, default=14.0, doc="Minimum magnitude to plot")
    magPlotMax = Field(dtype=float, default=28.0, doc="Maximum magnitude to plot")
    fluxColumn = Field(dtype=str, default="base_PsfFlux_flux", doc="Column to use for flux/magnitude plotting")
    zp = Field(dtype=float, default=27.0, doc="Magnitude zero point to apply")
    doPlotOldMagsHist = Field(dtype=bool, default=False, doc="Make older, separated, mag and hist plots?")
    doPlotFP = Field(dtype=bool, default=False, doc="Make FocalPlane plots?")

class Analysis(object):
    """Centralised base for plotting"""

    def __init__(self, catalog, func, quantityName, shortName, config, qMin=-0.2, qMax=0.2,
                 prefix="", flags=[], goodKeys=[], errFunc=None, labeller=AllLabeller()):
        self.catalog = catalog
        self.func = func
        self.quantityName = quantityName
        self.shortName = shortName
        self.config = config
        self.qMin = qMin
        self.qMax = qMax
        if (labeller.labels.has_key("galaxy") and "calib_psfUsed" not in goodKeys and
            self.quantityName != "pStar"):
            self.qMin, self.qMax = 2.0*qMin, 2.0*qMax
        if "galaxy" in labeller.plot and "calib_psfUsed" not in goodKeys and self.quantityName != "pStar":
            self.qMin, self.qMax = 2.0*qMin, 2.0*qMax
        self.prefix = prefix
        self.flags = flags # omit if flag = True
        self.goodKeys = goodKeys # include if goodKey = True
        self.errFunc = errFunc
        if type(func) == np.ndarray:
            self.quantity = func
        else:
            self.quantity = func(catalog)

        self.quantityError = errFunc(catalog) if errFunc is not None else None
        # self.mag = self.config/zp - 2.5*np.log10(catalog[prefix + self.config.fluxColumn])
        self.mag = -2.5*np.log10(catalog[prefix + self.config.fluxColumn])

        self.good = np.isfinite(self.quantity) & np.isfinite(self.mag)
        if errFunc is not None:
            self.good &= np.isfinite(self.quantityError)
        for ff in list(config.flags) + flags:
            if ff in catalog.schema:
                self.good &= ~catalog[prefix + ff]
            else:
                print "NOTE: Flag (in config.flags list to ignore) " + ff + " not in catalog.schema"
        for kk in goodKeys:
            self.good &= catalog[prefix + kk]

        labels = labeller(catalog)
        self.data = {name: Data(catalog, self.quantity, self.mag, self.good & (labels == value),
                                colorList[value], self.quantityError, name in labeller.plot) for
                     name, value in labeller.labels.iteritems()}


    def plotAgainstMag(self, filename, stats=None, camera=None, ccdList=None, skymap=None, patchList=None,
                       hscRun=None, matchRadius=None, zpLabel=None):
        """Plot quantity against magnitude"""
        fig, axes = plt.subplots(1, 1)
        plt.axhline(0, linestyle="--", color="0.4")
        magMin, magMax = self.config.magPlotMin, self.config.magPlotMax
        dataPoints = []
        for name, data in self.data.iteritems():
            if len(data.mag) == 0:
                continue
            dataPoints.append(axes.scatter(data.mag, data.quantity, s=4, marker="o", lw=0,
                                           c=data.color, label=name, alpha=0.3))
        axes.set_xlabel("Mag from %s" % self.config.fluxColumn)
        axes.set_ylabel(self.quantityName)
        axes.set_ylim(self.qMin, self.qMax)
        axes.set_xlim(magMin, magMax)
        if stats is not None:
            annotateAxes(plt, axes, stats, "star", self.config.magThreshold, hscRun=hscRun,
                              matchRadius=matchRadius)
        axes.legend(handles=dataPoints, loc=1, fontsize=8)
        labelVisit(filename, plt, axes, 0.5, 1.05)
        if zpLabel is not None:
            labelZp(zpLabel, plt, axes, 0.13, -0.09, color="green")
        fig.savefig(filename)
        plt.close(fig)

    def plotAgainstMagAndHist(self, filename, stats=None, camera=None, ccdList=None, skymap=None,
                              patchList=None, hscRun=None, matchRadius=None, zpLabel=None):
        """Plot quantity against magnitude with side histogram"""
        nullfmt = NullFormatter()   # no labels for histograms
        minorLocator = AutoMinorLocator(2) # minor tick marks
        # definitions for the axes
        left, width = 0.10, 0.62
        bottom, height = 0.08, 0.62
        left_h = left + width + 0.03
        bottom_h = bottom + width + 0.04
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.23]
        rect_histy = [left_h, bottom, 0.20, height]
        topRight = [left_h - 0.002, bottom_h + 0.01, 0.22, 0.22]
        # start with a rectangular Figure
        plt.figure(1)

        axScatter = plt.axes(rect_scatter)
        axScatter.axhline(0, linestyle="--", color="0.4")
        axHistx = plt.axes(rect_histx)
        axHisty = plt.axes(rect_histy)
        axHistx.tick_params(labelsize=9)
        axHisty.tick_params(labelsize=9)
        # no labels
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)

        # no labels
        # axTopRight.xaxis.set_major_formatter(nullfmt)
        # axTopRight.yaxis.set_major_formatter(nullfmt)

        axScatter.tick_params(labelsize=10)

        if camera is not None and len(ccdList) > 0:
            axTopRight = plt.axes(topRight)
            axTopRight.set_aspect("equal")
            plotCameraOutline(plt, axTopRight, camera, ccdList)

        if skymap is not None and len(patchList) > 0:
            axTopRight = plt.axes(topRight)
            axTopRight.set_aspect("equal")
            plotPatchOutline(axTopRight, skymap, patchList)

        starMagMax = self.data["star"].mag.max() - 0.1
        aboveStarMagMax = self.data["star"].mag > starMagMax
        while len(self.data["star"].mag[aboveStarMagMax]) < max(1.0, 0.008*len(self.data["star"].mag)):
            starMagMax -= 0.2
            aboveStarMagMax = self.data["star"].mag > starMagMax

        magMin, magMax = self.config.magPlotMin, self.config.magPlotMax
        magMax = max(self.config.magThreshold+1.0, min(magMax, starMagMax))

        axScatter.set_xlim(magMin, magMax)
        axScatter.set_ylim(0.99*self.qMin, 0.99*self.qMax)

        nxDecimal = int(-1.0*np.around(np.log10(0.05*abs(magMax - magMin)) - 0.5))
        xBinwidth = min(0.1, np.around(0.05*abs(magMax - magMin), nxDecimal))
        xBins = np.arange(magMin + 0.5*xBinwidth, magMax + 0.5*xBinwidth, xBinwidth)
        nyDecimal = int(-1.0*np.around(np.log10(0.05*abs(self.qMax - self.qMin)) - 0.5))
        yBinwidth = max(0.005, np.around(0.02*abs(self.qMax - self.qMin), nyDecimal))
        yBins = np.arange(self.qMin - 0.5*yBinwidth, self.qMax + 0.55*yBinwidth, yBinwidth)
        axHistx.set_xlim(axScatter.get_xlim())
        axHisty.set_ylim(axScatter.get_ylim())
        axHistx.set_yscale("log", nonposy="clip")
        axHisty.set_xscale("log", nonposy="clip")

        nxSyDecimal = int(-1.0*np.around(np.log10(0.05*abs(self.config.magThreshold - magMin)) - 0.5))
        xSyBinwidth = min(0.1, np.around(0.05*abs(self.config.magThreshold - magMin), nxSyDecimal))
        xSyBins = np.arange(magMin + 0.5*xSyBinwidth, self.config.magThreshold + 0.5*xSyBinwidth, xSyBinwidth)

        royalBlue = "#4169E1"
        cornflowerBlue = "#6495ED"

        dataPoints = []
        runStats = []
        for name, data in self.data.iteritems():
            if len(data.mag) == 0:
                continue
            alpha = min(0.75, max(0.25, 1.0 - 0.2*np.log10(len(data.mag))))
            # draw mean and stdev at intervals (defined by xBins)
            histColor = "red"
            if name == "split" :
                histColor = "green"
            if name == "star" :
                histColor = royalBlue
                # shade the portion of the plot fainter that self.config.magThreshold
                axScatter.axvspan(self.config.magThreshold, axScatter.get_xlim()[1], facecolor="k",
                                  edgecolor="none", alpha=0.15)
                # compute running stats (just for plotting)
                belowThresh = data.mag < magMax # set lower if you want to truncate plotted running stats
                numHist, dataHist = np.histogram(data.mag[belowThresh], bins=len(xSyBins))
                syHist, dataHist = np.histogram(data.mag[belowThresh], bins=len(xSyBins),
                                                weights=data.quantity[belowThresh])
                syHist2, datahist = np.histogram(data.mag[belowThresh], bins=len(xSyBins),
                                                 weights=data.quantity[belowThresh]**2)
                meanHist = syHist/numHist
                stdHist = np.sqrt(syHist2/numHist - meanHist*meanHist)
                runStats.append(axScatter.errorbar((dataHist[1:] + dataHist[:-1])/2, meanHist, yerr=stdHist,
                                                   fmt="o", mfc=cornflowerBlue, mec="k", ms=4, ecolor="k",
                                                   label="Running stats\n(all stars)"))

            # plot data.  Appending in dataPoints for the sake of the legend
            dataPoints.append(axScatter.scatter(data.mag, data.quantity, s=4, marker="o", lw=0,
                                           c=data.color, label=name, alpha=alpha))
            axHistx.hist(data.mag, bins=xBins, color=histColor, alpha=0.6, label=name)
            axHisty.hist(data.quantity, bins=yBins, color=histColor, alpha=0.6, orientation="horizontal",
                         label=name)
        # Make sure stars used histogram is plotted last
        for name, data in self.data.iteritems():
            if stats is not None and name == "star" :
                dataUsed = data.quantity[stats[name].dataUsed]
                axHisty.hist(dataUsed, bins=yBins, color=data.color, orientation="horizontal", alpha=1.0,
                             label="used in Stats")
        axHistx.xaxis.set_minor_locator(minorLocator)
        axHistx.tick_params(axis="x", which="major", length=5)
        axHisty.yaxis.set_minor_locator(minorLocator)
        axHisty.tick_params(axis="y", which="major", length=5)
        axScatter.yaxis.set_minor_locator(minorLocator)
        axScatter.xaxis.set_minor_locator(minorLocator)
        axScatter.tick_params(which="major", length=5)
        axScatter.set_xlabel("Mag from %s" % self.config.fluxColumn)
        axScatter.set_ylabel(self.quantityName)

        if stats is not None:
            l1, l2 = annotateAxes(plt, axScatter, stats, "star", self.config.magThreshold,
                                           hscRun=hscRun, matchRadius=matchRadius)
        dataPoints = dataPoints + runStats + [l1, l2]
        axScatter.legend(handles=dataPoints, loc=1, fontsize=8)
        axHistx.legend(fontsize=7, loc=2)
        axHisty.legend(fontsize=7)
        # Label total number of objects of each data type
        xLoc, yLoc = 0.16, 1.40
        for name, data in self.data.iteritems():
            yLoc -= 0.04
            plt.text(xLoc, yLoc, "Ntotal = " + str(len(data.mag)), ha="left", va="center",
                     fontsize=9, transform=axScatter.transAxes, color=data.color)

        labelVisit(filename, plt, axScatter, 1.18, -0.11, color="green")
        if zpLabel is not None:
            labelZp(zpLabel, plt, axScatter, 0.08, -0.11, color="green")
        plt.savefig(filename)
        plt.close()

    def plotHistogram(self, filename, numBins=51, stats=None, hscRun=None, matchRadius=None, zpLabel=None):
        """Plot histogram of quantity"""
        fig, axes = plt.subplots(1, 1)
        axes.axvline(0, linestyle="--", color="0.6")
        numMax = 0
        for name, data in self.data.iteritems():
            if len(data.mag) == 0:
                continue
            good = np.isfinite(data.quantity)
            if self.config.magThreshold is not None:
                good &= data.mag < self.config.magThreshold
            nValid = np.abs(data.quantity[good]) <= self.qMax # need to have datapoints lying within range
            if good.sum() == 0 or nValid.sum() == 0:
                continue
            num, _, _ = axes.hist(data.quantity[good], numBins, range=(self.qMin, self.qMax), normed=False,
                                  color=data.color, label=name, histtype="step")
            numMax = max(numMax, num.max()*1.1)
        axes.set_xlim(self.qMin, self.qMax)
        axes.set_ylim(0.9, numMax)
        axes.set_xlabel(self.quantityName)
        axes.set_ylabel("Number")
        axes.set_yscale("log", nonposy="clip")
        x0, y0 = 0.03, 0.96
        if self.qMin == 0.0 :
            x0, y0 = 0.68, 0.81
        if stats is not None:
            annotateAxes(plt, axes, stats, "star", self.config.magThreshold, x0=x0, y0=y0,
                              isHist=True, hscRun=hscRun, matchRadius=matchRadius)
        axes.legend()
        labelVisit(filename, plt, axes, 0.5, 1.05)
        if zpLabel is not None:
            labelZp(zpLabel, plt, axes, 0.13, -0.09, color="green")
        fig.savefig(filename)
        plt.close(fig)

    def plotSkyPosition(self, filename, cmap=plt.cm.Spectral, stats=None, dataId=None, butler=None,
                        camera=None, ccdList=None, skymap=None, patchList=None, hscRun=None,
                        matchRadius=None, zpLabel=None):
        """Plot quantity as a function of position"""
        ra = np.rad2deg(self.catalog[self.prefix + "coord_ra"])
        dec = np.rad2deg(self.catalog[self.prefix + "coord_dec"])
        raMin, raMax = np.round(ra.min() - 0.05, 2), np.round(ra.max() + 0.05, 2)
        decMin, decMax = np.round(dec.min() - 0.05, 2), np.round(dec.max() + 0.05, 2)
        good = (self.mag < self.config.magThreshold if self.config.magThreshold > 0 else
                np.ones(len(self.mag), dtype=bool))
        if self.data.has_key("galaxy") and "calib_psfUsed" not in self.goodKeys:
            vMin, vMax = 0.5*self.qMin, 0.5*self.qMax
        else:
            vMin, vMax = self.qMin, self.qMax

        fig, axes = plt.subplots(1, 1, subplot_kw=dict(axisbg="0.7"))
        for name, data in self.data.iteritems():
            if not data.plot:
                continue
            if len(data.mag) == 0:
                continue
            selection = data.selection & good
            axes.scatter(ra[selection], dec[selection], s=2, marker="o", lw=0,
                         c=data.quantity[good[data.selection]], cmap=cmap, vmin=vMin, vmax=vMax)

        if dataId is not None and butler is not None and len(ccdList) > 0:
            plotCcdOutline(axes, butler, dataId, ccdList, zpLabel=zpLabel)

        axes.set_xlabel("RA (deg)")
        axes.set_ylabel("Dec (deg)")

        axes.set_xlim(raMin, raMax)
        axes.set_ylim(decMin, decMax)

        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vMin, vmax=vMax))
        mappable._A = []        # fake up the array of the scalar mappable. Urgh...
        cb = plt.colorbar(mappable)
        cb.set_label(self.quantityName, rotation=270, labelpad=15)
        if hscRun is not None:
            axes.set_title("HSC stack run: " + hscRun, color="#800080")
        labelVisit(filename, plt, axes, 0.5, 1.07)
        if zpLabel is not None:
            labelZp(zpLabel, plt, axes, 0.13, -0.09, color="green")
        fig.savefig(filename)
        plt.close(fig)

    def plotRaDec(self, filename, stats=None, hscRun=None, matchRadius=None, zpLabel=None):
        """Plot quantity as a function of RA, Dec"""

        ra = np.rad2deg(self.catalog[self.prefix + "coord_ra"])
        dec = np.rad2deg(self.catalog[self.prefix + "coord_dec"])
        good = (self.mag < self.config.magThreshold if self.config.magThreshold is not None else
                np.ones(len(self.mag), dtype=bool))
        fig, axes = plt.subplots(2, 1)
        axes[0].axhline(0, linestyle="--", color="0.6")
        axes[1].axhline(0, linestyle="--", color="0.6")
        for name, data in self.data.iteritems():
            if len(data.mag) == 0:
                continue
            selection = data.selection & good
            kwargs = {"s": 4, "marker": "o", "lw": 0, "c": data.color, "alpha": 0.5}
            axes[0].scatter(ra[selection], data.quantity[good[data.selection]], label=name, **kwargs)
            axes[1].scatter(dec[selection], data.quantity[good[data.selection]], **kwargs)

        axes[0].set_xlabel("RA (deg)", labelpad=-1)
        axes[1].set_xlabel("Dec (deg)")
        fig.text(0.02, 0.5, self.quantityName, ha="center", va="center", rotation="vertical")

        axes[0].set_ylim(self.qMin, self.qMax)
        axes[1].set_ylim(self.qMin, self.qMax)

        axes[0].legend()
        if stats is not None:
            annotateAxes(plt, axes[0], stats, "star", self.config.magThreshold, x0=0.03, yOff=0.07,
                              hscRun=hscRun, matchRadius=matchRadius)
            annotateAxes(plt, axes[1], stats, "star", self.config.magThreshold, x0=0.03, yOff=0.07,
                              hscRun=hscRun, matchRadius=matchRadius)
        labelVisit(filename, plt, axes[0], 0.5, 1.1)
        if zpLabel is not None:
            labelZp(zpLabel, plt, axes[0], 0.13, -0.09, color="green")
        fig.savefig(filename)
        plt.close(fig)

    def plotAll(self, dataId, filenamer, log, enforcer=None, forcedMean=None, butler=None, camera=None,
                ccdList=None, skymap=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None):
        """Make all plots"""
        stats = self.stats(forcedMean=forcedMean)
        self.plotAgainstMagAndHist(filenamer(dataId, description=self.shortName, style="psfMagHist"),
                                   stats=stats, camera=camera, ccdList=ccdList, skymap=skymap,
                                   patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                   zpLabel=zpLabel)

        if self.config.doPlotOldMagsHist:
            self.plotAgainstMag(filenamer(dataId, description=self.shortName, style="psfMag"), stats=stats,
                                hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel)
            self.plotHistogram(filenamer(dataId, description=self.shortName, style="hist"), stats=stats,
                               hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel)

        self.plotSkyPosition(filenamer(dataId, description=self.shortName, style="sky"), stats=stats,
                             dataId=dataId, butler=butler, camera=camera, ccdList=ccdList,  skymap=skymap,
                             patchList=patchList, hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel)
        self.plotRaDec(filenamer(dataId, description=self.shortName, style="radec"), stats=stats,
                       hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel)
        log.info("Statistics from %s of %s: %s" % (dataId, self.quantityName, stats))
        if enforcer:
            enforcer(stats, dataId, log, self.quantityName)
        return stats

    def stats(self, forcedMean=None):
        """Calculate statistics on quantity"""
        stats = {}
        for name, data in self.data.iteritems():
            if len(data.mag) == 0:
                continue
            good = data.mag < self.config.magThreshold
            stats[name] = self.calculateStats(data.quantity, good, forcedMean=forcedMean)
            if self.quantityError is not None:
                stats[name].sysErr = self.calculateSysError(data.quantity, data.error,
                                                            good, forcedMean=forcedMean)
            if len(stats) == 0:
                stats = None
                print "WARNING stats: no usable data.  Returning stats = None"
        return stats

    def calculateStats(self, quantity, selection, forcedMean=None):
        total = selection.sum() # Total number we're considering
        if total == 0:
            return Stats(dataUsed=0, num=0, total=0, mean=np.nan, stdev=np.nan, forcedMean=np.nan,
                         median=np.nan, clip=np.nan)
        quartiles = np.percentile(quantity[selection], [25, 50, 75])
        assert len(quartiles) == 3
        median = quartiles[1]
        clip = self.config.clip*0.74*(quartiles[2] - quartiles[0])
        good = selection & np.logical_not(np.abs(quantity - median) > clip)
        actualMean = quantity[good].mean()
        mean = actualMean if forcedMean is None else forcedMean
        stdev = np.sqrt(((quantity[good].astype(np.float64) - mean)**2).mean())
        return Stats(dataUsed=good, num=good.sum(), total=total, mean=actualMean, stdev=stdev,
                     forcedMean=forcedMean, median=median, clip=clip)

    def calculateSysError(self, quantity, error, selection, forcedMean=None, tol=1.0e-3):
        import scipy.optimize
        def function(sysErr2):
            sigNoise = quantity/np.sqrt(error**2 + sysErr2)
            stats = self.calculateStats(sigNoise, selection, forcedMean=forcedMean)
            return stats.stdev - 1.0

        if True:
            result = scipy.optimize.root(function, 0.0, tol=tol)
            if not result.success:
                print "Warning: sysErr calculation failed: %s" % result.message
                answer = np.nan
            else:
                answer = np.sqrt(result.x[0])
        else:
            answer = np.sqrt(scipy.optimize.newton(function, 0.0, tol=tol))
        print "calculateSysError: ", (function(answer**2), function((answer+0.001)**2),
                                      function((answer-0.001)**2))
        return answer

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
            vMin = min(0,np.round(self.data["star"].quantity.min() - 10))
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
        self.analysisMatches.magThreshold = 21.0 # External catalogs like PS1 and SDSS used smaller telescopes


class CoaddAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        kwargs["cosmos"] = parsedCmd.cosmos

        # Partition all inputs by tract,filter
        FilterRefsDict = functools.partial(defaultdict, list) # Dict for filter-->dataRefs
        tractFilterRefs = defaultdict(FilterRefsDict) # tract-->filter-->dataRefs
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
###            catalog = catalog[catalog["deblend_nChild"] == 0].copy(True) # Don't care about blended objects
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
                centroidStr =  "base_TransformedCentroid"
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

    def plotCentroidXY(self, catalog, filenamer, dataId, camera=None, ccdList=None, skymap=None,
                       patchList=None, hscRun=None, matchRadius=None, zpLabel=None):
        enforcer = None # Enforcer(requireLess={"star": {"stdev": 0.02}})
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
                           qMin=-0.1, qMax=1.3, labeller=StarGalaxyLabeller()
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
        filterName = afwImage.Filter(afwImage.Filter(filterName).getId()).getName() # Get primary name
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
            self.plotCentroidXY(catalog, filenamer, dataId, camera=camera, ccdList=ccdList, hscRun=hscRun,
                                zpLabel=self.zpLabel)
        if self.config.doPlotStarGalaxy:
            if "ext_shapeHSM_HsmSourceMoments_xx" in catalog.schema:
                self.plotStarGal(catalog, filenamer, dataId, hscRun=hscRun, zpLabel=self.zpLabel)
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

class CompareAnalysisConfig(Config):
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

class CompareAnalysisRunner(TaskRunner):
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

class CompareAnalysisTask(CmdLineTask):
    ConfigClass = CompareAnalysisConfig
    RunnerClass = CompareAnalysisRunner
    _DefaultName = "compareAnalysis"

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
        distEnforcer = None # Enforcer(requireLess={"star": {"stdev": 0.005}})
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

class CompareVisitAnalysisTask(CompareAnalysisTask):
    _DefaultName = "compareVisitAnalysis"
    ConfigClass = CompareAnalysisConfig
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
                 matchRadius=None, zpLabel=None):
        enforcer = None # Enforcer(requireLess={"star": {"stdev": 0.02}})
        for col in self.config.fluxToPlotList:
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
        enforcer = None # Enforcer(requireLess={"star": {"stdev": 0.02}})
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
                    matchRadius=None, zpLabel=None):
        enforcer = None # Enforcer(requireLess={"star": {"stdev": 0.02}})
        for col in self.config.fluxToPlotList:
            if "first_" + col + "_apCorr" in catalog.schema and "second_" + col + "_apCorr" in catalog.schema:
                Analysis(catalog, ApCorrDiffCompare(col + "_apCorr"),
                         "Run Comparison: apCorr difference (%s)" % col, "diff_" + col + "_apCorr",
                         self.config.analysis,
                         prefix="first_", qMin=-0.025, qMax=0.025, flags=[col + "_flag_apCorr"],
                         errFunc=ApCorrDiffErr(col + "_apCorr"), labeller=OverlapsStarGalaxyLabeller(),
                         ).plotAll(dataId, filenamer, self.log, enforcer, butler=butler, camera=camera,
                                   ccdList=ccdList, hscRun=hscRun, matchRadius=matchRadius, zpLabel=None)
