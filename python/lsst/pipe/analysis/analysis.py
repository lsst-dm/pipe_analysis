import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, AutoMinorLocator
import numpy as np
np.seterr(all="ignore")
from eups import Eups
eups = Eups()

from lsst.pex.config import Config, Field, ListField
from .utils import *
from .plotUtils import *

__all__ = ["AnalysisConfig", "Analysis"]

colorList = ["blue", "red", "green", "black", "yellow", "cyan", "magenta", ]


class AnalysisConfig(Config):
    flags = ListField(dtype=str, doc="Flags of objects to ignore",
                      default=["base_SdssCentroid_flag", "base_PixelFlags_flag_saturatedCenter",
                               "base_PixelFlags_flag_interpolatedCenter", "base_PsfFlux_flag"])
    clip = Field(dtype=float, default=4.0, doc="Rejection threshold (stdev)")
    magThreshold = Field(dtype=float, default=21.0, doc="Magnitude threshold to apply")
    magPlotMin = Field(dtype=float, default=14.0, doc="Minimum magnitude to plot")
    magPlotMax = Field(dtype=float, default=26.0, doc="Maximum magnitude to plot")
    fluxColumn = Field(dtype=str, default="base_PsfFlux_flux", doc="Column to use for flux/mag plotting")
    coaddZp = Field(dtype=float, default=27.0, doc="Magnitude zero point to apply for coadds")
    commonZp = Field(dtype=float, default=33.0, doc="Magnitude zero point to apply for common ZP plots")
    doPlotOldMagsHist = Field(dtype=bool, default=False, doc="Make older, separated, mag and hist plots?")
    doPlotRaDec = Field(dtype=bool, default=False, doc="Make delta vs. Ra and Dec plots?")
    doPlotFP = Field(dtype=bool, default=False, doc="Make FocalPlane plots?")
    doPlotCcdXy = Field(dtype=bool, default=False, doc="Make plots as a function of CCD x and y?")


class Analysis(object):
    """Centralised base for plotting"""

    def __init__(self, catalog, func, quantityName, shortName, config, qMin=-0.2, qMax=0.2,
                 prefix="", flags=[], goodKeys=[], errFunc=None, labeller=AllLabeller(), flagsCat=None,
                 magThreshold=21):
        self.catalog = catalog
        self.func = func
        self.quantityName = quantityName
        self.shortName = shortName
        self.config = config
        if magThreshold is None:
            self.magThreshold = self.config.magThreshold
        else:
            self.magThreshold = magThreshold
        self.qMin = qMin
        self.qMax = qMax
        if (labeller.labels.has_key("galaxy") and "calib_psfUsed" not in goodKeys and
            self.quantityName != "pStar"):
            self.qMin, self.qMax = 2.0*qMin, 2.0*qMax
        if "galaxy" in labeller.plot and "calib_psfUsed" not in goodKeys and self.quantityName != "pStar":
            self.qMin, self.qMax = 2.0*qMin, 2.0*qMax
        self.prefix = prefix
        self.flags = flags  # omit if flag = True
        self.goodKeys = goodKeys  # include if goodKey = True
        self.errFunc = errFunc
        if type(func) == np.ndarray:
            self.quantity = func
        else:
            self.quantity = func(catalog)

        self.quantityError = errFunc(catalog) if errFunc is not None else None
        # self.mag = self.config/zp - 2.5*np.log10(catalog[prefix + self.config.fluxColumn])
        if prefix + self.config.fluxColumn in catalog.schema:
            self.fluxColumn = self.config.fluxColumn
        else:
            self.fluxColumn = "flux_psf_flux"
        self.mag = -2.5*np.log10(catalog[prefix + self.fluxColumn])

        self.good = np.isfinite(self.quantity) & np.isfinite(self.mag)
        if errFunc is not None:
            self.good &= np.isfinite(self.quantityError)
        if flagsCat is None:
            flagsCat = catalog
        if not checkIdLists(catalog, flagsCat, prefix=prefix):
            raise RuntimeError(
                "Catalog being used for flags does not have the same object list as the data catalog")
        # Don't have flags in match and overlap catalogs (already removed in the latter)
        if "matches" not in self.shortName and "overlap" not in self.shortName:
            for ff in list(config.flags) + flags:
                if prefix + ff in flagsCat.schema:
                    self.good &= ~flagsCat[prefix + ff]
        for kk in goodKeys:
            self.good &= catalog[prefix + kk]

        labels = labeller(catalog)
        self.data = {name: Data(catalog, self.quantity, self.mag, self.good & (labels == value),
                                colorList[value], self.quantityError, name in labeller.plot) for
                     name, value in labeller.labels.iteritems()}

    def plotAgainstMag(self, filename, stats=None, camera=None, ccdList=None, tractInfo=None, patchList=None,
                       hscRun=None, matchRadius=None, zpLabel=None):
        """Plot quantity against magnitude"""
        fig, axes = plt.subplots(1, 1)
        plt.axhline(0, linestyle="--", color="0.4")
        magMin, magMax = self.config.magPlotMin, self.config.magPlotMax
        dataPoints = []
        ptSize = None
        for name, data in self.data.iteritems():
            if len(data.mag) == 0:
                continue
            if ptSize is None:
                ptSize = setPtSize(len(data.mag))
            dataPoints.append(axes.scatter(data.mag, data.quantity, s=ptSize, marker="o", lw=0,
                                           c=data.color, label=name, alpha=0.3))
        axes.set_xlabel("Mag from %s" % self.fluxColumn)
        axes.set_ylabel(self.quantityName)
        axes.set_ylim(self.qMin, self.qMax)
        axes.set_xlim(magMin, magMax)
        if stats is not None:
            annotateAxes(plt, axes, stats, "star", self.magThreshold, hscRun=hscRun,
                         matchRadius=matchRadius)
        axes.legend(handles=dataPoints, loc=1, fontsize=8)
        labelVisit(filename, plt, axes, 0.5, 1.05)
        if zpLabel is not None:
            labelZp(zpLabel, plt, axes, 0.13, -0.09, color="green")
        fig.savefig(filename)
        plt.close(fig)

    def plotAgainstMagAndHist(self, log, filename, stats=None, camera=None, ccdList=None, tractInfo=None,
                              patchList=None, hscRun=None, matchRadius=None, zpLabel=None):
        """Plot quantity against magnitude with side histogram"""
        nullfmt = NullFormatter()  # no labels for histograms
        minorLocator = AutoMinorLocator(2)  # minor tick marks
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

        if tractInfo is not None and len(patchList) > 0:
            axTopRight = plt.axes(topRight)
            axTopRight.set_aspect("equal")
            plotTractOutline(axTopRight, tractInfo, patchList)

        inLimits = self.data["star"].quantity < self.qMax
        inLimits &= self.data["star"].quantity > self.qMin
        if len(self.data["star"].quantity) > 0:
            if len(self.data["star"].quantity[inLimits]) < max(1.0, 0.35*len(self.data["star"].quantity)):
                log.info("No data within limits...decreasing/increasing qMin/qMax")
            while (len(self.data["star"].quantity[inLimits]) < max(1.0, 0.35*len(self.data["star"].quantity))):
                self.qMin -= 0.1*np.abs(self.qMin)
                self.qMax += 0.1*self.qMax
                inLimits = self.data["star"].quantity < self.qMax
                inLimits &= self.data["star"].quantity > self.qMin

        starMagMax = self.data["star"].mag.max() - 0.1
        aboveStarMagMax = self.data["star"].mag > starMagMax
        while len(self.data["star"].mag[aboveStarMagMax]) < max(1.0, 0.008*len(self.data["star"].mag)):
            starMagMax -= 0.2
            aboveStarMagMax = self.data["star"].mag > starMagMax

        magMin, magMax = self.config.magPlotMin, self.config.magPlotMax
        # magMax = max(self.magThreshold + 1.0, min(magMax, starMagMax))

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

        nxSyDecimal = int(-1.0*np.around(np.log10(0.05*abs(self.magThreshold - magMin)) - 0.5))
        xSyBinwidth = min(0.1, np.around(0.05*abs(self.magThreshold - magMin), nxSyDecimal))
        xSyBins = np.arange(magMin + 0.5*xSyBinwidth, self.magThreshold + 0.5*xSyBinwidth, xSyBinwidth)

        royalBlue = "#4169E1"
        cornflowerBlue = "#6495ED"

        dataPoints = []
        runStats = []
        ptSize = None
        for name, data in self.data.iteritems():
            if len(data.mag) == 0:
                log.info("No data for dataset: {:s}".format(name))
                continue
            if ptSize is None:
                ptSize = setPtSize(len(data.mag))
            alpha = min(0.75, max(0.25, 1.0 - 0.2*np.log10(len(data.mag))))
            # draw mean and stdev at intervals (defined by xBins)
            histColor = "red"
            if name == "split" or name == "notStar":
                histColor = "green"
            if name == "star":
                histColor = royalBlue
                # shade the portion of the plot fainter that self.magThreshold
                axScatter.axvspan(self.magThreshold, axScatter.get_xlim()[1], facecolor="k",
                                  edgecolor="none", alpha=0.15)
                # compute running stats (just for plotting)
                belowThresh = data.mag < magMax  # set lower if you want to truncate plotted running stats
                numHist, dataHist = np.histogram(data.mag[belowThresh], bins=len(xSyBins))
                syHist, dataHist = np.histogram(data.mag[belowThresh], bins=len(xSyBins),
                                                weights=data.quantity[belowThresh])
                syHist2, datahist = np.histogram(data.mag[belowThresh], bins=len(xSyBins),
                                                 weights=data.quantity[belowThresh]**2)
                meanHist = syHist/numHist
                stdHist = np.sqrt(syHist2/numHist - meanHist*meanHist)
                runStats.append(axScatter.errorbar((dataHist[1:] + dataHist[:-1])/2, meanHist, yerr=stdHist,
                                                   fmt="o", mfc=cornflowerBlue, mec="k", ms=4,
                                                   ecolor="k", label="Running stats\n(all stars)"))

            # plot data.  Appending in dataPoints for the sake of the legend
            dataPoints.append(axScatter.scatter(data.mag, data.quantity, s=ptSize, marker="o", lw=0,
                                                c=data.color, label=name, alpha=alpha))
            axHistx.hist(data.mag, bins=xBins, color=histColor, alpha=0.6, label=name)
            axHisty.hist(data.quantity, bins=yBins, color=histColor, alpha=0.6, orientation="horizontal",
                         label=name)
        # Make sure stars used histogram is plotted last
        for name, data in self.data.iteritems():
            if stats is not None and name == "star":
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

        axScatter.set_xlabel("%s mag (%s)" % (fluxToPlotString(self.fluxColumn),
                                              filterStrFromFilename(filename)))
        axScatter.set_ylabel("%s (%s)" % (self.quantityName, filterStrFromFilename(filename)))

        if stats is not None:
            l1, l2 = annotateAxes(plt, axScatter, stats, "star", self.magThreshold,
                                  hscRun=hscRun, matchRadius=matchRadius)
        dataPoints = dataPoints + runStats + [l1, l2]
        axScatter.legend(handles=dataPoints, loc=1, fontsize=8)
        axHistx.legend(fontsize=7, loc=2)
        axHisty.legend(fontsize=7)
        # Label total number of objects of each data type
        xLoc, yLoc = 0.17, 1.435
        for name, data in self.data.iteritems():
            if len(data.mag) == 0:
                continue
            yLoc -= 0.04
            plt.text(xLoc, yLoc, "Ntotal = " + str(len(data.mag)), ha="left", va="center",
                     fontsize=9, transform=axScatter.transAxes, color=data.color)

        labelVisit(filename, plt, axScatter, 1.18, -0.09, color="green")
        if zpLabel is not None:
            labelZp(zpLabel, plt, axScatter, 0.07, -0.09, color="green")
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
            if self.magThreshold is not None:
                good &= data.mag < self.magThreshold
            nValid = np.abs(data.quantity[good]) <= self.qMax  # need to have datapoints lying within range
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
        if self.qMin == 0.0:
            x0, y0 = 0.68, 0.81
        if stats is not None:
            annotateAxes(plt, axes, stats, "star", self.magThreshold, x0=x0, y0=y0,
                         isHist=True, hscRun=hscRun, matchRadius=matchRadius)
        axes.legend()
        labelVisit(filename, plt, axes, 0.5, 1.05)
        if zpLabel is not None:
            labelZp(zpLabel, plt, axes, 0.13, -0.09, color="green")
        fig.savefig(filename)
        plt.close(fig)

    def plotSkyPosition(self, filename, cmap=plt.cm.Spectral, stats=None, dataId=None, butler=None,
                        camera=None, ccdList=None, tractInfo=None, patchList=None, hscRun=None,
                        matchRadius=None, zpLabel=None, dataName="star"):
        """Plot quantity as a function of position"""
        pad = 0.02 # Number of degrees to pad the axis ranges
        ra = np.rad2deg(self.catalog[self.prefix + "coord_ra"])
        dec = np.rad2deg(self.catalog[self.prefix + "coord_dec"])
        raMin, raMax = np.round(ra.min() - pad, 2), np.round(ra.max() + pad, 2)
        decMin, decMax = np.round(dec.min() - pad, 2), np.round(dec.max() + pad, 2)

        deltaRa = raMax - raMin
        deltaDec = decMax - decMin
        deltaDeg = max((deltaRa, deltaDec))
        raMin = (raMin + deltaRa/2.0) - deltaDeg/2.0
        raMax = raMin + deltaDeg
        decMin = (decMin + deltaDec/2.0) - deltaDeg/2.0
        decMax = decMin + deltaDeg

        good = (self.mag < self.magThreshold if self.magThreshold > 0 else
                np.ones(len(self.mag), dtype=bool))

        if dataName == "star" and "calib_psfUsed" not in self.goodKeys and "pStar" not in filename:
            vMin, vMax = 0.5*self.qMin, 0.5*self.qMax
        elif "CModel" in filename and "overlap" not in filename:
            vMin, vMax = 1.5*self.qMin, 0.5*self.qMax
        else:
            vMin, vMax = 1.5*self.qMin, 1.5*self.qMax

        fig, axes = plt.subplots(1, 1, subplot_kw=dict(axisbg="0.7"))
        ptSize = None

        if dataId is not None and butler is not None and ccdList is not None:
            plotCcdOutline(axes, butler, dataId, ccdList, zpLabel=zpLabel)

        if tractInfo is not None and patchList is not None:
            for ip, patch in enumerate(tractInfo):
                if str(patch.getIndex()[0])+","+str(patch.getIndex()[1]) in patchList:
                    raPatch, decPatch = bboxToRaDec(patch.getOuterBBox(), tractInfo.getWcs())
                    raMin = min(np.round(min(raPatch) - pad, 2), raMin)
                    raMax = max(np.round(max(raPatch) + pad, 2), raMax)
                    decMin = min(np.round(min(decPatch) - pad, 2), decMin)
                    decMax = max(np.round(max(decPatch) + pad, 2), decMax)
            plotPatchOutline(axes, tractInfo, patchList)

        for name, data in self.data.iteritems():
            if name is not dataName:
                continue
            if len(data.mag) == 0:
                continue
            if ptSize is None:
                ptSize = setPtSize(len(data.mag))
            selection = data.selection & good
            axes.scatter(ra[selection], dec[selection], s=ptSize, marker="o", lw=0, label=name,
                         c=data.quantity[good[data.selection]], cmap=cmap, vmin=vMin, vmax=vMax)

        axes.set_xlabel("RA (deg)")
        axes.set_ylabel("Dec (deg)")

        axes.set_xlim(raMax, raMin)
        axes.set_ylim(decMin, decMax)

        filterStr = filterStrFromFilename(filename)

        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vMin, vmax=vMax))
        mappable._A = []        # fake up the array of the scalar mappable. Urgh...
        cb = plt.colorbar(mappable)
        cb.set_label(self.quantityName + " (" + filterStr + ")", rotation=270, labelpad=15)
        if hscRun is not None:
            axes.set_title("HSC stack run: " + hscRun, color="#800080")
        labelVisit(filename, plt, axes, 0.5, 1.07)
        if zpLabel is not None:
            labelZp(zpLabel, plt, axes, 0.13, -0.09, color="green")
        axes.legend(loc='upper left', bbox_to_anchor=(0.0, 1.08), fancybox=True, shadow=True, fontsize=9)
        if stats is not None:
            axes.annotate("mean = {0.mean:.4f}".format(stats[dataName]), xy=(0.77, 1.08),
                          xycoords="axes fraction", ha="left", va="top", fontsize=10)
            axes.annotate("stdev = {0.stdev:.4f}".format(stats[dataName]), xy=(0.77, 1.04),
                          xycoords="axes fraction", ha="left", va="top", fontsize=10)
        fig.savefig(filename)
        plt.close(fig)

    def plotRaDec(self, filename, stats=None, hscRun=None, matchRadius=None, zpLabel=None):
        """Plot quantity as a function of RA, Dec"""

        ra = np.rad2deg(self.catalog[self.prefix + "coord_ra"])
        dec = np.rad2deg(self.catalog[self.prefix + "coord_dec"])
        good = (self.mag < self.magThreshold if self.magThreshold is not None else
                np.ones(len(self.mag), dtype=bool))
        fig, axes = plt.subplots(2, 1)
        axes[0].axhline(0, linestyle="--", color="0.6")
        axes[1].axhline(0, linestyle="--", color="0.6")
        ptSize = None
        for name, data in self.data.iteritems():
            if len(data.mag) == 0:
                continue
            if ptSize is None:
                ptSize = setPtSize(len(data.mag))
            selection = data.selection & good
            kwargs = {"s": ptSize, "marker": "o", "lw": 0, "c": data.color, "alpha": 0.5}
            axes[0].scatter(ra[selection], data.quantity[good[data.selection]], label=name, **kwargs)
            axes[1].scatter(dec[selection], data.quantity[good[data.selection]], **kwargs)

        axes[0].set_xlabel("RA (deg)", labelpad=-1)
        axes[1].set_xlabel("Dec (deg)")
        fig.text(0.02, 0.5, self.quantityName, ha="center", va="center", rotation="vertical")

        axes[0].set_ylim(self.qMin, self.qMax)
        axes[1].set_ylim(self.qMin, self.qMax)

        axes[0].legend()
        if stats is not None:
            annotateAxes(plt, axes[0], stats, "star", self.magThreshold, x0=0.03, yOff=0.07,
                         hscRun=hscRun, matchRadius=matchRadius)
            annotateAxes(plt, axes[1], stats, "star", self.magThreshold, x0=0.03, yOff=0.07,
                         hscRun=hscRun, matchRadius=matchRadius)
        labelVisit(filename, plt, axes[0], 0.5, 1.1)
        if zpLabel is not None:
            labelZp(zpLabel, plt, axes[0], 0.13, -0.09, color="green")
        fig.savefig(filename)
        plt.close(fig)

    def plotAll(self, dataId, filenamer, log, enforcer=None, forcedMean=None, butler=None, camera=None,
                ccdList=None, tractInfo=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None,
                postFix=""):
        """Make all plots"""
        stats = self.stats(forcedMean=forcedMean)
        self.plotAgainstMagAndHist(log, filenamer(dataId, description=self.shortName,
                                                  style="psfMagHist" + postFix),
                                   stats=stats, camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                   patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                   zpLabel=zpLabel)

        if self.config.doPlotOldMagsHist:
            self.plotAgainstMag(filenamer(dataId, description=self.shortName, style="psfMag" + postFix),
                                stats=stats, hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel)
            self.plotHistogram(filenamer(dataId, description=self.shortName, style="hist" + postFix),
                               stats=stats, hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel)

        self.plotSkyPosition(filenamer(dataId, description=self.shortName, style="sky-stars" + postFix),
                             stats=stats, dataId=dataId, butler=butler, camera=camera, ccdList=ccdList,
                             tractInfo=tractInfo, patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                             zpLabel=zpLabel, dataName="star")

        if (not any(ss in self.shortName for ss in
                    ["pStar", "race_", "psfUsed", "gri", "riz", "izy", "z9y", "color_"])):
            self.plotSkyPosition(filenamer(dataId, description=self.shortName, style="sky-gals" + postFix),
                                 stats=stats, dataId=dataId, butler=butler, camera=camera, ccdList=ccdList,
                                 tractInfo=tractInfo, patchList=patchList, hscRun=hscRun,
                                 matchRadius=matchRadius, zpLabel=zpLabel, dataName="galaxy")

        if self.config.doPlotRaDec:
            self.plotRaDec(filenamer(dataId, description=self.shortName, style="radec" + postFix),
                           stats=stats, hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel)
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
            good = data.mag < self.magThreshold
            stats[name] = self.calculateStats(data.quantity, good, forcedMean=forcedMean)
            if self.quantityError is not None:
                stats[name].sysErr = self.calculateSysError(data.quantity, data.error,
                                                            good, forcedMean=forcedMean)
            if len(stats) == 0:
                stats = None
        return stats

    def calculateStats(self, quantity, selection, forcedMean=None):
        total = selection.sum()  # Total number we're considering
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
                print "Warning: sysErr calculation failed: {:s}".format(result.message)
                answer = np.nan
            else:
                answer = np.sqrt(result.x[0])
        else:
            answer = np.sqrt(scipy.optimize.newton(function, 0.0, tol=tol))
        print "calculateSysError: {0:.4f}, {1:.4f}, {2:.4f}".format(function(answer**2),
                                                                        function((answer+0.001)**2),
                                                                        function((answer-0.001)**2))
        return answer
