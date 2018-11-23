from __future__ import print_function

import matplotlib
matplotlib.use("Agg")  # noqa E402
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, AutoMinorLocator
import numpy as np
np.seterr(all="ignore")  # noqa E402

import lsst.afw.geom as afwGeom
from lsst.display.matplotlib.matplotlib import AsinhNormalize
from lsst.pex.config import Config, Field, ListField, DictField

from .utils import Data, Stats, E1Resids, E2Resids, checkIdLists, fluxToPlotString
from .plotUtils import (annotateAxes, AllLabeller, setPtSize, labelVisit, plotText, plotCameraOutline,
                        plotTractOutline, plotPatchOutline, plotCcdOutline, labelCamera, getQuiver,
                        getRaDecMinMaxPatchList, bboxToXyCoordLists, makeAlphaCmap, buildTractImage)

__all__ = ["AnalysisConfig", "Analysis"]

colorList = ["blue", "red", "green", "black", "yellow", "cyan", "magenta", ]


class AnalysisConfig(Config):
    flags = ListField(dtype=str, doc="Flags of objects to ignore",
                      default=["base_SdssCentroid_flag", "slot_Centroid_flag", "base_PsfFlux_flag",
                               "base_PixelFlags_flag_saturatedCenter",
                               "base_ClassificationExtendedness_flag"])
    clip = Field(dtype=float, default=4.0, doc="Rejection threshold (stdev)")
    magThreshold = Field(dtype=float, default=21.0, doc="Magnitude threshold to apply")
    magPlotMin = Field(dtype=float, default=14.5, doc="Minimum magnitude to plot")
    magPlotMax = Field(dtype=float, default=26.0, doc="Maximum magnitude to plot")
    magPlotStarMin = DictField(
        keytype=str,
        itemtype=float,
        default={"HSC-G": 16.5, "HSC-R": 17.25, "HSC-I": 16.5, "HSC-Z": 15.5, "HSC-Y": 15.25,
                 "NB0921": 15.0, "g": 16.5, "r": 15.0, "i": 16.5, "z": 15.5, "y": 15.5},
        doc="Minimum magnitude to plot",
    )
    magPlotStarMax = DictField(
        keytype=str,
        itemtype=float,
        default={"HSC-G": 23.75, "HSC-R": 24.25, "HSC-I": 23.75, "HSC-Z": 23.0, "HSC-Y": 22.0,
                 "NB0921": 22.25, "g": 23.5, "r": 22.0, "i": 23.5, "z": 22.5, "y": 22.5},
        doc="Maximum magnitude to plot",
    )
    fluxColumn = Field(dtype=str, default="modelfit_CModel_instFlux",
                       doc="Column to use for flux/mag plotting")
    coaddZp = Field(dtype=float, default=27.0, doc="Magnitude zero point to apply for coadds")
    commonZp = Field(dtype=float, default=33.0, doc="Magnitude zero point to apply for common ZP plots")
    doPlotOldMagsHist = Field(dtype=bool, default=False, doc="Make older, separated, mag and hist plots?")
    doPlotRaDec = Field(dtype=bool, default=False, doc="Make delta vs. Ra and Dec plots?")
    doPlotFP = Field(dtype=bool, default=False, doc="Make FocalPlane plots?")
    doPlotCcdXy = Field(dtype=bool, default=False, doc="Make plots as a function of CCD x and y?")
    doPlotTractOutline = Field(dtype=bool, default=True, doc="Plot tract outline (may be a bit slow)?")
    visitClassFluxRatio = Field(dtype=float, default=0.95,
                                doc="Flux ratio for visit level star/galaxy classifiaction")
    coaddClassFluxRatio = Field(dtype=float, default=0.985,
                                doc="Flux ratio for coadd level star/galaxy classifiaction")


class Analysis(object):
    """Centralised base for plotting"""

    def __init__(self, catalog, func, quantityName, shortName, config, qMin=-0.2, qMax=0.2,
                 prefix="", flags=[], goodKeys=[], errFunc=None, labeller=AllLabeller(), flagsCat=None,
                 magThreshold=21, forcedMean=None, unitScale=1.0):
        self.catalog = catalog
        self.func = func
        self.quantityName = quantityName
        self.shortName = shortName
        self.config = config
        self.forcedMean = forcedMean
        if magThreshold is None:
            self.magThreshold = self.config.magThreshold
        else:
            self.magThreshold = magThreshold
        self.unitScale = unitScale
        self.qMin = qMin*self.unitScale
        self.qMax = qMax*self.unitScale
        if "modelfit" in self.shortName:  # Cmodel has smaller mean/scatter from psf mag for most objects
            self.qMin /= 2.0
            self.qMax /= 2.0
        self.goodKeys = goodKeys  # include if goodKey = True
        self.calibUsedOnly = len([key for key in self.goodKeys if "used" in key])
        if self.calibUsedOnly > 0:
            self.magThreshold = 99  # Want to plot all calib_used

        self.prefix = prefix
        self.errFunc = errFunc
        if func is not None:
            if type(func) == np.ndarray:
                self.quantity = func
            else:
                self.quantity = func(catalog)
        else:
            self.quantity = None

        self.quantityError = errFunc(catalog) if errFunc is not None else None
        if prefix + self.config.fluxColumn in catalog.schema:
            self.fluxColumn = self.config.fluxColumn
        else:
            self.fluxColumn = "flux_psf_flux"
        self.mag = -2.5*np.log10(catalog[prefix + self.fluxColumn])

        self.good = np.isfinite(self.quantity) & np.isfinite(self.mag) if self.quantity is not None else None
        if errFunc is not None:
            self.good &= np.isfinite(self.quantityError)
        if flagsCat is None:
            flagsCat = catalog
        if not checkIdLists(catalog, flagsCat, prefix=prefix):
            raise RuntimeError(
                "Catalog being used for flags does not have the same object list as the data catalog")
        # Don't have flags in match and overlap catalogs (already removed in the latter)
        if ("matches" not in self.shortName and "overlap" not in self.shortName and
                "quiver" not in self.shortName and "inputCounts" not in self.shortName):
            for ff in set(list(self.config.flags) + flags):
                if prefix + ff in flagsCat.schema:
                    self.good &= ~flagsCat[prefix + ff]
        for kk in goodKeys:
            self.good &= flagsCat[prefix + kk]

        if labeller is not None:
            labels = labeller(catalog)
            self.data = {name: Data(catalog, self.quantity, self.mag, self.good & (labels == value),
                                    colorList[value], self.quantityError, name in labeller.plot) for
                         name, value in labeller.labels.items()}
            # Sort data dict by number of points in each data type.
            self.data = {k: self.data[k] for _, k in sorted(((len(v.mag), k) for (k, v) in self.data.items()),
                                                            reverse=True)}
            self.stats = self.statistics(forcedMean=forcedMean)
            # Make sure you have some good data to plot: only check first dataset in labeller.plot
            # list as it is the most important one (and the only available in many cases where
            # StarGalaxyLabeller is used.
            if (self.stats[labeller.plot[0]].num) == 0:
                raise RuntimeError("No good data points to plot for sample labelled: {:}".
                                   format(labeller.plot[0]))
            # Ensure plot limits always encompass at least mean +/- 6.0*stdev, at most mean +/- 20.0*stddev,
            # and clipped stats range + 25%
            dataType = "all" if "all" in self.data else "star"
            if not any(ss in self.shortName for ss in ["footNpix", "distance", "pStar", "resolution",
                                                       "race"]):
                self.qMin = max(min(self.qMin, self.stats[dataType].mean - 6.0*self.stats[dataType].stdev,
                                self.stats[dataType].median - 1.25*self.stats[dataType].clip),
                                min(self.stats[dataType].mean - 20.0*self.stats[dataType].stdev,
                                    -0.005*self.unitScale))
            if not any(ss in self.shortName for ss in ["footNpix", "pStar", "resolution", "race"]):
                self.qMax = min(max(self.qMax, self.stats[dataType].mean + 6.0*self.stats[dataType].stdev,
                                self.stats[dataType].median + 1.25*self.stats[dataType].clip),
                                max(self.stats[dataType].mean + 20.0*self.stats[dataType].stdev,
                                    0.005*self.unitScale))

    def plotAgainstMag(self, filename, stats=None, camera=None, ccdList=None, tractInfo=None, patchList=None,
                       hscRun=None, matchRadius=None, zpLabel=None, forcedStr=None):
        """Plot quantity against magnitude"""
        fig, axes = plt.subplots(1, 1)
        plt.axhline(0, linestyle="--", color="0.4")
        magMin, magMax = self.config.magPlotMin, self.config.magPlotMax
        dataPoints = []
        ptSize = None
        for name, data in self.data.items():
            if not data.mag.any():
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
            annotateAxes(filename, plt, axes, stats, "star", self.magThreshold, hscRun=hscRun,
                         matchRadius=matchRadius, unitScale=self.unitScale)
        axes.legend(handles=dataPoints, loc=1, fontsize=8)
        labelVisit(filename, plt, axes, 0.5, 1.05)
        if zpLabel is not None:
            prefix = "" if "GE applied" in zpLabel else "zp: "
            plotText(zpLabel, plt, axes, 0.13, -0.09, prefix=prefix, color="green")
        if forcedStr is not None:
            plotText(forcedStr, plt, axes, 0.85, -0.09, prefix="cat: ", color="green")
        fig.savefig(filename, dpi=120)
        plt.close(fig)

    def plotAgainstMagAndHist(self, log, filename, stats=None, camera=None, ccdList=None, tractInfo=None,
                              patchList=None, hscRun=None, matchRadius=None, zpLabel=None, forcedStr=None,
                              plotRunStats=True, highlightList=None, filterStr=None, extraLabels=None):
        """Plot quantity against magnitude with side histogram"""
        if filterStr is None:
            filterStr = ""
            filterLabelStr = ""
        else:
            filterLabelStr = "[" + filterStr + "]" if "/color/" not in filename else ""

        nullfmt = NullFormatter()  # no labels for histograms
        # definitions for the axes
        left, width = 0.12, 0.62
        bottom, height = 0.10, 0.62
        left_h = left + width + 0.03
        bottom_h = bottom + width + 0.02
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.23]
        rect_histy = [left_h, bottom, 0.20, height]
        topRight = [left_h + 0.003, bottom_h + 0.01, 0.22, 0.22]
        # start with a rectangular Figure
        plt.figure(1)

        axScatter = plt.axes(rect_scatter)
        axScatter.axhline(0, linestyle="--", color="0.4")
        axHistx = plt.axes(rect_histx)
        axHisty = plt.axes(rect_histy)
        axHistx.tick_params(which="both", direction="in", top=True, right=True, labelsize=8)
        axHisty.tick_params(which="both", direction="in", top=True, right=True, labelsize=8)
        # no labels
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)

        axScatter.tick_params(which="both", direction="in", labelsize=9)

        if camera is not None and ccdList is not None:
            if ccdList:
                axTopRight = plt.axes(topRight)
                axTopRight.set_aspect("equal")
                plotCameraOutline(plt, axTopRight, camera, ccdList)

        if self.config.doPlotTractOutline and tractInfo is not None and len(patchList) > 0:
            axTopRight = plt.axes(topRight)
            axTopRight.set_aspect("equal")
            plotTractOutline(axTopRight, tractInfo, patchList)

        dataType = "all" if "all" in self.data else "star"
        inLimits = self.data[dataType].quantity < self.qMax
        inLimits &= self.data[dataType].quantity > self.qMin
        if self.data[dataType].quantity.any():
            if len(self.data[dataType].quantity[inLimits]) < max(1.0, 0.35*len(self.data[dataType].quantity)):
                log.info("No data within limits...decreasing/increasing qMin/qMax")
            while (len(self.data[dataType].quantity[inLimits]) <
                   max(1.0, 0.35*len(self.data[dataType].quantity))):
                self.qMin -= 0.1*np.abs(self.qMin)
                self.qMax += 0.1*self.qMax
                inLimits = self.data[dataType].quantity < self.qMax
                inLimits &= self.data[dataType].quantity > self.qMin

        # Make sure plot limit extends low enough to show well below the star/galaxy separation line.
        # Add delta as opposed to directly changing self.qMin to not affect other plots
        deltaMin = 0.0
        if "galaxy" in self.data and len(self.data["galaxy"].quantity) > 0 and "-mag_" in filename:
            if "GaussianFlux" in filename:
                galMin = np.round(2.5*np.log10(self.config.visitClassFluxRatio) - 0.08, 2)*self.unitScale
                deltaMin = max(0.0, self.qMin - galMin)
            if "CModel" in filename:
                galMin = np.round(2.5*np.log10(self.config.coaddClassFluxRatio) - 0.08, 2)*self.unitScale
                deltaMin = max(0.0, self.qMin - galMin)

        magMin, magMax = self.config.magPlotMin, self.config.magPlotMax
        if self.calibUsedOnly > 0 or "color" in filename or "visit" not in filename or "matches" in filename:
            if filterStr in self.config.magPlotStarMin.keys():
                magMin = self.config.magPlotStarMin[filterStr]
                if self.calibUsedOnly == 0 and ("plot-t" in filename or "compare-t" in filename):
                    magMin -= 1.5  # CModel flux for coadds can have brighter mags than the PSF equivalent
                    if "matches" in filename:  # But reference catalogs won't go quite so bright
                        magMin += 1.0
        if self.calibUsedOnly > 0 or "color" in filename or "matches" in filename:
            if filterStr in self.config.magPlotStarMax.keys():
                magMax = self.config.magPlotStarMax[filterStr]
        axScatter.set_xlim(magMin, magMax)
        yDelta = 0.01*(self.qMax - (self.qMin - deltaMin))
        axScatter.set_ylim((self.qMin - deltaMin) + yDelta, self.qMax - yDelta)

        # Get current y limits for scatter plot to base histogram bin sizing on
        axScatterY1 = axScatter.get_ylim()[0]
        axScatterY2 = axScatter.get_ylim()[1]

        nxDecimal = int(-1.0*np.around(np.log10(0.05*abs(magMax - magMin)) - 0.5))
        xBinwidth = min(0.1, np.around(0.05*abs(magMax - magMin), nxDecimal))
        xBins = np.arange(magMin + 0.5*xBinwidth, magMax + 0.5*xBinwidth, xBinwidth)
        nyDecimal = int(-1.0*np.around(np.log10(0.05*abs(axScatterY2 - (axScatterY1 - deltaMin))) - 0.5))
        yBinwidth = max(0.5/10**nyDecimal, np.around(0.02*abs(axScatterY2 - (axScatterY1 - deltaMin)),
                                                     nyDecimal))
        yBins = np.arange((axScatterY1 - deltaMin) - 0.5*yBinwidth, axScatterY2 + 0.55*yBinwidth, yBinwidth)
        axHistx.set_xlim(axScatter.get_xlim())
        axHisty.set_ylim(axScatter.get_ylim())
        axHistx.set_yscale("log", nonposy="clip")
        axHisty.set_xscale("log", nonposx="clip")
        nTotal = 0
        fullSampleMag = []
        fullSampleQuantity = []
        for name, data in self.data.items():
            if data.plot:
                nTotal += len(data.mag)
                fullSampleMag.extend(data.mag)
                fullSampleQuantity.extend(data.quantity)
        axScatterYlim = np.around(nTotal, -1*int(np.floor(np.log10(nTotal))))
        axHistx.set_ylim(0.8, axScatterYlim)
        axHisty.set_xlim(0.8, axScatterYlim)

        # Plot full sample histograms
        axHistx.hist(fullSampleMag, bins=xBins, color="black", alpha=0.4, label="All")
        axHisty.hist(fullSampleQuantity, bins=yBins, color="black", orientation="horizontal",
                     alpha=0.4, label="All")

        nxSyDecimal = int(-1.0*np.around(np.log10(0.05*abs(self.magThreshold - magMin)) - 0.5))
        xSyBinwidth = min(0.1, np.around(0.05*abs(self.magThreshold - magMin), nxSyDecimal))
        xSyBins = np.arange(magMin + 0.5*xSyBinwidth, self.magThreshold + 0.5*xSyBinwidth, xSyBinwidth)

        royalBlue = "#4169E1"
        cornflowerBlue = "#6495ED"

        dataPoints = []
        runStats = []
        ptSize = None
        for name, data in self.data.items():
            if not data.plot:
                log.info("Not plotting data for dataset: {0:s} (N = {1:d})".format(name, len(data.mag)))
                continue
            if not data.mag.any():
                log.info("No data for dataset: {:s}".format(name))
                continue
            if ptSize is None:
                ptSize = setPtSize(len(data.mag))
            alpha = min(0.75, max(0.25, 1.0 - 0.2*np.log10(len(data.mag))))
            # draw mean and stdev at intervals (defined by xBins)
            histColor = "red"
            if name == "split" or name == "notStar":
                histColor = "green"
            if name == "star" or name == "all":
                histColor = royalBlue
                # shade the portion of the plot fainter that self.magThreshold
                axScatter.axvspan(self.magThreshold, axScatter.get_xlim()[1], facecolor="k",
                                  edgecolor="none", alpha=0.15)
                # compute running stats (just for plotting)
                if self.calibUsedOnly == 0 and plotRunStats:
                    belowThresh = data.mag < magMax  # set lower if you want to truncate plotted running stats
                    numHist, dataHist = np.histogram(data.mag[belowThresh], bins=len(xSyBins))
                    # Only plot running stats if there are a significant number of data points per bin.
                    # Computed here as the mean number in the brightest 20% of the bins which we require
                    # to be greater than 12.
                    if numHist[0:max(1, int(0.2*len(xSyBins)))].mean() > 12:
                        syHist, dataHist = np.histogram(data.mag[belowThresh], bins=len(xSyBins),
                                                        weights=data.quantity[belowThresh])
                        syHist2, datahist = np.histogram(data.mag[belowThresh], bins=len(xSyBins),
                                                         weights=data.quantity[belowThresh]**2)
                        meanHist = syHist/numHist
                        stdHist = np.sqrt(syHist2/numHist - meanHist*meanHist)
                        runStats.append(axScatter.errorbar((dataHist[1:] + dataHist[:-1])/2, meanHist,
                                        yerr=stdHist, fmt="o", mfc=cornflowerBlue, mec="k",
                                        ms=2, ecolor="k", label="Running\nstats (all\nstars)"))

            if highlightList is not None:
                # Make highlight as a background ring of larger size than the data point size
                for flag, threshValue, color in highlightList:
                    label = flag.replace("merge_measurement", "ref")
                    highlightSelection = data.catalog[flag] > threshValue
                    if name == "star" or name == "all":
                        dataPoints.append(
                            axScatter.scatter(data.mag[highlightSelection],
                                              data.quantity[highlightSelection],
                                              s=1.3*ptSize, marker="o", facecolors="none",
                                              edgecolors=color, label=label))
                    else:
                        axScatter.scatter(data.mag[highlightSelection], data.quantity[highlightSelection],
                                          s=1.3*ptSize, marker="o", facecolors="none", edgecolors=color)

            # Plot data.  Appending in dataPoints for the sake of the legend
            dataPoints.append(axScatter.scatter(data.mag, data.quantity, s=ptSize, marker="o",
                                                facecolors=data.color, edgecolors="none", label=name,
                                                alpha=alpha))

            axHistx.hist(data.mag, bins=xBins, color=histColor, alpha=0.6, label=name)
            axHisty.hist(data.quantity, bins=yBins, color=histColor, alpha=0.6, orientation="horizontal",
                         label=name)
        # Make sure stars used histogram is plotted last
        for name, data in self.data.items():
            if stats is not None and (name == "star" or name == "all"):
                dataUsed = data.quantity[stats[name].dataUsed]
                axHisty.hist(dataUsed, bins=yBins, color=data.color, orientation="horizontal", alpha=1.0,
                             label="used in Stats")
        axHistx.tick_params(axis="x", which="major", direction="in", length=5)
        axHistx.xaxis.set_minor_locator(AutoMinorLocator(2))
        axHisty.tick_params(axis="y", which="major", direction="in", length=5)
        axHisty.yaxis.set_minor_locator(AutoMinorLocator(2))

        axScatter.tick_params(which="major", direction="in", length=5)
        axScatter.xaxis.set_minor_locator(AutoMinorLocator(2))
        axScatter.yaxis.set_minor_locator(AutoMinorLocator(2))

        yLabel = r"%s %s" % (self.quantityName, filterLabelStr)
        fontSize = min(11, max(6, 11 - int(np.log(max(1, len(yLabel) - 45)))))

        axScatter.set_xlabel("%s mag [%s]" % (fluxToPlotString(self.fluxColumn), filterStr), fontSize=11)
        axScatter.set_ylabel(yLabel, fontsize=fontSize)

        if stats is not None:
            l1, l2 = annotateAxes(filename, plt, axScatter, stats, dataType, self.magThreshold,
                                  hscRun=hscRun, matchRadius=matchRadius, unitScale=self.unitScale)
        dataPoints = dataPoints + runStats + [l1, l2]
        axScatter.legend(handles=dataPoints, loc=1, fontsize=8)
        axHistx.legend(fontsize=7, loc=2)
        axHisty.legend(fontsize=7)
        # Label total number of objects of each data type
        xLoc, yLoc = 0.09, 1.355
        lenNameMax = 0
        for name, data in self.data.items():
            if data.mag.any():
                lenNameMax = len(name) if len(name) > lenNameMax else lenNameMax
        xLoc += 0.02*lenNameMax

        plt.text(xLoc, yLoc, "N$_{all}$  = " + str(len(fullSampleMag)), ha="left", va="center",
                 fontsize=8, transform=axScatter.transAxes, color="black", alpha=0.6)
        for name, data in self.data.items():
            if not (data.mag.any() and data.plot):
                continue
            yLoc -= 0.05
            plt.text(xLoc, yLoc, "N$_{" + name[:4] + "}$ = " + str(len(data.mag)), ha="left", va="center",
                     fontsize=8, transform=axScatter.transAxes, color=data.color)

        labelVisit(filename, plt, axScatter, 1.18, -0.11, color="green")
        if zpLabel is not None:
            prefix = "" if "GE applied" in zpLabel else "zp: "
            plotText(zpLabel, plt, axScatter, 0.09, -0.11, prefix=prefix, color="green")
        if forcedStr is not None:
            plotText(forcedStr, plt, axScatter, 0.87, -0.11, prefix="cat: ", color="green")
        if extraLabels is not None:
            for i, extraLabel in enumerate(extraLabels):
                plotText(extraLabel, plt, axScatter, 0.29, 0.06 + i*0.05, fontSize=10, color="black")
        plt.savefig(filename, dpi=120)
        plt.close()

    def plotHistogram(self, filename, numBins=51, stats=None, hscRun=None, matchRadius=None, zpLabel=None,
                      forcedStr=None, camera=None, filterStr=None):
        """Plot histogram of quantity"""
        fig, axes = plt.subplots(1, 1)
        axes.axvline(0, linestyle="--", color="0.6")
        numMax = 0
        for name, data in self.data.items():
            if not data.mag.any():
                continue
            good = np.isfinite(data.quantity)
            if self.magThreshold is not None:
                good &= data.mag < self.magThreshold
            nValid = np.abs(data.quantity[good]) <= self.qMax  # need to have datapoints lying within range
            if good.sum() == 0 or nValid.sum() == 0:
                continue
            num, _, _ = axes.hist(data.quantity[good], numBins, range=(self.qMin, self.qMax), density=False,
                                  color=data.color, label=name, histtype="step")
            numMax = max(numMax, num.max()*1.1)
        axes.set_xlim(self.qMin, self.qMax)
        axes.set_ylim(0.9, numMax)
        if filterStr is None:
            filterStr = ''
        axes.set_xlabel("{0:s} [{1:s}]".format(self.quantityName, filterStr))
        axes.set_ylabel("Number")
        axes.set_yscale("log", nonposy="clip")
        x0, y0 = 0.03, 0.97
        if self.qMin == 0.0:
            x0, y0 = 0.68, 0.81
        if stats is not None:
            annotateAxes(filename, plt, axes, stats, "star", self.magThreshold, x0=x0, y0=y0,
                         isHist=True, hscRun=hscRun, matchRadius=matchRadius, unitScale=self.unitScale)
        axes.legend()
        if camera is not None:
            labelCamera(camera, plt, axes, 0.5, 1.09)
        labelVisit(filename, plt, axes, 0.5, 1.04)
        if zpLabel is not None:
            prefix = "" if "GE applied" in zpLabel else "zp: "
            plotText(zpLabel, plt, axes, 0.13, -0.09, prefix=prefix, color="green")
        if forcedStr is not None:
            plotText(forcedStr, plt, axes, 0.85, -0.09, prefix="cat: ", color="green")
        fig.savefig(filename, dpi=120)
        plt.close(fig)

    def plotSkyPosition(self, filename, cmap=plt.cm.Spectral, stats=None, dataId=None, butler=None,
                        camera=None, ccdList=None, tractInfo=None, patchList=None, hscRun=None,
                        matchRadius=None, zpLabel=None, highlightList=None, forcedStr=None, dataName="star"):
        """Plot quantity as a function of position"""
        pad = 0.02  # Number of degrees to pad the axis ranges
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

        magThreshold = self.magThreshold
        if dataName == "galaxy" and magThreshold < 99.0:
            magThreshold += 1.0  # plot to fainter mags for galaxies
        if dataName == "star" and "matches" in filename and magThreshold < 99.0:
            magThreshold += 1.0  # plot to fainter mags for matching against ref cat
        good = (self.mag < magThreshold if magThreshold > 0 else np.ones(len(self.mag), dtype=bool))
        if ((dataName == "star" or "matches" in filename or "compare" in filename) and
                "pStar" not in filename and "race" not in filename and "resolution" not in filename):
            vMin, vMax = 0.4*self.qMin, 0.4*self.qMax
            if "-mag_" in filename or any(ss in filename for ss in ["compareUnforced", "overlap"]):
                vMin, vMax = 0.6*vMin, 0.6*vMax
            if "-matches_mag" in filename:
                vMax = -vMin
        elif "CModel" in filename and "overlap" not in filename:
            vMin, vMax = 1.5*self.qMin, 0.5*self.qMax
        elif "raceDiff" in filename or "Resids" in filename:
            vMin, vMax = 0.5*self.qMin, 0.5*self.qMax
        elif "race" in filename:
            yDelta = 0.05*(self.qMax - self.qMin)
            vMin, vMax = self.qMin + yDelta, self.qMax - yDelta
        elif "pStar" in filename:
            vMin, vMax = 0.0, 1.0
        elif "resolution" in filename and "compare" not in filename:
            vMin, vMax = 0.0, 0.2
        elif "galacticExtinction" in filename and "compare" not in filename:
            vMin, vMax = min(1.05*self.qMin, 0.98*self.qMax), max(1.02*self.qMin, 0.95*self.qMax)
        else:
            vMin, vMax = self.qMin, self.qMax
        if "compare" not in filename:
            if dataName == "star" and "deconvMom" in filename:
                vMin, vMax = -0.1, 0.1
            if dataName == "galaxy" and "deconvMom" in filename:
                vMin, vMax = -0.1, 3.0*self.qMax
            if dataName == "galaxy" and "resolution" in filename:
                vMin, vMax = 0.0, 1.0
        if dataName == "galaxy" and "-mag_" in filename:
            vMin = 3.0*self.qMin
            if "GaussianFlux" in filename:
                vMin, vMax = 5.0*self.qMin, 0.0
        if (dataName == "galaxy" and ("CircularApertureFlux" in filename or "KronFlux" in filename) and
                "compare" not in filename and "overlap" not in filename):
            vMin, vMax = 4.0*self.qMin, 1.0*self.qMax

        fig, axes = plt.subplots(1, 1, subplot_kw=dict(facecolor="0.35"))
        axes.tick_params(which="both", direction="in", top=True, right=True, labelsize=8)
        ptSize = None

        if dataId is not None and butler is not None and ccdList is not None:
            plotCcdOutline(axes, butler, dataId, ccdList, zpLabel=zpLabel)

        if tractInfo is not None and patchList is not None:
            patchBoundary = getRaDecMinMaxPatchList(patchList, tractInfo, pad=pad, nDecimals=2, raMin=raMin,
                                                    raMax=raMax, decMin=decMin, decMax=decMax)
            raMin = patchBoundary.raMin
            raMax = patchBoundary.raMax
            decMin = patchBoundary.decMin
            decMax = patchBoundary.decMax
            plotPatchOutline(axes, tractInfo, patchList)

        stats0 = None
        for name, data in self.data.items():
            if name is not dataName:
                continue
            if not data.mag.any():
                continue
            if ptSize is None:
                ptSize = 0.7*setPtSize(len(data.mag))
            stats0 = self.calculateStats(data.quantity, good[data.selection])
            selection = data.selection & good
            if highlightList is not None:
                # Make highlight as a background ring of larger size than the data point size
                for flag, threshValue, color in highlightList:
                    label = flag.replace("merge_measurement", "ref")
                    # Only a white "halo" really shows up here, so ignore color
                    highlightSelection = (self.catalog[flag] > threshValue) & selection
                    axes.scatter(ra[highlightSelection], dec[highlightSelection], s=1.4*ptSize,
                                 marker="o", facecolors="none", edgecolors="white", label=label)

            axes.scatter(ra[selection], dec[selection], s=ptSize, marker="o", lw=0, label=name,
                         c=data.quantity[good[data.selection]], cmap=cmap, vmin=vMin, vmax=vMax)

        if stats0 is None:  # No data to plot
            return
        filterStr = dataId['filter'] if dataId is not None else ""
        filterLabelStr = "[" + filterStr + "]" if (filterStr and "/color/" not in filename) else ""
        axes.set_xlabel("RA (deg) {0:s}".format(filterLabelStr))
        axes.set_ylabel("Dec (deg) {0:s}".format(filterLabelStr))

        axes.set_xlim(raMax, raMin)
        axes.set_ylim(decMin, decMax)

        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vMin, vmax=vMax))
        mappable._A = []        # fake up the array of the scalar mappable. Urgh...
        cb = plt.colorbar(mappable)
        colorbarLabel = self.quantityName + " " + filterLabelStr
        fontSize = min(10, max(6, 10 - int(np.log(max(1, len(colorbarLabel) - 55)))))
        cb.ax.tick_params(labelsize=max(6, fontSize - 1))
        cb.set_label(colorbarLabel, fontsize=fontSize, rotation=270, labelpad=15)
        if hscRun is not None:
            axes.set_title("HSC stack run: " + hscRun, color="#800080")
        if camera is not None:
            labelCamera(camera, plt, axes, 0.5, 1.09)
        labelVisit(filename, plt, axes, 0.5, 1.04)
        if zpLabel is not None:
            prefix = "" if "GE applied" in zpLabel else "zp: "
            plotText(zpLabel, plt, axes, 0.13, -0.09, prefix=prefix, color="green")
        if forcedStr is not None:
            plotText(forcedStr, plt, axes, 0.85, -0.09, prefix="cat: ", color="green")
        if highlightList is not None:
            axes.legend(loc='upper left', bbox_to_anchor=(-0.05, 1.15), fancybox=True, shadow=True,
                        fontsize=7)
        else:
            axes.legend(loc='upper left', bbox_to_anchor=(-0.02, 1.08), fancybox=True, shadow=True,
                        fontsize=9)

        meanStr = "{0.mean:.4f}".format(stats0)
        stdevStr = "{0.stdev:.4f}".format(stats0)
        statsUnitStr = None
        if "(milli)" in self.quantityName:
            statsUnitStr = " (milli)"
        if "(mmag)" in self.quantityName:
            statsUnitStr = " (mmag)"
        if "(mas)" in self.quantityName:
            statsUnitStr = " (mas)"
        if statsUnitStr is not None:
            meanStr = "{0.mean:.2f}".format(stats0)
            stdevStr = "{0.stdev:.2f}".format(stats0)

        x0 = 0.86
        lenStr = 0.017*(max(len(meanStr), len(stdevStr)))
        axes.annotate("mean = ", xy=(x0, 1.08),
                      xycoords="axes fraction", ha="right", va="center", fontsize=8)
        axes.annotate(meanStr, xy=(x0 + lenStr, 1.08),
                      xycoords="axes fraction", ha="right", va="center", fontsize=8)
        if statsUnitStr is not None:
            axes.annotate(statsUnitStr, xy=(x0 + lenStr + 0.006, 1.08),
                          xycoords="axes fraction", ha="left", va="center", fontsize=8)
        axes.annotate("stdev = ", xy=(x0, 1.035),
                      xycoords="axes fraction", ha="right", va="center", fontsize=8)
        axes.annotate(stdevStr, xy=(x0 + lenStr, 1.035),
                      xycoords="axes fraction", ha="right", va="center", fontsize=8)
        axes.annotate(r"N = {0} [mag<{1:.1f}]".format(stats0.num, magThreshold),
                      xy=(x0 + lenStr + 0.012, 1.035),
                      xycoords="axes fraction", ha="left", va="center", fontsize=8)

        fig.savefig(filename, dpi=150)
        plt.close(fig)

    def plotRaDec(self, filename, stats=None, hscRun=None, matchRadius=None, zpLabel=None, forcedStr=None):
        """Plot quantity as a function of RA, Dec"""

        ra = np.rad2deg(self.catalog[self.prefix + "coord_ra"])
        dec = np.rad2deg(self.catalog[self.prefix + "coord_dec"])
        good = (self.mag < self.magThreshold if self.magThreshold is not None else
                np.ones(len(self.mag), dtype=bool))
        fig, axes = plt.subplots(2, 1)
        axes[0].axhline(0, linestyle="--", color="0.6")
        axes[1].axhline(0, linestyle="--", color="0.6")
        ptSize = None
        for name, data in self.data.items():
            if not data.mag.any():
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
            annotateAxes(filename, plt, axes[0], stats, "star", self.magThreshold, x0=0.03, yOff=0.07,
                         hscRun=hscRun, matchRadius=matchRadius, unitScale=self.unitScale)
            annotateAxes(filename, plt, axes[1], stats, "star", self.magThreshold, x0=0.03, yOff=0.07,
                         hscRun=hscRun, matchRadius=matchRadius, unitScale=self.unitScale)
        labelVisit(filename, plt, axes[0], 0.5, 1.1)
        if zpLabel is not None:
            prefix = "" if "GE applied" in zpLabel else "zp: "
            plotText(zpLabel, plt, axes[0], 0.13, -0.09, prefix=prefix, color="green")
        if forcedStr is not None:
            plotText(forcedStr, plt, axes[0], 0.85, -0.09, prefix="cat: ", color="green")
        fig.savefig(filename, dpi=120)
        plt.close(fig)

    def plotQuiver(self, catalog, filename, log, cmap=plt.cm.Spectral, stats=None, dataId=None, butler=None,
                   camera=None, ccdList=None, tractInfo=None, patchList=None, hscRun=None,
                   matchRadius=None, zpLabel=None, forcedStr=None, dataName="star", scale=1):
        """Plot ellipticity residuals quiver plot"""

        # Use HSM algorithm results if present, if not, use SDSS Shape
        if "ext_shapeHSM_HsmSourceMoments_xx" in catalog.schema:
            compareCol = "ext_shapeHSM_HsmSourceMoments"
            psfCompareCol = "ext_shapeHSM_HsmPsfMoments"
            shapeAlgorithm = "HSM"
            flags = ["ext_shapeHSM_HsmSourceMoments_flag", "ext_shapeHSM_HsmPsfMoments_flag"]
        else:
            compareCol = "base_SdssShape"
            psfCompareCol = "base_SdssShape_psf"
            shapeAlgorithm = "SDSS"
            flags = ["base_SdssShape_flag", "base_SdssShape_flag_psf"]

        # Cull the catalog of flagged sources
        bad = np.zeros(len(catalog), dtype=bool)
        bad |= catalog["deblend_nChild"] > 0
        for flag in flags:
            bad |= catalog[flag]
        # Cull the catalog down to calibration candidates (or stars if calibration flags not available)
        if "calib_psf_used" in catalog.schema:
            bad |= ~catalog["calib_psf_used"]
            catStr = "psf_used"
        elif "base_ClassificationExtendedness_value" in catalog.schema:
            bad |= catalog["base_ClassificationExtendedness_value"] > 0.5
            bad |= -2.5*np.log10(catalog[self.fluxColumn]) > self.magThreshold
            catStr = "ClassExtendedness"
        else:
            raise RuntimeError("Neither calib_psf_used nor base_ClassificationExtendedness_value in schema. "
                               "Skip quiver plot.")
        catalog = catalog[~bad].copy(deep=True)

        pad = 0.02  # Number of degrees to pad the axis ranges
        ra = np.rad2deg(catalog["coord_ra"])
        dec = np.rad2deg(catalog["coord_dec"])
        raMin, raMax = np.round(ra.min() - pad, 2), np.round(ra.max() + pad, 2)
        decMin, decMax = np.round(dec.min() - pad, 2), np.round(dec.max() + pad, 2)

        deltaRa = raMax - raMin
        deltaDec = decMax - decMin
        deltaDeg = max((deltaRa, deltaDec))
        raMin = (raMin + deltaRa/2.0) - deltaDeg/2.0
        raMax = raMin + deltaDeg
        decMin = (decMin + deltaDec/2.0) - deltaDeg/2.0
        decMax = decMin + deltaDeg

        fig, axes = plt.subplots(1, 1, subplot_kw=dict(facecolor="0.7"))
        axes.tick_params(which="both", direction="in", top=True, right=True, labelsize=8)

        if dataId is not None and butler is not None and ccdList is not None:
            plotCcdOutline(axes, butler, dataId, ccdList, zpLabel=zpLabel)

        if tractInfo is not None and patchList is not None:
            for ip, patch in enumerate(tractInfo):
                if str(patch.getIndex()[0])+","+str(patch.getIndex()[1]) in patchList:
                    raPatch, decPatch = bboxToXyCoordLists(patch.getOuterBBox(), wcs=tractInfo.getWcs())
                    raMin = min(np.round(min(raPatch) - pad, 2), raMin)
                    raMax = max(np.round(max(raPatch) + pad, 2), raMax)
                    decMin = min(np.round(min(decPatch) - pad, 2), decMin)
                    decMax = max(np.round(max(decPatch) + pad, 2), decMax)
            plotPatchOutline(axes, tractInfo, patchList)

        e1 = E1Resids(compareCol, psfCompareCol)
        e1 = e1(catalog)
        e2 = E2Resids(compareCol, psfCompareCol)
        e2 = e2(catalog)
        e = np.sqrt(e1**2 + e2**2)

        nz = matplotlib.colors.Normalize()
        nz.autoscale(e)
        cax, _ = matplotlib.colorbar.make_axes(plt.gca())
        cb = matplotlib.colorbar.ColorbarBase(cax, cmap=plt.cm.jet, norm=nz)
        cb.set_label(
            r"ellipticity residual: $\delta_e$ = $\sqrt{(e1_{src}-e1_{psf})^2 + (e2_{src}-e2_{psf})^2}$")

        getQuiver(ra, dec, e1, e2, axes, color=plt.cm.jet(nz(e)), scale=scale, width=0.002, label=catStr)

        filterStr = dataId['filter'] if dataId is not None else ''
        filterLabelStr = "[" + filterStr + "]"
        axes.set_xlabel("RA (deg) {0:s}".format(filterLabelStr))
        axes.set_ylabel("Dec (deg) {0:s}".format(filterLabelStr))

        axes.set_xlim(raMax, raMin)
        axes.set_ylim(decMin, decMax)

        good = np.ones(len(e), dtype=bool)
        stats0 = self.calculateStats(e, good)
        log.info("Statistics from %s of %s: %s" % (dataId, self.quantityName, stats0))
        meanStr = "{0.mean:.4f}".format(stats0)
        stdevStr = "{0.stdev:.4f}".format(stats0)

        x0 = 0.86
        lenStr = 0.1 + 0.022*(max(max(len(meanStr), len(stdevStr)) - 6, 0))
        axes.annotate("mean = ", xy=(x0, 1.08), xycoords="axes fraction",
                      ha="right", va="center", fontsize=8)
        axes.annotate(meanStr, xy=(x0 + lenStr, 1.08), xycoords="axes fraction",
                      ha="right", va="center", fontsize=8)
        axes.annotate("stdev = ", xy=(x0, 1.035), xycoords="axes fraction",
                      ha="right", va="center", fontsize=8)
        axes.annotate(stdevStr, xy=(x0 + lenStr, 1.035), xycoords="axes fraction",
                      ha="right", va="center", fontsize=8)
        axes.annotate(r"N = {0}".format(stats0.num), xy=(x0 + lenStr + 0.02, 1.035), xycoords="axes fraction",
                      ha="left", va="center", fontsize=8)

        if hscRun is not None:
            axes.set_title("HSC stack run: " + hscRun, color="#800080")
        if camera is not None:
            labelCamera(camera, plt, axes, 0.5, 1.09)
        labelVisit(filename, plt, axes, 0.5, 1.04)
        if zpLabel is not None:
            plotText(zpLabel, plt, axes, 0.13, -0.1, prefix="zp: ", color="green")
        plotText(shapeAlgorithm, plt, axes, 0.77, -0.1, prefix="Shape Alg: ", fontSize=8, color="green")
        if forcedStr is not None:
            plotText(forcedStr, plt, axes, 0.99, -0.1, prefix="cat: ", fontSize=8, color="green")
        axes.legend(loc='upper left', bbox_to_anchor=(0.0, 1.08), fancybox=True, shadow=True, fontsize=9)

        fig.savefig(filename, dpi=150)
        plt.close(fig)

    def plotInputCounts(self, catalog, filename, log, dataId, butler, tractInfo, patchList=None, camera=None,
                        forcedStr=None, cmap=plt.cm.viridis, alpha=0.5, doPlotTractImage=True,
                        doPlotPatchOutline=True, sizeFactor=5.0, maxDiamPix=1000):
        """Plot grayscale image of tract with base_InputCounts_value overplotted

        Parameters
        ----------
        catalog : `lsst.afw.table.SourceCatalog`
           The source catalog whose objects will be plotted as ellipses (scaled
           by a factor of ``sizeFactor`` but truncated to a maximum diameter
           of ``maxDiamPix``) and color-mapped by their base_InputCount_value.
        filename : `str`
           Full path and name of the file to which the plot will be written.
        log : `lsst.log.Log`
           Logger object for logging messages.
        dataId : `lsst.daf.persistence.DataId`
           An instance of `lsst.daf.persistence.DataId` from which to extract
           the filter name.
        butler : `lsst.daf.persistence.Butler`
        tractInfo : `lsst.skymap.tractInfo.ExplicitTractInfo`
           Tract information object.
        patchList : `list` of `str`, optional
           List of patch IDs with data to be plotted.
        camera : `lsst.afw.cameraGeom.Camera`, optional
           The base name of the coadd (e.g. deep or goodSeeing).
           Default is `None`.
        forcedStr : `str`, optional
           String to label the catalog type (forced vs. unforced) on the plot.
        cmap : `matplotlib.colors.ListedColormap`, optional
           The matplotlib colormap to use.  It will be given transparency level
           set by ``alpha``.  Default is `None`.
        alpha : `float`, optional
           The matplotlib blending value, between 0 (transparent) and 1 (opaque)
           Default is 0.5.
        doPlotTractImage : `bool`, optional
           A boolean indicating whether to plot the tract image (grayscale and
           asinh stretched).  Default is `True`.
        doPlotPatchOutline : `bool`, optional
           A boolean indicating whether to overplot the patch outlines and
           index labels.  Default is `True`.
        sizeFactor : `float`, optional
          Factor by which to multiply the source ellipse sizes for plotting
          (the nominal size is quite small).  Default is 5.0.
        maxDiamPix : `int`, optional
           A maximum diameter to be plotted for any source's ellipse (such that
           a single ellipse cannot overwhelm the plot and noting that this will
           not be indicative of the true input counts for the outer pixels as
           that number strictly applies to the objects centroid pixel).  If a
           given object gets truncated to this size, an opaque blue outline
           will be plotted around its ellipse.  Default is 1000.
        """
        tractBbox = tractInfo.getBBox()
        tractWcs = tractInfo.getWcs()

        fig, axes = plt.subplots(1, 1)
        axes.tick_params(which="both", direction="in", top=True, right=True, labelsize=7)
        if doPlotTractImage:
            image = buildTractImage(butler, dataId, tractInfo, patchList=patchList)
            med = np.nanmedian(image.array)
            mad = np.nanmedian(abs(image.array - med))
            imMin = med - 3.0*1.4826*mad
            imMax = med + 10.0*1.4826*mad
            norm = AsinhNormalize(minimum=imMin, dataRange=imMax - imMin, Q=8)
            extent = tractBbox.getMinX(), tractBbox.getMaxX(), tractBbox.getMinY(), tractBbox.getMaxY()
            axes.imshow(image.array, extent=extent, cmap="gray_r", norm=norm)

        centStr = "slot_Centroid"
        shapeStr = "slot_Shape"

        diamAs = []  # matplotlib.patches.Ellipse wants diameters for width and height
        diamBs = []
        thetas = []
        edgeColors = []  # to outline any ellipses truncated at maxDiamPix

        for src in catalog:
            edgeColor = "None"
            srcQuad = afwGeom.Quadrupole(src[shapeStr + "_xx"], src[shapeStr + "_yy"], src[shapeStr + "_xy"])
            srcEllip = afwGeom.ellipses.Axes(srcQuad)
            diamA = srcEllip.getA()*2.0*sizeFactor
            diamB = srcEllip.getB()*2.0*sizeFactor
            # Truncate ellipse size to a maximum width or height of maxDiamPix
            if diamA > maxDiamPix or diamB > maxDiamPix:
                edgeColor = "blue"
                if diamA >= diamB:
                    diamB = diamB*(maxDiamPix/diamA)
                    diamA = maxDiamPix
                else:
                    diamA = diamA*(maxDiamPix/diamB)
                    diamB = maxDiamPix
            diamAs.append(diamA)
            diamBs.append(diamB)
            thetas.append(np.degrees(srcEllip.getTheta()))
            edgeColors.append(edgeColor)

        xyOffsets = np.stack((catalog[centStr + "_x"], catalog[centStr + "_y"]), axis=-1)
        inputCounts = catalog["base_InputCount_value"]
        bounds = np.arange(inputCounts.max())
        bounds += 1
        alphaCmap = makeAlphaCmap(cmap=cmap, alpha=alpha)
        norm = matplotlib.colors.BoundaryNorm(bounds, alphaCmap.N)
        alphaCmap.set_under("r")

        ellipsePatchList = [matplotlib.patches.Ellipse(xy=xy, width=diamA, height=diamB, angle=theta)
                            for xy, diamA, diamB, theta in zip(xyOffsets, diamAs, diamBs, thetas)]
        ec = matplotlib.collections.PatchCollection(ellipsePatchList, cmap=alphaCmap, norm=norm,
                                                    edgecolors=edgeColors, linewidth=0.1)

        ec.set_array(inputCounts)
        axes.add_collection(ec)
        cbar = plt.colorbar(ec, extend="min", fraction=0.04)
        cbar.set_label("InputCount: ellipse size * {:} [maxDiam = {:}] (pixels)".
                       format(sizeFactor, maxDiamPix), fontsize=7)
        cbar.ax.tick_params(direction="in", labelsize=7)

        axes.set_xlim(tractBbox.getMinX(), tractBbox.getMaxX())
        axes.set_ylim(tractBbox.getMinY(), tractBbox.getMaxY())

        filterStr = dataId["filter"]
        filterLabelStr = "[" + filterStr + "]"
        axes.set_xlabel("xTract (pixels) {0:s}".format(filterLabelStr), size=9)
        axes.set_ylabel("yTract (pixels) {0:s}".format(filterLabelStr), size=9)

        # Get Ra and DEC tract limits to add to plot axis labels
        tract00 = tractWcs.pixelToSky(tractBbox.getMinX(),
                                      tractBbox.getMinY()).getPosition(units=afwGeom.degrees)
        tract0N = tractWcs.pixelToSky(tractBbox.getMinX(),
                                      tractBbox.getMaxY()).getPosition(units=afwGeom.degrees)
        tractN0 = tractWcs.pixelToSky(tractBbox.getMaxX(),
                                      tractBbox.getMinY()).getPosition(units=afwGeom.degrees)

        textKwargs = dict(ha="left", va="center", transform=axes.transAxes, fontsize=7, color="blue")
        plt.text(-0.05, -0.07, str("{:.2f}".format(tract00.getX())), **textKwargs)
        plt.text(-0.17, 0.00, str("{:.2f}".format(tract00.getY())), **textKwargs)
        plt.text(0.96, -0.07, str("{:.2f}".format(tractN0.getX())), **textKwargs)
        plt.text(-0.17, 0.97, str("{:.2f}".format(tract0N.getY())), **textKwargs)
        textKwargs["fontsize"] = 8
        plt.text(0.45, -0.11, "RA (deg)", **textKwargs)
        plt.text(-0.19, 0.5, "DEC (deg)", rotation=90, **textKwargs)

        if doPlotPatchOutline:
            plotPatchOutline(axes, tractInfo, patchList, plotUnits="pixel", idFontSize=5)
        if camera is not None:
            labelCamera(camera, plt, axes, 0.5, 1.09)
        labelVisit(filename, plt, axes, 0.5, 1.04)
        if forcedStr is not None:
            plotText(forcedStr, plt, axes, 0.99, -0.1, prefix="cat: ", fontSize=8, color="green")

        fig.savefig(filename, dpi=1200)  # Needs to be fairly hi-res to see enough detail
        plt.close(fig)

    def plotAll(self, dataId, filenamer, log, enforcer=None, butler=None, camera=None, ccdList=None,
                tractInfo=None, patchList=None, hscRun=None, matchRadius=None, zpLabel=None, forcedStr=None,
                postFix="", plotRunStats=True, highlightList=None, extraLabels=None):
        """Make all plots"""
        stats = self.stats
        if "galacticExtinction" not in self.shortName:
            self.plotAgainstMagAndHist(log, filenamer(dataId, description=self.shortName,
                                                      style="psfMagHist" + postFix),
                                       stats=stats, camera=camera, ccdList=ccdList, tractInfo=tractInfo,
                                       patchList=patchList, hscRun=hscRun, matchRadius=matchRadius,
                                       zpLabel=zpLabel, forcedStr=forcedStr, plotRunStats=plotRunStats,
                                       highlightList=highlightList, filterStr=dataId['filter'],
                                       extraLabels=extraLabels)

        if self.config.doPlotOldMagsHist and "galacticExtinction" not in self.shortName:
            self.plotAgainstMag(filenamer(dataId, description=self.shortName, style="psfMag" + postFix),
                                stats=stats, hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel,
                                forcedStr=forcedStr)
            self.plotHistogram(filenamer(dataId, description=self.shortName, style="hist" + postFix),
                               stats=stats, hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel,
                               forcedStr=forcedStr)

        skyPositionKwargs = dict(stats=stats, dataId=dataId, butler=butler, camera=camera, ccdList=ccdList,
                                 tractInfo=tractInfo, patchList=patchList, hscRun=hscRun,
                                 matchRadius=matchRadius, zpLabel=zpLabel, highlightList=highlightList,
                                 forcedStr=forcedStr)
        if "all" in self.data:
            styleStr = "sky-all"
            dataName = "all"
            self.plotSkyPosition(filenamer(dataId, description=self.shortName, style=styleStr + postFix),
                                 dataName=dataName, **skyPositionKwargs)
        if "star" in self.data:
            styleStr = "sky-stars"
            dataName = "star"
            self.plotSkyPosition(filenamer(dataId, description=self.shortName, style=styleStr + postFix),
                                 dataName=dataName, **skyPositionKwargs)
        if "galaxy" in self.data and (not any(ss in self.shortName for ss in
                                              ["pStar", "race", "Xx", "Yy", "Resids", "psf_used",
                                               "photometry_used", "gri", "riz", "izy", "z9y", "color_"])):
            styleStr = "sky-gals"
            dataName = "galaxy"
            self.plotSkyPosition(filenamer(dataId, description=self.shortName, style=styleStr + postFix),
                                 dataName=dataName, **skyPositionKwargs)
        if "diff_" in self.shortName and stats["split"].num > 0:
            styleStr = "sky-split"
            dataName = "split"
            self.plotSkyPosition(filenamer(dataId, description=self.shortName, style=styleStr + postFix),
                                 dataName=dataName, **skyPositionKwargs)

        if self.config.doPlotRaDec:
            self.plotRaDec(filenamer(dataId, description=self.shortName, style="radec" + postFix),
                           stats=stats, hscRun=hscRun, matchRadius=matchRadius, zpLabel=zpLabel,
                           forcedStr=forcedStr)
        log.info("Statistics from %s of %s: %s" % (dataId, self.quantityName, stats))
        if enforcer:
            enforcer(stats, dataId, log, self.quantityName)
        return stats

    def statistics(self, forcedMean=None):
        """Calculate statistics on quantity"""
        stats = {}
        for name, data in self.data.items():
            good = data.mag < self.magThreshold
            stats[name] = self.calculateStats(data.quantity, good, forcedMean=forcedMean)
            if self.quantityError is not None:
                stats[name].sysErr = self.calculateSysError(data.quantity, data.error,
                                                            good, forcedMean=forcedMean)
            if not stats:
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
                print("Warning: sysErr calculation failed: {:s}".format(result.message))
                answer = np.nan
            else:
                answer = np.sqrt(result.x[0])
        else:
            answer = np.sqrt(scipy.optimize.newton(function, 0.0, tol=tol))
        print("calculateSysError: {0:.4f}, {1:.4f}, {2:.4f}".format(function(answer**2),
                                                                    function((answer+0.001)**2),
                                                                    function((answer-0.001)**2)))
        return answer
