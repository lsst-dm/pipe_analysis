import matplotlib.pyplot as plt
import numpy as np
from  scipy.stats import median_absolute_deviation as sigmaMad
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from matplotlib.path import Path
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig

from .plotUtilsGen3 import generateSummaryStats, parsePlotInfo, addPlotInfo

class ScatterPlotWithTwoHistsTaskConnections(pipeBase.PipelineTaskConnections,
                                             dimensions=("tract", "skymap"),
                                             defaultTemplates={"inputCoaddName": "deep",
                                                               "plotName": "deltaCoords"}):

    catPlot = pipeBase.connectionTypes.Input(doc="The tract wide catalog to make plots from.",
                                             storageClass="DataFrame",
                                             name="qaTable_tract",
                                             dimensions=("tract", "skymap"))


    scatterPlot = pipeBase.connectionTypes.Output(doc="A scatter plot with histograms for both axes.",
                                                 storageClass="Plot",
                                                 name="scatterTwoHist_{plotName}",
                                                 dimensions=("tract", "skymap"))

    skymap = pipeBase.connectionTypes.Input(doc="The skymap for the tract",
                                            storageClass="SkyMap",
                                            name="{inputCoaddName}Coadd_skyMap",
                                            dimensions=("skymap",))

class ScatterPlotWithTwoHistsTaskConfig(pipeBase.PipelineTaskConfig,
                                        pipelineConnections=ScatterPlotWithTwoHistsTaskConnections):

    xColName = pexConfig.Field(
        doc="The column name for the values to be plotted on the x axis.",
        dtype=str,
        default="coord_ra",
    )

    yColName = pexConfig.Field(
        doc="The column name for the values to be plotted on the y axis.",
        dtype=str,
        default="coord_dec",
    )

    xLabel = pexConfig.Field(
        doc="The x axis label",
        dtype=str,
        default="Right Ascension (deg)",
    )

    yLabel = pexConfig.Field(
        doc="The y axis label",
        dtype=str,
        default="Declination (deg)",
    )

    title = pexConfig.Field(
        doc="The title of the plot.",
        dtype=str,
        default="Coordinates Plot",
    )

    sourceTypeColName = pexConfig.Field(
        doc="The column to use for star - galaxy separation.",
        dtype=str,
        default="iExtendedness",
    )

    objectsToPlot = pexConfig.Field(
        doc="Which types of objects to include on the plot, should be one of 'stars', 'galaxies' or 'all'.",
        dtype=str,
        default="stars",
    )


class ScatterPlotWithTwoHistsTask(pipeBase.PipelineTask):

    ConfigClass = ScatterPlotWithTwoHistsTaskConfig
    _DefaultName = "scatterPlotWithTwoHistsTask"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docs inherited from base class
        inputs = butlerQC.get(inputRefs)
        runName = inputRefs.catPlot.run
        dataId = butlerQC.quantum.dataId
        inputs["dataId"] = dataId
        inputs["runName"] = runName
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)


    def run(self, catPlot, dataId, runName, skymap):

        xLabel = self.config.xLabel
        yLabel = self.config.yLabel
        title = self.config.title

        plotInfo = parsePlotInfo(dataId, runName)
        sumStats = generateSummaryStats(catPlot, self.config.yColName, skymap, plotInfo)
        fig = self.scatterPlotWithTwoHists(catPlot, xLabel, yLabel, title, plotInfo, sumStats)

        return pipeBase.Struct(scatterPlot=fig)

    def scatterPlotWithTwoHists(self, catPlot, xLabel, yLabel, title, plotInfo, sumStats,
                                yLims=False, xLims=False):

        """Makes a generic plot with a 2D histogram and collapsed histograms of
        each axis.

        Parameters
        ----------
        xs : `numpy.ndarray`
            The array to be plotted on the x axis.
        ys : `numpy.ndarray`
            The array to be plotted on the y axis.
        xName : `str`
            The name to be used in the text for the x axis statistics.
        yName : `str`
            The name to be used in the text for the y axis statistics.
        xLabel : `str`
            The text to go on the xLabel of the plot.
        yLabel : `str`
            The text to go on the yLabel of the plot.
        title : `str`
            The text to be displayed as the plot title.
        plotInfoDict : `dict`
            A dictionary of information about the data being plotted with keys:
                ``camera``
                    The camera used to take the data
                    (`lsst.afw.cameraGeom.Camera`).
                ``"cameraName"``
                    The name of camera used to take the data (`str`).
                ``"filter"``
                    The filter used for this data (`str`).
                ``"visit"``
                    The visit of the data; only included if the data is from a
                    single epoch dataset (`str`).
                ``"patch"``
                    The patch that the data is from; only included if the data is
                    from a coadd dataset (`str`).
                ``"tract"``
                    The tract that the data comes from (`str`).
                ``"photoCalibDataset"``
                    The dataset used for the calibration, e.g. "jointcal" or "fgcm"
                    (`str`).
                ``"skyWcsDataset"``
                    The sky Wcs dataset used (`str`).
                ``"rerun"``
                    The rerun the data is stored in (`str`).
        statsUnit : `str`
            The text used to describe the units of the statistics calculated.
        yLims : `Bool` or `tuple`, optional
            The y axis limits to use for the plot.  If `False`, they are calculated
            from the data.  If being given a tuple of (yMin, yMax).
        xLims : `Bool` or `tuple`, optional
            The x axis limits to use for the plot.  If `False`, they are calculated
            from the data.  If being given a tuple of (xMin, xMax).

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            The resulting figure.
        xsMed : `float`
            The median of the values plotted on the x axis.
        xsSigmaMAD : `float`
            The sigma MAD from the x axis values.
        ysMed : `float`
            The median of the values plotted on the y axis.
        ysSigmaMAD : `float`
            The sigma MAD from the y axis values.

        Notes
        -----
        ``plotInfoDict`` needs to be a `dict` containing keys "camera", "filter",
        "visit", "tract" and "photoCalibDataset" (e.g. "jointcal" or "fgcm"), and
        "skyWcsDataset" ("jointcal" or not), it is used to add information to the
        plot.  Returns the median and sigma MAD for the x and y values.
        """

        fig = plt.figure()
        gs = gridspec.GridSpec(4, 4)

        # Cut the catalogue down to only valid sources
        catPlot = catPlot[catPlot["useForQAFlag"].values]

        # Need to separate stars and galaxies
        stars = (catPlot[self.config.sourceTypeColName] == 0.0)
        galaxies = (catPlot[self.config.sourceTypeColName] == 1.0)

        # For galaxies
        xsGalaxies = catPlot[self.config.xColName].values[galaxies]
        ysGalaxies = catPlot[self.config.yColName].values[galaxies]

        # For stars
        xsStars = catPlot[self.config.xColName].values[stars]
        ysStars = catPlot[self.config.yColName].values[stars]

        highStats = []
        lowStats = []
        # Calculate some statistics
        if self.config.objectsToPlot == "galaxies" or self.config.objectsToPlot == "all":
            highSnGals = ((catPlot["useForStats"] == 1) & galaxies)
            highSnGalMed = np.nanmedian(catPlot[self.config.yColName].values[highSnGals])
            highSnGalMad = sigmaMad(catPlot[self.config.yColName].values[highSnGals], nan_policy="omit")

            lowSnGals = (((catPlot["useForStats"] == 1) | (catPlot["useForStats"] == 2)) & galaxies)
            lowSnGalMed = np.nanmedian(catPlot[self.config.yColName].values[lowSnGals])
            lowSnGalMad = sigmaMad(catPlot[self.config.yColName].values[lowSnGals], nan_policy="omit")

            highStats += ["Median: {:0.2f}".format(highSnGalMed),
                          r"$\sigma_{MAD}$: " + "{:0.2f}".format(highSnGalMad)]
            lowStats += ["Median: {:0.2f}".format(lowSnGalMed),
                         r"$\sigma_{MAD}$: " + "{:0.2f}".format(lowSnGalMad)]

        if self.config.objectsToPlot == "stars" or self.config.objectsToPlot == "all":
            highSnStars = ((catPlot["useForStats"] == 1) & stars)
            highSnStarMed = np.nanmedian(catPlot[self.config.yColName].values[highSnStars])
            highSnStarMad = sigmaMad(catPlot[self.config.yColName].values[highSnStars], nan_policy="omit")
            approxHighMag = np.max(catPlot["iCModelMag"].values[highSnStars])

            lowSnStars = (((catPlot["useForStats"] == 1) | (catPlot["useForStats"] == 2)) & stars)
            lowSnStarMed = np.nanmedian(catPlot[self.config.yColName].values[lowSnStars])
            lowSnStarMad = sigmaMad(catPlot[self.config.yColName].values[lowSnStars], nan_policy="omit")
            approxLowMag = np.max(catPlot["iCModelMag"].values[lowSnStars])

            highStats  += ["Median: {:0.2f}".format(highSnStarMed),
                           r"$\sigma_{MAD}$: " + "{:0.2f}".format(highSnStarMad)]
            lowStats += ["Median: {:0.2f}".format(lowSnStarMed),
                         r"$\sigma_{MAD}$: " + "{:0.2f}".format(lowSnStarMad)]

        # Main scatter plot
        ax = fig.add_subplot(gs[1:, :-1])
        binThresh = 50

        [xs1, xs25, xs50, xs75, xs95, xs97] = np.nanpercentile(xsStars, [1, 25, 50, 75, 95, 97])
        xScale = (xs97 - xs1)/20.0  # This is ~5% of the data range
        # 40 was used as the number of bins because it looked good, might need to
        # be changed in the future
        xEdges = np.arange(xs1 - xScale, xs95, (xs95 - (xs1 - xScale))/40.0)

        yBinsOut = []
        linesForLegend = []

        if self.config.objectsToPlot == "stars":
            toPlotList = [(xsStars, ysStars, "C0")]
        elif self.config.objectsToPlot == "galaxies":
            toPlotList = [(xsGalaxies, ysGalaxies, "C3")]
        elif self.config.objectsToPlot == "all":
            toPlotList = [(xsGalaxies, ysGalaxies, "C3"), (xsStars, ysStars, "C0")]

        for (xs, ys, color) in toPlotList:
            medYs = np.nanmedian(ys)
            sigMadYs = sigmaMad(ys, nan_policy="omit")
            fiveSigmaHigh = medYs + 5.0*sigMadYs
            fiveSigmaLow = medYs - 5.0*sigMadYs
            binSize = (fiveSigmaHigh - fiveSigmaLow)/101.0
            yEdges = np.arange(fiveSigmaLow, fiveSigmaHigh, binSize)

            counts, xBins, yBins = np.histogram2d(xs, ys, bins=(xEdges, yEdges))
            yBinsOut.append(yBins)
            countsYs = np.sum(counts, axis=1)

            ids = np.where((countsYs > binThresh))[0]
            xEdgesPlot = xEdges[ids][1:]
            xEdges = xEdges[ids]

            # Create the codes needed to turn the sigmaMad lines into a path to speed
            # up checking which points are inside the area.
            codes = np.ones(len(xEdgesPlot)*2)*Path.LINETO
            codes[0] = Path.MOVETO
            codes[-1] = Path.CLOSEPOLY

            meds = np.zeros(len(xEdgesPlot))
            threeSigMadVerts = np.zeros((len(xEdgesPlot)*2, 2))
            sigMads = np.zeros(len(xEdgesPlot))

            for (i, xEdge) in enumerate(xEdgesPlot):
                ids = np.where((xs < xEdge) & (xs > xEdges[i]) & (np.isfinite(ys)))[0]
                med = np.median(ys[ids])
                sigMad = sigmaMad(ys[ids])
                meds[i] = med
                sigMads[i] = sigMad
                threeSigMadVerts[i, :] = [xEdge, med + 3*sigMad]
                threeSigMadVerts[-(i + 1), :] = [xEdge, med - 3*sigMad]

            medDiff = np.nanmedian(ys)
            sigMadDiff = sigmaMad(ys, nan_policy="omit")
            medLine, = ax.plot(xEdgesPlot, meds, color, label="Median: {:0.2f}".format(medDiff))
            linesForLegend.append(medLine)

            # Make path to check which points lie within one sigma mad
            threeSigMadPath = Path(threeSigMadVerts, codes)

            # Add lines for the median +/- 0.5 * sigma MAD
            #halfSigMadLine, = ax.plot(xEdgesPlot, meds + 0.5*sigMads, color, alpha=0.8)
            #ax.plot(xEdgesPlot, meds - 0.5*sigMads, color, alpha=0.8)

            # Add lines for the median +/- 3 * sigma MAD
            threeSigMadLine, = ax.plot(xEdgesPlot, threeSigMadVerts[:len(xEdgesPlot), 1], color, alpha=0.4)
            ax.plot(xEdgesPlot[::-1], threeSigMadVerts[len(xEdgesPlot):, 1], color, alpha=0.4)

            # Add lines for the median +/- 1 * sigma MAD
            sigMadLine, = ax.plot(xEdgesPlot, meds + 1.0*sigMads, color, alpha=0.8,
                                  label=r"$\sigma_{MAD}$: " + "{:0.2f}".format(sigMadDiff))
            linesForLegend.append(sigMadLine)
            ax.plot(xEdgesPlot, meds - 1.0*sigMads, color, alpha=0.8)

            # Add lines for the median +/- 2 * sigma MAD
            twoSigMadLine, = ax.plot(xEdgesPlot, meds + 2.0*sigMads, color, alpha=0.6)
            ax.plot(xEdgesPlot, meds - 2.0*sigMads, color, alpha=0.6)

            # Check which points are outside 3 sigma MAD of the median and plot these as
            # points.
            inside = threeSigMadPath.contains_points(np.array([xs, ys]).T)
            points, = ax.plot(xs[~inside], ys[~inside], ".", ms=3, alpha=0.3, mfc=color, mec=color, zorder=-1)

        # Set the scatter plot limits
        plotMed = np.nanmedian(ysStars)
        if yLims:
            ax.set_ylim(yLims[0], yLims[1])
        else:
            numSig = 3
            while np.max(meds) > plotMed + numSig*sigMadYs or np.min(meds) < plotMed - numSig*sigMadYs:
                numSig += 1

            numSig += 1
            ax.set_ylim(plotMed - numSig*sigMadYs, plotMed + numSig*sigMadYs)

        if xLims:
            ax.set_xlim(xLims[0], xLims[1])
        else:
            ax.set_xlim(xs1 - xScale, xs97)

        # Add axes labels
        ax.set_ylabel(yLabel, fontsize=12)
        ax.set_xlabel(xLabel, fontsize=12)

        # Top histogram
        topHist = plt.gcf().add_subplot(gs[0, :-1], sharex=ax)
        topHist.hist(catPlot[self.config.xColName].values, bins=100, color="grey", alpha=0.3, log=True,
                     label="All ({})".format(len(catPlot)))
        if self.config.objectsToPlot == "galaxies" or self.config.objectsToPlot == "all":
            topHist.hist(xsGalaxies, bins=100, color="C3", histtype="step", log=True,
                         label="Galaxies ({})".format(len(np.where(galaxies)[0])))
        if self.config.objectsToPlot == "stars" or self.config.objectsToPlot == "all":
            topHist.hist(xsStars, bins=100, color="C0", histtype="step", log=True,
                         label="Stars ({})".format(len(np.where(stars)[0])))
        topHist.axes.get_xaxis().set_visible(False)
        topHist.set_ylabel("Number")
        topHist.legend(fontsize=6, framealpha=0.9, borderpad=0.4, loc="lower left", ncol=3, edgecolor="k")

        # Side histogram
        sideHist = plt.gcf().add_subplot(gs[1:, -1], sharey=ax)
        finiteObjs = np.isfinite(catPlot[self.config.yColName].values)
        sideHist.hist(catPlot[self.config.yColName].values[finiteObjs], bins=100, color="grey", alpha=0.3,
                      orientation="horizontal", log=True)
        if self.config.objectsToPlot == "galaxies" or self.config.objectsToPlot == "all":
            sideHist.hist(ysGalaxies[np.isfinite(ysGalaxies)], bins=100, color="C3", histtype="step",
                          orientation="horizontal", log=True)
        if self.config.objectsToPlot == "stars" or self.config.objectsToPlot == "all":
            sideHist.hist(ysStars[np.isfinite(ysStars)], bins=100, color="C0", histtype="step",
                          orientation="horizontal", log=True)
        sideHist.axes.get_yaxis().set_visible(False)
        sideHist.set_xlabel("Number")

        # Corner plot of patches showing summary stat in each
        axCorner = plt.gcf().add_subplot(gs[0, -1])
        axCorner.yaxis.tick_right()
        axCorner.yaxis.set_label_position("right")
        axCorner.xaxis.tick_top()
        axCorner.xaxis.set_label_position("top")

        patches = []
        colors = []
        for dataId in sumStats.keys():
            (corners, stat) = sumStats[dataId]
            ra = corners[0][0].asDegrees()
            dec = corners[0][1].asDegrees()
            xy = (ra, dec)
            width = corners[2][0].asDegrees() - ra
            height = corners[2][1].asDegrees() - dec
            patches.append(Rectangle(xy, width, height))
            colors.append(stat)
            ras = [ra.asDegrees() for (ra, dec) in corners]
            decs = [dec.asDegrees() for (ra, dec) in corners]
            axCorner.plot(ras + [ras[0]], decs + [decs[0]], "k", lw=0.5)
            cenX = ra + width / 2
            cenY = dec + height / 2
            if dataId != "tract":
                axCorner.annotate(dataId, (cenX, cenY), color="k", fontsize=5, ha="center", va="center")

        cmapUse = plt.cm.coolwarm
        # Set the bad color to transparent and make a masked array
        cmapUse.set_bad(color="none")
        colors = np.ma.array(colors, mask=np.isnan(colors))
        collection = PatchCollection(patches, cmap=cmapUse)
        collection.set_array(colors)
        axCorner.add_collection(collection)

        axCorner.set_xlabel("R.A. (deg)", fontsize=8)
        axCorner.set_ylabel("Dec. (deg)", fontsize=8)
        axCorner.tick_params(axis="both", labelsize=8)

        # Add a colorbar
        divider = make_axes_locatable(axCorner)
        cax = divider.append_axes("left", size="14%")
        plt.colorbar(collection, cax=cax, orientation="vertical")
        cax.yaxis.set_ticks_position("left")
        for label in cax.yaxis.get_ticklabels():
            label.set_bbox(dict(facecolor="w", ec="none", alpha=0.5))
            label.set_fontsize(8)
        cax.text(0.6, 0.5, "Median Value", color="k", rotation="vertical", transform=cax.transAxes,
                 horizontalalignment="center", verticalalignment="center", fontsize=8)

        plt.draw()

        # Add a legend
        highLabels = [r"$\sigma_{MAD}$: " + "{:0.2f}".format(sigMadDiff)]
        for (i, newLabel) in enumerate(highStats):
            linesForLegend[i].set_label(newLabel)
        highLegendTitle = "S/N > 2700 Stats \n({} < {:0.2f})".format(self.config.xColName, approxHighMag)
        fig.legend(handles=linesForLegend, fontsize=6, bbox_to_anchor=(0.81, 0.655), edgecolor="k",
                   bbox_transform=fig.transFigure, title=highLegendTitle, title_fontsize=6)
        for (i, newLabel) in enumerate(lowStats):
            linesForLegend[i].set_label(newLabel)
        lowLegendTitle = "S/N > 500 Stats \n({} < {:0.2f})".format(self.config.xColName, approxLowMag)
        fig.legend(handles=linesForLegend, fontsize=6, bbox_to_anchor=(0.81, 0.355), edgecolor="k",
                   bbox_transform=fig.transFigure, title=lowLegendTitle, title_fontsize=6)

        # Add useful information to the plot

        plt.draw()
        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        fig = plt.gcf()

        fig = addPlotInfo(fig, plotInfo)

        return fig
