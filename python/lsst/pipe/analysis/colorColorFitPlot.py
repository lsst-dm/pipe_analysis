import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import median_absolute_deviation as sigmaMad
from scipy import stats as scipyStats

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig

from .plotUtilsGen3 import generateSummaryStats, parsePlotInfo, addPlotInfo


class ColorColorFitPlotTaskConnections(pipeBase.PipelineTaskConnections,
                                       dimensions=("tract", "skymap"),
                                       defaultTemplates={"inputCoaddName": "deep",
                                                         "plotName": "deltaCoords"}):

    catPlot = pipeBase.connectionTypes.Input(doc="The tract wide catalog to make plots from.",
                                             storageClass="DataFrame",
                                             name="qaTable_tract",
                                             dimensions=("tract", "skymap"))

    colorColorFitPlot = pipeBase.connectionTypes.Output(doc="A scatter plot with histograms for both axes.",
                                                        storageClass="Plot",
                                                        name="colorColor_{plotName}",
                                                        dimensions=("tract", "skymap"))

    skymap = pipeBase.connectionTypes.Input(doc="The skymap for the tract",
                                            storageClass="SkyMap",
                                            name="{inputCoaddName}Coadd_skyMap",
                                            dimensions=("skymap",))


class ColorColorFitPlotTaskConfig(pipeBase.PipelineTaskConfig,
                                  pipelineConnections=ColorColorFitPlotTaskConnections):

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

    name = pexConfig.Field(
        doc="The name of the subject of the plot.",
        dtype=str,
        default="yFit",
    )

    sourceTypeColName = pexConfig.Field(
        doc="The column to use for star - galaxy separation.",
        dtype=str,
        default="iExtendedness",
    )


class ColorColorFitPlotTask(pipeBase.PipelineTask):

    ConfigClass = ColorColorFitPlotTaskConfig
    _DefaultName = "ColorColorFitPlotTask"

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
        title = self.config.name

        plotInfo = parsePlotInfo(dataId, runName)
        sumStats = generateSummaryStats(catPlot, self.config.yColName, skymap, plotInfo)
        fig = self.colorColorFitPlot(catPlot, xLabel, yLabel, title, plotInfo, sumStats)

        return pipeBase.Struct(colorColorFitPlot=fig)

    def colorColorFitPlot(self, catPlot, xLabel, yLabel, name, plotInfo, sumStats, yLims=False, xLims=False):
        # TODO: Should the fitting stuff be here or is it done elsewhere?
        # TODO: Propogate the fit details from elsewhere?
        # TODO: This is slow, improve it once we figure out what it should do.

        # Get the limits of the box used for the statistics
        # Putting random values in here for now
        self.log.info(("Plotting {}: the values of {} against {} on a color-color plot with the area "
                       "used for calculating the stellar locus fits marked.".format(
                       self.config.connections.plotName, self.config.xColName, self.config.yColName)))

        if name == "yFit":
            xBoxLims = (0.8, 2.1)
            yBoxLims = (0.3, 0.9)
        elif name == "wFit":
            xBoxLims = (0.1, 1.0)
            yBoxLims = (0.0, 0.5)
        elif name == "xFit":
            xBoxLims = (1.0, 1.4)
            yBoxLims = (0.6, 1.6)

        fig = plt.figure()
        ax = fig.add_axes([0.12, 0.11, 0.3, 0.75])
        axHist = fig.add_axes([0.65, 0.11, 0.3, 0.75])

        # Cut the catalogue down to only valid sources
        catPlot = catPlot[catPlot["useForQAFlag"].values]

        # Only use sources brighter than 26th
        catPlot = catPlot[(catPlot["iCModelMag"] < 26.0)]

        # Need to separate stars and galaxies
        stars = (catPlot[self.config.sourceTypeColName] == 0.0)

        # For stars
        xsStars = catPlot[self.config.xColName].values[stars]
        ysStars = catPlot[self.config.yColName].values[stars]

        # Points to use for the fit
        fitPoints = np.where((xsStars > xBoxLims[0]) & (xsStars < xBoxLims[1])
                             & (ysStars > yBoxLims[0]) & (ysStars < yBoxLims[1]))[0]
        ax.plot([xBoxLims[0], xBoxLims[1], xBoxLims[1], xBoxLims[0], xBoxLims[0]],
                [yBoxLims[0], yBoxLims[0], yBoxLims[1], yBoxLims[1], yBoxLims[0]], "k")

        bbox = dict(alpha=0.9, facecolor="white", edgecolor="none")
        ax.text(0.05, 0.95, "N Used: {}".format(len(fitPoints)), color="k", transform=ax.transAxes,
                fontsize=8, bbox=bbox)

        xyStars = np.vstack([xsStars, ysStars])
        starsKde = scipyStats.gaussian_kde(xyStars)
        zStars = starsKde(xyStars)

        starPoints = ax.scatter(xsStars[~fitPoints], ysStars[~fitPoints], c=zStars[~fitPoints],
                                cmap="Greys_r", label="Stars", s=0.5)
        starFitPoints = ax.scatter(xsStars[fitPoints], ysStars[fitPoints], c=zStars[fitPoints], cmap="winter",
                                   label="Used for Fit", s=0.5)

        # Add colorbar
        starCbAx = fig.add_axes([0.43, 0.11, 0.04, 0.75])
        plt.colorbar(starFitPoints, cax=starCbAx)
        starCbAx.text(0.6, 0.5, "Number Density", color="k", rotation="vertical",
                      transform=starCbAx.transAxes, ha="center", va="center", fontsize=10)

        ax.set_xlabel(self.config.xLabel)
        ax.set_ylabel(self.config.yLabel)

        # Set useful axis limits
        starPercsX = np.nanpercentile(xsStars, [1, 99.5])
        starPercsY = np.nanpercentile(ysStars, [1, 99.5])
        ax.set_xlim(starPercsX[0], starPercsX[1])
        ax.set_ylim(starPercsY[0], starPercsY[1])

        # Fit a line to the points in the box
        (m, b) = np.polyfit(xsStars, ysStars, 1)
        xsFitLine = [xBoxLims[0], xBoxLims[1]]
        ysFitLine = [m*xsFitLine[0] + b, m*xsFitLine[1] + b]
        ax.plot(xsFitLine, ysFitLine, "w", lw=3, ls="--")
        ax.plot(xsFitLine, ysFitLine, "k", lw=1, ls="--")

        p1 = np.array([xsFitLine[0], ysFitLine[0]])
        p2 = np.array([xsFitLine[1], ysFitLine[1]])

        dists = []
        for point in zip(xsStars[fitPoints], ysStars[fitPoints]):
            point = np.array(point)
            distToLine = np.cross(p1 - point, p2 - point)/np.linalg.norm(p2 - point)
            dists.append(distToLine)

        # Add a histogram
        axHist.set_ylabel("Number")
        axHist.set_xlabel("Distance to Line Fit")
        axHist.hist(dists, bins=100)
        medDists = np.median(dists)
        madDists = sigmaMad(dists)
        axHist.set_xlim(medDists - 4.0*madDists, medDists + 4.0*madDists)
        axHist.axvline(medDists, color="k", label="Median: {:0.3f}".format(medDists))
        axHist.axvline(medDists + madDists, color="k", ls="--", label="Sigma Mad: {:0.3f}".format(madDists))
        axHist.axvline(medDists - madDists, color="k", ls="--")
        axHist.legend(fontsize=8, loc="lower left")

        fig = addPlotInfo(plt.gcf(), plotInfo)

        return fig
