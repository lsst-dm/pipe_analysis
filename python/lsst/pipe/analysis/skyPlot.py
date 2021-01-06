import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import median_absolute_deviation as sigmaMad
from matplotlib import patheffects
from matplotlib.patches import Rectangle

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig

from .plotUtilsGen3 import generateSummaryStats, parsePlotInfo, addPlotInfo


class SkyPlotTaskConnections(pipeBase.PipelineTaskConnections, dimensions=("tract", "skymap"),
                             defaultTemplates={"inputCoaddName": "deep", "plotName": "deltaCoords"}):

    catPlot = pipeBase.connectionTypes.Input(doc="The tract wide catalog to make plots from.",
                                             storageClass="DataFrame",
                                             name="qaTable_tract",
                                             dimensions=("tract", "skymap"))

    skyPlot = pipeBase.connectionTypes.Output(doc="A plot showing the on sky distribution of a value.",
                                              storageClass="Plot",
                                              name="skyPlot_{plotName}",
                                              dimensions=("tract", "skymap"))

    skymap = pipeBase.connectionTypes.Input(doc="The skymap for the tract",
                                            storageClass="SkyMap",
                                            name="{inputCoaddName}Coadd_skyMap",
                                            dimensions=("skymap",))


class SkyPlotTaskConfig(pipeBase.PipelineTaskConfig, pipelineConnections=SkyPlotTaskConnections):

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

    colorCodeValueColName = pexConfig.Field(
        doc="Which column to use to color code the points.",
        dtype=str,
        default="iPsMag",
    )


class SkyPlotTask(pipeBase.PipelineTask):

    ConfigClass = SkyPlotTaskConfig
    _DefaultName = "skyPlotTask"

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

        plotInfo = parsePlotInfo(dataId, runName)
        sumStats = generateSummaryStats(catPlot, self.config.colorCodeValueColName, skymap, plotInfo)
        fig = self.skyPlot(catPlot, xLabel, yLabel, plotInfo, sumStats)

        return pipeBase.Struct(skyPlot=fig)

    def skyPlot(self, catPlot, xLabel, yLabel, plotInfo, sumStats, yLims=False, xLims=False):

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
                    from a coadd dataset (`str`).
                ``"tract"``
                    The tract that the data comes from (`str`).
                ``"photoCalibDataset"``
                    (`str`).
                ``"skyWcsDataset"``
                    The sky Wcs dataset used (`str`).
                ``"rerun"``
                    The rerun the data is stored in (`str`).
        statsUnit : `str`
            The text used to describe the units of the statistics calculated.
        yLims : `Bool` or `tuple`, optional
            The y axis limits to use for the plot. If `False`,
            they are calculated from the data.
            If being given a tuple of (yMin, yMax).
        xLims : `Bool` or `tuple`, optional
            The x axis limits to use for the plot. If `False`, they are
            calculated from the data.
            If being given a tuple of (xMin, xMax).

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
        ``plotInfoDict`` needs to be a `dict` containing keys "camera",
        "filter", "visit", "tract" and "photoCalibDataset"
        (e.g. "jointcal" or "fgcm"), and "skyWcsDataset"
        ("jointcal" or not), it is used to add information to the
        plot.  Returns the median and sigma MAD for the x and y values.
        """

        self.log.info("Plotting {}: the values of {} for {} on a sky plot.".format(
                      self.config.connections.plotName, self.config.colorCodeValueColName,
                      self.config.objectsToPlot))

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Cut the catalogue down to only valid sources
        sourcesToUse = ((catPlot["useForQAFlag"].values) & (catPlot["useForStats"].values != 0))
        catPlot = catPlot[sourcesToUse]

        # Need to separate stars and galaxies
        stars = (catPlot[self.config.sourceTypeColName] == 0.0)
        galaxies = (catPlot[self.config.sourceTypeColName] == 1.0)

        # For galaxies
        xsGalaxies = catPlot[self.config.xColName].values[galaxies]
        ysGalaxies = catPlot[self.config.yColName].values[galaxies]
        colorValsGalaxies = catPlot[self.config.colorCodeValueColName].values[galaxies]

        # For stars
        xsStars = catPlot[self.config.xColName].values[stars]
        ysStars = catPlot[self.config.yColName].values[stars]
        colorValsStars = catPlot[self.config.colorCodeValueColName].values[stars]

        # Calculate some statistics
        if self.config.objectsToPlot == "galaxies" or self.config.objectsToPlot == "all":
            lowSnGals = (((catPlot["useForStats"] == 1) | (catPlot["useForStats"] == 2)) & galaxies)
            lowSnGalMed = np.nanmedian(catPlot[self.config.yColName].values[lowSnGals])
            lowSnGalMad = sigmaMad(catPlot[self.config.yColName].values[lowSnGals], nan_policy="omit")

            galStatsText = ("Median: {:0.2f}\n".format(lowSnGalMed) + r"$\sigma_{MAD}$: "
                            + "{:0.2f}".format(lowSnGalMad))
            # Add statistics
            bbox = dict(facecolor="C1", alpha=0.3, edgecolor="none")
            ax.text(0.7, 0.92, galStatsText, transform=fig.transFigure, fontsize=8, bbox=bbox)

        if self.config.objectsToPlot == "stars" or self.config.objectsToPlot == "all":

            lowSnStars = (((catPlot["useForStats"] == 1) | (catPlot["useForStats"] == 2)) & stars)
            lowSnStarMed = np.nanmedian(catPlot[self.config.yColName].values[lowSnStars])
            lowSnStarMad = sigmaMad(catPlot[self.config.yColName].values[lowSnStars], nan_policy="omit")
            approxLowMag = np.max(catPlot["iCModelMag"].values[lowSnStars])

            starStatsText = ("Median: {:0.2f}\n".format(lowSnStarMed) + r"$\sigma_{MAD}$: "
                             + "{:0.2f}".format(lowSnStarMad))
            # Add statistics
            bbox = dict(facecolor="C0", alpha=0.3, edgecolor="none")
            ax.text(0.8, 0.92, starStatsText, transform=fig.transFigure, fontsize=8, bbox=bbox)

        if self.config.objectsToPlot == "stars":
            toPlotList = [(xsStars, ysStars, colorValsStars, "winter_r", "Stars")]
        elif self.config.objectsToPlot == "galaxies":
            toPlotList = [(xsGalaxies, ysGalaxies, colorValsGalaxies, "autumn_r", "Galaxies")]
        elif self.config.objectsToPlot == "all":
            toPlotList = [(xsGalaxies, ysGalaxies, colorValsGalaxies, "autumn_r", "Galaxies"),
                          (xsStars, ysStars, colorValsStars, "winter_r", "Stars")]

        # Corner plot of patches showing summary stat in each
        patches = []
        for dataId in sumStats.keys():
            (corners, stat) = sumStats[dataId]
            ra = corners[0][0].asDegrees()
            dec = corners[0][1].asDegrees()
            xy = (ra, dec)
            width = corners[2][0].asDegrees() - ra
            height = corners[2][1].asDegrees() - dec
            patches.append(Rectangle(xy, width, height, alpha=0.3))
            ras = [ra.asDegrees() for (ra, dec) in corners]
            decs = [dec.asDegrees() for (ra, dec) in corners]
            ax.plot(ras + [ras[0]], decs + [decs[0]], "k", lw=0.5)
            cenX = ra + width / 2
            cenY = dec + height / 2
            if dataId != "tract":
                ax.annotate(dataId, (cenX, cenY), color="k", fontsize=7, ha="center", va="center",
                            path_effects=[patheffects.withStroke(linewidth=2, foreground="w")])

        for (i, (xs, ys, colorVals, cmap, label)) in enumerate(toPlotList):
            med = np.median(colorVals)
            mad = sigmaMad(colorVals)
            vmin = med - 3*mad
            vmax = med + 3*mad
            scatterOut = ax.scatter(xs, ys, c=colorVals, cmap=cmap, s=10.0, vmin=vmin, vmax=vmax)
            cax = fig.add_axes([0.87 + i*0.04, 0.11, 0.04, 0.77])
            plt.colorbar(scatterOut, cax=cax, extend="both")
            colorBarLabel = "{}: {}".format(self.config.colorCodeValueColName, label)
            cax.text(0.6, 0.5, colorBarLabel, color="k", rotation="vertical", transform=cax.transAxes,
                     ha="center", va="center", fontsize=10)

            if i == 0 and len(toPlotList) > 1:
                cax.yaxis.set_ticks_position("left")

        ax.set_xlabel("R.A. (deg)")
        ax.set_ylabel("Dec. (deg)")

        plt.draw()

        # Add useful information to the plot
        plt.subplots_adjust(wspace=0.0, hspace=0.0, right=0.85)
        fig = plt.gcf()
        fig = addPlotInfo(fig, plotInfo)

        return fig
