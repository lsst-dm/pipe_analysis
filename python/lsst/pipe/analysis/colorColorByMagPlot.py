import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig

from .plotUtilsGen3 import generateSummaryStats, parsePlotInfo, addPlotInfo


class ColorColorByMagPlotTaskConnections(pipeBase.PipelineTaskConnections,
                                         dimensions=("tract", "skymap"),
                                         defaultTemplates={"inputCoaddName": "deep",
                                                           "plotName": "deltaCoords"}):

    catPlot = pipeBase.connectionTypes.Input(doc="The tract wide catalog to make plots from.",
                                             storageClass="DataFrame",
                                             name="qaTable_tract",
                                             dimensions=("tract", "skymap"))

    colorColorByMagPlot = pipeBase.connectionTypes.Output(doc="A scatter plot with histograms for both axes.",
                                                          storageClass="Plot",
                                                          name="colorColorByMagPlot_{plotName}",
                                                          dimensions=("tract", "skymap"))

    skymap = pipeBase.connectionTypes.Input(doc="The skymap for the tract",
                                            storageClass="SkyMap",
                                            name="{inputCoaddName}Coadd_skyMap",
                                            dimensions=("skymap",))


class ColorColorByMagPlotTaskConfig(pipeBase.PipelineTaskConfig,
                                    pipelineConnections=ColorColorByMagPlotTaskConnections):

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

    magCol = pexConfig.Field(
        doc="The column to use for magnitude cuts.",
        dtype=str,
        default="iCModelMag",
    )


class ColorColorByMagPlotTask(pipeBase.PipelineTask):

    ConfigClass = ColorColorByMagPlotTaskConfig
    _DefaultName = "ColorColorByMagPlotTask"

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
        sumStats = generateSummaryStats(catPlot, self.config.yColName, skymap, plotInfo)
        fig = self.colorColorByMagPlot(catPlot, xLabel, yLabel, plotInfo, sumStats)

        return pipeBase.Struct(colorColorByMagPlot=fig)

    def colorColorByMagPlot(self, catPlot, xLabel, yLabel, plotInfo, sumStats, yLims=False, xLims=False):

        self.log.info(("Plotting {}: the values of {} against {} on color-color plots sub divided by "
                       "magnitude cut.".format(self.config.connections.plotName, self.config.xColName,
                                               self.config.yColName)))

        fig = plt.figure()
        gs = gridspec.GridSpec(2, 2)

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

        # calculate useful axis limits
        starPercsX = np.nanpercentile(xsStars, [1, 99.5])
        starPercsY = np.nanpercentile(ysStars, [1, 99.5])

        magLims = [26.0, 25.0, 24.0, 23.0, 15.5]
        for (i, axLoc) in enumerate(gs):
            ax = fig.add_subplot(axLoc)
            galsToPlot = np.where((catPlot[self.config.magCol][galaxies] <= magLims[i])
                                  & (catPlot[self.config.magCol][galaxies] > magLims[i + 1]))[0]
            galMags = catPlot[self.config.magCol].values[galaxies]
            galPoints = ax.scatter(xsGalaxies[galsToPlot], ysGalaxies[galsToPlot],
                                   c=galMags[galsToPlot], cmap="autumn_r", label="Galaxies", s=0.5,
                                   vmin=15.5, vmax=26.0)

            starsToPlot = np.where((catPlot[self.config.magCol][stars] <= magLims[i])
                                   & (catPlot[self.config.magCol][stars] > magLims[i + 1]))[0]
            starMags = catPlot[self.config.magCol].values[stars]
            starPoints = ax.scatter(xsStars[starsToPlot], ysStars[starsToPlot], c=starMags[starsToPlot],
                                    cmap="winter_r", label="Stars", s=0.5, vmin=15.5, vmax=26.0)

            # Set useful axis limits
            ax.set_xlim(starPercsX[0], starPercsX[1])
            ax.set_ylim(starPercsY[0], starPercsY[1])

            if i == 0 or i == 1:
                ax.xaxis.set_visible(False)
            if i == 1 or i == 3:
                ax.yaxis.set_visible(False)

            # Add text details
            bbox = dict(alpha=0.9, facecolor="white", edgecolor="none")
            ax.text(0.05, 0.9, "NGals: {}".format(len(galsToPlot)), color="C1", bbox=bbox,
                    transform=ax.transAxes, fontsize=8)
            ax.text(0.6, 0.9, "NStars: {}".format(len(starsToPlot)), color="C0", bbox=bbox,
                    transform=ax.transAxes, fontsize=8)
            ax.text(0.05, 0.05, "{} <= {} < {}".format(magLims[i], self.config.magCol, magLims[i + 1]),
                    bbox=bbox, transform=ax.transAxes, fontsize=8)

        # Add colorbars
        galCbAx = fig.add_axes([0.85, 0.11, 0.04, 0.77])
        plt.colorbar(galPoints, cax=galCbAx)
        galCbAx.yaxis.set_visible(False)
        starCbAx = fig.add_axes([0.89, 0.11, 0.04, 0.77])
        plt.colorbar(starPoints, cax=starCbAx)
        galCbAx.text(0.6, 0.5, "HSC-I [CModel]: Galaxies", color="k", rotation="vertical",
                     transform=galCbAx.transAxes, ha="center", va="center", fontsize=10)
        starCbAx.text(0.6, 0.5, "HSC-I [CModel]: Stars", color="k", rotation="vertical",
                      transform=starCbAx.transAxes, ha="center", va="center", fontsize=10)

        fig.text(0.5, 0.04, self.config.xLabel, va="center", ha="center", fontsize=12)
        fig.text(0.04, 0.5, self.config.yLabel, va="center", ha="center", rotation="vertical",
                 fontsize=12)

        plt.subplots_adjust(hspace=0, wspace=0, right=0.83)

        fig = addPlotInfo(plt.gcf(), plotInfo)

        return fig
