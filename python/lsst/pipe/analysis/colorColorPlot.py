import matplotlib.pyplot as plt
import numpy as np

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig

from .plotUtilsGen3 import generateSummaryStats, parsePlotInfo, addPlotInfo


class ColorColorPlotTaskConnections(pipeBase.PipelineTaskConnections,
                                    dimensions=("tract", "skymap"),
                                    defaultTemplates={"inputCoaddName": "deep", "plotName": "deltaCoords"}):

    catPlot = pipeBase.connectionTypes.Input(doc="The tract wide catalog to make plots from.",
                                             storageClass="DataFrame",
                                             name="qaTable_tract",
                                             dimensions=("tract", "skymap"))

    colorColorPlot = pipeBase.connectionTypes.Output(doc="A scatter plot with histograms for both axes.",
                                                     storageClass="Plot",
                                                     name="colorColor_{plotName}",
                                                     dimensions=("tract", "skymap"))

    skymap = pipeBase.connectionTypes.Input(doc="The skymap for the tract",
                                            storageClass="SkyMap",
                                            name="{inputCoaddName}Coadd_skyMap",
                                            dimensions=("skymap",))


class ColorColorPlotTaskConfig(pipeBase.PipelineTaskConfig,
                               pipelineConnections=ColorColorPlotTaskConnections):

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


class ColorColorPlotTask(pipeBase.PipelineTask):

    ConfigClass = ColorColorPlotTaskConfig
    _DefaultName = "ColorColorPlotTask"

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
        fig = self.colorColorPlot(catPlot, xLabel, yLabel, plotInfo, sumStats)

        return pipeBase.Struct(colorColorPlot=fig)

    def colorColorPlot(self, catPlot, xLabel, yLabel, plotInfo, sumStats, yLims=False, xLims=False):

        self.log.info(("Plotting {}: {} against {} on a color-color plot.".format(
                       self.config.connections.plotName, self.config.xColName, self.config.yColName)))

        fig = plt.figure()
        ax = fig.add_axes([0.12, 0.11, 0.65, 0.75])

        # Cut the catalogue down to only valid sources
        catPlot = catPlot[catPlot["useForQAFlag"].values]

        # Only use sources brighter than 26th
        catPlot = catPlot[(catPlot["iCModelMag"] < 26.0)]

        # Need to separate stars and galaxies
        stars = (catPlot[self.config.sourceTypeColName] == 0.0)
        galaxies = (catPlot[self.config.sourceTypeColName] == 1.0)

        # For galaxies
        xsGalaxies = catPlot[self.config.xColName].values[galaxies]
        ysGalaxies = catPlot[self.config.yColName].values[galaxies]

        # For stars
        xsStars = catPlot[self.config.xColName].values[stars]
        ysStars = catPlot[self.config.yColName].values[stars]

        galPoints = ax.scatter(xsGalaxies, ysGalaxies, c=catPlot["iCModelMag"].values[galaxies],
                               cmap="autumn_r", label="Galaxies", s=0.5)
        starPoints = ax.scatter(xsStars, ysStars, c=catPlot["iCModelMag"].values[stars], cmap="winter_r",
                                label="Stars", s=0.5)

        # Add text details
        fig.text(0.70, 0.9, "Num. Galaxies: {}".format(len(np.where(galaxies)[0])), color="C1")
        fig.text(0.70, 0.93, "Num. Stars: {}".format(len(np.where(stars)[0])), color="C0")

        # Add colorbars
        galCbAx = fig.add_axes([0.85, 0.11, 0.04, 0.75])
        plt.colorbar(galPoints, cax=galCbAx)
        galCbAx.yaxis.set_ticks_position("left")
        starCbAx = fig.add_axes([0.89, 0.11, 0.04, 0.75])
        plt.colorbar(starPoints, cax=starCbAx)
        galCbAx.text(0.6, 0.5, "HSC-I [CModel]: Galaxies", color="k", rotation="vertical",
                     transform=galCbAx.transAxes, ha="center", va="center", fontsize=10)
        starCbAx.text(0.6, 0.5, "HSC-I [CModel]: Stars", color="k", rotation="vertical",
                      transform=starCbAx.transAxes, ha="center", va="center", fontsize=10)

        ax.set_xlabel(self.config.xLabel)
        ax.set_ylabel(self.config.yLabel)

        # Set useful axis limits
        starPercsX = np.nanpercentile(xsStars, [1, 99.5])
        starPercsY = np.nanpercentile(ysStars, [1, 99.5])
        ax.set_xlim(starPercsX[0], starPercsX[1])
        ax.set_ylim(starPercsY[0], starPercsY[1])

        fig = addPlotInfo(plt.gcf(), plotInfo)

        return fig
