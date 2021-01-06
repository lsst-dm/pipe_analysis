import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import median_absolute_deviation as sigmaMad
from scipy import stats as scipyStats
import scipy.odr as scipyODR

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig

from .plotUtilsGen3 import generateSummaryStats, parsePlotInfo, addPlotInfo


class CalcStellarLocusParamsTaskConnections(pipeBase.PipelineTaskConnections,
                                            dimensions=("tract", "skymap"),
                                            defaultTemplates={"inputCoaddName": "deep", "fitType": "wFit"}):

    catPlot = pipeBase.connectionTypes.Input(doc="The tract wide catalog to make plots from.",
                                             storageClass="DataFrame",
                                             name="qaTable_tract",
                                             dimensions=("tract", "skymap"))

    fitParams = pipeBase.connectionTypes.Output(doc="The parameters from the fit to the stellar locus.",
                                                        storageClass="StructuredDataDict",
                                                        name="stellarLocusParams_{fitType}",
                                                        dimensions=("tract", "skymap"))

    skymap = pipeBase.connectionTypes.Input(doc="The skymap for the tract",
                                            storageClass="SkyMap",
                                            name="{inputCoaddName}Coadd_skyMap",
                                            dimensions=("skymap",))


class CalcStellarLocusParamsTaskConfig(pipeBase.PipelineTaskConfig,
                                  pipelineConnections=CalcStellarLocusParamsTaskConnections):

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

    sourceTypeColName = pexConfig.Field(
        doc="The column to use for star - galaxy separation.",
        dtype=str,
        default="iExtendedness",
    )

    name = pexConfig.Field(
        doc="The name of the subject of the plot.",
        dtype=str,
        default="wFit",
    )


class CalcStellarLocusParamsTask(pipeBase.PipelineTask):

    ConfigClass = CalcStellarLocusParamsTaskConfig
    _DefaultName = "CalcStellarLocusParamsTask"

    def run(self, catPlot, skymap):

        fitParams = self.calcStellarLocusParams(catPlot)

        return pipeBase.Struct(fitParams=fitParams)

    def calcStellarLocusParams(self, catPlot):

        # Get the limits of the box used for the statistics
        # Putting random values in here for now
        self.log.info(("Fitting: the values of {} against {}".format(
                       self.config.xColName, self.config.yColName)))

        if self.config.name == "yFit":
            xBoxLims = (0.82, 2.01)
            yBoxLims = (0.37, 0.9)
        elif self.config.name == "wFit":
            xBoxLims = (0.28, 1.0)
            yBoxLims = (0.02, 0.48)
        elif self.config.name == "xFit":
            xBoxLims = (1.05, 1.45)
            yBoxLims = (0.78, 1.62)

        # Cut the catalogue down to only valid sources
        catPlot = catPlot[catPlot["useForQAFlag"].values]

        # Only use sources brighter than something
        # TODO: what is that something
        catPlot = catPlot[(catPlot["iCModelMag"] < 22.0)]

        # Need to separate stars and galaxies
        stars = (catPlot[self.config.sourceTypeColName] == 0.0)

        # For stars
        xsStars = catPlot[self.config.xColName].values[stars]
        ysStars = catPlot[self.config.yColName].values[stars]

        # Points to use for the fit
        fitPoints = np.where((xsStars > xBoxLims[0]) & (xsStars < xBoxLims[1])
                             & (ysStars > yBoxLims[0]) & (ysStars < yBoxLims[1]))[0]

        # Fit a line to the points in the box
        (m, b) = np.polyfit(xsStars, ysStars, 1)
        print(m, b, type(m), type(b))
        xsFitLine = [xBoxLims[0], xBoxLims[1]]
        ysFitLine = [m*xsFitLine[0] + b, m*xsFitLine[1] + b]

        def linearFit(B, x):
            return B[0]*x + B[1]
        linear = scipyODR.Model(linearFit)
        linear = scipyODR.polynomial(1)

        data = scipyODR.Data(xsStars, ysStars)
        odr = scipyODR.ODR(data, linear)
        params = odr.run()
        print(params.pprint())

        odr2 = scipyODR.ODR(data, linear, beta0=[params.beta[1], params.beta[0]])
        params2 = odr.run()
        print(params2.pprint())

        linParams = scipyStats.linregress(xsStars, ysStars)

        p1 = np.array([xsFitLine[0], ysFitLine[0]])
        p2 = np.array([xsFitLine[1], ysFitLine[1]])

        dists = []
        for point in zip(xsStars[fitPoints], ysStars[fitPoints]):
            point = np.array(point)
            distToLine = np.cross(p1 - point, p2 - point)/np.linalg.norm(p2 - point)
            dists.append(distToLine)

        print(params.beta)
        paramDict = {"m": float(m), "b": float(b), "m_odr": float(params.beta[0]), "b_odr": float(params.beta[1]),
                "m_odr2": float(params2.beta[0]), "b_odr2": float(params2.beta[1]), "x1": xBoxLims[0], "x2": xBoxLims[1],
                "y1": yBoxLims[0], "y2": yBoxLims[1], "magLim": 22.0}

        return paramDict


        def hardwiredFits(name):
            """Hardwired fits taken from pipe_analysis"""

            # Straight line fits for the perpendicular ranges
            # The following fits were derived in the process of calibrating
            # the above coeffs (all three RC2 tracts gave ~ the same fits).
            # May remove later if deemed no longer useful.

            wFit = {"name": "wFit", "range": "griBlue", "HSC-G": 0.52, "HSC-R": -0.52,
                    "HSC-I": 0.0, "HSC-Z": 0.0, "const": -0.08}
            xFit = {"name": "xFit", "range": "griRed", "HSC-G": 13.35, "HSC-R": -13.35,
                    "HSC-I": 0.0, "HSC-Z": 0.0, "const": -15.54}
            yFit = {"name": "yFit", "range": "rizRed", "HSC-G": 0.0, "HSC-R": 0.40,
                    "HSC-I": -0.40, "HSC-Z": 0.0, "const": 0.03}

            if name == "wFit":
                return wFit
            elif name == "xFit":
                return xFit
            elif name == "yFit":
                return yFit
