import numpy as np
import matplotlib.pyplot as plt
from lsst.geom import SpherePoint, radians, degrees, Box2D

def parsePlotInfo(dataId, runName):

    plotInfo = {"run": runName}

    for dataInfo in dataId:
        plotInfo[dataInfo.name] = dataId[dataInfo.name]

    if "filter" not in plotInfo.keys():
        plotInfo["filter"] = "N/A"

    return plotInfo

def generateSummaryStats(cat, colName, skymap, plotInfo):

    tractInfo = skymap.generateTract(plotInfo["tract"])
    tractWcs = tractInfo.getWcs()

    # For now also convert the gen 2 patchIds to gen 3

    patchInfoList = {}
    for patch in cat.patchId.unique():
        if patch is None:
            continue
        # Once the objectTable_tract catalogues are using gen 3 patches this will go away
        patchTuple = (int(patch.split(",")[0]), int(patch.split(",")[-1]))
        onPatch = (cat["patchId"] == patch)
        stat = np.nanmedian(cat[colName].values[onPatch])
        patchInfo = tractInfo.getPatchInfo(patchTuple)
        corners = Box2D(patchInfo.getInnerBBox()).getCorners()
        skyCoords = tractWcs.pixelToSky(corners)

        gen3PatchId = tractInfo.getSequentialPatchIndex(patchInfo)

        patchInfoList[gen3PatchId] = (skyCoords, stat)

    tractCorners = Box2D(tractInfo.getBBox()).getCorners()
    skyCoords = tractWcs.pixelToSky(tractCorners)
    patchInfoList["tract"] = (skyCoords, np.nan)

    """
    coords = []
    for (ra, dec) in zip(cat["coord_ra"].values, cat["coord_dec"].values):
        coords.append(SpherePoint(ra, dec, degrees))

    def checkOnPatch(row):
        coord = SpherePoint(row["coord_ra"], row["coord_dec"], degrees)
        return patchPolygon.contains(coord.getVector())

    patchList = tractInfo.findPatchList(coords)
    patchInfoList = []
    for patchInfo in patchList:
        corners = Box2D(patchInfo.getInnerBBox()).getCorners()
        skyCoords = tractWcs.pixelToSky(corners)
        #patchPolygon = patchInfo.getInnerSkyPolygon(tractWcs)
        #onPatch = cat.apply(checkOnPatch, axis=1)
        #sumStat = np.nanmedian(cat[colName].values[onPatch])
        sumStat = 3
        patchInfoList.append((patchInfo.getIndex(), skyCoords, sumStat))
    """

    return patchInfoList

def addPlotInfo(fig, plotInfo):

    # TO DO: figure out how to get this information
    photocalibDataset = "None"
    astroDataset = "None"

    plt.text(0.02, 0.98, "Run:" + plotInfo["run"], fontsize=8, alpha=0.8, transform=fig.transFigure)
    datasetType = "Datasets used, photocalib: " + photocalibDataset + ", astrometry: " + astroDataset
    plt.text(0.02, 0.95, datasetType, fontsize=8, alpha=0.8, transform=fig.transFigure)

    plt.text(0.02, 0.92, "Tract: " + str(plotInfo["tract"]), fontsize=8, alpha=0.8, transform=fig.transFigure)
    plt.text(0.02, 0.89, "Filter: " + plotInfo["filter"], fontsize=8, alpha=0.8, transform=fig.transFigure)
    return fig

