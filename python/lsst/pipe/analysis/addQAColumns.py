from __future__ import annotations

import numpy as np

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig

from .qaUtils import (addMagnitudes, addColors, stellarLocusFits, addUseForQAFlag, addUseForStatsColumn)

class AddQAColumnsTaskConnections(pipeBase.PipelineTaskConnections, dimensions=("tract", "skymap")):

    cat = pipeBase.connectionTypes.Input(doc="The tract wide catalog to make plots from.",
                                         storageClass="DataFrame",
                                         name="objectTable_tract",
                                         dimensions=("tract", "skymap"))

    qaCat = pipeBase.connectionTypes.Output(doc="The input catalog with additional columns for QA added",
                                            storageClass="DataFrame",
                                            name="qaTable_tract",
                                            dimensions=("tract", "skymap"))


class AddQAColumnsTaskConfig(pipeBase.PipelineTaskConfig, pipelineConnections=AddQAColumnsTaskConnections):

    pass


class AddQAColumnsTask(pipeBase.PipelineTask):

    ConfigClass = AddQAColumnsTaskConfig
    _DefaultName = "addQAColumnsTask"

    def run(self, cat):
        # TODO: Get rid of warnings
        # TODO: raise exceptions if needed e.g. all nan column

        self.log.info("Adding additional QA columns to the object table")
        cat = addMagnitudes(cat)
        cat = addMagnitudes(cat, fluxColName="CModelFlux")
        cat = addColors(cat)
        cat = addColors(cat, magColName="CModelMag")
        cat = stellarLocusFits(cat)
        cat = stellarLocusFits(cat, magColName="CModelMag")
        cat = addUseForQAFlag(cat)
        cat = addUseForStatsColumn(cat)

        return pipeBase.Struct(qaCat=cat)
