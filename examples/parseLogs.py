#!/usr/bin/env python

# This file is part of pipe_analysis.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import ast
import copy


class VisitEntry(object):
    def __init__(self, field, tract, patch, visit, band, quantityName, starStatsDict, galStatsDict):
        self.field = field
        self.tract = tract
        self.patch = patch
        self.visit = visit
        self.band = band
        self.quantityName = quantityName
        self.starStatsDict = starStatsDict
        self.galStatsDict = galStatsDict


def insertNullEntry(insertFromEntry, insertToList, insertAtIndex):
    """Insert a null entry for any missing entries in a given entry list.
    """
    insertToEntry = copy.deepcopy(insertFromEntry)
    for k in insertToEntry.starStatsDict.iterkeys():
        if k == "num" or k == "total":
            insertToEntry.starStatsDict[k] = 0
        else:
            insertToEntry.starStatsDict[k] = float("nan")
    for k in insertToEntry.galStatsDict.iterkeys():
        if k == "num" or k == "total":
            insertToEntry.galStatsDict[k] = 0
        else:
            insertToEntry.galStatsDict[k] = float("nan")
    insertToList.insert(insertAtIndex, insertToEntry)

    return insertToList


def createDataList(analysisStr, logFile, tracts, band):
    field = None
    tract = None
    patch = None
    visit = 0
    nameStartStr = analysisStr + " INFO: shortName = "
    statsStartStr = analysisStr + " INFO: Statistics from DataId(initialdata="
    visitEntryList = []
    quantityList = []
    shortNameColorsList = []
    # This is for the Distance to polynomial entry that comes with colorPerp
    # but is associated with the same shortName (which is just the color,
    # e.g. gri).
    with open(logFile, "r") as inFile:
        for line in inFile:
            if line.find(nameStartStr) >= 0:
                tempName = line[line.index(nameStartStr) + len(nameStartStr):]
                if tempName.strip() in colorsList:
                    shortNameColorsList.append(line[line.index(nameStartStr) + len(nameStartStr):])
    with open(logFile, "r") as inFile:
        i = 0
        for line in inFile:
            if line.find(nameStartStr) >= 0:
                shortName = line[line.index(nameStartStr) + len(nameStartStr):]
            if line.startswith(statsStartStr):
                id1 = len(statsStartStr)
                id2 = line.index("}", len(statsStartStr)) + 1
                dataIdDictStr = line[id1: id2]
                dataIdDict = dict(ast.literal_eval(dataIdDictStr))
                tract = dataIdDict["tract"]
                if "patch" in dataIdDict:
                    patch = dataIdDict["patch"]
                if "field" in dataIdDict:
                    field = dataIdDict["field"]
                if "visit" in dataIdDict:
                    visit = dataIdDict["visit"]
                qn1 = line.index(" of ", id2) + 4
                qn2 = line.index(": {", qn1)
                quantityName = line[qn1: qn2]
                if "polynomial" in quantityName:
                    quantityName += " " + shortNameColorsList[i].strip()
                    i += 1
                if i == len(shortNameColorsList):
                    i = 0
                if "commonZp" in shortName:
                    quantityName += " (commonZP)"
                if "_distance" in shortName:
                    quantityName = shortName.strip() + " (arcsec)"
                if quantityName not in quantityList:
                    quantityList.append(quantityName)
                statsDictStr = line[qn2 + 2:]
                statsDictStr = statsDictStr.replace("Stats(", "{\'")
                statsDictStr = statsDictStr.replace("=", "\': ")
                statsDictStr = statsDictStr.replace("; ", ", '")
                statsDictStr = statsDictStr.replace(")", "}")
                starStatsDict = None
                if statsDictStr.find("star"):
                    ss1 = statsDictStr.index("{", statsDictStr.index("star") + len("star"))
                    ss2 = statsDictStr.index("}", ss1) + 1
                    starStatsStr = statsDictStr[ss1: ss2]
                    starStatsDict = dict(ast.literal_eval(starStatsStr))
                if statsDictStr.find("galaxy") > 0 and statsDictStr.find("\'num\': 0") < 0:
                    gs1 = statsDictStr.index("{", statsDictStr.index("galaxy") + len("galaxy"))
                    gs2 = statsDictStr.index("}", gs1) + 1
                    galStatsStr = statsDictStr[gs1: gs2]
                    galStatsDict = dict(ast.literal_eval(galStatsStr))
                else:
                    galStatsDict = {'mean': 0, 'stdev': 0, 'num': 0, 'total': 0,
                                    'median': 0, 'clip': 0, 'forcedMean': 0}

                entry = VisitEntry(field, tract, patch, visit, band, quantityName, starStatsDict,
                                   galStatsDict)
                visitEntryList.append(entry)

    return quantityList, visitEntryList


def printStatsSum(analysisStr, lsstLogFile, hscLogFile, tracts, band, outLogFile, outShortLogFile):
    lsstQuantityList, lsstVisitEntryList = createDataList(analysisStr, lsstLogFile, tracts, band)
    hscQuantityList, hscVisitEntryList = createDataList(analysisStr, hscLogFile, tracts, band)

    if lsstQuantityList != hscQuantityList:
        print("WARNING: quantityName lists differ between LSST and HSC.  Will try to accommodate...")
        # print("lsstQuantityEntryList = ", lsstQuantityList)
        # print(" hscQuantityEntryList = ", hscQuantityList)

    if lsstVisitEntryList[0].patch is not None:
        for index, (lsstVisitEntry, hscVisitEntry) in enumerate(zip(lsstVisitEntryList, hscVisitEntryList)):
            # HSC forced catalogs do not include Gaussian fluxes, so need to
            # insert Null entry.
            if "Gaussian" in lsstVisitEntry.quantityName:
                insertNullEntry(lsstVisitEntry, hscVisitEntryList, index)

    if len(lsstVisitEntryList) != len(hscVisitEntryList):
        print("WARNING: visit entry lists differ between LSST and HSC")
        print("len(lsstVisitEntryList) = ", len(lsstVisitEntryList))
        print("len(hscVisitEntryList)  = ", len(hscVisitEntryList))
        print("len(lsstQuantityEntryList) = ", len(lsstQuantityList))
        print("len(hscQuantityEntryList) = ", len(hscQuantityList))

        # Insert Nan entries where HSC has missing ones (something is going
        # wrong with reading in some of the match files...or perhaps they just
        # do not exist for certain visit/ccd.
        index = 0
        looping = True
        while looping:
            for index, (lsstVisitEntry, hscVisitEntry) in enumerate(zip(lsstVisitEntryList,
                                                                        hscVisitEntryList)):
                if lsstVisitEntry.quantityName != hscVisitEntry.quantityName:
                    if hscVisitEntry.quantityName == "Mag(Gaussian) - PSFMag (commonZP)":
                        insertNullEntry(lsstVisitEntry, hscVisitEntryList, index)
                    else:
                        raise RuntimeError("Case not accounted for...")
                    break
            if len(hscVisitEntryList) == len(lsstVisitEntryList) or index > len(lsstVisitEntryList) - 7:
                looping = False

        if len(hscVisitEntryList) < len(lsstVisitEntryList):
            for lsstVisitEntry in lsstVisitEntryList[len(hscVisitEntryList): len(lsstVisitEntryList)]:
                insertHscEntry = copy.deepcopy(lsstVisitEntry)
                for k in insertHscEntry.starStatsDict.iterkeys():
                    insertHscEntry.starStatsDict[k] = 0
                for k in insertHscEntry.galStatsDict.iterkeys():
                    insertHscEntry.galStatsDict[k] = 0
                hscVisitEntryList.append(insertHscEntry)

    if len(lsstVisitEntryList) != len(hscVisitEntryList):
        raise RuntimeError("Something went wrong...")

    with open(outLogFile, "w") as outFile, open(outShortLogFile, "w") as outShortFile:
        for lsstQuantityName in lsstQuantityList:
            outShortFile.write("# Stars: " + lsstQuantityName + "\n")
            outShortFile.write("#                filter   mean    [HSC]     stdev   [HSC]    "
                               "median   [HSC]      num     [HSC]     numTot    [HSC]  NumEntries" + "\n")
            outFile.write("# Stars: " + lsstQuantityName + "\n")
            outFile.write("# tract  visit   filter   mean    [HSC]     stdev   [HSC]    median   [HSC] "
                          "     num     [HSC]     numTot    [HSC]" + "\n")
            lsstMeanMean, hscMeanMean, lsstWgtMeanMean, hscWgtMeanMean = 0.0, 0.0, 0.0, 0.0
            lsstMeanStdev, hscMeanStdev, lsstWgtMeanStdev, hscWgtMeanStdev = 0.0, 0.0, 0.0, 0.0
            lsstMeanMedian, hscMeanMedian, lsstWgtMeanMedian, hscWgtMeanMedian = 0.0, 0.0, 0.0, 0.0
            lsstNumSum, hscNumSum = 0, 0
            lsstTotalSum, hscTotalSum = 0, 0
            lsstNumVisits, hscNumVisits = 0, 0

            for lsstVisitEntry, hscVisitEntry in zip(lsstVisitEntryList, hscVisitEntryList):
                if lsstVisitEntry.quantityName != hscVisitEntry.quantityName:
                    raise RuntimeError("Quantity names for visit {0:d} do not match! {1:s} vs {2:s}".
                                       format(lsstVisitEntry.visit, lsstVisitEntry.quantityName, band,
                                              hscVisitEntry.quantityName))
                # outFile.write("# {:s}".format(lsstVisitEntry.field)
                formatStr = ("{0:6d} {1:7d} {2:>7s} {3:>7.3F} [{4:7.3F}] {5:>7.3F} [{6:7.3F}] "
                             "{7:>7.3F} [{8:7.3F}] {9:>8d} [{10:8d}] {11:>8d} [{12:8d}]\n")
                if lsstVisitEntry.quantityName == lsstQuantityName:
                    outFile.write((formatStr.format(
                        lsstVisitEntry.tract, lsstVisitEntry.visit, bandStrDict[band],
                        lsstVisitEntry.starStatsDict["mean"], hscVisitEntry.starStatsDict["mean"],
                        lsstVisitEntry.starStatsDict["stdev"], hscVisitEntry.starStatsDict["stdev"],
                        lsstVisitEntry.starStatsDict["median"], hscVisitEntry.starStatsDict["median"],
                        lsstVisitEntry.starStatsDict["num"], hscVisitEntry.starStatsDict["num"],
                        lsstVisitEntry.starStatsDict["total"], hscVisitEntry.starStatsDict["total"])))

                    if lsstVisitEntry.starStatsDict["num"] > 0:
                        lsstMeanMean += lsstVisitEntry.starStatsDict["mean"]
                        lsstMeanStdev += lsstVisitEntry.starStatsDict["stdev"]
                        lsstMeanMedian += lsstVisitEntry.starStatsDict["median"]
                        lsstWgtMeanMean += (
                            lsstVisitEntry.starStatsDict["mean"]*lsstVisitEntry.starStatsDict["num"])
                        lsstWgtMeanStdev += (
                            lsstVisitEntry.starStatsDict["stdev"]*lsstVisitEntry.starStatsDict["num"])
                        lsstWgtMeanMedian += (
                            lsstVisitEntry.starStatsDict["median"]*lsstVisitEntry.starStatsDict["num"])
                        lsstNumSum += lsstVisitEntry.starStatsDict["num"]
                        lsstTotalSum += lsstVisitEntry.starStatsDict["total"]
                        lsstNumVisits += 1
                    if hscVisitEntry.starStatsDict["num"] > 0:
                        hscMeanMean += hscVisitEntry.starStatsDict["mean"]
                        hscMeanStdev += hscVisitEntry.starStatsDict["stdev"]
                        hscMeanMedian += hscVisitEntry.starStatsDict["median"]
                        hscWgtMeanMean += (
                            hscVisitEntry.starStatsDict["mean"]*hscVisitEntry.starStatsDict["num"])
                        hscWgtMeanStdev += (
                            hscVisitEntry.starStatsDict["stdev"]*hscVisitEntry.starStatsDict["num"])
                        hscWgtMeanMedian += (
                            hscVisitEntry.starStatsDict["median"]*hscVisitEntry.starStatsDict["num"])
                        hscNumSum += hscVisitEntry.starStatsDict["num"]
                        hscTotalSum += hscVisitEntry.starStatsDict["total"]
                        hscNumVisits += 1
            if lsstNumVisits > 0:
                lsstMeanMean /= lsstNumVisits
                lsstMeanStdev /= lsstNumVisits
                lsstMeanMedian /= lsstNumVisits
            else:
                lsstMeanMean, lsstMeanStdev, lsstMeanMedian = float("nan"), float("nan"), float("nan")
            if hscNumVisits > 0:
                hscMeanMean /= hscNumVisits
                hscMeanStdev /= hscNumVisits
                hscMeanMedian /= hscNumVisits
            else:
                hscMeanMean, hscMeanStdev, hscMeanMedian = float("nan"), float("nan"), float("nan")
            if lsstNumSum > 0:
                lsstWgtMeanMean /= lsstNumSum
                lsstWgtMeanStdev /= lsstNumSum
                lsstWgtMeanMedian /= lsstNumSum
            else:
                lsstWgtMeanMean, lsstWgtMeanStdev, lsstWgtMeanMedian = (float("nan"), float("nan"),
                                                                        float("nan"))
            if hscNumSum > 0:
                hscWgtMeanMean /= hscNumSum
                hscWgtMeanStdev /= hscNumSum
                hscWgtMeanMedian /= hscNumSum
            else:
                hscWgtMeanMean, hscWgtMeanStdev, hscWgtMeanMedian = float("nan"), float("nan"), float("nan")

            if lsstNumVisits > 1:
                outFile.write(lineSplitStr)
                outFile.write(("{0:14s} {1:>7s} {2:>7.3F} [{3:7.3F}] {4:>7.3F} [{5:7.3F}] "
                               "{6:>7.3F} [{7:7.3F}] {8:>8d} [{9:8d}] {10:>8d} [{11:8d}]\n".
                               format("# straight avg", bandStrDict[band], lsstMeanMean, hscMeanMean,
                                      lsstMeanStdev, hscMeanStdev, lsstMeanMedian, hscMeanMedian,
                                      lsstNumSum, hscNumSum, lsstTotalSum, hscTotalSum)))
                outFile.write(("{0:14s} {1:>7s} {2:>7.3F} [{3:7.3F}] {4:>7.3F} [{5:7.3F}] "
                               "{6:>7.3F} [{7:7.3F}] {8:>8d} [{9:8d}] {10:>8d} [{11:8d}]\n".
                               format("# weighted avg", bandStrDict[band], lsstWgtMeanMean, hscWgtMeanMean,
                                      lsstWgtMeanStdev, hscWgtMeanStdev, lsstWgtMeanMedian, hscWgtMeanMedian,
                                      lsstNumSum, hscNumSum, lsstTotalSum, hscTotalSum)))
            outFile.write(lineSplitStr)

            outShortFile.write(("{0:14s} {1:>7s} {2:>7.3F} [{3:7.3F}] {4:>7.3F} [{5:7.3F}] "
                                "{6:>7.3F} [{7:7.3F}] {8:>8d} [{9:8d}] {10:>8d} [{11:8d}] {12:>8d}\n".
                                format("# straight avg", bandStrDict[band], lsstMeanMean, hscMeanMean,
                                       lsstMeanStdev, hscMeanStdev, lsstMeanMedian, hscMeanMedian,
                                       lsstNumSum, hscNumSum, lsstTotalSum, hscTotalSum, lsstNumVisits)))
            if lsstNumVisits > 1:
                outShortFile.write(("{0:14s} {1:>7s} {2:>7.3F} [{3:7.3F}] {4:>7.3F} [{5:7.3F}] "
                                    "{6:>7.3F} [{7:7.3F}] {8:>8d} [{9:8d}] {10:>8d} [{11:8d}] {12:>8d}\n".
                                    format("# weighted avg", bandStrDict[band], lsstWgtMeanMean,
                                           hscWgtMeanMean, lsstWgtMeanStdev, hscWgtMeanStdev,
                                           lsstWgtMeanMedian, hscWgtMeanMedian, lsstNumSum, hscNumSum,
                                           lsstTotalSum, hscTotalSum, lsstNumVisits)))
            outShortFile.write(lineSplitStr)


def printCompareStatsSum(analysisStr, lsstLogFile, tracts, band, outLogFile, outShortLogFile):
    lsstQuantityList, lsstVisitEntryList = createDataList(analysisStr, lsstLogFile, tracts, band)

    with open(outLogFile, "w") as outFile, open(outShortLogFile, "w") as outShortFile:
        for lsstQuantityName in lsstQuantityList:
            outShortFile.write("# Stars: " + lsstQuantityName + "\n")
            outShortFile.write("#               filter   mean    stdev   median     num    numTot  "
                               "NumEntries" + "\n")
            outFile.write("# Stars: " + lsstQuantityName + "\n")
            outFile.write("# tract  visit  filter   mean    stdev   median     num    numTot" + "\n")
            lsstMeanMean, lsstWgtMeanMean = 0.0, 0.0
            lsstMeanStdev, lsstWgtMeanStdev = 0.0, 0.0
            lsstMeanMedian, lsstWgtMeanMedian = 0.0, 0.0
            lsstNumSum, lsstTotalSum, lsstNumVisits = 0, 0, 0

            for lsstVisitEntry in lsstVisitEntryList:
                formatStr = "{0:6d} {1:7d} {2:>7s} {3:>7.3F} {4:>7.3F} {5:>7.3F} {6:>8d} {7:>8d}\n"
                if lsstVisitEntry.quantityName == lsstQuantityName:
                    outFile.write((formatStr.format(
                        lsstVisitEntry.tract, lsstVisitEntry.visit, bandStrDict[band],
                        lsstVisitEntry.starStatsDict["mean"],
                        lsstVisitEntry.starStatsDict["stdev"],
                        lsstVisitEntry.starStatsDict["median"],
                        lsstVisitEntry.starStatsDict["num"],
                        lsstVisitEntry.starStatsDict["total"])))

                    if lsstVisitEntry.starStatsDict["num"] > 0:
                        lsstMeanMean += lsstVisitEntry.starStatsDict["mean"]
                        lsstMeanStdev += lsstVisitEntry.starStatsDict["stdev"]
                        lsstMeanMedian += lsstVisitEntry.starStatsDict["median"]
                        lsstWgtMeanMean += (
                            lsstVisitEntry.starStatsDict["mean"]*lsstVisitEntry.starStatsDict["num"])
                        lsstWgtMeanStdev += (
                            lsstVisitEntry.starStatsDict["stdev"]*lsstVisitEntry.starStatsDict["num"])
                        lsstWgtMeanMedian += (
                            lsstVisitEntry.starStatsDict["median"]*lsstVisitEntry.starStatsDict["num"])
                        lsstNumSum += lsstVisitEntry.starStatsDict["num"]
                        lsstTotalSum += lsstVisitEntry.starStatsDict["total"]
                        lsstNumVisits += 1

            if lsstNumVisits > 0:
                lsstMeanMean /= lsstNumVisits
                lsstMeanStdev /= lsstNumVisits
                lsstMeanMedian /= lsstNumVisits
            else:
                lsstMeanMean, lsstMeanStdev, lsstMeanMedian = float("nan"), float("nan"), float("nan")

            if lsstNumSum > 0:
                lsstWgtMeanMean /= lsstNumSum
                lsstWgtMeanStdev /= lsstNumSum
                lsstWgtMeanMedian /= lsstNumSum
            else:
                lsstWgtMeanMean, lsstWgtMeanStdev, lsstWgtMeanMedian = (float("nan"), float("nan"),
                                                                        float("nan"))

            if lsstNumVisits > 1:
                outFile.write(lineSplitStr)
                outFile.write(("{0:14s} {1:>7s} {2:>7.3F} {3:>7.3F} {4:>7.3F} {5:>8d} {6:>8d}\n".
                               format("# straight avg", bandStrDict[band], lsstMeanMean, lsstMeanStdev,
                                      lsstMeanMedian, lsstNumSum, lsstTotalSum)))
                outFile.write(("{0:14s} {1:>7s} {2:>7.3F} {3:>7.3F} {4:>7.3F} {5:>8d} {6:>8d}\n".
                               format("# weighted avg", bandStrDict[band], lsstWgtMeanMean, lsstWgtMeanStdev,
                                      lsstWgtMeanMedian, lsstNumSum, lsstTotalSum)))
            outFile.write(lineSplitStr)

            outShortFile.write(("{0:14s} {1:>7s} {2:>7.3F} {3:>7.3F} {4:>7.3F} {5:>8d} {6:>8d} {7:>8d}\n".
                                format("# straight avg", bandStrDict[band], lsstMeanMean, lsstMeanStdev,
                                       lsstMeanMedian, lsstNumSum, lsstTotalSum, lsstNumVisits)))
            if lsstNumVisits > 1:
                outShortFile.write(("{0:14s} {1:>7s} {2:>7.3F} {3:>7.3F} {4:>7.3F} {5:>8d} {6:>8d} {7:>8d}\n".
                                    format("# weighted avg", bandStrDict[band], lsstWgtMeanMean,
                                           lsstWgtMeanStdev, lsstWgtMeanMedian, lsstNumSum, lsstTotalSum,
                                           lsstNumVisits)))
            outShortFile.write(lineSplitStr)


logRootDir = "/tigress/HSC/users/lauren/"
lsstTicket = "DM-6816"
hscTicket = "HSC-1382"
rcFields = ["cosmos", "wide"]
bands = ["g", "r", "i", "z", "y", "n921"]
allBands = "HSC-G^HSC-R^HSC-I^HSC-Z^HSC-Y^NB0921"
bandStrDict = {"g": "HSC-G", "r": "HSC-R", "i": "HSC-I", "z": "HSC-Z", "y": "HSC-Y", "n921": "NB0921",
               "HSC-G^HSC-R^HSC-I^HSC-Z^HSC-Y^NB0921": "GRIZY9", "HSC-G^HSC-R^HSC-I^HSC-Z^HSC-Y": "GRIZY"}
colorsList = ["gri", "riz", "izy", "z9y"]
tracts = [0, ]

analysisStrs = ["visitAnalysis", "compareVisitAnalysis", "coaddAnalysis", "compareCoaddAnalysis",
                "colorAnalysis"]
lineSplitStr = ("#==========================================================================="
                "========================================" + "\n")

for rcField in rcFields:
    if rcField == "wide":
        bands = ["g", "r", "i", "z", "y"]
        allBands = "HSC-G^HSC-R^HSC-I^HSC-Z^HSC-Y"
        tracts = [8766, 8767]

    for analysisStr in analysisStrs:

        if analysisStr == "colorAnalysis":
            bands = [allBands, ]

        if "compare" in analysisStr:
            lineSplitStr = ("#==================================================================="
                            "=========" + "\n")

        for band in bands:
            lsstLogFile = (logRootDir + lsstTicket + "/" + rcField + "_noJunk/hsc" + analysisStr[0].upper()
                           + analysisStr[1:] + "-" + lsstTicket + "-" + rcField + "-" + band + "_all.log")
            hscLogFile = (logRootDir + lsstTicket + "/" + rcField + "_noJunk/hsc" + analysisStr[0].upper()
                          + analysisStr[1:] + "-" + hscTicket + "-" + rcField + "-" + band + "_all.log")
            print("lsstLogFile = ", lsstLogFile)
            print("hscLogFile = ", hscLogFile)

            outLogFile = lsstLogFile[:-7] + "sum.txt"
            outShortLogFile = lsstLogFile[:-7] + "shortSum.txt"
            print("outLogFile = ", outLogFile)
            print("outShortLogFile = ", outShortLogFile)
            if "compare" in analysisStr:
                printCompareStatsSum(analysisStr, lsstLogFile, tracts, band, outLogFile, outShortLogFile)
            else:
                printStatsSum(analysisStr, lsstLogFile, hscLogFile, tracts, band, outLogFile, outShortLogFile)
