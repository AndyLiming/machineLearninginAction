import matplotlib as mpl
from math import log
import operator

def calShannonEnt(dataset):
  num = len(dataset)
  labelCounts = {}
  for feaVec in dataset:
    currentLabel = feaVec[-1]
    if currentLabel not in labelCounts.keys():
      labelCounts[currentLabel] = 0
    labelCounts[currentLabel] += 1
  shannonEnt = 0.0
  for key in labelCounts:
    prob = float(labelCounts[key]) / num
    shannonEnt -= prob * log(prob, 2)
  return shannonEnt

def createDataset():
  dataset = [[1,1,'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
  labels = ['no surfacing', 'flippers']
  return dataset, labels

def splitDataset(dataset, axis, value):
  retDataset = []
  for feaVec in dataset:
    if feaVec[axis] == value:
      reducedVec = feaVec[:axis]
      reducedVec.extend(feaVec[axis + 1:])
      retDataset.append(reducedVec)
  return retDataset

def chooseBestToSplit(dataset):
  numFeatures = len(dataset[0]) - 1
  baseEnt = calShannonEnt(dataset)
  bestInfoGain = 0.0
  bestFeature = -1
  for i in range(numFeatures):
    feaList = [example[i] for example in dataset]
    uniqVals = set(feaList)
    newEnt = 0.0
    for value in uniqVals:
      subDataset = splitDataset(dataset, i, value)
      prob = len(subDataset) / float(len(dataset))
      newEnt += prob * calShannonEnt(subDataset)
    infoGain = baseEnt - newEnt
    if (infoGain > bestInfoGain):
      bestInfoGain = infoGain
      bestFeature = i
  return bestInfoGain, bestFeature

def majorityCnt(classList):
  classCount = {}
  for vote in classList:
    if vote not in classCount.keys():
      classCount[vote] = 0
    classCount[vote] += 1
  sortedClassCount = sorted(classCount, items(), key=operator.itemgetter(1), reverse=true)
  return sortedClassCount[0][0]

def createTree(dataset, labels):
  classList = [example[-1] for example in dataset]
  if classList.count(classList[0]) == len(classList):
    return classList[0]
  if len(dataset[0]) == 1:
    return majorityCnt(classList)
  bestInfoGain, bestFeature = chooseBestToSplit(dataset)
  bestFeatureLabel = labels[bestFeature]
  myTree = {bestFeatureLabel: {}}
  del (labels[bestFeature])
  feaValues = [example[bestFeature] for example in dataset]
  uniqVals = set(feaValues)
  for value in uniqVals:
    subLabels = labels[:]
    myTree[bestFeatureLabel][value] = createTree(splitDataset(dataset, bestFeature, value), subLabels)
  return myTree

if __name__ == '__main__':
  myDat, myLabels = createDataset()
  myTree = createTree(myDat, myLabels)
  print(myTree)