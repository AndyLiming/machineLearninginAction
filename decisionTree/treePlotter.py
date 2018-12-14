import matplotlib.pyplot as plt
import decisionTrees as dt
import copy
import pickle

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def getNumLeafs(myTree):
  numLeafs = 0
  firstStr = list(myTree.keys())[0]
  secondDict = myTree[firstStr]
  for key in secondDict.keys():
    if type(secondDict[key]) is dict:
      numLeafs += getNumLeafs(secondDict[key])
    else:
      numLeafs += 1
  return numLeafs

def getTreeDepth(myTree):
  maxDepth = 0
  firstStr = list(myTree.keys())[0]
  secondDict = myTree[firstStr]
  for key in secondDict.keys():
    if type(secondDict[key]) is dict:
      thisDepth = 1 + getTreeDepth(secondDict[key])
    else:
      thisDepth = 1
    if thisDepth > maxDepth: maxDepth = thisDepth
  return maxDepth

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
  createPlot.axl.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

def plotMidText(cntrPt, parentPt, txtString):
  xMid = (parentPt[0] - cntrPt[0]) / 2.0+ cntrPt[0]
  yMid = (parentPt[1] - cntrPt[1]) / 2.0+ cntrPt[1]
  createPlot.axl.text(xMid,yMid,txtString)

def plotTree(myTree, parentPt, nodeTxt):
  numLeafs = getNumLeafs(myTree)
  depth = getTreeDepth(myTree)
  firstStr = list(myTree.keys())[0]
  cntrPt = (plotTree.xOff + (1.0+ float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
  plotMidText(cntrPt, parentPt, nodeTxt)
  plotNode(firstStr, cntrPt, parentPt, decisionNode)
  secondDict = myTree[firstStr]
  plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
  for key in secondDict.keys():
    if type(secondDict[key]) is dict:
      plotTree(secondDict[key], cntrPt, str(key))
    else:
      plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
      plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
      plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
  plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

def createPlot(inTree):
  fig = plt.figure(1, facecolor='white')
  fig.clf()
  axprops = dict(xticks=[], yticks=[])
  #createPlot.axl = plt.subplot(111, frameon=False)
  createPlot.axl = plt.subplot(111, frameon=False, **axprops)
  plotTree.totalW = float(getNumLeafs(inTree))
  plotTree.totalD = float(getTreeDepth(inTree))
  plotTree.xOff = -0.5 / plotTree.totalW
  plotTree.yOff = 1.0
  plotTree(inTree,(0.5,1.0),'')
  #plotNode("decision node", (0.5, 0.1), (0.1, 0.5), decisionNode)
  #plotNode("leaf node", (0.8, 0.1), (0.3, 0.8), leafNode)
  plt.show()

def classifyByTree(decTree, feaLabels, testVec):
  firstStr = list(decTree.keys())[0]
  print(firstStr)
  print(feaLabels)
  secondDict = decTree[firstStr]
  feaId = feaLabels.index(firstStr)
  for key in secondDict.keys():
    if testVec[feaId] == key:
      if type(secondDict[key]) is dict:
        classLabel = classifyByTree(secondDict[key], feaLabels, testVec)
      else:
        classLabel = secondDict[key]
  return classLabel

def storeTree(inputTree, filename):
  with open(filename,mode='wb') as fw:
    pickle.dump(inputTree, fw)

def grabTree(filename):
  with open(filename,mode='rb') as fr:
    return pickle.load(fr)

if __name__ == "__main__":
  myDat, myLabels = dt.createDataset()
  feaLabels = copy.deepcopy(myLabels)
  myTree = dt.createTree(myDat, myLabels)
  print(myTree)
  print(myLabels)
  depth = getTreeDepth(myTree)
  leafs = getNumLeafs(myTree)
  print("depth is %d, num of leafs is %d" % (depth, leafs))
  classLabel = classifyByTree(myTree, feaLabels, [1, 0])
  print("vec [1,1] is classified to %s"%classLabel)
  #createPlot(myTree)
  storeTree(myTree, '123.txt')
  newTree = grabTree('123.txt')
  print(newTree)