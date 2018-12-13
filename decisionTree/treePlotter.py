import matplotlib.pyplot as plt
import decisionTrees as dt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def getNumLeafs(myTree):
  numLeafs = 0
  firstStr = list(myTree.keys())[0]
  secondDict = myTree[firstStr]
  for key in secondDict.keys():
    if type(secondDict[key]).__name__ == 'dict':
      numLeafs += getNumLeafs(secondDict[key])
    else:
      numLeafs += 1
  return numLeafs

def getTreeDepth(myTree):
  maxDepth = 0
  firstStr = list(myTree.keys())[0]
  secondDict = myTree[firstStr]
  for key in secondDict.keys():
    if type(secondDict[key]).__name__ == 'dict':
      thisDepth = 1 + getTreeDepth(secondDict[key])
    else:
      thisDepth = 1
    if thisDepth > maxDepth: maxDepth = thisDepth
  return maxDepth

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
  createPlot.axl.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

def plotMidText(cntrPt, parentPt, txtString):
  xMid = (parentPt[0] - cntrPt[0]) / 2.0+ cntrPt[0]
  yMid = (parentPt[1] - cntrPt[1]) / 2.0+ cntrPt[1]
  createPlot.axl.text(xMid,yMid,txtString)

def createPlot():
  fig = plt.figure(1, facecolor='white')
  fig.clf()
  createPlot.axl = plt.subplot(111, frameon=False)
  plotNode("decision node", (0.5, 0.1), (0.1, 0.5), decisionNode)
  plotNode("leaf node", (0.8, 0.1), (0.3, 0.8), leafNode)
  plt.show()

if __name__ == "__main__":
  myDat, myLabels = dt.createDataset()
  myTree = dt.createTree(myDat, myLabels)
  depth = getTreeDepth(myTree)
  leafs = getNumLeafs(myTree)
  print("depth is %d, num of leafs is %d"%(depth,leafs))