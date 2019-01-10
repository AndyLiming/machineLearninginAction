import numpy as np
import math
import random
import os
import matplotlib.pyplot as plt

def loadDataset(fileName):
  numFeat = len(open(fileName).readline().split('\t'))
  dataMat = []
  labelMat = []
  fr = open(fileName)
  for line in fr.readlines():
    lineArr = []
    curLine = line.strip().split('\t')
    for i in range(numFeat - 1):
      lineArr.append(float(curLine[i]))
    dataMat.append(lineArr)
    labelMat.append(float(curLine[-1]))
  return dataMat, labelMat

def standRegres(xArr, yArr):
  xMat = np.mat(xArr)
  yMat = np.mat(yArr).T
  xTx = xMat.T * xMat
  if np.linalg.det(xTx) == 0.0:
    print("this matrix is singular, cannot do inverse")
    return
  ws = xTx.I * (xMat.T * yMat)
  return ws

if __name__ == '__main__':
  xArr, yArr = loadDataset('ex0.txt')
  ws = standRegres(xArr, yArr)
  print(ws)
  xMat = np.mat(xArr)
  yMat = np.mat(yArr)
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
  xCopy = xMat.copy()
  xCopy.sort(0)
  yHat = xCopy * ws
  ax.plot(xCopy[:, 1], yHat)
  plt.show()