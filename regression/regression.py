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

def lwlr(testPoint, xArr, yArr, k=1.0):
  xMat = np.mat(xArr)
  yMat = np.mat(yArr).T
  m = np.shape(xMat)[0]
  weights = np.mat(np.eye((m)))
  for j in range(m):
    diffMat = testPoint - xMat[j,:]
    weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
  xTx = xMat.T * (weights * xMat)
  if np.linalg.det(xTx) == 0.0:
    print("this matrix is singular, cannot do inverse")
    return
  ws = xTx.I * (xMat.T * (weights*yMat))
  return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
  m = np.shape(testArr)[0]
  yHat = np.zeros(m)
  for i in range(m):
    yHat[i] lwlr(testArr[i], xArr, yArr, k)
  return yHat

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