import numpy as np
import math
import random

def loadDataset(fileName):
  dataMat = []
  labelMat = []
  with open(fileName) as fr:
    for line in fr.readlines():
      lineArr = line.strip().split('\t')
      dataMat.append([float(lineArr[0]), float(lineArr[1])])
      labelMat.append(float(lineArr[2]))
  return dataMat, labelMat

def selectJrand(i, m):
  j = i
  while (j == i):
    j = int(random.uniform(0, m))
  return j

def clipAlpha(aj, H, L):
  if aj > H:
    aj = H
  if aj < L:
    aj = L
  return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
  dataMatrix = np.mat(dataMatIn)
  labelMatrix = np.mat(classLabels).transpose()
  b = 0
  m, n = np.shape(dataMatrix)
  alphas = np.mat(np.zeros((m, 1)))
  iter = 0
  while (iter < maxIter):
    alphaPairsChanged = 0
    for i in range(m):
      fXi = float(np.multiply(alphas, labelMatrix).T * (dataMatrix * dataMatrix[i,:].T)) + b
      Ei = fXi - float(labelMatrix[i])
      if ((labelMatrix[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMatrix[i] * Ei > toler) and (alphas[i] > 0)):
        j = selectJrand(i, m)
        fXj = float(np.multiply(alphas, labelMatrix).T * (dataMatrix * dataMatrix[j,:].T)) + b
        Ej = fXj - float(labelMatrix[j])
        alphaIold = alphas[i].copy()
        alphaJold = alphas[j].copy()
        if labelMatrix[i] != labelMatrix[j]:
          L = max(0, alphas[j] - alphas[i])
          H = min(C, C + alphas[j] - alphas[i])
        else:
          L = max(0, alphas[j] + alphas[i] - C)
          H = min(C, alphas[j] + alphas[i])
        if L == H: print("L == H"); continue
        eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - dataMatrix[i,:] * dataMatrix[i,:].T - dataMatrix[j,:] * dataMatrix[j,:].T
        if eta >= 0: print("eta >= 0"); continue
        alphas[j] -= labelMatrix[j] * (Ei - Ej) / eta
        alphas[j] = clipAlpha(alphas[j], H, L)
        if abs(alphas[j] - alphaJold) < 0.00001:
          print("j not moving enough")
          continue
        alphas[i] += labelMatrix[j] * labelMatrix[i] * (alphaJold - alphas[j])
        b1 = b - Ei - labelMatrix[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[i,:].T - labelMatrix[j] * (alphas[j] - alphaJold) * dataMatrix[i,:] * dataMatrix[j,:].T
        b2 = b - Ej - labelMatrix[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[j,:].T - labelMatrix[j] * (alphas[j] - alphaJold) * dataMatrix[j,:] * dataMatrix[j,:].T
        if 0 < alphas[i] and C > alphas[i]: b = b1
        elif 0 < alphas[j] and C > alphas[j]: b = b2
        else: b = (b1 + b2) / 2.0
        alphaPairsChanged += 1
        print("iter: %d, i:%d, pairs changed: %d" % (iter, i, alphaPairsChanged))
    if alphaPairsChanged == 0:
      iter += 1
    else:
      iter=0
    print("iteration num: %d" % iter)
  print(m,n)
  return b, alphas

class optStruct:
  def __init__(self, dataMatIn, classLabels, C, toler):
    self.X = dataMatIn
    self.labelMat = classLabels
    self.C = C
    self.tol = toler
    self.m = np.shape(dataMatIn)[0]
    self.alphas = np.mat(np.zeros((self.m, 1)))
    self.b = 0
    self.eCache = np.mat(np.zeros((self.m, 2)))

def calcEk(oS, k):
  fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.X * oS.X[k,:].T) + oS.b
  Ek = fXk - float(oS.labelMat[k])
  return Ek

def selectJ(i, oS, Ei):
  maxK = -1; maxDeltaE = 0; Ej = 0
  oS.eCache[i] = [1,Ei]
  validEcacheList = nonzero(oS.eCache[:,0].A)[0]
  if (len(validEcacheList)) > 1:
    for k in validEcacheList:
      if k == i: continue
      Ek = calcEk(oS, k)
      deltaE = abs(Ei - Ek)
      if (deltaE > maxDeltaE):
      axK = k; maxDeltaE = deltaE; Ej = Ek
    return maxK, Ej
  else:
    j = selectJrand(i, oS.m)
    Ej = calcEk(oS, j)
  return j, Ej

def updateEk(oS, k):
  Ek = calcEk(oS, k)
  os.eCache[k]=[1,Ek]

if __name__ == '__main__':
  dataMat, labelMat = loadDataset('testSet.txt')
  print(labelMat)
  #dataMat = np.array(dataMat)
  #labelMat = np.array(labelMat)
  b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
  print(b)
  print(alphas[alphas>0])