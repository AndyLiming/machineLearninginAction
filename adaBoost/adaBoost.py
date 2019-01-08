import numpy as np
import math
import random
import os

def loadSimpData():
  dataMat = np.matrix([[1., 2.],
                      [2., 1.1],
                      [1.3, 1.],
                      [1., 1.],
                      [2., 1.]])
  classLabels=[1.0, 1.0, -1.0, -1.0, 1.0]
  return dataMat, classLabels

if __name__=='__main__':
