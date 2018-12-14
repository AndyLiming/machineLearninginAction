import decisionTrees as dt
import treePlotter as tp

if __name__ == '__main__':
  lenses=[]
  with open('lenses.txt', 'r') as fr:
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
  lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
  lensesTree = dt.createTree(lenses, lensesLabels)
  tp.createPlot(lensesTree)