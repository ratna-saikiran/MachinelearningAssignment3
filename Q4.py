from Q1 import MyNeuralNetwork
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import import TSNE
import seaborn as sn
train = pd.read_pickle("train_CIFAR.pickle")
test = pd.read_pickle("test_CIFAR.pickle")
#traindata=pickledtrain.to_numpy()
#testdata = pickledtest.to_numpy()
print(len(train['Y']))
trainx=train['X']
trainy=train['Y']
testx=test['X']
testy=test['Y']
print(trainy,testy)
unique,counts = np.unique(testy,return_counts=True)
print(dict(zip(unique,counts)))
