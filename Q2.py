import numpy as np
from Q1 import MyNeuralNetwork
import pandas as pd
from sample import Neural_Net
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import h5py
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
f=h5py.File('part_A_train.h5','r')
n1=np.array(f['X'][:])
n2=np.array(f['Y'][:])
k1=[]
for i in range(4200):
    ress=np.where(n2[i]==1)
    k1.append(ress[0])
n2=np.array(k1)

xtrain,xtest,ytrain,ytest=train_test_split(n1,n2,test_size=0.6,random_state=20)
#####reg1.fit(trainx, trainy)
clf = MyNeuralNetwork(5,[784,256,128,64,10],"relu",0.01,"normal",128,20)
clf.fit(xtrain,ytrain,xtest,ytest)
pd2=clf.predict(xtest)
arr=clf.a[-2]
##pd=clf.predict(xtest)
##print(len(clf.a[-2]),len(clf.a[-2][0]))
##print(len(pd))
####print(len(clf.a[0]))
##pwds=clf.a
#####a1=clf.training_losses
######a2=clf.testing_losses
######print("test accuracy",a2[-1])
##fgfile=open('reluweights','ab')
##pickle.dump(pwds,fgfile)
##fgfile.close()
##plt.plot(np.arange(100),a1)
##plt.plot(np.arange(100),a2)
##plt.title("sigmoid function as acctivation function")
##plt.xlabel('epochs')
##plt.ylabel('accuracy')
##plt.legend(["training accuracy","testing accuracy"])
##plt.show()

# question 2 part 6 changed parameters in activation
##ytrain=ytrain.T[0]
##reg1 = MLPClassifier(hidden_layer_sizes=(256, 128, 64),activation='logistic', max_iter=100,verbose=True)
##reg1.fit(xtrain,ytrain)
##valid_=reg1.score(xtest,ytest)
##print(valid_)
##data=pd.read_pickle('reluweights')
###print(data)
##arr=data[-2]
##print(len(arr),len(arr[0]))
print(len(arr))
tsne=TSNE(n_components=2).fit_transform(arr)
dfg=pd.DataFrame(data=tsne,columns=['cp1','cp2'])
targetdf=pd.DataFrame(data=ytrain,columns=['target'])
df3=pd.concat([dfg,targetdf],axis=1)
ax=sns.scatterplot(x='cp1',y='cp2',data=df3,hue='target').plot()
plt.show()
