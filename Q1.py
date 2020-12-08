import numpy as np
from sklearn.metrics import accuracy_score,log_loss,mean_squared_log_error
class MyNeuralNetwork():
    """
    My implementation of a Neural Network Classifier.
    """

    acti_fns = ['relu', 'sigmoid', 'linear', 'tanh']
    weight_inits = ['zero', 'random', 'normal']

    def __init__(self, n_layers, layer_sizes, activation, learning_rate, weight_init, batch_size, num_epochs):
        """
        Initializing a new MyNeuralNetwork object

        Parameters
        ----------
        n_layers : int value specifying the number of layers

        layer_sizes : integer array of size n_layers specifying the number of nodes in each layer

        activation : string specifying the activation function to be used
                     possible inputs: relu, sigmoid, linear, tanh

        learning_rate : float value specifying the learning rate to be used

        weight_init : string specifying the weight initialization function to be used
                      possible inputs: zero, random, normal

        batch_size : int value specifying the batch size to be used

        num_epochs : int value specifying the number of epochs to be used
        """
        self.n_layers=n_layers
        self.layer_sizes=layer_sizes
        self.activation=activation
        self.learning_rate = learning_rate
        self.weight_init=weight_init
        self.batch_size = batch_size
        self.num_epochs=num_epochs
        self.training_losses=[]
        self.testing_losses=[]
        self.training_accuracy=[]
        self.label=[]
        self.weightupdate=[None for i in range(self.n_layers-1)]
        self.errors=[None for i in range(self.n_layers-1)]

        if activation not in self.acti_fns:
            raise Exception('Incorrect Activation Function')

        if weight_init not in self.weight_inits:
            raise Exception('Incorrect Weight Initialization Function')
        pass

    def relu(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        
        return np.maximum(0,X)

    def relu_grad(self, X):
        """
        Calculating the gradient of ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return (X>0).astype(int)

    def sigmoid(self, X):
        """
        Calculating the Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1/(1+np.exp(-X))

    def sigmoid_grad(self, X):
        """
        Calculating the gradient of Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return self.sigmoid(X)*(1-self.sigmoid(X))

    def linear(self, X):
        """
        Calculating the Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return X

    def linear_grad(self, X):
        """
        Calculating the gradient of Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1

    def tanh(self, X):
        """
        Calculating the Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return np.tanh(X)

    def tanh_grad(self, X):
        """
        Calculating the gradient of Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1-self.tanh(X)**2

    def softmax(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        e1=np.exp(X-X.max())
        esum=np.sum(e1,axis=1,keepdims=True)
        return e1/esum

    def softmax_grad(self, X):
        """
        Calculating the gradient of Softmax activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        out=self.softmax(X)*(1-self.softmax(X))
        return out
    def bias_init(self):
        b=[]
        for ly in range(self.n_layers-1):
            b.append(np.zeros((1,self.layer_sizes[ly+1])))
        return b
    def zero_init(self):
        """
        Calculating the initial weights after Zero Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 1-dimensional numpy array which contains the initial weights for the requested layer
        """
        W=[]
        for layer_index in range(self.n_layers-1):
            W.append(np.zeros((self.layer_sizes[layer_index],self.layer_sizes[layer_index+1])))
        
        return W

    def random_init(self):
        """
        Calculating the initial weights after Random Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 1-dimensional numpy array which contains the initial weights for the requested layer
        """
        W=[]
        for layer_index in range(self.n_layers-1):
            W.append(np.random.randn((self.layer_sizes[layer_index],self.layer_sizes[layer_index+1]))*0.01)
        
        return W

    def normal_init(self):
        """
        Calculating the initial weights after Normal(0,1) Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 1-dimensional numpy array which contains the initial weights for the requested layer
        """
        W=[]
        for layer_index in range(self.n_layers-1):
            W.append(np.random.normal(0,1,size=(self.layer_sizes[layer_index],self.layer_sizes[layer_index+1]))*0.01)
        
        return W

    def fit(self, X, y,xtest,ytest):
        """
        Fitting (training) the linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self
        """
        if(self.weight_init=="zero"):
            self.W=self.zero_init()
        elif(self.weight_init=="normal"):
            self.W=self.normal_init()
        else:
            self.W=self.random_init()
        self.a=[0]*(len(self.W))
        self.b=self.bias_init()
        totalbatches = X.shape[0]//self.batch_size

        for iters in range(self.num_epochs):
            print("epoch:",iters)
            print("training ...")
            # print trianing accuracy
           # print(self.cross_entropy(self.predict(X),y))
          #  print(self.a[-1])
           # print(self.predict(X))
          #  print(self.a[-2])
            #pd=self.predict(xtest)
            #print(self.errorp(self.a[-1],y))
            self.testing_losses.append(self.score(xtest,ytest))
            #pd2=self.predict(X)
            self.training_losses.append(self.score(X,y))
            total_accuracy=0
            for j in range(totalbatches):
                train_X=X[self.batch_size*j:self.batch_size*(j+1),:]
                train_Y=y[self.batch_size*j:self.batch_size*(j+1)]
                ojk=np.zeros((train_X.shape[0],10))
                for h in range(self.batch_size):
                    ojk[h,train_Y[h]]=1
                self.label=ojk
                self.forwardpropagation(train_X)
                self.backpropagation(train_X,train_Y)
                #self.backpropagation(train_X,train_Y)
                
                
                
                
            
            
            

        # fit function has to return an instance of itself or else it won't work with test.py
        return self

    def predict_proba(self, X):
        """
        Predicting probabilities using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the prediction probabilities.
        """

        # return the numpy array y which contains the predicted values
        self.forwardpropagation(X)
        arf=self.a[-1]
        afr1=np.sum(arf,axis=1)
        return arf/afr1.reshape(arf.shape[0],1)

    def predict(self, X):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """
        self.forwardpropagation(X)

        # return the numpy array y which contains the predicted values
        return np.argmax(self.a[-1],axis=1)

    def score(self, X, y):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as testing labels.

        Returns
        -------
        acc : float value specifying the accuracy of the model on the provided testing set
        """

        # return the numpy array y which contains the predicted values
        self.forwardpropagation(X)
        arf = self.a[-1]
        pred=np.argmax(arf,axis=1).T
        accuracy=accuracy_score(pred,y)
        return accuracy
    def cross_entropy(self,predicted,actual):
        #pred=np.argmax(predicted,axis=1).T
        pred=predicted.T.reshape(actual.shape[0],1)
        print(pred.shape,actual.shape)
        cost = np.sum(actual*np.log(pred))
        samples = actual.shape[0]
        return -1*cost/samples
    
        
    def forwardpropagation(self,X):
        #self.a[0]=X
        for layer_index,(w,b) in enumerate(zip(self.W,self.b)):
            if(layer_index==len(self.W)-1):
                out1=np.dot(self.a[layer_index-1],w)+b
##                if(self.activation=="relu"):
##                    self.a[layer_index]=self.relu(out1)
##                elif(self.activation=="sigmoid"):
##                    self.a[layer_index]=self.sigmoid(out1)
##                elif(self.activation=="linear"):
##                    self.a[layer_index]=self.linear(out1)
##                else:
##                    self.a[layer_index]=self.tanh(out1)
               # print(out1)
                self.a[layer_index]=self.softmax(out1)
            elif(layer_index==0):
                out1=np.dot(X,w)+b
                if(self.activation=="relu"):
                   self.a[layer_index]=self.relu(out1)
                elif(self.activation=="sigmoid"):
                    self.a[layer_index]=self.sigmoid(out1)
                elif(self.activation=="linear"):
                    self.a[layer_index]=self.linear(out1)
                else:
                    self.a[layer_index]=self.tanh(out1)
                #self.a[layer_index]=self.tanh(np.dot(X,w)+b)
            else:
                out1=np.dot(self.a[layer_index-1],w)+b
                if(self.activation=="relu"):
                    self.a[layer_index]=self.relu(out1)
                elif(self.activation=="sigmoid"):
                    self.a[layer_index]=self.sigmoid(out1)
                elif(self.activation=="linear"):
                    self.a[layer_index]=self.linear(out1)
                else:
                    self.a[layer_index]=self.tanh(out1)
        
    def backpropagation(self,X,y):
        delta_a = [0] * len(self.a)

        delta_z = [0] * (len(self.a)-1)
        
        delta_a[0] = (self.a[-1]-self.label) 
        delta_z[0] = np.dot(delta_a[0], self.W[-1].T)

        for layer_ind in range(1, len(self.a)):
            if(self.activation=="sigmoid"):
                delta_a[layer_ind] = delta_z[layer_ind-1] * self.sigmoid_grad(self.a[-1 - layer_ind])
            elif(self.activation=="relu"):
                delta_a[layer_ind] = delta_z[layer_ind-1] * self.relu_grad(self.a[-1 - layer_ind])
            elif(self.activation=="tanh"):
                delta_a[layer_ind] = delta_z[layer_ind-1] * self.tanh_grad(self.a[-1 - layer_ind])
            else:
                delta_a[layer_ind] = delta_z[layer_ind-1] * self.linear_grad(self.a[-1 - layer_ind])
            if layer_ind < (len(self.a)-1):
                delta_z[layer_ind] = np.dot(delta_a[layer_ind], self.W[-1-layer_ind].T)

        # a4_delta = self.cross_entropy(self.a4, self.y) # w4
        # z3_delta = np.dot(a4_delta, self.w4.T)
        # a3_delta = z3_delta * self.sigmoid_derv(self.a3) # w3
        # z2_delta = np.dot(a3_delta, self.w3.T)
        # a2_delta = z2_delta * self.sigmoid_derv(self.a2) # w2
        # z1_delta = np.dot(a2_delta, self.w2.T)
        # a1_delta = z1_delta * self.sigmoid_derv(self.a1) # w1

        self.W[-1] -= (self.learning_rate/self.batch_size) * np.dot(self.a[-2].T, delta_a[0])
        self.b[-1] -= (self.learning_rate/self.batch_size) * np.sum(delta_a[0], axis=0, keepdims=True)
        for layer_ind in range(len(self.W), 1, -1):
            layer_ind -= 2
            if layer_ind > 0:
                # print(layer_ind)
                # print(self.W[layer_ind+1].shape, self.W[layer_ind].shape, self.a[layer_ind-1].T.shape, delta_a[-layer_ind-1].shape, "LOL")
                self.W[layer_ind] -= (self.learning_rate/self.batch_size) * np.dot(self.a[layer_ind-1].T, delta_a[-layer_ind-1])
                self.b[layer_ind] -= (self.learning_rate/self.batch_size) * np.sum(delta_a[-layer_ind-1])
            else:
                self.W[layer_ind] -= (self.learning_rate/self.batch_size) * np.dot(X.T, delta_a[-layer_ind-1])
                self.b[layer_ind] -= (self.learning_rate/self.batch_size) * np.sum(delta_a[-layer_ind-1])
        

                                            
