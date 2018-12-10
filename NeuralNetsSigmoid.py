
# coding: utf-8

# In[121]:


#####################################################################################################################
#   CS 6375.003 - Assignment 3, Neural Network Programming
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import Normalizer

class NeuralNet:
    def __init__(self, train, header = True, h1 = 30, h2 = 30):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers
        
        raw_input = pd.read_csv('http://mlr.cs.umass.edu/ml/machine-learning-databases/car/car.data',names=['buying','maintenance','doors','persons','lug_boot','safety','abcd'])
        # TODO: Remember to implement the preprocess method
        train_dataset = self.preprocess(raw_input)
        
        ncols = len(train_dataset.columns)
        print (ncols)
        nrows = len(train_dataset.index)
        print (nrows)
        self.X = train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
       # scaler = StandardScaler()
       # scaler.fit(self.X)
       # self.X = scaler.transform(self.X)
        print (self.X)
        
        
        

        self.y = train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,test_size=0.3)
        self.X= self.X_train
        self.y= self.y_train
        print (self.X)
        print ("class",self.y)
        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))
    #
    # TODO: I have coded the sigmoid activation function, you need to do the same for tanh and ReLu
    #

    def __activation(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        elif activation =="tanh":
            self.TanH(self, x)
        elif activation=="relu":
            self.ReLu(self, x)

    #
    # TODO: Define the function for tanh, ReLu and their derivatives
    #

    def __activation_derivative(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        elif activation=="tanh":
            self.DerivativeTanH(self,x)
        elif activation=="relu":
            self.DerivativeReLu(self,x)
#    def __activation(self, x, activation="tanh"):
#        if activation=="tanh":
#            self.TanH(self, x)

    #
    # TODO: Define the function for tanh, ReLu and their derivatives
    #
#    def __activation_derivative(self,x, activation = "tanh"):
#        if activation=="tanh":
#            self.DerivativeTanH(self,x)  

#    def __activation(self,x,activation="relu"):
#        if activation=="relu":
#            self.ReLu(self,x)

 #   def __activation_derivative(self,x, activation="relu"):
 #       if activation=="relu":
 #           self.DerivativeReLu(self,x)


    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def TanH(self,x):
        return np.tanh(x)
    # derivative of sigmoid function, indicates confidence about existing weight
    
    def ReLu(self,x):
        return np.maximum(x, 0)

    def __sigmoid_derivative(self, x):
        return x * (1 - x)
        
    def DerivativeTanH(self,x):
        return (1 - x**2)
    
    def DerivativeReLu(self,x):
        return 1. * (x > 0)


    #
    # TODO: Write code for pre-processing the dataset, which would include standardization, normalization,
    #   categorical to numerical, etc
    #

    def preprocess(self, X):
            

        le= preprocessing.LabelEncoder()

        #to convert into numbers
        X.buying = le.fit_transform(X.buying)
        X.maintenance = le.fit_transform(X.maintenance)
        X.doors = le.fit_transform(X.doors)
        X.persons = le.fit_transform(X.persons)
        X.lug_boot = le.fit_transform(X.lug_boot)
        X.safety = le.fit_transform(X.safety)
        X.abcd = le.fit_transform(X.abcd)
        
        X = X.astype(float)
        scaler = MinMaxScaler(feature_range=(0, 1))
        X = scaler.fit_transform(X)
        
        #train.Sex = le_sex.inverse_transform(train.Sex)
       


        return pd.DataFrame(X)

    # Below is the training function

    def train(self, max_iterations = 1000, learning_rate = 0.01):
        for iteration in range(max_iterations):
            out = self.forward_pass()
            error = 0.5 * np.power((out - self.y), 2)
            #print(error);
            self.backward_pass(out, activation="sigmoid")
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)
        
    def forward_pass(self):
        # pass our inputs through our neural network
        in1 = np.dot(self.X, self.w01 )
        self.X12 = self.__sigmoid(in1)
        in2 = np.dot(self.X12, self.w12)
        self.X23 = self.__sigmoid(in2)
        in3 = np.dot(self.X23, self.w23)
        out = self.__sigmoid(in3)
        return out



    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    # TODO: Implement other activation functions

#    def compute_output_delta(self, out, activation="tanh"):
#        if activation == "tanh":
#            delta_output = (self.y - out) * (self.DerivativeTanH(out))
#        self.deltaOut = delta_output

    def compute_output_delta(self, out, activation):
        if activation =="sigmoid":
            delta_output = (self.y-out)*(self.__sigmoid_derivative(out))
            self.deltaOut = delta_output
        elif activation == "relu":
            delta_output = (self.y-out)
            self.deltaOut = delta_output
        elif activation =="tanh":
            delta_output = (self.y - out) * (self.DerivativeTanH(out))
            self.deltaOut = delta_output
            
#    def compute_output_delta(self, out, activation="relu"):
#        if activation == "relu":
#            delta_output = (self.y-out)*(self.DerivativeReLu(out))
#        self.deltaOut=delta_output

            # TODO: Implement other activation functions

    def compute_hidden_layer2_delta(self, activation):
        if activation=="sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
            self.delta23 = delta_hidden_layer2
        elif activation=="relu":
            delta_hidden_layer2=(self.deltaOut.dot(self.w23.T)) * (self.DerivativeReLu(self.X23))
            self.delta23=delta_hidden_layer2
        elif activation=="tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.DerivativeTanH(self.X23))
            self.delta23 = delta_hidden_layer2
#    def compute_hidden_layer2_delta(self, activation="relu"):
#        if activation=="relu":
#            delta_hidden_layer2=(self.deltaOut.dot(self.w23.T)) * (self.DerivativeReLu(self.X23))
#        self.delta23=delta_hidden_layer2
        
        # TODO: Implement other activation functions

    def compute_hidden_layer1_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
            self.delta12 = delta_hidden_layer1
        elif activation == "relu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.DerivativeReLu(self.X12))
            self.delta12 = delta_hidden_layer1
        elif activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.DerivativeTanH(self.X12))
            self.delta12 = delta_hidden_layer1
            

    def compute_input_layer_delta(self, activation):
        if activation == "relu":
            delta_input_layer = np.multiply(self.DerivativeReLu(self.X01), self.delta01.dot(self.w01.T))
            self.delta01 = delta_input_layer
        elif activation == "tanh":
            delta_input_layer = np.multiply(self.DerivativeTanH(self.X01), self.delta01.dot(self.w01.T))
            self.delta01 = delta_input_layer
        elif activation == "sigmoid":
            delta_input_layer = np.multiply(self.__sigmoid_derivative(self.X01), self.delta01.dot(self.w01.T))
            self.delta01 = delta_input_layer
        
            
            
    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    def predict(self, test, header = True):
        #raw_input = pd.read_csv(test)
        # TODO: Remember to implement the preprocess method
        #test_dataset = self.preprocess(raw_input)
        #ncols = len(test_dataset.columns)
        #nrows = len(test_dataset.index)
        
        self.X = self.X_test
        self.y = self.y_test;
        
        out = self.forward_pass()
        #print("start",out,"end")
        error = 0.5 * np.power((out - self.y), 2)
        print("Error on test data is: ",str(np.sum(error)))   
        return 0




# In[122]:



if __name__ == "__main__":
    neural_network = NeuralNet("train.csv")
    neural_network.train()
    testError = neural_network.predict("test.csv")


