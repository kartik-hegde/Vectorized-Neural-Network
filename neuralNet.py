##############  Vectorized Neural Network learning implementation ###############
## Author: Karthik Hegde, hegdekartik7@gmail.com

import numpy as np


## Supporting functions

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def diffSigmoid(x):
    #return x*(1.0-x)
    return 1.0 - np.power(x,2)

## Neural Network class

class neuralNet:

    ## Set up the neural network
    
    def __init__(self, noInputNeurons, noHiddenNeurons, noOutputNeurons , alpha=0.7, momentum=0.0):
        
        self.noInputNeurons = noInputNeurons+1    ## for bias mode
        self.noHiddenNeurons = noHiddenNeurons
        self.noOutputNeurons = noOutputNeurons

        ##Learning rate and momentum factors
        self.alpha = alpha
        self.momentum = momentum
        
        ##initialise activation functions for all layers
        self.actIn = np.reshape(np.ones(self.noInputNeurons),(1,self.noInputNeurons))
        self.actHid = np.reshape(np.ones(noHiddenNeurons),(1,noHiddenNeurons))
        self.actOut = np.reshape(np.ones(noOutputNeurons),(1,noOutputNeurons))

        ##initialise theta matrix ( weight )
        self.thetaIn = np.random.uniform(-0.2,0.2,(self.noInputNeurons,noHiddenNeurons))
        self.thetaOut = np.random.uniform(0.0,1.0,(noHiddenNeurons,noOutputNeurons))

        ## Momentum matrix
        self.momentumIn = np.zeros((self.noInputNeurons,noHiddenNeurons))
        self.momentumOut = np.zeros((noHiddenNeurons,noOutputNeurons))

    ##forward propagation, X - training set input

    def forwardProp(self, X):

        ##assign the input to the activation of input layer
        self.actIn = X.copy()

        ##perform vectorized forward propagation
        self.actHid = sigmoid(np.dot(self.actIn,self.thetaIn))
        self.actOut = sigmoid(np.dot(self.actHid,self.thetaOut))

        return self.actOut

    ##Back Propagation, y - training set output

    def backProp(self, y):

        ##calculate delta at output
        ## delta is used to the correction needed for each neuron
        deltaOut = diffSigmoid(self.actOut)*(y - self.actOut)
        
        ##delta caclulation for hidden layer
        deltaHid = diffSigmoid(self.actHid) * (np.dot(deltaOut,self.thetaOut.T))

        ##update the theta matrix
        ## Note : Abrupt changes is limited by momentum factor(uses previous result) and
        ##         Learning rate is decided by alpha
        
        ## mod represents the matrix that decides the changes in the original theta matrix       
        modOut = np.dot(self.actHid.T,deltaOut)
        modIn =  np.dot(self.actIn.T,deltaHid)
        
        ##update
        self.thetaOut = self.thetaOut + self.alpha * modOut + self.momentum * self.momentumOut
        self.ThetaIn  = self.thetaIn + self.alpha * modIn + self.momentum * self.momentumIn

        ## Store the momentum values
        self.momentumOut = np.copy(modOut)
        self.momentumIn  = np.copy(modIn)

    def rmsError(self,y):
        ## calculate RMS Error
        return np.sqrt(np.sum(np.power((y - self.actOut),2)))/self.noOutputNeurons

    def learn(self, dataTrain, maxIter = 50000):
            ## learn function takes in dataIn, with each row as an example and columns representing features
            ## dataOut as a matrix with expected output in each row, concatenated in
            ## a single numpy matrix dataTrain, the training data.
        for i in range(maxIter):
            cumErr = 0.0  ## cumulative error
            for j in dataTrain:
                dataIn = np.reshape(j[0],(1,self.noInputNeurons))
                dataOut = np.reshape(j[1],(1,self.noOutputNeurons))
                ## perform forward propagation
                self.forwardProp(dataIn)
                cumErr = cumErr + self.rmsError(dataOut)
                self.backProp(dataOut)

            if i%5000 == 0:
                print cumErr

    def test(self,dataVal):
        ## This function can be used after running the learn function on the training data
        ## Try tis on the validation data to check the correctness of training.
        for i in dataVal:
            ##reshape the array
            dataIn = np.reshape(i[0],(1,self.noInputNeurons))
            ##run forward propagation
            dataOut = self.forwardProp(dataIn)
            print('input,',list(dataIn),' output: ',list(dataOut))
