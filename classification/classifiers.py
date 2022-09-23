# classifiers.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import util
from classificationMethod import ClassificationMethod
import numpy as np


class LinearRegressionClassifier(ClassificationMethod):
    """
    Classifier with Linear Regression.
    """
    def __init__(self, legalLabels):
        """

        :param legalLabels: Labels to predict (for digit data, legalLabels = range(10))
        """
        super(LinearRegressionClassifier, self).__init__(legalLabels)
        self.legalLabels = legalLabels
        self.type = 'lr'
        self.lambda_ = 1e-4
        self.weights = None

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Train the Linear Regression Classifier.

        For digit data, trainingData/validationData are all in numpy format with size ([number of data], 784)
        For doc data, trainingData/validationData should also be in numpy format.
        """
        n, dim = trainingData.shape
        X = trainingData
        Y = np.zeros((n, len(self.legalLabels)))
        Y[np.arange(n), trainingLabels] = 1
        self.weights = np.dot(np.linalg.inv(np.dot(X.T, X) + self.lambda_*np.eye(dim)), np.dot(X.T, Y))
    
    def classify(self, data):
        """
        Predict which class is in.
        :param data: data to classify which class is in. (in numpy format)
        :return list or numpy array
        """
        return np.argmax(np.dot(data, self.weights), axis=1)
        
        
class KNNClassifier:
    """
    KNN Classifier.
    """
    
    def __init__(self, legalLabels, num_neighbors):
        """

        :param legalLabels: Labels to predict (for digit data, legalLabels = range(10))
        :param num_neighbors: number of nearest neighbors.
        """
        self.legalLabels = legalLabels
        self.type = 'knn'
        self.num_neighbors = num_neighbors
    
    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Train the Linear Regression Classifier by just storing the trainingData and trainingLabels.

        For digit data, trainingData/validationData are all in numpy format with size ([number of data], 784)
        For doc data, trainingData/validationData should also be in numpy format.
        """
        self.trainingData = trainingData
        self.trainingLabels = trainingLabels
    
    def classify(self, data):
        """
        Predict which class is in.

        Some numpy functions that may be of use (we consider np as short of numpy)
        np.sum(a, axis): sum of array elements over a given axis.
        np.dot(A, B): dot product of two arrays, or matrix multiplication between A and B.
        np.sort, np.argsort: return a sorted copy (or indices) of an array.

        :param data: Data to classify which class is in. (in numpy format)
        :return Determine the class of the given data (list or numpy array)
        """

        "*** YOUR CODE HERE ***"
        # should compute (validationData[i] - trainingData[j])^2
        nTraining = self.trainingData.shape[0]
        nTest = data.shape[0]
        testLabels = []
        for i in range(nTest):
            distances = np.array([0]*nTraining)
            for j in range(nTraining):
                temp = data[i] - self.trainingData[j]
                distances[j] = np.dot(temp, temp)
            sortResult = np.argsort(distances)
            
            neighborLabels = []
            for k in range(0, self.num_neighbors, 1):
                index = sortResult[k]
                neighborLabels.append(self.trainingLabels[index])
            
            iLabel = max(neighborLabels, key = neighborLabels.count)
            testLabels.append(iLabel)

        return testLabels  




class PerceptronClassifier:
    """
    Perceptron classifier.
    """
    def __init__( self, legalLabels, max_iterations):
        """
        self.weights/self.bias: parameters to train, can be considered as parameter W and b in a perception.
        self.batchSize: batch size in a mini-batch, used in SGD method
        self.weight_decay: weight decay parameters.
        self.learningRate: learning rate parameters.

        :param legalLabels: Labels to predict (for digit data, legalLabels = range(10))
        :param max_iterations: maximum epoches
        """
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.batchSize = 100
        self.weight_decay = 1e-3
        self.learningRate = 1
        
    def setWeights(self, input_dim):
        self.weights = np.random.randn(input_dim, len(self.legalLabels))/np.sqrt(input_dim)
        self.bias = np.zeros(len(self.legalLabels))
    
    def prepareDataBatches(self, traindata, trainlabel):
        """
        Generate data batches with given batch size(self.batchsize)

        :return a list in which each element are in format (batch_data, batch_label). E.g.:
            [(batch_data_1), (batch_label_1), (batch_data_2, batch_label_2), ..., (batch_data_n, batch_label_n)]

        """
        index = np.random.permutation(len(traindata))
        traindata = traindata[index]
        trainlabel = trainlabel[index]
        split_no = int(len(traindata) / self.batchSize)
        return zip(np.split(traindata[:split_no*self.batchSize], split_no), np.split(trainlabel[:split_no*self.batchSize], split_no))

    def train(self, trainingData, trainingLabels, validationData, validationLabels ):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the project description for details.

        For digit data, trainingData/validationData are all in numpy format with size ([number of data], 784)
        For doc data, trainingData/validationData should also be in numpy format.

        Some data structures that may be in use:
        self.weights/self.bias (numpy format): parameters to train,
            can be considered as parameter W and b in a perception.
        self.batchSize (scalar): batch size in a mini-batch, used in SGD method
        self.weight_decay (scalar): weight decay parameters.
        self.learningRate (scalar): learning rate parameters.

        Some numpy functions that may be of use (we consider np as short of numpy)
        np.sum(a, axis): sum of array elements over a given axis.
        np.dot(A, B): dot product of two arrays, or matrix multiplication between A and B.
        np.mean(a, axis): mean value of array elements over a given axis
        np.exp(a)
        """

        self.setWeights(trainingData.shape[1])
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.
        
        # Hyper-parameters. Your can reset them. Default batchSize = 100, weight_decay = 1e-3, learningRate = 1e-2
        "*** YOU CODE HERE ***"
        self.batchSize = 100
        self.weight_decay = 1e-3
        self.learningRate = 1

        for iteration in range(self.max_iterations):
            if iteration % 10 == 0: print ("Starting iteration ", iteration, "...")
            dataBatches = self.prepareDataBatches(trainingData, trainingLabels)
            for batchData, batchLabel in dataBatches: # batchData/batchLabel contains batchSize number of elements
                "*** YOUR CODE HERE ***"
                WByCol = (np.transpose(self.weights)).copy() # transpose weights matrix in order to visit its column arrays
                probs = np.zeros((self.batchSize, len(self.legalLabels))) # record the probabilities calculated
                for i in range(self.batchSize):
                    deno = float(0)
                    for j in range(len(self.legalLabels)):
                        deno += np.exp(np.dot(WByCol[j], batchData[i]) + self.bias[j])
                    if deno == 0.0:
                        deno = 1.0
                    for j in range(len(self.legalLabels)):
                        probs[i][j] = np.exp(np.dot(WByCol[j], batchData[i]) + self.bias[j])/deno
                
                for j in range(len(self.legalLabels)):
                    trialSumW = np.zeros(batchData.shape[1])
                    trialSumb = 0.0
                    for i in range(self.batchSize):
                        if(j == batchLabel[i]):
                            trialSumW += (probs[i][j] - 1)*batchData[i]
                            trialSumb += probs[i][j] - 1
                        else:
                            trialSumW += probs[i][j]*batchData[i]
                            trialSumb += probs[i][j]
                    WByCol[j] -= self.learningRate*self.weight_decay*WByCol[j] + (self.learningRate/self.batchSize)*trialSumW
                    self.bias[j] -= self.learningRate*self.weight_decay*self.bias[j] + (self.learningRate/self.batchSize)*trialSumb
                self.weights = (np.transpose(WByCol)).copy()

    def classify(self, data):
        """
        :param data: Data to classify which class is in. (in numpy format)
        :return Determine the class of the given data (list or numpy array)
        """
        
        return np.argmax(np.dot(data, self.weights) + self.bias, axis=1)

    def visualize(self):
        sort_weights = np.sort(self.weights, axis=0)
        _min = 0
        _max = sort_weights[-10]
        return np.clip(((self.weights-_min) / (_max-_min)).T, 0, 1)


class SVMClassifier(ClassificationMethod):
    """
    SVM Classifier
    """
    def __init__(self, legalLabels):
        """
        :param legalLabels: Labels to predict (for digit data, legalLabels = range(10))
        """
        super(SVMClassifier, self).__init__(legalLabels)
        self.type = 'svm'
        self.legalLabels = legalLabels
        
        # you may use this for constructing the svm classifier with sklearn
        self.sklearn_svm = None 
        
    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        """
        training with SVM using sklearn API

        For digit data, trainingData/validationData are all in numpy format with size ([number of data], 784)

        sklearn.svm.SVC should be used in this algorithm. The following parameters should be taken into account:
        C: float
        kernel: string
        gamma: float
        decision_function_shape:  'ovo' or 'ovr'
        """
        from sklearn import svm
         
        "*** YOUR CODE HERE ***"
        self.sklearn_svm = svm.SVC(C=5.0 , kernel='rbf' , gamma=0.005 , decision_function_shape='ovr')
        self.sklearn_svm.fit(trainingData, trainingLabels)

    
    def classify(self, data):
        """
        classification with SVM using sklearn API
        """
        "*** YOUR CODE HERE ***"
        return self.sklearn_svm.predict(data)

        
class BestClassifier(ClassificationMethod):
    """
    SVM Classifier
    """
    def __init__(self, legalLabels):
        """
        :param legalLabels: Labels to predict (for digit data, legalLabels = range(10))
        """
        super(BestClassifier, self).__init__(legalLabels)
        self.type = 'best'
        self.legalLabels = legalLabels
        
        "*** YOUR CODE HERE (If needed) ***"
        # For passing the autograder, you may import sklearn package HERE.
        from sklearn import svm
        self.sklearn_classifier = None
    
    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        """
        design a classifier using sklearn API

        For digit data, trainingData/validationData are all in numpy format with size ([number of data], 784)
        For passing the autograder, you may import sklearn package HERE. 
        """
        from sklearn import svm
        "*** YOUR CODE HERE ***"
        self.sklearn_classifier = svm.SVC(C=4.0 , kernel='rbf' , gamma=0.02 , decision_function_shape='ovr')
        self.sklearn_classifier.fit(trainingData , trainingLabels)
            
    def classify(self, data):
        """
        classification with the designed classifier
        """
        "*** YOUR CODE HERE ***"
        return self.sklearn_classifier.predict(data)

