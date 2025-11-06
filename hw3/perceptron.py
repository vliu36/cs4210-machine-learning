#-------------------------------------------------------------------------
# AUTHOR: Vinson Liu
# FILENAME: perceptron.py
# SPECIFICATION: Trains two perceptron models and compares their performances while revolving parameters
# FOR: CS 4210- Assignment #3
# TIME SPENT: 30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

accPerceptron = 0
accMLP = 0

for rate in n: #iterates over n
    for isShuffle in r: #iterates over r

        #iterates over both algorithms
        #-->add your Python code here

        clfPerceptron = Perceptron(eta0=rate, shuffle=isShuffle, max_iter=1000)
        clfMLP = MLPClassifier(activation="logistic", learning_rate_init=rate, hidden_layer_sizes=[25], shuffle=isShuffle, max_iter=1000)
        #Create a Neural Network classifier
        #if Perceptron then
        #   clf = Perceptron()    #use those hyperparameters: eta0 = learning rate, shuffle = shuffle the training data, max_iter=1000
        #else:
        #   clf = MLPClassifier() #use those hyperparameters: activation='logistic', learning_rate_init = learning rate,
        #                          hidden_layer_sizes = number of neurons in the ith hidden layer - use 1 hidden layer with 25 neurons,
        #                          shuffle = shuffle the training data, max_iter=1000
        #-->add your Python code here

        #Fit the Neural Network to the training data
        clfPerceptron.fit(X_training, y_training)
        clfMLP.fit(X_training, y_training)

        #make the classifier prediction for each test sample and start computing its accuracy
        #hint: to iterate over two collections simultaneously with zip() Example:
        #for (x_testSample, y_testSample) in zip(X_test, y_test):
        #to make a prediction do: clf.predict([x_testSample])
        #--> add your Python code here
        numTruePerceptron = 0
        totalPerceptron = 0
        numTrueMLP = 0
        totalMLP = 0
        for (x_sample, y_sample) in zip(X_test, y_test):
            if y_sample == clfPerceptron.predict([x_sample]):
                numTruePerceptron += 1
            
            if y_sample == clfMLP.predict([x_sample]):
                numTrueMLP += 1

            totalPerceptron += 1
            totalMLP += 1

        #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
        #and print it together with the network hyperparameters
        #Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
        #Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
        #--> add your Python code here
        if accPerceptron <= (numTruePerceptron / totalPerceptron):
            accPerceptron = numTruePerceptron / totalPerceptron
            print(f"Highest Perceptron accuracy so far: {accPerceptron}, Parameters: learning rate={rate}, shuffle={isShuffle}")
        if accMLP <= (numTrueMLP / totalMLP):
            accMLP = numTrueMLP / totalMLP
            print(f"Highest MLP accuracy so far: {accMLP}, Parameters: learning rate={rate}, shuffle={isShuffle}")












