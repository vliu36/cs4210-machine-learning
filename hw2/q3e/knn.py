#-------------------------------------------------------------------------
# AUTHOR: Vinson Liu
# FILENAME: knn.py
# SPECIFICATION: Computes the error rate for a 1NN classifier
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

#Reading the data in a csv file using pandas
db = []
df = pd.read_csv('email_classification.csv')
for _, row in df.iterrows():
    db.append(row.tolist())

correctCount = 0
errorCount = 0

#Loop your data to allow each instance to be your test set
for i in range(len(db)):

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    X = []
    for j in range(len(db)):
        row = []
        if i != j:
            for k in range(20):
                row.append(float(db[j][k]) + 2.0)
            X.append(row)

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    Y = []
    for j in range(len(db)):
        if j != i:
            Y.append(0) if db[j][-1] == "ham" else Y.append(1)
            
    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = [float(db[i][j]) + 2.0 for j in range(20)]

    #Fitting the knn to the data using k = 1 and Euclidean distance (L2 norm)
    #--> add your Python code here
    clf = KNeighborsClassifier(n_neighbors=1,algorithm='brute')
    clf.fit(X, Y)


    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    testVal = 0 if db[i][20] == "ham" else 1
    if class_predicted == testVal:
        correctCount += 1
    else:
        errorCount += 1


#Print the error rate
#--> add your Python code here
print(f"The error rate is {errorCount / (correctCount + errorCount)}")






