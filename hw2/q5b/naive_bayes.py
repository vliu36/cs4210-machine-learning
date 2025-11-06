#-------------------------------------------------------------------------
# AUTHOR: Vinson Liu
# FILENAME: naive_bayes.py
# SPECIFICATION: Naive Bayes classification
# FOR: CS 4210- Assignment #2
# TIME SPENT: 40 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd

dbTraining = []
dbTest = []

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
valuesDict = dict()
code = 2
X = []
for row in dbTraining:
    tempArr = []
    for i in range(1, len(row) - 1):
        if not (row[i] in valuesDict):
            valuesDict.update({row[i] : code})
            code += 1
        tempArr.append(valuesDict[row[i]])
    X.append(tempArr)

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = [0 if dbTraining[i][-1] == "No" else 1 for i in range(len(dbTraining))]

#Fitting the naive bayes to the data using smoothing
#--> add your Python code here
clf = GaussianNB()
clf.fit(X, Y)

#Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header os the solution
#--> add your Python code here
output = {"Day": [], "Outlook": [], "Temperature": [], "Humidity": [], "Wind": [], "PlayTennis": [], "Confidence": []}

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for row in dbTest:
    confidence = clf.predict_proba([[valuesDict[row[i]] for i in range(1, len(row) - 1)]])[0]
    if confidence[0] >= 0.75:
        output["Day"].append(row[0])
        output["Outlook"].append(row[1])
        output["Temperature"].append(row[2])
        output["Humidity"].append(row[3])
        output["Wind"].append(row[4])
        output["PlayTennis"].append("No")
        output["Confidence"].append(round(confidence[0], 2))
    elif confidence[1] >= 0.75:
        output["Day"].append(row[0])
        output["Outlook"].append(row[1])
        output["Temperature"].append(row[2])
        output["Humidity"].append(row[3])
        output["Wind"].append(row[4])
        output["PlayTennis"].append("Yes")
        output["Confidence"].append(round(confidence[1], 2))
        
print(pd.DataFrame(output))

