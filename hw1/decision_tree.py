#-------------------------------------------------------------------------
# AUTHOR: Vinson Liu
# FILENAME: decision_tree.py
# SPECIFICATION: Trains a decision tree model from the contact_lens.csv dataset.
# FOR: CS 4210- Assignment #1
# TIME SPENT: 55 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#encode the original categorical training features into numbers and add to the 4D array X.
#--> add your Python code here
valuesDict = dict()
code = 2
for row in db:
   tempArr = []
   for i in range(len(row)):
      if i > 3:
         break
      if not (row[i] in valuesDict):
         valuesDict.update({row[i] : code})
         code += 1
      tempArr.append(valuesDict[row[i]])
   X.append(tempArr)
  
#encode the original categorical training classes into numbers and add to the vector Y.
#--> addd your Python code here
Y = [0 if db[i][-1] == "No" else 1 for i in range(10)]

#fitting the decision tree to the data using entropy as your impurity measure
#--> addd your Python code here
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

# # #plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()