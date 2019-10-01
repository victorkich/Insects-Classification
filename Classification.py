import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Values to predict
P = [[3.2, 4.2], [7.2, 4.1]]

# Select the data base
df = pd.read_csv('Insetos00.csv', encoding='iso-8859-1', header=None)
df.drop(0, inplace=True)

x_train = df.iloc[:7,1:3]
y_train = df.iloc[:7,3]

x_test = df.iloc[7:,1:3]
y_test = df.iloc[7:,3]

print(df)

# Classification using DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

# Classification using RandomForestClassifier
rfor = RandomForestClassifier()
rfor.fit(x_train, y_train)

# Classification using KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

# Show the models and yours respective acurracy 
print ("\nUsing DecisionTree classifier prediction is " + str(clf.predict(P)))
print("Accuracy: " + str(clf.score(x_test, y_test)) + "\n")
print ("Using MLPC classifier prediction is " + str(rfor.predict(P)))
print("Accuracy: "+ str(rfor.score(x_test, y_test)) + "\n")
print ("Using MLPC classifier prediction is " + str(knn.predict(P)))
print("Accuracy: " + str(knn.score(x_test, y_test)) + "\n")

#y_train = y_train.replace('Grasshopper', 1).copy()
#y_train = y_train.replace('Katydid', 2).copy()