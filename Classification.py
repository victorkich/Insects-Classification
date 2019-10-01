import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn as sk
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('Insetos00.csv', encoding='iso-8859-1', header=None)
df.drop(0, inplace=True)
df.sort_values(1, ascending = True,inplace=True)
df = df.replace('Grasshopper', 1).copy()
df = df.replace('Katydid', 1).copy()

x = pd.DataFrame(df[1], dtype=float)
y = pd.DataFrame(df[2], dtype=float)

x_train = df.iloc[:6,0:3]
y_train = df.iloc[:6,3]

x_test = df.iloc[6:,0:3]
y_test = df.iloc[6:,3]

classifier = MLPClassifier(activation='relu', solver='sgd', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1, verbose=1)
classifier.fit(x_train, y_train)
print(classifier.score(x_test, y_test), 4)

colors = (0,0,0)
plt.scatter(x, y, alpha=0.5, c = colors)
plt.title('Gafanhotos')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(classifier)
plt.show(block=True)