import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# import CSV:
dataset = pd.read_csv("./sf-crime/train.csv")

# Getting size of dataset: eg, (878049, 9)
print(dataset.shape)

# Print content of CSV
# print(dataset.head())

X = dataset.drop(['Resolution', 'Dates'], axis=1)
y = dataset['Resolution']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)