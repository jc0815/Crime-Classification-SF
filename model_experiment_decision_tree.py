import os
import data_cleaning
import numpy as np
import pandas as pd
from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier 

if __name__== "__main__":
    # Use current script dir as working dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    data = pd.read_csv('./cleanedDataset.csv')
    y = data.Category
    X = data.drop('Category', axis=1)
    labels = list(set(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    # training a DescisionTreeClassifier 
    dtree_model = DecisionTreeClassifier(max_depth = 30).fit(X_train, y_train) 
    dtree_predictions = dtree_model.predict(X_test) 
    print(classification_report(y_test, dtree_predictions, labels=labels))
