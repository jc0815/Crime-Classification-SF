import os
import numpy as np
import pandas as pd
from catboost import Pool, cv
from sklearn.model_selection import train_test_split

# TO LOAD MODEL:
# from catboost import CatBoostClassifier
# model = CatBoostClassifier()      # parameters not required.
# model.load_model('model_name')

if __name__== "__main__":
    # Use current script dir as working dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    data = pd.read_csv('./cleanedDataset.csv')
    data = data.drop(['Descript', 'Resolution', 'Address'], axis=1)
    labels = data.Category # get the class labels
    X = data.drop('Category', axis=1) # all features without class labels
    X_train, X_validation, y_train, y_validation = train_test_split(X, labels, train_size = 0.8, random_state = 42)

    cv_dataset = Pool(data = X, label = labels, cat_features = [0])
    params = {"logging_level": "Verbose", "iterations": 2, "depth": 2, "random_seed": 42, "loss_function": "MultiClass"}

    scores = cv(cv_dataset, params, fold_count = 5, plot="True")
    
    print(scores)