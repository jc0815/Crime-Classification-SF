import os
import data_cleaning
import numpy as np
import pandas as pd
from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import train_test_split

if __name__== "__main__":
    # Use current script dir as working dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    data = pd.read_csv('./cleanedDataset.csv')
    #  data = data.drop(['Descript', 'Resolution', 'Address'], axis=1)
    data = data.drop('Descript', axis=1)
    data = data.drop('Resolution', axis=1)
    data = data.drop('Address', axis=1)
    y = data.Category
    X = data.drop('Category', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

    cat_features = [0]

    train_dataset = Pool(data=X_train,
                        label=y_train,
                        cat_features=cat_features)

    eval_dataset = Pool(data=X_test,
                        label=y_test,
                        cat_features=cat_features)

    # Initialize CatBoostClassifier
    model = CatBoostClassifier(iterations=10,
                            learning_rate=0.03,
                            depth=2,
                            loss_function='MultiClass')
    # Fit model
    model.fit(train_dataset)
    # Get predicted classes
    preds_class = model.predict(eval_dataset)
    # Get predicted probabilities for each class
    preds_proba = model.predict_proba(eval_dataset)
    # Get predicted RawFormulaVal
    preds_raw = model.predict(eval_dataset, 
                            prediction_type='RawFormulaVal')
    
    print(model.get_best_score())
