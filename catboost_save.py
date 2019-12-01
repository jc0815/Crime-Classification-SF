import os
import numpy as np
import pandas as pd
from catboost import Pool, CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split

if __name__== "__main__":
    # Use current script dir as working dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    model = CatBoostClassifier()
    model.load_model("models/model_01_10000_6_cleaned")
    X_test = pd.read_csv("./sf-crime/test.csv")
    crime_id = X_test['Id']
    X_test = X_test.drop('Id', axis = 1)
    cat_features = np.where(X_test.dtypes != float)[0] # get features that aren't floats
    test_dataset = Pool(data = X_test, cat_features = cat_features)
    pred_proba = model.predict_proba(test_dataset)

    # Save results
    result = pd.DataFrame(pred_proba)
    result.insert(0, 'Id', crime_id)
    column_names = np.insert(category[1], 0, 'Id')
    result.columns = column_names
    result.to_csv('submission.csv', index = False)