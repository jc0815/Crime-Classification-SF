import os
import numpy as np
import pandas as pd
from catboost import Pool, CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split

if __name__== "__main__":
    # Use current script dir as working dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    data = pd.read_csv("./sf-crime/train.csv") # convert csv to df
    y = data.Category # get the class labels
    X = data.drop("Category", axis=1) # all features without class labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) # split test and train df
    cat_features = np.where(X.dtypes != float)[0] # get features that aren't floats

    train_dataset = Pool(data = X_train, label = y_train, cat_features = cat_features)

    eval_dataset = Pool(data = X_test, label = y_test, cat_features = cat_features)

    # Initialize CatBoostClassifier
    model = CatBoostClassifier(l2_leaf_reg = 5, learning_rate = 0.01, logging_level = "Verbose", iterations = 10000, depth = 6, random_seed = 42,
                                loss_function = "MultiClass")

    model.fit(train_dataset) # Fit model
    preds_class = model.predict(eval_dataset) # Get predicted classes
    # Get predicted probabilities for each class
    preds_proba = model.predict_proba(eval_dataset)
    # Get predicted RawFormulaVal
    preds_raw = model.predict(eval_dataset, prediction_type = "RawFormulaVal")
    
    print(model.get_best_score())
    model.save_model("./models/model_01_10000_6_final") # save the model