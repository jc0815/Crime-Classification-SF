import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# plt.style.use('ggplot') 
import seaborn as sns
from catboost import Pool, CatBoostClassifier, cv
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

# plots feature variance
def get_feature_variance(df):
    # drop unnecessary features
    features_raw = df.drop(['Descript', 'Resolution', 'Address', 'Category', 'X', 'Y', 'Dates'], axis = 1)
    features = pd.get_dummies(features_raw)
    scaler = MinMaxScaler()
    scaler.fit(features)
    MinMaxScaler(copy = True, feature_range = (0, 1))
    scaled_features = scaler.transform(features)
    threshold = .85 * (1 - .85) # variance threshold
    select = VarianceThreshold(threshold=threshold)
    select.fit(scaled_features)
    indices = np.argsort(select.variances_)[::-1]
    feature_list = list(features)
    sorted_feature_list = []
    threshold_list = []
    for f in range(len(features.columns)):
        sorted_feature_list.append(feature_list[indices[f]])
        threshold_list.append(threshold)
    plt.figure(figsize=(14,6))
    plt.title("Feature Variance: ", fontsize = 14)
    plt.bar(range(len(features.columns)), select.variances_[indices], color = "g")
    plt.xticks(range(len(features.columns)), sorted_feature_list, rotation = 90)
    plt.xlim([-0.5, len(features.columns)])
    plt.tight_layout()
    plt.show()


# plots feature correlation
def get_correlation_plot(df):
    # drop unnecessary features
    features_raw = df.drop(['Descript', 'Resolution', 'Address', 'Category', 'Dates'], axis=1)
    features = pd.get_dummies(features_raw)
    plt.figure(figsize = (15, 15))
    sns.heatmap(features.corr(), annot=True, fmt=".2f")
    plt.show()


if __name__== "__main__":
    # Use current script dir as working dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    data = pd.read_csv('./sf-crime/train.csv') # convert csv to df
    get_feature_variance(data) # feature variance
    get_correlation_plot(data) # feature correlation
