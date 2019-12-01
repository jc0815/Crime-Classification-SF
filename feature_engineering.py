import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
import zipfile

# plot feature variance
def get_feature_variance(df, engineered):
    if engineered:
        df = df.drop("Category", axis=1) # drop labels
    else:
        df = df.drop(['Descript', 'Resolution', 'Address', 'Category', 'X', 'Y', 'Dates'], axis = 1)
        df = pd.get_dummies(df)
    scaler = MinMaxScaler()
    scaler.fit(df)
    MinMaxScaler(copy=True, feature_range = (0, 1))
    scaled_features = scaler.transform(df)
    threshold = .85 * (1 - .85) # variance threshold
    select = VarianceThreshold(threshold = threshold)
    select.fit(scaled_features)
    indices = np.argsort(select.variances_)[::-1]
    feature_list = list(df)
    sorted_feature_list = []
    threshold_list = []
    for f in range(len(df.columns)):
        sorted_feature_list.append(feature_list[indices[f]])
        threshold_list.append(threshold)

    plt.figure(figsize = (14, 6))
    plt.title("Feature Variance: ", fontsize = 14)
    plt.bar(range(len(df.columns)), select.variances_[indices], color = "g")
    plt.xticks(range(len(df.columns)), sorted_feature_list, rotation = 90)
    plt.xlim([-0.5, len(df.columns)])
    plt.tight_layout()
    plt.show()


# plot feature correlation
def get_correlation_plot(df, engineered):
    if engineered:
        df = df.drop("Category", axis=1) # drop labels
    else:
        df = df.drop(['Descript', 'Resolution', 'Address', 'Category', 'Dates'], axis=1)
        df = pd.get_dummies(df)
    plt.figure(figsize = (30, 30))
    sns.heatmap(df.corr(), annot = True, fmt = ".1f")
    plt.show()


if __name__== "__main__":
    feature_engineered = False # change this to False for raw df
    zipLocation = "./sf-crime/train_engineered.csv.zip"
    csvFile = "train_engineered.csv"
    if not feature_engineered:
        zipLocation = "./sf-crime/train_raw.csv.zip"
        csvFile = "train_raw.csv"

    data = pd.read_csv(zipfile.ZipFile(zipLocation).open(csvFile)) # convert zip csv to df
    get_feature_variance(data, feature_engineered) # feature variance
    get_correlation_plot(data, feature_engineered) # feature correlation
