import numpy as np
import pandas as pd
import time
import os
import argparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier

# File name for  data set, reduced
TRAIN_FILE = 'act_train_features_reduced.csv'
# File name for test data set, reduced
TEST_FILE = 'act_test_features_reduced.csv'
# output for train data set
OUTPUT ='act_train_output.csv'

# Path to the output file
# Non feature
NON_FEATURE=['activity_id','people_id','date','people_date']

# Categorical data that is only label encoded
CATEGORICAL_DATA = ['people_char_1', 'people_char_2','people_group_1',
                    'people_char_3', 'people_char_4', 'people_char_5',
                    'people_char_6', 'people_char_7', 'people_char_8',
                    'people_char_9', 'activity_category',
                    'char_1', 'char_2', 'char_3', 'char_4', 'char_5', 'char_6',
                    'char_7', 'char_8', 'char_9', 'char_10']

# Already in a one-hot encoded form
CATEGORICAL_BINARY = ['people_char_10', 'people_char_11', 'people_char_12',
                      'people_char_13', 'people_char_14', 'people_char_15',
                      'people_char_16', 'people_char_17', 'people_char_18',
                      'people_char_19', 'people_char_20', 'people_char_21',
                      'people_char_22', 'people_char_23', 'people_char_24',
                      'people_char_25', 'people_char_26', 'people_char_27',
                      'people_char_28', 'people_char_29', 'people_char_30',
                      'people_char_31', 'people_char_32', 'people_char_33',
                      'people_char_34', 'people_char_35', 'people_char_36',
                      'people_char_37' ]

# Continuous categories
CONT = ['people_days', 'days',
        'people_month',  'month',
        'people_quarter', 'quarter',
        'people_week', 'week',
        'people_dayOfMonth', 'dayOfMonth',
        'people_year', 'year',
        'people_char_38']


def get_file_path(directory, filename):
    """ Combines file path directory/filename
    """
    return os.path.join(directory, filename)


def category_to_one_hot(dataset, non_feature, continuous_feature):
    """
    Uses scikit learn's one hot encoding to generate sparse matrix
    Note: Certain models might not have sparse matrix support, do check

    :param dataset: the data set to one hot encode
    :param non_feature: A list of columns in the data set that are not features
    :param continuous_feature: A list of columns in the data set that are features but continuous values
    :return:
    """
    ds = dataset.drop(non_feature, axis=1)
    boolean_column = []
    counter = 0
    # Find the positional index of each categorical column
    for column in ds.columns:
        if column not in continuous_feature:
            boolean_column.append(counter)
        counter += 1
    grd_enc = OneHotEncoder(categorical_features=boolean_column)
    encoded_arr = grd_enc.fit_transform(ds)
    return encoded_arr


def normalize_matrix(arr):
    """
    Function normalizes a matrix

    :param arr: A sparse matrix to normalize
    :return:  A normalized sparse matrix
    """
    norm = Normalizer(copy=False)
    norm.fit(arr)
    return norm.transform(arr)


def write_out(df, output):
    """ Writes out the data frame to rhe output file
    """
    df[['outcome', 'activity_id']].set_index('activity_id').drop('act_0').to_csv(output)


def knn(data_directory, n_neighbours):

    # Start reading files
    start = time.time()
    # Read data frame
    file_path = get_file_path(data_directory, TRAIN_FILE)
    train_data_df = pd.read_csv(file_path, parse_dates=["date"])
    train_data_df.sort_values(by=['activity_id'], ascending=True, inplace=True)

    # Read train output
    file_path = get_file_path(data_directory, OUTPUT)
    train_output = pd.read_csv(file_path)
    train_output.sort_values(by='activity_id', ascending=True, inplace=True)

    # End reading files
    end = time.time()
    print("Training files read. Time taken:"+str(end-start))

    # Start one hot encoding
    start = time.time()
    train_enc = category_to_one_hot(train_data_df , NON_FEATURE, CONT)
    end = time.time()
    print("One hot encoded the values. Time taken: "+str(end-start))

    # Delete redundant data
    del train_data_df

    # Normalize
    start = time.time()
    train_norm = normalize_matrix(train_enc)
    end = time.time()
    print("Normalized the matrix. Time taken: "+str(end-start))

    # Fit knn model
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=n_neighbours)
    knn.fit(train_norm, train_output['output'].as_matrix())
    print("Fit the knn model on train set. Time taken: "+str(end-start))

    # Delete redundant data
    del train_output

    # Read test data frame
    file_path = get_file_path(data_directory, TEST_FILE)
    test_data_df = pd.read_csv(file_path, parse_dates=["date"])
    test_data_df.sort_values(by=['activity_id'], ascending=True, inplace=True)

    # Start one hot encoding
    start = time.time()
    test_enc = category_to_one_hot(test_data_df , NON_FEATURE, CONT)
    end = time.time()
    print("One hot encoded the values. Time taken: "+str(end-start))

    # Normalize
    start = time.time()
    test_norm = normalize_matrix(test_enc)
    end = time.time()
    print("Normalized the matrix. Time taken: "+str(end-start))

    # predict
    y_pred = knn.predict_proba(test_norm)

    test_data_df['outcome'] = y_pred[:, 1]

    file_path = get_file_path(data_directory, "KNN_results.csv")
    write_out(test_data_df,file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Determine and graph top 10 trending places')
    parser.add_argument('--data_directory', default=None, help='The directory pointing to the data')
    parser.add_argument('--K', type=int, default=5, help='Value of k for k-neighbours')

    knn(**parser.parse_args().__dict__)