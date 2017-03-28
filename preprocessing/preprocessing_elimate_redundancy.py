import numpy as np
import pandas as pd
import os
import argparse
import time

# Path to people.csv from ReadHatKaggle data set
TRAIN_FILE = 'act_train_features.csv'

# Path to the test file
TEST_FILE = 'act_test_features.csv'

# Non feature
NON_FEATURE = ['activity_id', 'people_id', 'date', 'people_date']

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
                      'people_char_37']

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


def write_out(df, output):
    """ Writes out the data frame to rhe output file
    """
    df.set_index(['activity_id']).to_csv(output + "_features_reduced.csv")


def remove_redundant(df_train, df_test, column, replacement):
    """

    :param df_train: training dataset
    :param df_test: test dataset
    :param column: the categorical column to perform redundancy elimination
    :param replacement: the replacement value for all redundant columns
    :return:
    """
    intersection_test_train=list(\
                                 set(df_test[column].astype('int64').unique())\
                                 .intersection\
                                 (set(df_train[column].astype('int64').unique())))
    df_train[column] = df_train[column].apply(lambda x: replacement if x not in intersection_test_train else x)
    df_test[column] = df_test[column].apply(lambda x: replacement if x not in intersection_test_train else x)
    return df_train, df_test


def eliminate_values(data_directory):
    """
    Main function to perform elimination
    :param data_directory:
    """

    # path to train file
    train_file_path = get_file_path(data_directory, TRAIN_FILE)

    # path to test file
    test_file_path = get_file_path(data_directory, TEST_FILE)

    # Read the train data set
    train_data_df = pd.read_csv(train_file_path,parse_dates=["date"])

    # Read the test data set
    test_data_df = pd.read_csv(test_file_path,parse_dates=["date"])

    # Function to help reduce exploding dimensions
    start = time.time()
    for column in CATEGORICAL_DATA:
        train_data_df, test_data_df = remove_redundant(train_data_df, test_data_df, column, 99999)
    end = time.time()
    print("Time taken to reduce: "+str(end-start))

    # Save files
    file_save = get_file_path(data_directory, 'act_train')
    write_out(train_data_df, file_save)
    file_save = get_file_path(data_directory, 'act_test')
    write_out(test_data_df, file_save)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Determine and graph top 10 trending places')
    parser.add_argument('--data_directory', default=None, help='The directory pointing to the data')

    eliminate_values(**parser.parse_args().__dict__)