#################################
# Takes in the train and test data 
# set and transforms it 
# into merged, label encoded values
#################################

import os
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import LabelEncoder

# Path to people.csv from ReadHatKaggle data set
PEOPLE_FILE_PATH = '../Data/people.csv'
# Path to act_train.csv from RedHatKaggle data set
ACTIVITIES_FILE_PATH = '../Data/act_train.csv'
# Path to test.csv from RedHatKaggle data set
TEST_DATA_FILE_PATH = '../Data/act_test.csv'

# Value to assign to null categories
NULL_VALUE = 'type 0'

# COLUMNS in people.csv and activity.csv 
PEOPLE_ID = "people_people_id"
ACTIVITY_ID = "people_id"

# Identity columns (Non-feature)
ID = ['activity_id','people_id']


def get_file_path(directory, filename):
    """ Combines file path directory/filename
    """
    return os.path.join(directory, filename)


def fill_null(df, fill_with = NULL_VALUE):
    """ Fills null values with 'type 0'
    This assumes null values have no significance
    """
    return df.fillna(fill_with)


def convert_dates(df):
    """
    Converts the dates of a data frame into different columns
    """
    df['days'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    df['week'] = df['date'].dt.week
    df['dayOfMonth'] = df['date'].dt.day
    return df


def merge(people_df, activity_df, left_on_id=PEOPLE_ID, right_on_id=ACTIVITY_ID):
    """ Merge people_df with either activity.csv or test.csv both of which contain activity data
    """
    df = pd.merge(people_df, activity_df, how='right', left_on=left_on_id, right_on=right_on_id)
    return df.drop(left_on_id, axis=1, errors='ignore')


def category_to_label_encoding(dataset, identity_columns=ID):
    """Converts data set to label encoding
    """
    for column in dataset.columns:
        if column not in identity_columns:
            if dataset[column].dtype == 'O':
                dataset[column] = dataset[column].apply(lambda x: str(x).split(' ')[1]).astype(np.int32)
            # This converts bool to int, sklearn anyway treats bool as int
            elif dataset[column].dtype == 'bool':
                le = LabelEncoder()
                le.fit(['True', 'False'])
                dataset[column] = le.transform(dataset[column])
        elif column==ACTIVITY_ID:
            dataset[column] = dataset[column].apply(lambda x: str(x).split('_')[1]).astype(np.float)
    return dataset


def write_out(df, output):
    df.drop('outcome', errors='ignore', axis=1).set_index(['activity_id']).to_csv(output + "_features.csv")
    if 'outcome' in df.columns:
        df[['activity_id','outcome']].set_index(['activity_id']).to_csv(output + "_output.csv")


def encode_labels(data_directory):
    """
    The 3 files to be read are hardcoded.
    Converts each category to label encoding and fills null values with 0.
    This assumes null values have no significance
    """
    # Read the data set people.csv, common for both files
    file_path_people = get_file_path(data_directory, PEOPLE_FILE_PATH)
    people_df = pd.read_csv(file_path_people, parse_dates=["date"], true_values=["True"], false_values=["False"])

    # Introduce a category for null values called category 0 since scikit needs numeric data
    people_df = fill_null(people_df)

    # Add columns for dates, does not drop the actual dates columns in case required
    people_df = convert_dates(people_df)

    # Rename columns for people.csv to avoid confusion
    people_df = people_df.rename(columns=lambda x: "people_" + x)

    for key, file_path in {"train": ACTIVITIES_FILE_PATH, "test": TEST_DATA_FILE_PATH}.items():
        file_path_activity = get_file_path(data_directory, file_path)
        activity_df = pd.read_csv(file_path_activity, parse_dates=["date"])

        # Introduce a category for null values called category 0 since scikit needs numeric data
        activity_df = fill_null(activity_df)

        # Add columns for dates, does not drop the actual dates columns in case required
        activity_df = convert_dates(activity_df)

        # merge the two data frames
        train_dataset = merge(people_df, activity_df)

        # encode with labels
        del activity_df
        train_dataset = category_to_label_encoding(train_dataset)

        # Write to output
        output_file = get_file_path(data_directory, file_path[:-4])
        write_out(train_dataset, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Determine and graph top 10 trending places')
    parser.add_argument('--data_directory', default=None, help='The directory pointing to the data')

    encode_labels(**parser.parse_args().__dict__)
