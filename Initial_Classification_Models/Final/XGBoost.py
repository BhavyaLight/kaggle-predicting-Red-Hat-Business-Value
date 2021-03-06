import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.cross_validation import train_test_split
import time

# Non feature
NON_FEATURE=['activity_id','people_id','date','people_date']

# Categorical data that is only label encoded
CATEGORICAL_DATA = ['people_char_1', 'people_char_2','people_group_1',
                    'people_char_3', 'people_char_4', 'people_char_5',
                    'people_char_6', 'people_char_7', 'people_char_8',
                    'people_char_9', 'activity_category',
                    'char_1', 'char_2', 'char_3', 'char_4', 'char_5', 'char_6',
                    'char_7', 'char_8', 'char_9']

#removed char_10 to check xgb

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


# Path to people.csv from ReadHat Kaggle data set with reduced dimensions
FEATURE_FILE ='Data/act_train_features_reduced.csv'
# Path to act_train.csv from RedHat Kaggle data set with reduced dimensions
OUTPUT ='Data/act_train_output.csv'

TEST_FILE = 'Data/act_test_features_reduced.csv'

def category_to_one_hot(dataset, non_feature, continuous_feature):
    # Function to change labels of categories to one-hot encoding using scikit's OneHot Encoding sparse matrix
    # pd.get_dummies(df) does the same, provides sweet header's as well but it kill's memory
    ds = dataset.drop(non_feature, axis=1)
    boolean_column = []
    counter = 0
    for column in ds.columns:
        if column not in continuous_feature:
            boolean_column.append(counter)
        counter += 1
    # boolean_column is not the column name but index
    print("Done filtering columns...")
    grd_enc = OneHotEncoder(categorical_features=boolean_column)
    encoded_arr = grd_enc.fit_transform(ds)
    return encoded_arr

# Read the data set. Note this dataset does not contain the 'outcome' columns
train_data_df = pd.read_csv(FEATURE_FILE,parse_dates=["date"])
train_data_df.sort_values(by=['activity_id'],ascending=True, inplace=True)


# Read the train data output
train_output = pd.read_csv(OUTPUT)
train_output.sort_values(by='activity_id',ascending=True, inplace=True)
v_out=train_output['outcome'].as_matrix()

test_data_df = pd.read_csv(TEST_FILE,parse_dates=["date"])
test_data_df.sort_values(by=['activity_id'],ascending=True, inplace=True)

############################ UNTOUCHED ######################################
def normalize(df,non_features):
    df=df.drop(non_features,axis=1)
    features=df.columns
    # Normalize categorical values to range betwwen 0-1, time usually : 3secs
    start=time.time()
    scaler = Normalizer()
    df[features] = scaler.fit_transform(df[features])
    end = time.time()
    print (end-start)
    return df

start = time.time()
## SAMPLE: without dropping char_10
train_arr = category_to_one_hot(train_data_df, NON_FEATURE, CONT)
#train_arr.set_index(['activity_id']).to_csv("act_train_features_onehot.csv")
## SAMPLE: try to run with char_10 first, if it does crash, you add it in NON_FEATURE and then run this code. Okay?
end = time.time()
print(end-start)

start = time.time()
## SAMPLE: without dropping char_10
test_arr = category_to_one_hot(test_data_df, NON_FEATURE, CONT)
## SAMPLE: try to run with char_10 first, if it does crash, you add it in NON_FEATURE and then run this code. Okay?
end = time.time()
print(end-start)

scal=Normalizer()
arr_train_norm=scal.fit_transform(train_arr)

scal=Normalizer()
arr_test_norm=scal.fit_transform(test_arr)