import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from Classification import Utility

# Path to people.csv from ReadHatKaggle data set
FEATURE_FILE ='../../UntitledFolder/merged_old_and_new_train.csv'
OUTPUT_FILE ='../../Data/act_train_output.csv'
# Path to the test file
TEST_FILE = '../../UntitledFolder/merged_old_and_new_test.csv'

# Path to the files with manipulated data
SAVE_AS_DIR = '../../Data/manipulation/New/manipulated_results.csv'

# Non feature
NON_FEATURE = ['activity_id', 'people_id', 'date', 'people_date', "char_10"]

# Categorical data that is only label encoded
CATEGORICAL_DATA = ['people_char_1', 'people_char_2','people_group_1',
                    'people_char_3', 'people_char_4', 'people_char_5',
                    'people_char_6', 'people_char_7', 'people_char_8',
                    'people_char_9', 'activity_category',
                    'char_1', 'char_2', 'char_3', 'char_4', 'char_5', 'char_6',
                    'char_7', 'char_8', 'char_9']

# Already in a one-hot encoded form
CATEGORICAL_BINARY = ['people_char_10', 'people_char_11', 'people_char_12',
                      'people_char_13', 'people_char_14', 'people_char_15',
                      'people_char_16', 'people_char_18',
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
        'people_char_38', "hat_trick"]

# Function to change labels of categories to one-hot encoding using scikit's OneHot Encoding
# pd.get_dummies(df) does the same, provides sweet header's as well but it it not fast enough, kill's memory
def category_to_one_hot(dataset, non_feature, continuous_feature):
    ds = dataset.drop(non_feature, axis=1)
    boolean_column = []
    counter = 0
    for column in ds.columns:
        if column not in continuous_feature:
            boolean_column.append(counter)
        counter += 1
    # boolean_colum is not the column name but index
    print ("Done filtering columns...")
    grd_enc = OneHotEncoder(categorical_features=boolean_column)
    encoded_arr = grd_enc.fit_transform(ds)
    print("Done converting OHW")
    return encoded_arr


# Read the train data set
train_data_df = pd.read_csv(FEATURE_FILE, parse_dates=["date", "people_date"])
train_data_df.sort_values(by=['activity_id'], ascending=True, inplace=True)
# train_data_df['diff_dates'] = (train_data_df['date'] - train_data_df['people_date']).apply(lambda x: x.days)
# Read the train data output
train_output = pd.read_csv(OUTPUT_FILE)
train_output.sort_values(by='activity_id', ascending=True, inplace=True)

test_data_df = pd.read_csv(TEST_FILE, parse_dates=["date", "people_date"])
test_data_df.sort_values(by=['activity_id'], ascending=True, inplace=True)
# test_data_df['diff_dates'] = (test_data_df['date'] - test_data_df['people_date']).apply(lambda x: x.days)
combined = pd.merge(train_data_df, train_output, on='activity_id', how='left')
# Delete redundant values
v_out = combined['outcome']

# Function to one hot encode all values ~ 120 secs
start = time.time()
arr = category_to_one_hot(combined, NON_FEATURE+['outcome'], CONT + CATEGORICAL_BINARY)
end = time.time()
print(end - start)

test_data_df.drop('outcome', inplace=True, errors='ignore')
start = time.time()
arr_b = category_to_one_hot(test_data_df, NON_FEATURE, CONT + CATEGORICAL_BINARY)
end = time.time()
print(end - start)

print (arr.shape)
print (arr_b.shape)

# 90 secs
start = time.time()
norm = StandardScaler(with_mean=False, with_std=True)
norm.fit(arr)
train_arr_n = norm.transform(arr)
end = time.time()
print(end - start)

# 16 secs
start = time.time()
norm = StandardScaler(with_mean=False, with_std=True)
norm.fit(arr_b)
test_arr_n = norm.transform(arr_b)
end = time.time()
print(end - start)

# instantiate a logistic regression model, and fit with X and y - 1552.088
start = time.time()
model = LogisticRegression(penalty='l1')
model = model.fit(arr, v_out)
end = time.time()
print(end - start)

start = time.time()
y_pred = model.predict_proba(arr_b)
end = time.time()
print(end - start)
test_data_df['outcome'] = y_pred[:, 1]

test_data_df[['outcome', 'activity_id']].set_index('activity_id').drop('act_0',axis=0).to_csv("../../Data/outputs/LRresults3_hattrick_NEWEST _WO_norm.csv")

# test_data_df_group = pd.read_csv('kaggle-predicting-Red-Hat-Business-Value/Data/act_the_17304group_features.csv')
#
# test_data_df.set_index('activity_id', inplace=True)
# jj_outcome = pd.read_csv('kaggle-predicting-Red-Hat-Business-Value/Data/manipulated_results.csv')
# jj_outcome.set_index('activity_id', inplace=True)
# jj_outcome['outcome'] = jj_outcome['outcome'].astype('float')
# len(jj_outcome)
# len(test_data_df)
# test_data_df_group.set_index('activity_id', inplace=True)
# counter = 0
# for index, items in jj_outcome.groupby('outcome').get_group(-1)['outcome'].iteritems():
#     if index in test_data_df_group.index:
#         jj_outcome.set_value(index, 'outcome', 0)
#         continue
#     original = test_data_df['outcome'].loc[index]
#     counter += 1
#     jj_outcome.set_value(index, 'outcome', original)
# jj_outcome.reset_index(inplace=True)
# test_data_df.reset_index(inplace=True)
#
# jj_outcome[['outcome', 'activity_id']].set_index('activity_id').drop('act_0').to_csv("LRresults3.csv")
# # 5hours to run
# new_df = []
# start=time.time()
# for (date,group),items in test_data_df.groupby(['date','people_group_1']):
#     if group == 99999:
#         new_df.append(items)
#         continue
#     if len(combined[(combined['people_group_1'] == group) & (combined['date'] == date) ]) == 0:
#         new_df.append(items)
#         continue
#     outcome = combined[(combined['people_group_1'] == group) & (combined['date'] == date) ]['outcome'][:1]
#     items['outcome']=outcome.as_matrix()[0]
#     new_df.append(items)
# end=time.time()
# print(end-start)
