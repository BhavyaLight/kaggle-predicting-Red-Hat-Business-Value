import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
import time


# Non feature
NON_FEATURE=['activity_id','people_id','date','people_date','char_10']

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
                      'people_char_16', 'people_char_17', 'people_char_18',
                      'people_char_19', 'people_char_20', 'people_char_21',
                      'people_char_22', 'people_char_23', 'people_char_24',
                      'people_char_25', 'people_char_26', 'people_char_27',
                      'people_char_28', 'people_char_29', 'people_char_30',
                      'people_char_31', 'people_char_32', 'people_char_33',
                      'people_char_34', 'people_char_35', 'people_char_36',
                      'people_char_37','weekend' ]

# Continuous categories
CONT = ['people_days', 'days',
      'people_month',  'month',
      'people_quarter', 'quarter',
      'people_week', 'week',
      'people_dayOfMonth', 'dayOfMonth',
      'people_year', 'year',
      'people_char_38','diff_dates']

# Path to people.csv from ReadHat Kaggle data set with reduced dimensions
FEATURE_FILE ='Data/act_train_features_reduced_17304_non_cont.csv'
# Path to act_train.csv from RedHat Kaggle data set with reduced dimensions
OUTPUT ='Data/act_train_output.csv'
TEST_FILE = 'Data/act_test_features_reduced_17304_non_cont.csv'


def category_to_one_hot(dataset,non_feature,continuous_feature):
    print non_feature
    ds = dataset.drop(non_feature,axis=1)
    boolean_column = []
    counter=0
    for column in ds.columns:
        if column not in continuous_feature:
            boolean_column.append(counter)
        counter += 1
    # boolean_colum is not the column name but index
    print ("Done filtering columns...")
    grd_enc = OneHotEncoder(categorical_features=boolean_column)
    encoded_arr=grd_enc.fit_transform(ds)
    return encoded_arr



# Read the data set. Note this dataset does not contain the 'outcome' columns
train_data_df = pd.read_csv(FEATURE_FILE,parse_dates=["date","people_date"])
train_data_df.sort_values(by=['activity_id'],ascending=True, inplace=True)

print("train_data_df done...")

# Read the train data output
train_output = pd.read_csv(OUTPUT)
train_output.sort_values(by='activity_id',ascending=True, inplace=True)
train_data_df['diff_dates']=(train_data_df['date']-train_data_df['people_date']).apply(lambda x: x.days)

print("train_output done...")

test_data_df = pd.read_csv(TEST_FILE,parse_dates=["date","people_date"])
test_data_df.sort_values(by=['activity_id'],ascending=True, inplace=True)
test_data_df['diff_dates']=(test_data_df['date']-test_data_df['people_date']).apply(lambda x: x.days)
print("test_data_df done...")

combined = pd.merge(train_data_df,train_output,on='activity_id')
v_out=combined['outcome'].as_matrix()



print(combined.shape)
print(test_data_df.shape)

set(test_data_df['char_2'].unique()).difference(train_data_df['char_2'].unique())

start = time.time()
train_arr = category_to_one_hot(combined,NON_FEATURE+['outcome'],CONT+CATEGORICAL_BINARY)
end = time.time()
print(end-start)



start = time.time()
## SAMPLE: with dropping char_10
test_data_df.drop('outcome',inplace=True,errors='ignore')
test_arr = category_to_one_hot(test_data_df, NON_FEATURE, CONT+CATEGORICAL_BINARY)
## SAMPLE: try to run with char_10 first, if it does crash, you add it in NON_FEATURE and then run this code. Okay?
end = time.time()
print(end-start)

print (train_arr.shape)
print (test_arr.shape)

#Trial. Run instead of above three
dtrain = xgb.DMatrix(train_arr,label=v_out)
dtest = xgb.DMatrix(test_arr)

param = {'max_depth':10, 'eta':0.02, 'silent':1, 'objective':'binary:logistic' }
param['nthread'] = 4
param['eval_metric'] = 'auc'
param['subsample'] = 0.7
param['colsample_bytree']= 0.7
param['min_child_weight'] = 0
param['booster'] = "gblinear"

watchlist  = [(dtrain,'train')]
num_round = 300
early_stopping_rounds=10
bst = xgb.train(param, dtrain, num_round, watchlist,early_stopping_rounds=early_stopping_rounds)

ypred = bst.predict(dtest)

test_data_df['outcome']=ypred
test_data_df.set_index('activity_id',inplace=True)

jj_outcome = pd.read_csv('Data/manipulated_results.csv')
jj_outcome.set_index('activity_id',inplace=True)
jj_outcome['outcome']=jj_outcome['outcome'].astype('float')
counter = 0
# Uncommebt if using removed 17304 files
test_data_df_group = pd.read_csv('Data/act_the_17304group_features.csv')
test_data_df_group.set_index('activity_id',inplace=True)
for index, items in jj_outcome.groupby('outcome').get_group(-1)['outcome'].iteritems():
# UNCOMMENT if using removed 17304
    if index not in test_data_df.index:
        jj_outcome.set_value(index,'outcome',0)
        continue
    original = test_data_df['outcome'].loc[index]
    counter +=1
    jj_outcome.set_value(index,'outcome',original)
jj_outcome.reset_index(inplace=True)
test_data_df.reset_index(inplace=True)
jj_outcome[['outcome','activity_id']].set_index('activity_id').drop('act_0').to_csv("XGBOOST_results.csv")