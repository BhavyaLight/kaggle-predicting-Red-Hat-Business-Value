#################################
# Takes in the train and test data
# set and transforms it
# into merged, label encoded values
#################################

import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
import pickle
import time
from Classification import Utility

def category_to_one_hot(dataset,non_feature,continuous_feature):
    ds = dataset.drop(non_feature,axis=1,errors='ignore')
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


# from xgboost import XGBClassifier

start_time = time.time()

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


train_dataset = pd.read_csv("../Data/act_train_features_reduced.csv")
test_dataset = pd.read_csv("../Data/act_test_features_reduced.csv")
train_output = pd.read_csv("../Data/act_train_output.csv")

train_dataset = pd.merge(train_dataset, train_output, on="activity_id", how='inner')
print("--- %s seconds ---" % (time.time() - start_time))

print("Starting")
print("Read --- %s seconds ---" % (time.time() - start_time))
cols = (train_dataset.columns.tolist())
for x in NON_FEATURE:
    cols.remove(x)
for x in CONT:
    cols.remove(x)
cols.remove("outcome")
val = len(cols)
cols.extend(CONT)
val2 = len(cols)
print(val)
print(val2)

discrete = []

for x in range(0, 62):
    if(x<49):
        discrete.append(1)
    else:
        discrete.append(0)

# trainingData = trainingData[cols]
# X = trainingData[trainingData.columns[:-4]]
# Y = trainingData[trainingData.columns[-1:]].values.ravel()

# print(X.shape)

# X_new = SelectKBest(chi2, k=20).fit_transform(X, Y)
# print(X_new.shape)
# print(X.columns.tolist())
# print(X)
# Create the RFE object and compute a cross-validated score.
# gradientBooster = GradientBoostingClassifier(n_estimators=100)
# The "accuracy" scoring is proportional to the number of correct
# classifications
# print("Start")
# rfecv = mutual_info_classif(train_dataset[cols], train_dataset["outcome"], discrete_features=discrete, n_neighbors=3, copy=True, random_state=None)
rfecv = Utility.loadModel("MIC")
# Utility.saveModel(rfecv, "MIC")
data = (zip(cols, rfecv))
data.sort(key=lambda tup: tup[1])
print(data)
#[source]
# rfecv.fit(X, Y)
# print("Fit  --- %s seconds ---" % (time.time() - start_time))
#
# pickle.dump(rfecv, open("rfecv_gradientBooster", 'wb'))
# print sorted(zip(map(lambda x: round(x, 4), rfecv.ranking_), list(range(0, X.shape[1]))))
#
# print("Optimal number of features : %d" % rfecv.n_features_)