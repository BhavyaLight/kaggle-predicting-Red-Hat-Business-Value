from sklearn.ensemble import GradientBoostingClassifier
import pickle
import pandas as pd
from Classification import Utility
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

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

filename = "gradientBoostModelStd"

start_time = time.time()

# Non feature
NON_FEATURE=['activity_id','people_id','date','people_date', 'outcome']

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

train_dataset = pd.read_csv("../../Data/act_train_features_reduced.csv")
test_dataset = pd.read_csv("../../Data/act_test_features_reduced.csv")
train_output = pd.read_csv("../../Data/act_train_output.csv")

train_dataset = pd.merge(train_dataset, train_output, on="activity_id", how='inner')
print("--- %s seconds ---" % (time.time() - start_time))

norm = StandardScaler(with_mean=False, with_std=True)

Y = train_dataset[["outcome"]].values.ravel()
X = category_to_one_hot(train_dataset, NON_FEATURE, CONT)
norm.fit(X)
X = norm.transform(X)

adaBoostModel = GradientBoostingClassifier(n_estimators=100)

print("--- %s seconds ---" % (time.time() - start_time))

X = X.toarray()
adaBoostModel.fit(X, Y)
print("--- %s seconds ---" % (time.time() - start_time))

Utility.saveModel(adaBoostModel, filename)
print("--- %s seconds ---" % (time.time() - start_time))

# adaBoostModel = Utility.loadModel(filename)
X_test = category_to_one_hot(test_dataset, NON_FEATURE, CONT)
norm.fit(X_test)
X_test = norm.transform(X_test)
X_test = X_test.toarray()
prob = adaBoostModel.predict_proba(X_test)
test_dataset["outcome"] = prob[:,1]
print("--- %s seconds ---" % (time.time() - start_time))

Utility.saveInOutputForm(test_dataset, filename + ".csv", "ensemble")
# test_dataset[["activity_id", "outcome"]].set_index[["activity_id"]].to_csv("../Data/randomForest.csv")