import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from Classification import Utility
import pickle
import time
from sklearn.preprocessing import StandardScaler


# Function to change labels of categories to one-hot encoding using scikit's OneHot Encoding
# pd.get_dummies(df) does the same, provides sweet header's as well but it it not fast enough, kill's memory
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

start_time = time.time()

# features = [(1.0, 'people_group_1')]
# columns = []
#
# filename = "randomForest500Model"
#
# for val in features:
#     if (val[0] == 1.0):
#         columns.append(val[1])
#
# RandomForestFilename = "randomForest500Model"
#
# train_dataset = pd.read_csv("../../Data/act_train_features_reduced.csv")
# test_dataset = pd.read_csv("../../Data/act_test_features_reduced.csv")
# train_output = pd.read_csv("../../Data/act_train_output.csv")
#
# train_dataset = pd.merge(train_dataset, train_output, on="activity_id", how='inner')
# print("--- %s seconds ---" % (time.time() - start_time))
#
# randomForestModel = Utility.loadModel("randomForestModel_OHE")
# # randomForestModel = RandomForestClassifier(n_estimators=500)
# #
# # randomForestModel.fit(train_dataset[columns], train_dataset[["outcome"]].values.ravel())
#
# prob_train = randomForestModel.predict_proba(train_dataset[columns])
# prob_test = randomForestModel.predict_proba(test_dataset[columns])
# # Utility.saveModel(randomForestModel, "randomForestModel_OHE")
#
# train_dataset["Random_Forest_1"] = prob_train[:,1]
#
# test_dataset["Random_Forest_1"] = prob_test[:,1]
#
# Utility.saveModel(train_dataset, "train_randomforest")
# Utility.saveModel(test_dataset, "test_randomforest")

# train_dataset = Utility.loadModel("train_randomforest")
test_dataset = Utility.loadModel("test_randomforest")
print("Random Forest Done")
print("--- %s seconds ---" % (time.time() - start_time))

features = [(1.0, 'Random_Forest_1'), (1.0, 'char_3'), (1.0, 'char_4'), (1.0, 'char_5'),
            (1.0, 'char_6'), (1.0, 'char_8'), (1.0, 'char_9'), (1.0, 'days'), (1.0, 'month'), (1.0, 'people_char_1'),
            (1.0, 'people_char_10'), (1.0, 'people_char_11'), (1.0, 'people_char_12'), (1.0, 'people_char_13'),
            (1.0, 'people_char_14'), (1.0, 'people_char_15'), (1.0, 'people_char_16'), (1.0, 'people_char_17'),
            (1.0, 'people_char_18'), (1.0, 'people_char_19'), (1.0, 'people_char_2'), (1.0, 'people_char_20'),
            (1.0, 'people_char_21'), (1.0, 'people_char_22'), (1.0, 'people_char_23'), (1.0, 'people_char_24'),
            (1.0, 'people_char_25'), (1.0, 'people_char_26'), (1.0, 'people_char_27'), (1.0, 'people_char_28'),
            (1.0, 'people_char_29'), (1.0, 'people_char_3'), (1.0, 'people_char_30'), (1.0, 'people_char_31'),
            (1.0, 'people_char_32'), (1.0, 'people_char_33'), (1.0, 'people_char_34'), (1.0, 'people_char_35'),
            (1.0, 'people_char_36'), (1.0, 'people_char_37'), (1.0, 'people_char_38'), (1.0, 'people_char_4'),
            (1.0, 'people_char_5'), (1.0, 'people_char_6'), (1.0, 'people_char_7'), (1.0, 'people_char_8'),
            (1.0, 'people_char_9'), (1.0, 'people_dayOfMonth'), (1.0, 'people_month'), (1.0, 'people_quarter'),
            (1.0, 'people_week'), (1.0, 'people_year'), (1.0, 'quarter'), (1.0, 'week'), (1.0, 'year'), (2.0, 'char_7'),
            (3.0, 'char_1'), (4.0, 'dayOfMonth'), (5.0, 'activity_category'), (6.0, 'people_days'), (7.0, 'char_2'),
            (8.0, 'people_group_1'), (9.0, 'char_10'), (10.0, 'people_id')]
columns = []

filename = 'randomPlusLogisticOHE'

for val in features:
    # if(val[0] == 1.0):
    columns.append(val[1])

# train_dataset_outcome = train_dataset[["outcome"]]
# train_dataset = train_dataset[columns]

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

# train_dataset_array = (category_to_one_hot(train_dataset, NON_FEATURE, CONT))
# Utility.saveModel(train_dataset_array, "ohe_log")
# norm = StandardScaler(with_mean=False, with_std=True)
norm1 = StandardScaler(with_mean=False, with_std=True)

# train_dataset_array =
print("--- %s seconds ---" % (time.time() - start_time))

print("Starting Log Reg")

# logisticModel = LogisticRegression()
# X = train_dataset_array
# norm.fit(X)
# X = norm.transform(X)
# Y = train_dataset_outcome.values.ravel()

# logisticModel.fit(X, Y)
print("--- %s seconds ---" % (time.time() - start_time))

# Utility.saveModel(logisticModel, filename)
# pickle.dump(logisticModel, open(filename, 'wb'))

logisticModel = Utility.loadModel(filename)
test_dataset_act_id = test_dataset[["activity_id"]]
test_dataset = test_dataset[columns]
test_dataset_array = (category_to_one_hot(test_dataset, NON_FEATURE, CONT))
norm1.fit(test_dataset_array)
test_dataset_array = norm1.transform(test_dataset_array)
probs = (logisticModel.predict_proba(test_dataset_array))
test_dataset_act_id["outcome"] = probs[:,1]
print("--- %s seconds ---" % (time.time() - start_time))
Utility.saveInOutputForm(test_dataset_act_id, ".csv", "ensemble")
# test_dataset_act_id[["activity_id", "outcome"]].set_index(["activity_id"]).to_csv("../../Data/" + filename + ".csv")

