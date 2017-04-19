import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from Classification import Utility
import pickle
import time
from sklearn.preprocessing import StandardScaler

def input_fn(df, LABEL_COLUMN, NON_FEATURE, CONT, cat):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    ds = df.drop(NON_FEATURE,axis=1,errors='ignore')
    continuous_cols = {k: tf.constant(df[k].values)
                       for k in CONT}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        dense_shape=[df[k].size, 1])
                        for k in cat}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols.items() + categorical_cols.items())
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label

LABEL_COLUMN = "outcome"
NON_FEATURE = ['activity_id', 'people_id', 'date', 'people_date']
CONT = ['people_days', 'days',
            'people_month', 'month',
            'people_quarter', 'quarter',
            'people_week', 'week',
            'people_dayOfMonth', 'dayOfMonth',
            'people_year', 'year',
            'people_char_38']

CONT_VALUES = {}

# Categorical data that is only label encoded
CATEGORICAL_DATA = ['people_char_1', 'people_char_2', 'people_group_1',
                        'people_char_3', 'people_char_4', 'people_char_5',
                        'people_char_6', 'people_char_7', 'people_char_8',
                        'people_char_9', 'activity_category',
                        'char_1', 'char_2', 'char_3', 'char_4', 'char_5', 'char_6',
                        'char_7', 'char_8', 'char_9', 'char_10', 'people_char_10', 'people_char_11', 'people_char_12',
                        'people_char_13', 'people_char_14', 'people_char_15',
                        'people_char_16', 'people_char_17', 'people_char_18',
                        'people_char_19', 'people_char_20', 'people_char_21',
                        'people_char_22', 'people_char_23', 'people_char_24',
                        'people_char_25', 'people_char_26', 'people_char_27',
                        'people_char_28', 'people_char_29', 'people_char_30',
                        'people_char_31', 'people_char_32', 'people_char_33',
                        'people_char_34', 'people_char_35', 'people_char_36',
                        'people_char_37']


start_time = time.time()

train_dataset = pd.read_csv("../../Data/act_train_features_reduced.csv")
test_dataset = pd.read_csv("../../Data/act_test_features_reduced.csv")
train_output = pd.read_csv("../../Data/act_train_output.csv")
train_dataset = pd.merge(train_dataset, train_output, on="activity_id")
test_dataset[LABEL_COLUMN] = -1

print("read")
print("--- %s seconds ---" % (time.time() - start_time))

train_dataset = input_fn(train_dataset, LABEL_COLUMN, NON_FEATURE, CONT, CATEGORICAL_DATA)
test_dataset = input_fn(test_dataset, LABEL_COLUMN, NON_FEATURE, CONT, CATEGORICAL_DATA)

Utility.saveModel(train_dataset, "train_dnn")
Utility.saveModel(test_dataset, "test_dnn")

print("--- %s seconds ---" % (time.time() - start_time))

features = [(1.0, 'char_3'), (1.0, 'char_4'), (1.0, 'char_5'),
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

filename = 'DNN'

for var in CONT:
    CONT_VALUES[var] = tf.contrib.layers.real_valued_column(var)


print("--- %s seconds ---" % (time.time() - start_time))

print("Starting DNN")

# # Build 3 layer DNN with 10, 20, 10 units respectively.
# classifier = tf.contrib.learn.DNNClassifier(feature_columns=columns,
#                                             hidden_units=[1024, 1024],
#                                             optimizer=tf.train.GradientDescentOptimizer(
#                                                 learning_rate=0.01
#                                             ),
#                                             n_classes=12,
#                                             model_dir="model/dnn_model")
#
# iterations = 10
# steps = 200
#
# for i in xrange(iterations):
#     classifier.fit(x=train_dataset_array,
#                    y=train_dataset_outcome.values.ravel(),
#                    steps=steps,
#                    batch_size=200)
#     # Evaluate accuracy.
#     accuracy_score = classifier.evaluate(x=train_dataset_array,
#                                          y=train_dataset_outcome.values.ravel())["accuracy"]
#     print("Iterations: {%d}, Accuracy: {%f}" % ((i + 1) * steps, accuracy_score))
#
# test_dataset_act_id = test_dataset[["activity_id"]]
# test_dataset = test_dataset[columns]
# test_dataset_array = (category_to_one_hot(test_dataset, NON_FEATURE, CONT))
# norm1.fit(test_dataset_array)
# test_dataset_array = norm1.transform(test_dataset_array)
# probs = classifier.predict_proba
# test_dataset_act_id["outcome"] = probs[:,1]
# print("--- %s seconds ---" % (time.time() - start_time))
# Utility.saveInOutputForm(test_dataset_act_id, ".csv", "ensemble")