from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import Utility
import time

start_time = time.time()

features = [(1.0, 'people_group_1'), (1.0, 'hat_trick')]
columns = []

filename = "randomForest500Model_new"

for val in features:
    if(val[0] == 1.0):
        columns.append(val[1])

train_dataset = pd.read_csv("../UntitledFolder/merged_old_and_new_train.csv")
test_dataset = pd.read_csv("../UntitledFolder/merged_old_and_new_test.csv")
train_output = pd.read_csv("../Data/act_train_output.csv")

combined_train = pd.merge(train_dataset, train_output, on = "activity_id", how="left")

randomForestModel = RandomForestClassifier(n_estimators=500)
X = combined_train[columns]
Y = combined_train[["outcome"]].values.ravel()

randomForestModel.fit(X, Y)
print("--- %s seconds ---" % (time.time() - start_time))

Utility.saveModel(randomForestModel, filename)
print("--- %s seconds ---" % (time.time() - start_time))

# randomForestModel = Utility.loadModel(filename)

prob = randomForestModel.predict_proba(test_dataset[columns])
test_dataset["outcome"] = prob[:,1]
print("--- %s seconds ---" % (time.time() - start_time))

Utility.saveInOutputForm(test_dataset, filename + ".csv")
# test_dataset[["activity_id", "outcome"]].set_index[["activity_id"]].to_csv("../Data/randomForest.csv")

