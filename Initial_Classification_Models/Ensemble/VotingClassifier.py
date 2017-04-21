import pandas as pd
from Classification import Utility

# test_dataset = Utility.loadModel("../Final/test_randomforest")
# test_dataset.set_index(["activity_id"]).drop('act_0')
test_dataset = pd.read_csv("../../Data/Outputs/Best/randomForest500Model_new.csv")
test_dataset = test_dataset[["activity_id", "outcome"]]
test_dataset["outcome_RF"] = test_dataset["outcome"]

# print(len(test_dataset["outcome_RF"]))

xgb = pd.read_csv("../../Data/XGB.csv")
xgb["out_xgb"] = xgb["outcome"]
lr = pd.read_csv("../../Data/LR.csv")
lr["out_lr"] = lr["outcome"]
manipulation = pd.read_csv("../../Data/Outputs/manipulation.csv")
manipulation["out_man"] = manipulation["outcome"]
# print(len(lr["out_lr"]))

output = pd.merge(xgb, lr, on="activity_id")
output = pd.merge(test_dataset, output, on="activity_id")
output = pd.merge(output, manipulation, on='activity_id')
# print(output)'
print(output.columns)
output["outcome"] = (0.60*output["out_xgb"] + 0.35*output["outcome_RF"] + 0.05*output["out_lr"])
output.set_index(["activity_id"])
# output.loc[len(output)] = ["act_0", "act_0", "act_0", "act_0", "act_0", "act_0"]

Utility.saveInOutputForm(output, "60XGB_35rf_5lr.csv", "ensemble")