import pickle

def saveInOutputForm(df, file, type = "hello"):
    if(type=="ensemble"):
        df[["activity_id","outcome"]].set_index(["activity_id"]).to_csv("../../Data/Outputs/" + file)
    else:
        df[["activity_id","outcome"]].set_index(["activity_id"]).drop('act_0', axis=0).to_csv("../Data/Outputs/" + file)

def saveModel(model, file):
    pickle.dump(model, open(file, 'wb'))

def loadModel(file):
    return pickle.load(open(file, 'rb'))