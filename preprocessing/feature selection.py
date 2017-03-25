# code to upload ( Assuming pre-processing on merged dataset done )
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
rf=RandomForestClassifier()
rf.fit(x_train,y_train)

results=[]
def topkfeatures(rf,k):
    important_features = []
    unsortarr=[]

    sortarr=[]
    for x,i in enumerate(rf.feature_importances_):
        unsortarr.append([x,i])
        sortarr=sorted(unsortarr, key=lambda x: x[1], reverse=True)
    #Iterate depending on variable k
    for j in sortarr[:k]:
        important_features.append(j[0])
    return important_features 

#Can try smaller intervals to find optimal range
for j in range(5,60,5):
    print(j)
    feature_names=x_train.columns
    important_names = feature_names[topkfeatures(rf,j)]
    rf1=RandomForestClassifier()
    x_train_opt=x_train[important_names]
    x_test_opt=x_test[important_names]
    rf1.fit(x_train_opt,y_train)
    results.append([j,rf1.score(x_test_opt,y_test)])
    print(important_names)

print(result)


