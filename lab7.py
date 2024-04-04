#0
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


cancer = load_breast_cancer()
#print(len(cancer['feature_names']))

def question1():
    df = pd.DataFrame(data=cancer['data'],columns=cancer['feature_names'])
    df['target'] = cancer['target']
    return df

def question2():
    cancerdf = question1()
    malignant = (cancerdf['target']==0).sum()
    benign = (cancerdf['target']==1).sum()
    ans = [malignant, benign]
    return ans

def question3():
    cancerdf = question1()
    x = cancerdf.iloc[:,:-1]
    y = cancerdf['target']
    return x,y

def question4():
    x , y = question3()
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)

    return x_train,x_test,y_train,y_test

def question5():
     x_train, x_test, y_train, y_test = question4()
     model = KNeighborsClassifier(n_neighbors=1)
     model.fit(x_train,y_train)
     return model

def question6():
    cancerdf = question1()
    means = cancerdf.mean()[:-1].values.reshape(1,-1)
    model = question5()
    return model.predict(means)

def question7():
    x_train, x_test, y_train, y_test = question4()
    knn = question5()
    return knn.predict(x_test)
def question8():
    x_train, x_test, y_train, y_test = question4()
    knn = question5()
    return knn.score(x_test,y_test)



