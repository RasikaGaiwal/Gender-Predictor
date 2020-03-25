import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

df = pd.read_csv('names_dataset.csv') 
df.head()
df.isnull().sum()
df.sex.value_counts()               
df_names = df.copy()

df_names.sex.replace({'F':0,'M':1},inplace=True)
print(df_names.head())

def features(name):
    name = name.lower()
    return {
        'first-letter': name[0], 
        'first2-letters': name[0:2], 
        'first3-letters': name[0:3],
        'last-letter': name[-1],
        'last2-letters': name[-2:],
        'last3-letters': name[-3:],
    }


features = np.vectorize(features)
print(features(["Shivani", "Rasika", "Deepti"]))
X = features(df_names['name'])
df_names.name[0],  X[0]
X_train, X_test, y_train, y_test = train_test_split(X, df_names.sex,test_size=0.33,random_state=42)
dv = DictVectorizer()
dv.fit_transform(X_train)


from sklearn import tree
dt_clf = DecisionTreeClassifier().fit(dv.transform(X_train), y_train)
pred = dt_clf.predict(dv.transform(X_test))

print("Test accuracy:",dt_clf.score(dv.transform(X_test), y_test)*100,"%")

print("Train accuracy:",dt_clf.score(dv.transform(X_train), y_train)*100,"%")

print(confusion_matrix(pred, y_test))

from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(random_state=101).fit(dv.transform(X_train), y_train)
print("Test Accuracy",rf_clf.score(dv.transform(X_test), y_test)*100,"%")
print("Train Accuracy",rf_clf.score(dv.transform(X_train), y_train)*100,"%")

pred = rf_clf.predict(dv.transform(X_test))
print(confusion_matrix(pred, y_test))

from sklearn.naive_bayes import MultinomialNB
nb_clf = MultinomialNB().fit(dv.transform(X_train), y_train)
print("Test accuracy",nb_clf.score(dv.transform(X_test), y_test)*100,"%")
print("Train accuracy",nb_clf.score(dv.transform(X_train), y_train)*100,"%")
pred = nb_clf.predict(dv.transform(X_test))
print(confusion_matrix(pred, y_test))

def genderpredictor(a):
    test_name1 = [a]
    transform_dv =dv.transform(features(test_name1))
    vector = transform_dv.toarray()
    if dt_clf.predict(vector) == 0:
        print("Female")
    else:
        print("Male")
        
genderpredictor("Vikas")
genderpredictor("Rasika")

                        
