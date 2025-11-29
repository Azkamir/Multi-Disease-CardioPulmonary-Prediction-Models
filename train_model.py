import pandas as pd
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC

df = pd.read_csv('LungcancerDs.csv')

df.drop(['Patient Id'], axis=1, inplace=True)

df['Level'] = df['Level'].replace(['Low', 'Medium', 'High'], [0, 1, 2])

X = df.drop('Level', axis=1)
Y = df['Level']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

selector = SelectKBest(f_classif, k=9)
selector.fit(X, Y)

joblib.dump(selector, 'selector.sav')

x_train_selected = selector.transform(x_train)
x_test_selected = selector.transform(x_test)

clf1 = LogisticRegression(penalty='l2', verbose=0, n_jobs=-1)
clf2 = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
clf3 = SVC()

model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)], voting='hard')

model.fit(x_train_selected, y_train)

joblib.dump(model, 'model.sav')

print("Successful")
