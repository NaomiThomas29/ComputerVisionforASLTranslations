from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
import numpy as np
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

data_dict = pickle.load(open('./data.pickle', 'rb'))

X =  (data_dict['data'])
y =  (data_dict['labels'])


homogeneous_data = [x for x in X if len(x) == 42]

filtered_y = [y[i] for i in range(len(y)) if len(X[i]) == len(X[0])]
X = homogeneous_data

y = filtered_y

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.8,
                       test_size=0.2,shuffle=True, stratify=y)

X_train = np.array(X_train)
X_test = np.array(X_test)

'''
attempted knn and logistic regression both had a lower accuracy of 93%
'''
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
score = rfc.score(X_test, y_test)
print(score)

print(classification_report(y_test, y_pred))
 
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
precision = metrics.precision_score(y_test, y_pred, average='weighted')
print("Precision:", precision)
recall = metrics.recall_score(y_test, y_pred, average='weighted')
print("Recall:", recall)
f1 = metrics.f1_score(y_test, y_pred, average='weighted')
print("F1 Score:", f1)

f = open('model.p', 'wb')
pickle.dump({'model': rfc}, f)
f.close()