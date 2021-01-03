import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error

# a.
train_set = pd.read_csv('data/faces_train_data.csv')
test_set = pd.read_csv('data/faces_test_data.csv')

train_set_x = train_set.drop(['smiling'], axis=1)
train_set_y = train_set['smiling']

test_set_x = test_set.drop(['smiling'], axis=1)
test_set_y = test_set['smiling']

print("a.", test_set.info())

# b.
smiley = train_set[train_set['smiling']==1].copy()
smiley.drop(['smiling'], axis=1, inplace=True)

meds=list(smiley.mean())
plt.scatter(meds[0],meds[1], color='blue', label = 'smiling')
for i in range(2,len(meds),2):
    plt.scatter(meds[i],meds[i+1], color="blue")

non_smiley = train_set[train_set['smiling']==0].copy()
non_smiley.drop(['smiling'], axis=1, inplace=True)
meds2=list(non_smiley.mean())
plt.scatter(meds2[0],meds2[1], color="red", label = 'non-smiling')
for i in range(2,len(meds2),2):
    plt.scatter(meds2[i],meds2[i+1], color="red")
plt.legend()
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()

#c. d. pe foaie

#e.
dtc2 = DecisionTreeClassifier(random_state=2001, max_depth=2)
dtc2.fit(train_set_x, train_set_y)
y_train_pred = dtc2.predict(train_set_x)
y_test_pred = dtc2.predict(test_set_x)
print("max depth: 2")
print("training: ", accuracy_score(train_set_y, y_train_pred), dtc2.score(train_set_x,train_set_y))
print("test: ", accuracy_score(test_set_y, y_test_pred), dtc2.score(test_set_x,test_set_y))

dtc8 = DecisionTreeClassifier(random_state=2001, max_depth=8)
dtc8.fit(train_set_x, train_set_y)
y_train_pred = dtc8.predict(train_set_x)
y_test_pred = dtc8.predict(test_set_x)
print("max depth: 8")
print("training: ", accuracy_score(train_set_y, y_train_pred), dtc8.score(train_set_x,train_set_y))
print("test: ", accuracy_score(test_set_y, y_test_pred), dtc8.score(test_set_x,test_set_y))

dtc20 = DecisionTreeClassifier(random_state=2001, max_depth=20)
dtc20.fit(train_set_x, train_set_y)
y_train_pred = dtc20.predict(train_set_x)
y_test_pred = dtc20.predict(test_set_x)
print("max depth: 20")
print("training: ", accuracy_score(train_set_y, y_train_pred), dtc20.score(train_set_x,train_set_y))
print("test: ", accuracy_score(test_set_y, y_test_pred), dtc20.score(test_set_x,test_set_y))


#f.
d = dict(zip(train_set_x.columns, dtc8.feature_importances_))
d_sorted = {k:v for k,v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
l = list(d_sorted.keys())
print(l[0], l[1], l[2])
