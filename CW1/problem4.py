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
from sklearn.metrics import roc_auc_score, roc_curve

df = pd.read_csv('data/classification_eval_1.csv')
print("a.", df.info())

print('a.')
df2 = df.copy(deep=True)
df2['gt'].astype('float64')
df2['alg_1']=np.round(df2['alg_1'])
df2['alg_2']=np.round(df2['alg_2'])
df2['alg_3']=np.round(df2['alg_3'])
df2['alg_4']=np.round(df2['alg_4'])

#print(df2.head())
print("alg_1 = ", accuracy_score(df2['alg_1'], df2['gt']))
print("alg_2 = ", accuracy_score(df2['alg_2'], df2['gt']))
print("alg_3 = ", accuracy_score(df2['alg_3'], df2['gt']))
print("alg_4 = ", accuracy_score(df2['alg_4'], df2['gt']))

# b.
print('b.')
roc1 = roc_auc_score(df['gt'], df['alg_1'])
roc2 = roc_auc_score(df['gt'], df['alg_2'])
roc3 = roc_auc_score(df['gt'], df['alg_3'])
roc4 = roc_auc_score(df['gt'], df['alg_4'])
print("alg_1_roc = ", roc1)
print("alg_2_roc = ", roc2)
print("alg_3_roc = ", roc3)
print("alg_4_roc = ", roc4)

# c.
fpr1, tpr1, _ = roc_curve(df['gt'], df['alg_1'])
fpr2, tpr2, _ = roc_curve(df['gt'], df['alg_2'])
fpr3, tpr3, _ = roc_curve(df['gt'], df['alg_3'])
fpr4, tpr4, _ = roc_curve(df['gt'], df['alg_4'])
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.3])
plt.plot(fpr1, tpr1, color='darkorange', linewidth=2, label='Alg_1 (area = %0.2f)' % roc1)
plt.plot(fpr2, tpr2, color='red', linewidth=2, label='Alg_2 (area = %0.2f)' % roc2)
plt.plot(fpr3, tpr3, color='yellowgreen', linewidth=2, label='Alg_3 (area = %0.2f)' % roc3)
plt.plot(fpr4, tpr4, color='blue', linewidth=2, label='Alg_4 (area = %0.2f)' % roc4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc = 'upper left')
plt.show()