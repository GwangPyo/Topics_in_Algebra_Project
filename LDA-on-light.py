# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:13:05 2019

@author: marti
"""

#%%
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

data = pd.read_csv('preprocessed.csv')

#%%y_data: lightcategories, x_data: all but light and appliances (maybe sort time).
y_data = np.zeros(len(data))
for i in range(len(y_data)):
    if data['light1'][i] == 1:
        y_data[i] = 1
    elif data['light2'][i] == 1:
        y_data[i] = 2
    elif data['light3'][i] == 1:
        y_data[i] = 3

X_data = data.drop(['Appliances', 'lights0', 'light1', 'light2', 'light3'], axis=1)

clf = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001)
clf.fit(X_data, y_data)

#test = 1
#print('Test:', clf.predict([X_data.iloc[test,:]]))
#print('Real:', y_data[test])

# Testing accuracy
Acc = 0
for i in range(len(y_data)):
    if clf.predict([X_data.iloc[i,:]]) == y_data[i]:
        Acc = Acc+1

Acc = Acc/len(y_data)
print('Accuracy:', Acc)

#%% Splitting for training and test
X_train, X_test, Y_train, Y_test = train_test_split(X_data,y_data, test_size=0.2, random_state=42)
clf.fit(X_train, Y_train)

#test = 1
#print('Test:', clf.predict([X_test.iloc[test,:]]))
#print('Real:', Y_test[test])

Acc = 0
for i in range(len(Y_test)):
    if clf.predict([X_test.iloc[i,:]]) == Y_test[i]:
        Acc = Acc+1

Acc = Acc/len(Y_test)
print('Accuracy:', Acc)