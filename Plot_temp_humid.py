# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:22:35 2019

@author: marti
"""

#%% Histogram
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

data = pd.read_csv('energydata_complete.csv')
#x_data=data[['T1','RH_1','T2','RH_2','T3','RH_3','T4','RH_4','T5','RH_5','T6','RH_6','T7','RH_7','T8','RH_8','T9','RH_9','T_out','RH_out']].values

#%% Test on T1
sns.distplot(data['T1'], fit=norm);
fig = plt.figure()
res = stats.probplot(data['T1'], plot=plt)

print("Skewness: %f" % data['T1'].skew())
print("Kurtosis: %f" % data['T1'].kurt())

#%% 
for i in list(data.columns)[3:27]:
    bob = plt.figure()
    sns.distplot(data[i], fit=norm);
    fig = plt.figure()
    res = stats.probplot(data[i], plot=plt)
    plt.show()
    print(i, "skewness: %f" % data[i].skew())
    print(i, "kurtosis: %f" % data[i].kurt())


#%% Problematic: T2, (RH_2), (RH_3), (RH_4), T5?, RH_5, T6, (?RH_6?), (RH_7), (RH_9), T_out?, RH_out?, ?Windspeed?.
# Solve shit by applying log transformation.
data = pd.read_csv('energydata_complete.csv')

data['T2'] = stats.boxcox(data['T2'], -1)
data['RH_3'] = stats.boxcox(data['RH_3'], 1)
data['RH_4'] = stats.boxcox(data['RH_4'], -1)
data['T5'] = stats.boxcox(data['T5'], -1)
data['RH_5'] = stats.boxcox(data['RH_5'], -1)
data['T6'] = stats.boxcox(data['T6']+10, 0.5)
data['RH_9'] = stats.boxcox(data['RH_9'], -0.5)
data['T_out'] = stats.boxcox(data['T_out']+10, 0.5)
#data['RH_out'] = stats.boxcox(data['RH_out'], 1)
#data['Windspeed'] = stats.boxcox(data['Windspeed'], 1)

for i in list(data.columns)[3:27]:
    bob = plt.figure()
    sns.distplot(data[i], fit=norm);
    fig = plt.figure()
    res = stats.probplot(data[i], plot=plt)
    plt.show()
    print(i, "skewness: %f" % data[i].skew())
    print(i, "kurtosis: %f" % data[i].kurt())

#%%
data = pd.read_csv('energydata_complete.csv')
sns.distplot(data['Windspeed']+5, fit=norm);
fig = plt.figure()
res = stats.probplot(data['Windspeed'], plot=plt)
plt.show()

print("Skewness: %f" % data['Windspeed'].skew())
print("Kurtosis: %f" % data['Windspeed'].kurt())

data['Windspeed'] = stats.boxcox(data['Windspeed']+5, -1)
sns.distplot(data['Windspeed'], fit=norm);
fig = plt.figure()
res = stats.probplot(data['Windspeed'], plot=plt)
plt.show()

print("Skewness: %f" % data['Windspeed'].skew())
print("Kurtosis: %f" % data['Windspeed'].kurt())

data = pd.read_csv('energydata_complete.csv')
data['Windspeed'] = stats.boxcox(data['Windspeed']+5, -0.5)
sns.distplot(data['Windspeed'], fit=norm);
fig = plt.figure()
res = stats.probplot(data['Windspeed'], plot=plt)
plt.show()

print("Skewness: %f" % data['Windspeed'].skew())
print("Kurtosis: %f" % data['Windspeed'].kurt())

data = pd.read_csv('energydata_complete.csv')
data['Windspeed'] = stats.boxcox(data['Windspeed']+5, 0)
sns.distplot(data['Windspeed'], fit=norm);
fig = plt.figure()
res = stats.probplot(data['Windspeed'], plot=plt)
plt.show()

print("Skewness: %f" % data['Windspeed'].skew())
print("Kurtosis: %f" % data['Windspeed'].kurt())

data = pd.read_csv('energydata_complete.csv')
data['Windspeed'] = stats.boxcox(data['Windspeed']+5, 0.5)
sns.distplot(data['Windspeed'], fit=norm);
fig = plt.figure()
res = stats.probplot(data['Windspeed'], plot=plt)
plt.show()

print("Skewness: %f" % data['Windspeed'].skew())
print("Kurtosis: %f" % data['Windspeed'].kurt())