# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:36:25 2023

"""
import matplotlib.pyplot as plt

import numpy as np
from numpy import mean, std
 
import seaborn as sns

import pandas as pd
from pandas import read_csv

from sklearn import preprocessing, model_selection, feature_selection, metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn import neighbors, tree, linear_model, ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.wrappers.scikit_learn import KerasRegressor

import statsmodels.formula.api as sm

"""
####reading data
df = read_csv("C:/Users/zia_s/OneDrive/Desktop/Mahsa/Machine learning paper/Mahsadataset3.csv")
X = df.drop(['Number', 'Running_time(s)', 'inst_power(watt)'], axis = 1)	
y = df['inst_power(watt)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 20)

"""
####reading data
df = pd.read_csv("C:/Users/zia_s/OneDrive/Desktop/Mahsa/Machine learning paper/Mahsadataset3_training.csv")
X_train = df.drop(['Number', 'Running_time(s)', 'inst_power(watt)'], axis = 1)  
Y_train = df['inst_power(watt)']
df2 = pd.read_csv("C:/Users/zia_s/OneDrive/Desktop/Mahsa/Machine learning paper/Mahsadataset3_test.csv")
X_test = df2.drop(['Number', 'Running_time(s)', 'inst_power(watt)'], axis = 1)  
Y_test = df2['inst_power(watt)']


#####normalising

scaler=MinMaxScaler()

scaler.fit(X_train)
scaler.fit(X_test)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

###################################################  MLP   ##################################################

#           model defining
print('\n\n******** MLP *******')

model = Sequential()
model.add(Dense(128, input_dim=10, activation='relu'))
model.add(Dense(64, activation='relu'))
#Output layer
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.summary()
history = model.fit(X_train_scaled, Y_train, batch_size=500, epochs =50, validation_split=0.2)


#    plot the training and validation loss at each epoch

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#    Predict on test data - metrics results and plotting
y_pred_mlp = model.predict(X_test_scaled)
y_pred_mlp = np.squeeze(y_pred_mlp)

print(mean_squared_error(Y_test, y_pred_mlp))
print(mean_absolute_error(Y_test, y_pred_mlp))
print(r2_score(Y_test, y_pred_mlp))

plt.plot(Y_test, 'b.')
plt.plot(y_pred_mlp, 'r.')
plt.show()


###################################################  LSTM   ##################################################

#           model defining
print('\n\n******** LSTM *******')


model = Sequential()
model.add(LSTM(100, input_shape=(10,1)))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.summary()
history = model.fit(X_train_scaled, Y_train, batch_size=500, epochs =50, validation_split=0.2)

#    plot the training and validation loss at each epoch

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#    Predict on test data - metrics results and plotting
y_pred_lstm = model.predict(X_test_scaled)
y_pred_lstm = np.squeeze(y_pred_lstm)

print(mean_squared_error(Y_test, y_pred_lstm))
print(mean_absolute_error(Y_test, y_pred_lstm))
print(r2_score(Y_test, y_pred_lstm))

plt.plot(Y_test, 'b.')
plt.plot(y_pred_lstm, 'r.')
plt.show()






