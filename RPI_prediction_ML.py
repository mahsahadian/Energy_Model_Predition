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
from sklearn import neighbors, linear_model, ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression

import xgboost

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.wrappers.scikit_learn import KerasRegressor

import statsmodels.formula.api as sm
from sklearn.model_selection import cross_val_score, cross_validate
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

###################################################  Linear regression   ##################################################

#           model defining
print('\n\n******** Linear Regression *******')

lr_model = linear_model.LinearRegression()
lr_model.fit(X_train_scaled, Y_train)


cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=1)
n_scores_lr = cross_validate(lr_model, X_train_scaled , Y_train, scoring=('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'), cv=cv, n_jobs=-1)
print('MSE: %.9f (%.9f)' % (mean(n_scores_lr['test_neg_mean_squared_error']), std(n_scores_lr['test_neg_mean_squared_error'])))
print('MAE: %.9f (%.9f)' % (mean(n_scores_lr['test_neg_mean_absolute_error']), std(n_scores_lr['test_neg_mean_absolute_error'])))  
print('R2: %.9f (%.9f)' % (mean(n_scores_lr['test_r2']), std(n_scores_lr['test_r2']))) 



#    Predict on test data - metrics results and plotting
y_pred_lr = lr_model.predict(X_test_scaled)

print(mean_squared_error(Y_test, y_pred_lr))
print(mean_absolute_error(Y_test, y_pred_lr))
print(r2_score(Y_test, y_pred_lr))

plt.plot(Y_test, 'b.')
plt.plot(y_pred_lr, 'r.')
plt.show()


###################################################  Decision tree   ##################################################
#           model defining
print('\n\n******** Decision tree *******')

tree = DecisionTreeRegressor()
tree.fit(X_train_scaled, Y_train)



cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=1)
n_scores_tree = cross_validate(tree, X_train_scaled , Y_train, scoring=('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'), cv=cv, n_jobs=-1)
print('MSE: %.9f (%.9f)' % (mean(n_scores_tree['test_neg_mean_squared_error']), std(n_scores_tree['test_neg_mean_squared_error'])))
print('MAE: %.9f (%.9f)' % (mean(n_scores_tree['test_neg_mean_absolute_error']), std(n_scores_tree['test_neg_mean_absolute_error'])))  
print('R2: %.9f (%.9f)' % (mean(n_scores_tree['test_r2']), std(n_scores_tree['test_r2']))) 


#    Predict on test data - metrics results and plotting

y_pred_tree = tree.predict(X_test_scaled)


print(mean_squared_error(Y_test, y_pred_tree))
print(mean_absolute_error(Y_test, y_pred_tree))
print(r2_score(Y_test, y_pred_tree))

plt.plot(Y_test, 'b.')
plt.plot(y_pred_tree, 'r.')
plt.show()





###################################################  Random Forest   ##################################################
#           model defining
print('\n\n******** Random Forest *******')

model_RF = RandomForestRegressor(n_estimators = 30, random_state=30)
model_RF.fit(X_train_scaled, Y_train)



cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=1)
n_scores_rf = cross_validate(model_RF, X_train_scaled , Y_train, scoring=('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'), cv=cv, n_jobs=-1)
print('MSE: %.9f (%.9f)' % (mean(n_scores_rf['test_neg_mean_squared_error']), std(n_scores_rf['test_neg_mean_squared_error'])))
print('MAE: %.9f (%.9f)' % (mean(n_scores_rf['test_neg_mean_absolute_error']), std(n_scores_rf['test_neg_mean_absolute_error'])))  
print('R2: %.9f (%.9f)' % (mean(n_scores_rf['test_r2']), std(n_scores_rf['test_r2']))) 




y_pred_RF = model_RF.predict(X_test_scaled)


print(mean_squared_error(Y_test, y_pred_RF))
print(mean_absolute_error(Y_test, y_pred_RF))
print(r2_score(Y_test, y_pred_RF))

plt.plot(Y_test, 'b.')
plt.plot(y_pred_RF, 'r.')
plt.show()


#Feature ranking...
feature_list = list(X_train.columns)
feature_imp = pd.Series(model_RF.feature_importances_, index=feature_list).sort_values(ascending=False)
print('Random Forest feature importance: ',feature_imp)




###################################################  XGBoost   ##################################################
#           model defining
print('\n\n******** XGBoost regressor *******')

xgb_model=xgboost.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
xgb_model.fit(X_train_scaled,Y_train)




cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=1)
n_scores_xgb = cross_validate(xgb_model, X_train_scaled , Y_train, scoring=('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'), cv=cv, n_jobs=-1)
print('MSE: %.9f (%.9f)' % (mean(n_scores_xgb['test_neg_mean_squared_error']), std(n_scores_xgb['test_neg_mean_squared_error'])))
print('MAE: %.9f (%.9f)' % (mean(n_scores_xgb['test_neg_mean_absolute_error']), std(n_scores_xgb['test_neg_mean_absolute_error'])))  
print('R2: %.9f (%.9f)' % (mean(n_scores_xgb['test_r2']), std(n_scores_xgb['test_r2']))) 




#    Predict on test data - metrics results and plotting
y_pred_xgb = xgb_model.predict(X_test_scaled)

print(mean_squared_error(Y_test, y_pred_xgb))
print(mean_absolute_error(Y_test, y_pred_xgb))
print(r2_score(Y_test, y_pred_xgb))

plt.plot(Y_test, 'b.')
plt.plot(y_pred_xgb, 'r.')
plt.show()



###################################################  ensemble adaboost   ##################################################
#           model defining
print('\n\n******** ensemble adaboost regressor *******')


ada_model = ensemble.AdaBoostRegressor()
ada_model.fit(X_train_scaled, Y_train)


cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=1)
n_scores_ada = cross_validate(ada_model, X_train_scaled , Y_train, scoring=('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'), cv=cv, n_jobs=-1)
print('MSE: %.9f (%.9f)' % (mean(n_scores_ada['test_neg_mean_squared_error']), std(n_scores_ada['test_neg_mean_squared_error'])))
print('MAE: %.9f (%.9f)' % (mean(n_scores_ada['test_neg_mean_absolute_error']), std(n_scores_ada['test_neg_mean_absolute_error'])))  
print('R2: %.9f (%.9f)' % (mean(n_scores_ada['test_r2']), std(n_scores_ada['test_r2']))) 



 #    Predict on test data - metrics results and plotting
y_pred_ada = ada_model.predict(X_test_scaled)

print(mean_squared_error(Y_test, y_pred_ada))
print(mean_absolute_error(Y_test, y_pred_ada))
print(r2_score(Y_test, y_pred_ada))

plt.plot(Y_test, 'b.')
plt.plot(y_pred_ada, 'r.')
plt.show()      
        
      
   ###################################################  KNN regressor   ##################################################
   #           model defining
print('\n\n******** KNN regressor *******')    
        

knn_model = neighbors.KNeighborsRegressor()
knn_model.fit(X_train_scaled, Y_train)



cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=1)
n_scores_knn = cross_validate(knn_model, X_train_scaled , Y_train, scoring=('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'), cv=cv, n_jobs=-1)
print('MSE: %.9f (%.9f)' % (mean(n_scores_knn['test_neg_mean_squared_error']), std(n_scores_knn['test_neg_mean_squared_error'])))
print('MAE: %.9f (%.9f)' % (mean(n_scores_knn['test_neg_mean_absolute_error']), std(n_scores_knn['test_neg_mean_absolute_error'])))  
print('R2: %.9f (%.9f)' % (mean(n_scores_knn['test_r2']), std(n_scores_knn['test_r2']))) 


 #    Predict on test data - metrics results and plotting
y_pred_knn = knn_model.predict(X_test_scaled)

print(mean_squared_error(Y_test, y_pred_knn))
print(mean_absolute_error(Y_test, y_pred_knn))
print(r2_score(Y_test, y_pred_knn))

plt.plot(Y_test, 'b.')
plt.plot(y_pred_knn, 'r.')
plt.show()  





