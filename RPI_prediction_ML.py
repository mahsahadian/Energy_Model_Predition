import matplotlib.pyplot as plt
import numpy as np
from numpy import mean, std
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn import neighbors, linear_model, ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.dummy import DummyRegressor
import xgboost
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.wrappers.scikit_learn import KerasRegressor

# Read the data
df = pd.read_csv("C:/Users/zia_s/OneDrive/Desktop/Mahsa/Machine learning paper/mahsadataset3_training.csv")
X_train = df.drop(['Number', 'Running_time(s)', 'inst_power(watt)'], axis = 1)  
Y_train = df['inst_power(watt)']
df2 = pd.read_csv("C:/Users/zia_s/OneDrive/Desktop/Mahsa/Machine learning paper/mahsadataset3_test.csv")
X_test = df2.drop(['Number', 'Running_time(s)', 'inst_power(watt)'], axis = 1)  
Y_test = df2['inst_power(watt)']

# Normalizing the data
scaler=MinMaxScaler()
scaler.fit(X_train)
scaler.fit(X_test)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model
print('\n\n******** Linear Regression *******')
lr_model = linear_model.LinearRegression()
lr_model.fit(X_train_scaled, Y_train)

# Predict Using Cross Validate
cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)
n_scores_lr = cross_validate(lr_model, X_train_scaled , Y_train, scoring=('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'), cv=cv, n_jobs=-1)

lr_mean_squared_error=pd.Series(-n_scores_lr['test_neg_mean_squared_error'])
lr_mean_absolute_error=pd.Series(-n_scores_lr['test_neg_mean_absolute_error'])
lr_r2_error=pd.Series(n_scores_lr['test_r2'])

print('MSE: %.9f (%.9f)' % (mean(-n_scores_lr['test_neg_mean_squared_error']), std(-n_scores_lr['test_neg_mean_squared_error'])))
print('MAE: %.9f (%.9f)' % (mean(-n_scores_lr['test_neg_mean_absolute_error']), std(-n_scores_lr['test_neg_mean_absolute_error'])))  
print('R2: %.9f (%.9f)' % (mean(n_scores_lr['test_r2']), std(n_scores_lr['test_r2']))) 

lr_mean_squared_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("mean_squared_error")
plt.show()

lr_mean_absolute_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("mean_absolute_error")
plt.show()

lr_r2_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("R2")
plt.show()

# Predict on distinct test data
y_pred_lr = lr_model.predict(X_test_scaled)

print(mean_squared_error(Y_test, y_pred_lr))
print(mean_absolute_error(Y_test, y_pred_lr))
print(r2_score(Y_test, y_pred_lr))


# Feature Importance
importance = lr_model.coef_
for i,v in enumerate(importance):
 print('DD: %0d, Score: %.9f' % (i,v))

# Define the model
print('\n\n******** Decision tree *******')
tree = DecisionTreeRegressor()
tree.fit(X_train_scaled, Y_train)

# Hyperparameter search space
param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

# Randomized search
random_search = RandomizedSearchCV(
    estimator=tree,
    param_distributions=param_grid,
    cv=3,
    n_iter=10,
    scoring='neg_mean_squared_error',
    random_state=42
)

# Fit the random search
random_search.fit(X_train_scaled, Y_train)

# Best hyperparameters
print("Best Hyperparameters:")
print(random_search.best_params_)

# Decision TREE with best hyperparameters
best_tree = random_search.best_estimator_

# Predict Using Cross Validate
cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)
n_scores_tree = cross_validate(best_tree, X_train_scaled , Y_train, scoring=('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'), cv=cv, n_jobs=-1)

tree_mean_squared_error=pd.Series(-n_scores_tree['test_neg_mean_squared_error'])
tree_mean_absolute_error=pd.Series(-n_scores_tree['test_neg_mean_absolute_error'])
tree_r2_error=pd.Series(n_scores_tree['test_r2'])

print('MSE: %.9f (%.9f)' % (mean(-n_scores_tree['test_neg_mean_squared_error']), std(-n_scores_tree['test_neg_mean_squared_error'])))
print('MAE: %.9f (%.9f)' % (mean(-n_scores_tree['test_neg_mean_absolute_error']), std(-n_scores_tree['test_neg_mean_absolute_error'])))  
print('R2: %.9f (%.9f)' % (mean(n_scores_tree['test_r2']), std(n_scores_tree['test_r2']))) 


tree_mean_squared_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("mean_squared_error")
plt.show()

tree_mean_absolute_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("mean_absolute_error")
plt.show()

tree_r2_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("R2")
plt.show()

# Predict on distinct test data
y_pred_tree = best_tree.predict(X_test_scaled)

print(mean_squared_error(Y_test, y_pred_tree))
print(mean_absolute_error(Y_test, y_pred_tree))
print(r2_score(Y_test, y_pred_tree))

#Feature ranking...
feature2_list = list(X_train.columns)
feature2_imp = pd.Series(best_tree.feature_importances_, index=feature2_list).sort_values(ascending=False)
print('Decision tree feature importance: ',feature2_imp)

# Define the model
print('\n\n******** Random Forest *******')
model_RF = RandomForestRegressor(n_estimators = 30, random_state=30)
model_RF.fit(X_train_scaled, Y_train)

# Hyperparameter search space
param_grid = {
    'n_estimators': [10, 30, 50, 100],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Randomized search
random_search = RandomizedSearchCV(
    estimator=model_RF,
    param_distributions=param_grid,
    cv=3,
    n_iter=10,
    scoring='neg_mean_squared_error',
    random_state=42
)

# Fit the random search
random_search.fit(X_train_scaled, Y_train)

# Best hyperparameters
print("Best Hyperparameters:")
print(random_search.best_params_)

# Random Forest with best hyperparameters
best_model_RF = random_search.best_estimator_

# Predict Using Cross Validate
cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)
n_scores_rf = cross_validate(best_model_RF, X_train_scaled , Y_train, scoring=('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'), cv=cv, n_jobs=-1)

rf_mean_squared_error=pd.Series(-n_scores_rf['test_neg_mean_squared_error'])
rf_mean_absolute_error=pd.Series(-n_scores_rf['test_neg_mean_absolute_error'])
rf_r2_error=pd.Series(n_scores_rf['test_r2'])

print('MSE: %.9f (%.9f)' % (mean(-n_scores_rf['test_neg_mean_squared_error']), std(-n_scores_rf['test_neg_mean_squared_error'])))
print('MAE: %.9f (%.9f)' % (mean(-n_scores_rf['test_neg_mean_absolute_error']), std(-n_scores_rf['test_neg_mean_absolute_error'])))  
print('R2: %.9f (%.9f)' % (mean(n_scores_rf['test_r2']), std(n_scores_rf['test_r2']))) 

rf_mean_squared_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("mean_squared_error")
plt.show()

rf_mean_absolute_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("mean_absolute_error")
plt.show()

rf_r2_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("R2")
plt.show()

# Predict on distinct test data
y_pred_RF = best_model_RF.predict(X_test_scaled)

print(mean_squared_error(Y_test, y_pred_RF))
print(mean_absolute_error(Y_test, y_pred_RF))
print(r2_score(Y_test, y_pred_RF))

#Feature ranking...
feature_list = list(X_train.columns)
feature_imp = pd.Series(best_model_RF.feature_importances_, index=feature_list).sort_values(ascending=False)
print('Random Forest feature importance: ',feature_imp)

# Define the model
print('\n\n******** XGBoost regressor *******')
xgb_model=xgboost.XGBRegressor(objective ='reg:linear', colsample_bytree = 1, learning_rate = 0.3, max_depth = 6, alpha = 0, n_estimators = 100)
xgb_model.fit(X_train_scaled,Y_train)

# Hyperparameter search space
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.001, 0.01, 0.1, 1.0],
    'max_depth': [3, 6, 9],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}

# Randomized search
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    cv=3,
    n_iter=10,
    scoring='neg_mean_squared_error',
    random_state=42
)

# Fit the random search
random_search.fit(X_train_scaled, Y_train)

# Best hyperparameters
print("Best Hyperparameters:")
print(random_search.best_params_)

# XGBoost with best hyperparameters
best_model_xgb = random_search.best_estimator_

cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)
n_scores_xgb = cross_validate(best_model_xgb, X_train_scaled , Y_train, scoring=('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'), cv=cv, n_jobs=-1)

xgb_mean_squared_error=pd.Series(-n_scores_xgb['test_neg_mean_squared_error'])
xgb_mean_absolute_error=pd.Series(-n_scores_xgb['test_neg_mean_absolute_error'])
xgb_r2_error=pd.Series(n_scores_xgb['test_r2'])

print('MSE: %.9f (%.9f)' % (mean(-n_scores_xgb['test_neg_mean_squared_error']), std(-n_scores_xgb['test_neg_mean_squared_error'])))
print('MAE: %.9f (%.9f)' % (mean(-n_scores_xgb['test_neg_mean_absolute_error']), std(-n_scores_xgb['test_neg_mean_absolute_error'])))  
print('R2: %.9f (%.9f)' % (mean(n_scores_xgb['test_r2']), std(n_scores_xgb['test_r2']))) 

xgb_mean_squared_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("mean_squared_error")
plt.show()

xgb_mean_absolute_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("mean_absolute_error")
plt.show()

xgb_r2_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("R2")
plt.show()

# Predict on distinct test data
y_pred_xgb = best_model_xgb.predict(X_test_scaled)

print(mean_squared_error(Y_test, y_pred_xgb))
print(mean_absolute_error(Y_test, y_pred_xgb))
print(r2_score(Y_test, y_pred_xgb))

# Define the model
print('\n\n******** ensemble adaboost regressor *******')
ada_model = ensemble.AdaBoostRegressor()
ada_model.fit(X_train_scaled, Y_train)

# Hyperparameter search space
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.001, 0.01, 0.1, 1.0],
    'loss': ['linear', 'square', 'exponential']
}

# Randomized search
random_search = RandomizedSearchCV(
    estimator=ada_model,
    param_distributions=param_grid,
    cv=3,
    n_iter=10,
    scoring='neg_mean_squared_error',
    random_state=42
)

# Fit the random search
random_search.fit(X_train_scaled, Y_train)

# Best hyperparameters
print("Best Hyperparameters:")
print(random_search.best_params_)

# ensemble adaboost with best hyperparameters
best_model_ada = random_search.best_estimator_

# Predict Using Cross Validate
cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)
n_scores_ada = cross_validate(best_model_ada, X_train_scaled , Y_train, scoring=('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'), cv=cv, n_jobs=-1)
ada_mean_squared_error=pd.Series(-n_scores_ada['test_neg_mean_squared_error'])
ada_mean_absolute_error=pd.Series(-n_scores_ada['test_neg_mean_absolute_error'])
ada_r2_error=pd.Series(n_scores_ada['test_r2'])

print('MSE: %.9f (%.9f)' % (mean(-n_scores_ada['test_neg_mean_squared_error']), std(-n_scores_ada['test_neg_mean_squared_error'])))
print('MAE: %.9f (%.9f)' % (mean(-n_scores_ada['test_neg_mean_absolute_error']), std(-n_scores_ada['test_neg_mean_absolute_error'])))  
print('R2: %.9f (%.9f)' % (mean(n_scores_ada['test_r2']), std(n_scores_ada['test_r2']))) 

ada_mean_squared_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("mean_squared_error")
plt.show()

ada_mean_absolute_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("mean_absolute_error")
plt.show()

ada_r2_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("R2")
plt.show()


# Predict on distinct test data
y_pred_ada = best_model_ada.predict(X_test_scaled)

print(mean_squared_error(Y_test, y_pred_ada))
print(mean_absolute_error(Y_test, y_pred_ada))
print(r2_score(Y_test, y_pred_ada))
    
# Define the model
print('\n\n******** KNN regressor *******')    
knn_model = neighbors.KNeighborsRegressor()
knn_model.fit(X_train_scaled, Y_train)

# Hyperparameter search space
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

# Randomized search
random_search = RandomizedSearchCV(
    estimator=knn_model,
    param_distributions=param_grid,
    cv=3,
    n_iter=10,
    scoring='neg_mean_squared_error',
    random_state=42
)

# Fit the random search
random_search.fit(X_train_scaled, Y_train)

# Best hyperparameters
print("Best Hyperparameters:")
print(random_search.best_params_)

# KNN regressor with best hyperparameters
best_model_knn = random_search.best_estimator_

# Predict Using Cross Validate
cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)
n_scores_knn = cross_validate(best_model_knn, X_train_scaled , Y_train, scoring=('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'), cv=cv, n_jobs=-1)
knn_mean_squared_error=pd.Series(-n_scores_knn['test_neg_mean_squared_error'])
knn_mean_absolute_error=pd.Series(-n_scores_knn['test_neg_mean_absolute_error'])
knn_r2_error=pd.Series(n_scores_knn['test_r2'])

print('MSE: %.9f (%.9f)' % (mean(-n_scores_knn['test_neg_mean_squared_error']), std(-n_scores_knn['test_neg_mean_squared_error'])))
print('MAE: %.9f (%.9f)' % (mean(-n_scores_knn['test_neg_mean_absolute_error']), std(-n_scores_knn['test_neg_mean_absolute_error'])))  
print('R2: %.9f (%.9f)' % (mean(n_scores_knn['test_r2']), std(n_scores_knn['test_r2']))) 

knn_mean_squared_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("mean_squared_error")
plt.show()

knn_mean_absolute_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("mean_absolute_error")
plt.show()

knn_r2_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("R2")
plt.show()

# Predict on distinct test data
y_pred_knn = best_model_knn.predict(X_test_scaled)

print(mean_squared_error(Y_test, y_pred_knn))
print(mean_absolute_error(Y_test, y_pred_knn))
print(r2_score(Y_test, y_pred_knn))

# Define the model
print('\n\n******** MLP *******')  
def create_model1(neurons=128, hidden_layers=1):
    model_mlp = Sequential()
    model_mlp.add(Dense(neurons, input_dim=10, activation='relu'))
    for _ in range(hidden_layers):
        model_mlp.add(Dense(neurons, activation='relu'))
    model_mlp.add(Dense(1, activation='linear'))
    model_mlp.compile(loss='mse', optimizer='adam')
    return model_mlp

# Hyperparameter search space
param_grid = {
    'neurons': [64, 128, 256],
    'hidden_layers': [1, 2, 3],
    'batch_size': [100, 500, 1000],
    'epochs': [50, 100, 200]
}

# Wrap the model creation function using KerasRegressor
keras_regressor1 = KerasRegressor(build_fn=create_model1)


# Randomized search
random_search = RandomizedSearchCV(
    estimator=keras_regressor1,
    param_distributions=param_grid,
    cv=3,
    n_iter=10,
    scoring='neg_mean_squared_error',
    verbose=1,
    random_state=42
)


# Fit the random search
random_search.fit(X_train_scaled, Y_train)

# Best hyperparameters
print("Best Hyperparameters:")
print(random_search.best_params_)


# MLP with best hyperparameters
best_model_mlp = random_search.best_estimator_


# Predict Using Cross Validate
cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)
n_scores_mlp = cross_validate(best_model_mlp, X_train_scaled , Y_train, scoring=('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'), cv=cv, n_jobs=-1)

mlp_mean_squared_error=pd.Series(-n_scores_mlp['test_neg_mean_squared_error'])
mlp_mean_absolute_error=pd.Series(-n_scores_mlp['test_neg_mean_absolute_error'])
mlp_r2_error=pd.Series(n_scores_mlp['test_r2'])

print('MSE: %.9f (%.9f)' % (mean(-n_scores_mlp['test_neg_mean_squared_error']), std(-n_scores_mlp['test_neg_mean_squared_error'])))
print('MAE: %.9f (%.9f)' % (mean(-n_scores_mlp['test_neg_mean_absolute_error']), std(-n_scores_mlp['test_neg_mean_absolute_error'])))  
print('R2: %.9f (%.9f)' % (mean(n_scores_mlp['test_r2']), std(n_scores_mlp['test_r2'])))

mlp_mean_squared_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("mean_squared_error")
plt.show()

mlp_mean_absolute_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("mean_absolute_error")
plt.show()

mlp_r2_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("R2")
plt.show()

# Predict on distinct test data

y_pred_mlp = best_model_mlp.predict(X_test_scaled)
y_pred_mlp = np.squeeze(y_pred_mlp)

print("Mean Squared Error:", mean_squared_error(Y_test, y_pred_mlp))
print("Mean Absolute Error:", mean_absolute_error(Y_test, y_pred_mlp))
print("R2 Score:", r2_score(Y_test, y_pred_mlp))



# Define the model
print('\n\n******** LSTM *******')    
def create_model2(neurons=100):
    model_lstm = Sequential()
    model_lstm.add(LSTM(neurons, input_shape=(10, 1)))
    model_lstm.add(Dense(1, activation='linear'))
    model_lstm.compile(loss='mse', optimizer='adam')
    return model_lstm


# Hyperparameter search space
param_grid = {
    'neurons': [50, 100, 200],
    'batch_size': [100, 500, 1000],
    'epochs': [50, 100, 200]
}


# Wrap the model creation function using KerasRegressor
keras_regressor2 = KerasRegressor(build_fn=create_model2)


# Randomized search
random_search = RandomizedSearchCV(
    estimator=keras_regressor2,
    param_distributions=param_grid,
    cv=3,
    n_iter=10,
    scoring='neg_mean_squared_error',
    verbose=1,
    random_state=42
)


# Fit the random search
random_search.fit(X_train_scaled, Y_train)

# Best hyperparameters
print("Best Hyperparameters:")
print(random_search.best_params_)

# LSTM with best hyperparameters
best_model_lstm = random_search.best_estimator_

# Predict Using Cross Validate
cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)
n_scores_lstm = cross_validate(best_model_lstm, X_train_scaled , Y_train, scoring=('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'), cv=cv, n_jobs=-1)

lstm_mean_squared_error=pd.Series(-n_scores_lstm['test_neg_mean_squared_error'])
lstm_mean_absolute_error=pd.Series(-n_scores_lstm['test_neg_mean_absolute_error'])
lstm_r2_error=pd.Series(n_scores_lstm['test_r2'])

print('MSE: %.9f (%.9f)' % (mean(-n_scores_lstm['test_neg_mean_squared_error']), std(-n_scores_lstm['test_neg_mean_squared_error'])))
print('MAE: %.9f (%.9f)' % (mean(-n_scores_lstm['test_neg_mean_absolute_error']), std(-n_scores_lstm['test_neg_mean_absolute_error'])))  
print('R2: %.9f (%.9f)' % (mean(n_scores_lstm['test_r2']), std(n_scores_lstm['test_r2']))) 

lstm_mean_squared_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("mean_squared_error")
plt.show()

lstm_mean_absolute_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("mean_absolute_error")
plt.show()

lstm_r2_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("R2")
plt.show()

# Predict on distinct test data
y_pred_lstm = best_model_lstm.predict(X_test_scaled)
y_pred_lstm = np.squeeze(y_pred_lstm)

print("Mean Squared Error:", mean_squared_error(Y_test, y_pred_lstm))
print("Mean Absolute Error:", mean_absolute_error(Y_test, y_pred_lstm))
print("R2 Score:", r2_score(Y_test, y_pred_lstm))

# Define the model
print('\n\n******** Dummy Regressor *******')
dummy_regressor = DummyRegressor(strategy='constant', constant=0.0007)
dummy_regressor.fit(X_train_scaled, Y_train)

# Predict Using Cross Validate
cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)
n_scores_dm = cross_validate(dummy_regressor, X_train_scaled , Y_train, scoring=('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'), cv=cv, n_jobs=-1)

dm_mean_squared_error=pd.Series(-n_scores_dm['test_neg_mean_squared_error'])
dm_mean_absolute_error=pd.Series(-n_scores_dm['test_neg_mean_absolute_error'])
dm_r2_error=pd.Series(n_scores_dm['test_r2'])

print('MSE: %.9f (%.9f)' % (mean(-n_scores_dm['test_neg_mean_squared_error']), std(-n_scores_dm['test_neg_mean_squared_error'])))
print('MAE: %.9f (%.9f)' % (mean(-n_scores_dm['test_neg_mean_absolute_error']), std(-n_scores_dm['test_neg_mean_absolute_error'])))  
print('R2: %.9f (%.9f)' % (mean(n_scores_dm['test_r2']), std(n_scores_dm['test_r2']))) 

dm_mean_squared_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("mean_squared_error")
plt.show()

dm_mean_absolute_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("mean_absolute_error")
plt.show()

dm_r2_error.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("R2")
plt.show()

# Predict on distinct test data
y_pred_dm = dummy_regressor.predict(X_test_scaled)
y_pred_dm = np.squeeze(y_pred_dm)

print("Mean Squared Error:", mean_squared_error(Y_test, y_pred_dm))
print("Mean Absolute Error:", mean_absolute_error(Y_test, y_pred_dm))
print("R2 Score:", r2_score(Y_test, y_pred_dm))




# Models and baseline Cross Validate results comparison

mean_squared_error_df=pd.concat([dm_mean_squared_error,lr_mean_squared_error,tree_mean_squared_error,rf_mean_squared_error,xgb_mean_squared_error,ada_mean_squared_error,knn_mean_squared_error,lstm_mean_squared_error,mlp_mean_squared_error], axis=1)
mean_absolute_error_df=pd.concat([dm_mean_absolute_error,lr_mean_absolute_error,tree_mean_absolute_error,rf_mean_absolute_error,xgb_mean_absolute_error,ada_mean_absolute_error,knn_mean_absolute_error,lstm_mean_absolute_error,mlp_mean_absolute_error], axis=1)
r2_error_df=pd.concat([dm_r2_error,lr_r2_error,tree_r2_error,rf_r2_error,xgb_r2_error,ada_r2_error,knn_r2_error,lstm_r2_error,mlp_r2_error], axis=1)

mean_squared_error_df.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("mean_squared_error")
plt.show()

mean_absolute_error_df.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("mean_absolute__error")
plt.show()

r2_error_df.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(loc="best")
plt.xlabel("r2__error")
plt.show()

# Models and baseline distinct test data results comparison


