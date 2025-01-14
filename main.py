import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from tabpfn.scripts.decision_boundary import DecisionBoundaryDisplay
from tabpfn import TabPFNClassifier

#Connect to Google Drive
from google.colab import drive
drive.mount('/content/drive')

#Load the data sets (pre)
tabpfn_train_df = pd.read_csv('train.csv')
tabpfn_test_df = pd.read_csv('test.csv')
xgboost_train_df = pd.read_csv('train.csv')
xgboost_test_df = pd.read_csv('test.csv')
FINAL_df = pd.read_csv('final.csv')

#Remove the row label column in the training and testing data sets for TabPFN
tabpfn_train_df.drop(tabpfn_train_df.columns[0], axis=1, inplace=True)
tabpfn_test_copy = tabpfn_test_df.copy()
tabpfn_test_df.drop(tabpfn_test_df.columns[0], axis=1, inplace=True)

#Remove the row label column in the training and testing data sets for XGBoost
xgboost_train_df.drop(xgboost_train_df.columns[0], axis=1, inplace=True)
xgboost_test_copy = xgboost_test_df.copy()
xgboost_test_df.drop(xgboost_test_df.columns[0], axis=1, inplace=True)

#Now for XGBoost side
#assign x and y
xgboost_y_train = xgboost_train_df['Score']
xgboost_train_df.pop('Score')
xgboost_test_df.pop('Score')

# Create and configure the XGBoost regressor
xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', seed=42)

# Define the parameter grid for tuning
param_grid = {
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.5, 0.6, 0.7],
    'colsample_bytree': [0.5, 0.6, 0.7],
    'learning_rate': [0.01, 0.05, 0.1],
}

# Set up GridSearchCV with the XGBoost regressor and the parameter grid
grid_search = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=1)

# Fit the grid search to the training data
xgboost_best_params = grid_search.fit(xgboost_train_df, xgboost_y_train)

#Fit the best parameters onto the test set
best_xgb_regressor = grid_search.best_estimator_
best_xgb_regressor.fit(xgboost_train_df, xgboost_y_train)
xgboost_Final_pred = best_xgb_regressor.predict(xgboost_test_df)

#TabPFN prediction First.

#adjust the training data via selecting the amount of columns required for
#training features, and for target.  Not deleting.
tabpfn_train_y = pd.cut(tabpfn_train_df['Score'], bins=10, labels=['3.45','3.5',
                                                          '3.55','3.60',
                                                           '3.65','3.70',
                                                           '3.75','3.80',
                                                           '3.90', '3.95'])
tabpfn_train_y = tabpfn_train_y.astype('category')
tabpfn_train_df = tabpfn_train_df.iloc[:,:-1]
scaler = StandardScaler()
tabpfn_train_df_scaled = scaler.fit_transform(tabpfn_train_df)

#adjust test dataframe by selecting features, and leaving out target
tabpfn_test_df = tabpfn_test_df.iloc[:,:-1]
tabpfn_test_df_scaled = scaler.transform(tabpfn_test_df)

#Define Training
classifier = TabPFNClassifier(device='cuda', N_ensemble_configurations=4)

#Begin training
start = time.time()
classifier.fit(tabpfn_train_df_scaled, tabpfn_train_y)

#Run Predict on test set
tabpfn_Final_pred, tabpfn_p_eval = classifier.predict(tabpfn_test_df_scaled, return_winning_probability=True)

#Now Average the two together, and place them into the Final dataframe for review
tabpfn_Final_pred = tabpfn_Final_pred.astype('float')
Final_y = np.mean([xgboost_Final_pred, tabpfn_Final_pred], axis = 0)
FINAL_df['Score'] = Final_y

#Export Final CSV into Colab Testing Folder - Change to name you want
FINAL_df.to_csv('final.csv',index=False)
