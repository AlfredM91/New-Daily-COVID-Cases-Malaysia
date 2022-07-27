# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 02:04:22 2022

@author: aaron
"""

from covid_module import ExploratoryDataAnalysis,ModelDevelopment,ModelEvaluation
import pandas as pd
import numpy as np
import os

#%% Constants

CSV_PATH_TRAIN = os.path.join(os.getcwd(),'dataset','cases_malaysia_train.csv')
CSV_PATH_TEST = os.path.join(os.getcwd(),'dataset','cases_malaysia_test.csv')

#%% Step 1) Data Loading

df_train = pd.read_csv(CSV_PATH_TRAIN)
df_test = pd.read_csv(CSV_PATH_TEST)
target = 'cases_new'
df_concat = pd.concat((df_train[target],df_test[target]))

#%% Step 2) Data Inspection/Data Visualization

eda = ExploratoryDataAnalysis()
eda.data_inspection(df_concat) # There's 1 NaN, and data type is in object

np.unique(df_train[target]) # There's blanks space and ? in the data
np.unique(df_test[target]) # There's 1 NaN

#%% Step 3) Data Cleaning

df_train[target].replace(' ',np.nan,inplace=True)
df_train[target].replace('?',np.nan,inplace=True)

df_train['date'] = pd.to_datetime(df_train['date'], format='%d/%m/%Y')
df_train = df_train.set_index('date')
df_test['date'] = pd.to_datetime(df_test['date'], format='%d/%m/%Y')
df_test = df_test.set_index('date')

df_train[target].interpolate(method='pad',inplace=True)
df_test[target].interpolate(method='pad',inplace=True)

df_train[target].isna().sum()
df_test[target].isna().sum()

df_concat = pd.concat((df_train[target],df_test[target]))

#%% Step 4) Features Selection

X_train, X_test, y_train, y_test = eda.timeseries_features(df_train, df_test,
                                                           df_concat,target,
                                                           'mms',win_size=30)

#%% Step 5) Data Preprocessing - You do not train test split time series data

#%% Model Development

md = ModelDevelopment() 

model = md.dl_lstm_timeseries_model(X_train, y_train,dense_node=64,
                                    dropout_rate=0,LSTM_layer=4)

md.dl_model_compilation(model,metrics=['mse','mean_absolute_percentage_error'])

#%% Model Training

hist = md.dl_model_training(X_train, X_test, y_train, y_test, model, epochs=200,
                            monitor='val_mean_absolute_percentage_error',
                            use_model_checkpoint=(True))

#%% Model Evaluation

me = ModelEvaluation()

me.dl_plot_hist(hist)

me.timeseries_actual_predicted(df_train,df_test,df_concat,target,30,
                                X_test,y_test,model,'mms')












