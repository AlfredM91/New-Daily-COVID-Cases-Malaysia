# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 02:04:47 2022

@author: aaron
"""

from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras import Sequential, Input
from tensorflow.keras.utils import plot_model

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import os

#%% Constants

BEST_MODEL_PATH = os.path.join(os.getcwd(),'models','best_model.h5')
MMS_PATH = os.path.join(os.getcwd(),'models','mms.pkl')
SS_PATH = os.path.join(os.getcwd(),'models','ss.pkl')
PLOT_PATH = os.path.join(os.getcwd(),'statics','model.png')
LOGS_PATH = os.path.join(os.getcwd(),'logs',datetime.now().
                         strftime('%Y%m%d-%H%M%S'))
MODEL_PATH = os.path.join(os.getcwd(),'models','model.h5')
TIME_SERIES_ACTUAL_PREDICTED_PATH = os.path.join(os.getcwd(),'statics',
                                                 'time_series_actual_predicted.png')

#%% Classes

class ExploratoryDataAnalysis:
    def data_inspection(self,df):
        pd.set_option('display.max_columns',None)
        print(df.describe(include='all').T)
        print(df.info())
        print(df.isna().sum())
        print(df.isnull().sum())
        print(df.duplicated().sum())
    
    def min_max_scaler(self,X):
        mms = MinMaxScaler()    
        
        if X.ndim == 1:
            X = mms.fit_transform(np.expand_dims(X,axis=-1))
        else:    
            X = mms.fit_transform(X)
            
        with open(MMS_PATH,'wb') as file:
            pickle.dump(mms,file)
        
        return X
    
    def timeseries_features(self,df_train,df_test,df_concat,target,ss_or_mms,
                            win_size=60):
        
        df_train = df_train[target] 
        
        if ss_or_mms == 'mms':
            mms = MinMaxScaler()
            df_train = mms.fit_transform(np.expand_dims(df_train,axis=-1))
            
            MMS_TRAIN_PATH = os.path.join(os.getcwd(),'models','mms_train.pkl')
            
            with open(MMS_TRAIN_PATH,'wb') as file:
                pickle.dump(mms,file)
        
        elif ss_or_mms =='ss':
            ss = StandardScaler()
            df_train = ss.fit_transform(np.expand_dims(df_train,axis=-1))
            
            SS_TRAIN_PATH = os.path.join(os.getcwd(),'models','ss_train.pkl')
            
            with open(SS_TRAIN_PATH,'wb') as file:
                pickle.dump(ss,file)
        else:
            'Please put either "ss" or "mms" for the ss_or_mms argument'
                
        X_train = []
        y_train = []
        
        for i in range(win_size,len(df_train)):
            X_train.append(df_train[i-win_size:i])
            y_train.append(df_train[i])
        
        X_train  = np.array(X_train)
        y_train = np.array(y_train)
        
        length_days = win_size+len(df_test)
        df_test = df_concat[-length_days:]
        
        if ss_or_mms == 'mms':
            df_test = mms.transform(np.expand_dims(df_test,axis=-1))
    
            MMS_TEST_PATH = os.path.join(os.getcwd(),'models','mms_test.pkl')
    
            with open(MMS_TEST_PATH,'wb') as file:
                pickle.dump(mms,file)
        
        elif ss_or_mms =='ss':
            df_test = ss.transform(np.expand_dims(df_test,axis=-1))
    
            SS_TEST_PATH = os.path.join(os.getcwd(),'models','ss_test.pkl')
    
            with open(SS_TEST_PATH,'wb') as file:
                pickle.dump(ss,file)
        else:
            'Please put either "ss" or "mms" for the ss_or_mms argument'
        
        X_test = []
        y_test = []
        
        for i in range(win_size,len(df_test)):
            X_test.append(df_test[i-win_size:i])
            y_test.append(df_test[i])
        
        X_test  = np.array(X_test)
        y_test = np.array(y_test)
        
        return X_train, X_test, y_train, y_test
    
class ModelDevelopment:
    def dl_lstm_timeseries_model(self,X_train,y_train,activation_output='relu',
                        dense_node=128,dropout_rate=0.3,no_of_target=1,
                        LSTM_layer=3):
                
        model = Sequential()   
        model.add(Input(shape=np.shape(X_train)[1:]))
        for layer in range(LSTM_layer-1):
            model.add(LSTM(dense_node,return_sequences=(True)))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        model.add(LSTM(dense_node))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
        model.add(Dense(no_of_target,activation=activation_output))
        model.summary()
        
        plot_model(model,show_layer_names=(True),show_shapes=True,
                   to_file=PLOT_PATH)
        
        return model
    
    def dl_model_compilation(self,model,cat_or_con='custom',
                             loss='mse',metrics='mse'):
        
        if cat_or_con=='cat':
            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics='acc')
        elif cat_or_con=='con':
            model.compile(optimizer='adam',
                          loss='mse',
                          metrics='mse')
        elif cat_or_con=='custom':
            model.compile(optimizer='adam',
                          loss=loss,
                          metrics=metrics)
        else:
            print('Please enter either ''cat'' or ''con'' in the second argument')

    def dl_model_training(self,X_train,X_test,y_train,y_test,model,epochs=10,
                       monitor='val_loss',use_early_callback=False,
                       use_model_checkpoint=False):
        
        tensorboard_callback = TensorBoard(log_dir=LOGS_PATH,histogram_freq=1)
        callbacks = [tensorboard_callback]
        
        if use_early_callback==True:
            if epochs <= 30:
                early_callback = EarlyStopping(monitor=monitor,patience=3)
                callbacks.extend([early_callback])
            else:
                early_callback = EarlyStopping(monitor=monitor,
                                               patience=np.floor(0.1*epochs))
                callbacks.extend([early_callback])
        elif use_early_callback==False:
            early_callback=None
        else:
            print('Please put only True or False for use_early_callback argument')
        
        if monitor=='val_acc':
            mode='max'
        elif monitor=='val_loss':
            mode='min'
        else:
            mode='auto'
        
        if use_model_checkpoint==True:
            model_checkpoint = ModelCheckpoint(BEST_MODEL_PATH, monitor=monitor,
                                               save_best_only=(True),
                                               mode=mode,verbose=1)
            callbacks.extend([model_checkpoint])
        elif use_model_checkpoint==False:
            model_checkpoint=None
        else:
            print('Please put only True or False for use_model_checkpoint argument')
        
        hist = model.fit(X_train,y_train,epochs=epochs,verbose=1,
                        validation_data=(X_test,y_test),
                         callbacks=callbacks)
        
        model.save(MODEL_PATH)
        
        return hist
        
class ModelEvaluation:
    def dl_plot_hist(self,hist):
        
        keys = list(hist.history.keys())
        len(keys)

        for i in range(int(len(keys)/2)):
            plt.figure()
            plt.plot(hist.history[keys[i]])
            plt.plot(hist.history[keys[i+int(len(keys)/2)]])
            plt.xlabel('Epoch')
            plt.legend(['Training '+keys[i],'Validation '+keys[i]])
            plt.show()
            
    def timeseries_actual_predicted(self,df_train,df_test,df_concat,target,
                                    win_size,X_test,y_test,model,ss_or_mms):
        
        df_train = df_train[target] 
        
        length_days = win_size+len(df_test)
        df_test = df_concat[-length_days:]
        
        
        if ss_or_mms == 'ss':
            SS_TEST_PATH = os.path.join(os.getcwd(),'models','ss_test.pkl')
            with open(SS_TEST_PATH,'rb') as file:
                ss = pickle.load(file)
            plt.figure()
            plt.plot(ss.inverse_transform(y_test),color='red')
            plt.plot(ss.inverse_transform(model.predict(
                np.reshape(X_test,X_test.shape[:-1]))),color='blue')
            plt.xlabel(pd.DataFrame(X_test).columns[0])
            plt.ylabel(pd.DataFrame(y_test).columns[0])
            plt.legend(['Actual','Predicted'])
            plt.show()
        if ss_or_mms == 'mms':
            MMS_TEST_PATH = os.path.join(os.getcwd(),'models','mms_test.pkl')
            with open(MMS_TEST_PATH,'rb') as file:
                mms = pickle.load(file)
            y_pred = model.predict(X_test)
            plt.figure()
            plt.plot(mms.inverse_transform(y_test),color='red')
            plt.plot(mms.inverse_transform(y_pred),color='blue')
            plt.xlabel('Days')
            plt.ylabel(target)
            plt.legend(['Actual','Predicted'])
            plt.savefig(TIME_SERIES_ACTUAL_PREDICTED_PATH)
            plt.show()
        else:
            print('Please put either "ss" or "mms" for the ss_or_mms argument')