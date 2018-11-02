import glob 
import os
import re
import math
import json

import array
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras
from keras import utils
from keras.layers import Input, Dense, Concatenate, BatchNormalization, LeakyReLU, Lambda, Dropout
from keras.losses import categorical_crossentropy, mean_squared_error 
from keras.activations import relu, elu, selu, softmax, tanh
from keras.models import Model, model_from_json, load_model
from keras import losses, optimizers
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau, ModelCheckpoint, History, Callback
from keras.regularizers import l1,l2
import keras.backend as K

import talos


def interpolation_model(x_train,y_train,x_val,y_val,params):
    """
    Keras model for the Neural Network, used to scan the hyperparameter space by Talos
    Inputs :
        - x_train = training inputs (aka : mH,mA)
        - y_train = training outputs (aka : rho distribution -> 6 bins content)
        - x_test = test inputs
        - y_test = test outputs
        - params = dict of parameters for the talos scan
    Outputs :
        - out =  predicted outputs from network
        - model = fitted models with weights
    """     
    # Design network #
    inputs = Input(shape=(x_train.shape[1],),name='input')
    L1 = Dense(params['first_neuron'],
               activation=params('first_activation'),
               name='L1')(inputs)
    L2 = Dense(params['second_neuron'],
               activation=params['second_activation'],
               name='L2')(L1)
    OUT_1 = Dense(1,activation=params['output_activation'],name='OUT_1')(L2)
    OUT_2 = Dense(1,activation=params['output_activation'],name='OUT_2')(L2)
    OUT_3 = Dense(1,activation=params['output_activation'],name='OUT_3')(L2)
    OUT_4 = Dense(1,activation=params['output_activation'],name='OUT_4')(L2)
    OUT_5 = Dense(1,activation=params['output_activation'],name='OUT_5')(L2)
    OUT_6 = Dense(1,activation=params['output_activation'],name='OUT_6')(L2)

    # Define model #    
    model = Model(inputs=[inputs], outputs=[OUT_1,OUT_2,OUT_3,OUT_4,OUT_5,OUT_6])
    utils.print_summary(model=model)

    # Compile #
    adam = optimizers.Adam(lr=params('lr'), beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False, clipvalue=0.5)
    model.compile(optimizer=adam,
                  loss={'OUT_1':'binary_crossentropy',
                        'OUT_2':'binary_crossentropy',                        
                        'OUT_3':'binary_crossentropy',                        
                        'OUT_4':'binary_crossentropy',                        
                        'OUT_5':'binary_crossentropy',                        
                        'OUT_6':'binary_crossentropy',                        
                       },
                  metrics=['accuracy']) 

    # Fit #
    out = model.fit({'inputs':x_train},
                    {'OUT_1':y_train[0],
                     'OUT_2':y_train[1],
                     'OUT_3':y_train[2],
                     'OUT_4':y_train[3],
                     'OUT_5':y_train[4],
                     'OUT_6':y_train[5],
                    },
                    sample_weight=None,  # TO BE CHANGED
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    verbose=2,
                    validation_data=({'inputs':x_test},
                                     {'OUT_1':y_test[0],
                                      'OUT_2':y_test[1],
                                      'OUT_3':y_test[2],
                                      'OUT_4':y_test[3],
                                      'OUT_5':y_test[4],
                                      'OUT_6':y_test[5],
                                     }#,
                                     ########### -> Must put test weight
                                    ))
                    
    return out,model

def HyperScan(x,y):  # MUST ADD THE WEIGHTS
    """ 
    Performs the split  and preprocessing of data and scan for hyperparameters 
    Inputs :
        - x = inputs (aka : mH,mA)
        - y = outputs (aka : rho distribution -> 6 bins content)
    """ 
    # Splitting and preprocessing
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4) # MUST ADD WEIGHTS

    scaler = preprocessing.StandardScaler().fit(x_train,axis=0)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    # Talos hyperscan #
    p = {
            'lr' : (0.01,0.1,10),
            'first_neuron' : [10,20,30],
            'first_activation' : [relu,selu,tanh],
            'second_neuron' : [10,20,30],
            'second_activation' : [relu,selu,tanh],
            'output_activation' : [relu,selu,tanh],
            'epochs' : [50],
            'batch_size' : [1000,10000,100000],
        }
    
    h = talos.Scan(x=x_train,
                   y=y_train,
                   params=p,
                   dataset_name='first test',
                   experiment_no='1',
                   model=interpolation_model,
                   x_val=x_test,
                   y_val=y_test,
                   val_split=0.3
                  )

    
    
