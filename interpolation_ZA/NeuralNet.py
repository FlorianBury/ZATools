import glob 
import os
import re
import math
import sys
import json

import array
import numpy as np


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
from talos import Scan, Reporting, Predict, Evaluate

import matplotlib.pyplot as plt

def interpolation_model(x_train,y_train,x_val,y_val,params):
    """
    Keras model for the Neural Network, used to scan the hyperparameter space by Talos
    Inputs :
        - x_train = training inputs (aka : mH,mA)
        - y_train = training outputs (aka : rho distribution -> 6 bins content)
        - x_val = test inputs
        - y_val = test outputs
        - params = dict of parameters for the talos scan
    Outputs :
        - out =  predicted outputs from network
        - model = fitted models with weights
    """     
    # Design network #
    inputs = Input(shape=(x_train.shape[1],),name='inputs')
    L1 = Dense(params['first_neuron'],
               activation=params['first_activation'],
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
    #utils.print_summary(model=model) #used to print model

    # Compile #
    adam = optimizers.Adam(lr=params['lr'], beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False, clipvalue=0.5)
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
                    {'OUT_1':y_train[:,0],
                     'OUT_2':y_train[:,1],
                     'OUT_3':y_train[:,2],
                     'OUT_4':y_train[:,3],
                     'OUT_5':y_train[:,4],
                     'OUT_6':y_train[:,5],
                    },
                    sample_weight=None,  # TODO TO BE CHANGED
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    verbose=2,
                    validation_data=({'inputs':x_val},
                                     {'OUT_1':y_val[:,0],
                                      'OUT_2':y_val[:,1],
                                      'OUT_3':y_val[:,2],
                                      'OUT_4':y_val[:,3],
                                      'OUT_5':y_val[:,4],
                                      'OUT_6':y_val[:,5],
                                     }#,
                                     ########### -> TODO Must put test weight
                                    ))
                    
    return out,model

def HyperScan(x,y,name):  # MUST ADD THE WEIGHTS
    """ 
    Performs the split  and preprocessing of data and scan for hyperparameters 
    Inputs :
        - x = inputs (aka : mH,mA)
        - y = outputs (aka : rho distribution -> 6 bins content)
    """ 
   
    # Talos hyperscan #
    p = {
            'lr' : (0.001,0.1,1),
            'first_neuron' : [20],
            'first_activation' : [relu],
            'second_neuron' : [20],
            'second_activation' : [relu],
            'output_activation' : [relu],
            'epochs' : [100],
            'batch_size' : [5]
        }
    h = Scan(  x=x,
               y=y,
               params=p,
               dataset_name=name,
               model=interpolation_model,
               val_split=0.2,
               reduction_metric='val_loss',
               last_epoch_value=True,
               print_params=True
            )
            # returns the results dataframe
    print
    print ('='*80,end='\n\n')
    print ('Complete data ',end='\n\n')
    print (h.data)

    # returns the experiment configuration details
    print 
    print ('-'*80)
    print ('Details',end='\n\n')
    print (h.details)

    # returns the saved models (json)
    #h.saved_models
    # returns the saved model weights
    #h.saved_weights
    # returns x data
    #h.x
    # returns y data
    #h.y

    return h

def HyperReport(name,x,y):
    """
     See /home/ucl/cp3/fbury/.local/lib/python3.6/site-packages/talos/commands/reporting.py 
    """
    r = Reporting(name)     

    # returns the results dataframe
    print
    print ('='*80,end='\n\n')
    print ('Complete data after n_round = ',r.rounds(),':\n',r.data,end='\n\n')

    # Lowest val_loss #
    print 
    print ('-'*80)
    print ('Lowest val_loss = ',r.low('val_loss'),' obtained after ',r.rounds2high('val_loss'))

    # Best params #
    print 
    print ('-'*80)
    print ('Best parameters :\n',r.best_params(metric='val_loss',n=1),end='\n\n')
    
    # Few plots #
    r.plot_corr(metric='val_loss',color_grades=20)
    plt.savefig(name+'_corrplot.png')
    
#print (r.rounds())
#print (r.rounds2high('val_loss'))
#print (r.table(metric='val_loss',sort_by='val_loss',ascending=True))
#print (r.correlate('val_loss'))


#r.correlate(metric='val_loss')
#r.plot_corr(metric='val_loss',color_grades=20)
#r.plot_regs(x='first_neuron',y='val_loss')
#r.plot_kde('val_loss')
    #r.plot_hist(metric='val_loss',bins=100)
    #r.plot_bars('lr','val_loss','batch_size','second_neuron')
    #print (type(r.plot_bars('lr','val_loss','batch_size','second_neuron')))
    #plt.show()
    #r.plot_line('val_loss')
    #print (r.best_params(metric='val_loss',n=10))
