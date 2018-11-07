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
from talos.utils.best_model import *

import matplotlib.pyplot as plt

#################################################################################################
# InterpolationModel #
#################################################################################################
def InterpolationModel(x_train,y_train,x_val,y_val,params):
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
    #OUT_1 = Dense(1,activation=params['output_activation'],name='OUT_1')(L2)
    #OUT_2 = Dense(1,activation=params['output_activation'],name='OUT_2')(L2)
    #OUT_3 = Dense(1,activation=params['output_activation'],name='OUT_3')(L2)
    #OUT_4 = Dense(1,activation=params['output_activation'],name='OUT_4')(L2)
    #OUT_5 = Dense(1,activation=params['output_activation'],name='OUT_5')(L2)
    #OUT_6 = Dense(1,activation=params['output_activation'],name='OUT_6')(L2)
    OUT = Dense(6,activation=params['output_activation'],name='OUT')(L2)

    # Define model #    
    #model = Model(inputs=[inputs], outputs=[OUT_1,OUT_2,OUT_3,OUT_4,OUT_5,OUT_6])
    model = Model(inputs=[inputs], outputs=[OUT])
    #utils.print_summary(model=model) #used to print model

    # Compile #
    adam = optimizers.Adam(lr=params['lr'], beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False, clipvalue=0.5)
    model.compile(optimizer=adam,
                  loss={#'OUT_1':'binary_crossentropy',
                        #'OUT_2':'binary_crossentropy',                        
                        #'OUT_3':'binary_crossentropy',                        
                        #'OUT_4':'binary_crossentropy',                        
                        #'OUT_5':'binary_crossentropy',                        
                        #'OUT_6':'binary_crossentropy',                        
                        'OUT':'binary_crossentropy',                        
                       },
                  metrics=['accuracy']) 

    # Fit #
    out = model.fit({'inputs':x_train},
                    {#'OUT_1':y_train[:,0],
                     #'OUT_2':y_train[:,1],
                     #'OUT_3':y_train[:,2],
                     #'OUT_4':y_train[:,3],
                     #'OUT_5':y_train[:,4],
                     #'OUT_6':y_train[:,5],
                     'OUT':y_train,
                    },
                    sample_weight=None,  # TODO TO BE CHANGED
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    verbose=0,
                    validation_data=({'inputs':x_val},
                                     {#'OUT_1':y_val[:,0],
                                      #'OUT_2':y_val[:,1],
                                      #'OUT_3':y_val[:,2],
                                      #'OUT_4':y_val[:,3],
                                      #'OUT_5':y_val[:,4],
                                      #'OUT_6':y_val[:,5],
                                      'OUT':y_val,
                                     }#,
                                     ########### -> TODO Must put test weight
                                    )
                    )
                    
    return out,model

#################################################################################################
# HyperScan #
#################################################################################################
def HyperScan(x_train,y_train,name):  # MUST ADD THE WEIGHTS
    """ 
    Performs the scan for hyperparameters 
    Inputs :
        - x_train : numpy array [:,2] 
            input training values (aka : mH,mA) 
        - y_train : numpy array [:,6]
            output training values (aka : rho distribution -> 6 bins content)
        - name : str
            name of the dataset
    Outputs : 
        - h = Class Scan() object
            object from class Scan to be used by other functions    
    Reference : /home/ucl/cp3/fbury/.local/lib/python3.6/site-packages/talos/scan/Scan.py
    """ 
   
    # Talos hyperscan parameters #
    p = {
            'lr' : (0.001,0.1,5),
            'first_neuron' : [10,20],
            'first_activation' : [relu,tanh],
            'second_neuron' : [10,20],
            'second_activation' : [relu,tanh],
            'output_activation' : [relu,tanh],
            'epochs' : [100],
            'batch_size' : [5]
        }
    h = Scan(  x=x_train,
               y=y_train,
               params=p,
               dataset_name=name,
               model=InterpolationModel,
               val_split=0.2,
               reduction_metric='val_loss',
               last_epoch_value=True,
               print_params=True
            )

    # returns the experiment configuration details
    print 
    print ('='*80,end='\n\n')
    print ('Details',end='\n\n')
    print (h.details)

    return h

#################################################################################################
# HyperEvaluate #
#################################################################################################
def HyperEvaluate(h,x_test,y_test,folds=5):
    """
    Performs the cross-validation of the different models 
    Inputs :
        - h = Class Scan() object
            object from class Scan coming from HyperScan 
        - x_test : numpy array [:,2] 
            input testing values (aka : mH,mA), not used during learning 
        - y_test : numpy array [:,6]
            output testing values (aka : rho distribution -> 6 bins content), not used during learning 
        - folds : int (default = 5)
            Number of cross-validation folds
    Outputs : 
        - idx_best_eval : idx
            Index of best model according to cross-validation

    Reference : 
        /home/ucl/cp3/fbury/.local/lib/python3.6/site-packages/talos/commands/evaluate.py 
    """


    # Predict to get number of round #
    r = Reporting(h)
    n_rounds = r.rounds()

    # Evaluation #
    print 
    print ('='*80,end='\n\n')
    scores = []
    idx_best_model = best_model(h, 'val_loss', asc=True)

    for i in range(0,n_rounds):
        e = Evaluate(h)
        score = e.evaluate(x=x_test,
                           y=y_test,
                           model_id = i,
                           folds=folds,
                           shuffle=True,
                           metric='val_loss',
                           average='macro',
                           asc=True  # because loss 
                          )
        score.append(i) # score = [mean(error),std(error),model_index]
        scores.append(score)

    # Sort scores #
    scores.sort(key=lambda x : x[0])
    idx_best_eval = scores[0][2]

    # Print ordered scores #
    count =0
    for m_err,std_err, idx in scores:
        # Avoid useless info  #
        if count >10:
            print ('...')
            break
        # Print model and error in order #
        print ('Model index %d -> Error = %0.5f (+/- %0.5f))'%(idx,m_err,std_err))
        if idx==idx_best_model:
           print ('\t-> Best model from val_loss')

    print 
    print ('='*80,end='\n\n')
        
    # Prints best model accordind to cross-validation and val_loss #

    print ('Best model from val_loss -> id ',idx_best_model)
    print ('Eval error : %0.5f (+/- %0.5f))'%(scores[idx_best_model][0],scores[idx_best_model][1]))
    print (h.data.iloc[idx_best_model])
    print ('-'*80,end='\n\n')

    print ('Best model from cross validation -> id ',idx_best_eval)
    if idx_best_eval==idx_best_model:
        print ('Same model')
    else:
        print ('Eval error : %0.5f (+/- %0.5f))'%(scores[idx_best_eval][0],scores[idx_best_eval][1]))
        print (h.data.iloc[idx_best_eval])
        print ('-'*80,end='\n\n')


    return idx_best_eval

#################################################################################################
# HyperDeploy #
#################################################################################################
def HyperDeploy(h,name,best):
    """
    Performs the cross-validation of the different models 
    Inputs :
        - h = Class Scan() object
            object from class Scan coming from HyperScan 
        - name : str
            Name of the model package to be saved on disk
        - best : int
            index of the best model 
                -> -1 : not used HyperEvaluate => select the one with lowest val_loss
                -> >0 : comes from HyperEvaluate => the one with best error from cross-validation

    Reference : 
        /home/ucl/cp3/fbury/.local/lib/python3.6/site-packages/talos/commands/deploy.py 
    """
    if best == -1:
        idx = best_model(h, 'val_loss', asc=True)
    else: # From HyperEvaluate
        idx = best

    Deploy(h,model_name=name,metric='val_loss',asc=True)


#################################################################################################
# HyperReport #
#################################################################################################
def HyperReport(name):
    """
    Reports the model from csv file of previous scan 
    Plot several quantities and comparisons in dir /$name/
    Inputs :
        - name : str
            Name of the csv file 

    Reference : 
        /home/ucl/cp3/fbury/.local/lib/python3.6/site-packages/talos/commands/reporting.py
    """
    r = Reporting(name+'.csv')     

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
    path = os.path.join(os.getcwd(),name)
    if not os.path.isdir(path):
        os.makedirs(os.path.join(os.getcwd(),name))

    print ('[INFO] Starting plot section')
    # Correlation #
    r.plot_corr(metric='val_loss',color_grades=20)
    plt.savefig(path+'/correlation.png')

    # val_loss VS loss #
    r.plot_regs('loss','val_loss')
    plt.savefig(path+'/val_loss_VS_loss.png')

    # val_loss KDE #
    r.plot_kde('val_loss')
    plt.savefig(path+'/KDE_val_loss.png')

    # Plot bars #
    #r.plot_bars('lr','val_loss','batch_size','second_neuron')


