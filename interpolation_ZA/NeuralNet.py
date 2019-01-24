import glob
import os
import re
import math
import sys
import json

import array
import numpy as np
import itertools


import keras
from keras import utils
from keras.layers import Input, Dense, Concatenate, BatchNormalization, LeakyReLU, Lambda, Dropout
from keras.losses import binary_crossentropy, mean_squared_error
from keras.optimizers import RMSprop, Adam, Nadam, SGD
from keras.activations import relu, elu, selu, softmax, tanh
from keras.models import Model, model_from_json, load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l1,l2
import keras.backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # removes annoying warning

# Keras local version #
from talos import Scan, Reporting, Predict, Evaluate, Deploy, Restore, Autom8
from talos.utils.best_model import *
from talos.model.layers import *
from talos.model.normalizers import lr_normalizer
from talos.utils.gpu_utils import parallel_gpu_jobs

#import astetik as ast # For the plot section

import matplotlib.pyplot as plt

from plot_scans import PlotScans

#################################################################################################
# InterpolationModel #
#################################################################################################
def InterpolationModel(x_train,y_train,x_val,y_val,params):
    """
    Keras model for the Neural Network, used to scan the hyperparameter space by Talos
    Inputs :
        - x_train = training inputs (aka : mA,mH)
        - y_train = training outputs (aka : rho distribution -> 6 bins content)
        - x_val = test inputs
        - y_val = test outputs
        - params = dict of parameters for the talos scan
    Outputs :
        - out =  predicted outputs from network
        - model = fitted models with weights
    """
    # Design network #
    IN = Input(shape=(x_train.shape[1],),name='IN')
    L1 = Dense(params['first_neuron'],
               activation=params['activation'],
               kernel_regularizer=l2(params['l2']))(IN)
    DROP_L1 = Dropout(params['dropout'])(L1)
    HIDDEN = hidden_layers(params,6).API(DROP_L1) # Carefull : number of layers = initial layer + hidden layers !!!
    # Dropout is already cared about in HIDDEN 
    OUT = Dense(6,activation=params['output_activation'],name='OUT')(HIDDEN)

    # Define model #
    model = Model(inputs=[IN], outputs=[OUT])
    #utils.print_summary(model=model) #used to print model

    # Callbacks #
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=1, mode='min')
    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min', epsilon=0.001, cooldown=0, min_lr=0.00001)
    Callback_list = [early_stopping]

    # Compile #
    model.compile(optimizer=params['optimizer'](lr_normalizer(params['lr'], params['optimizer'])),
                  loss={'OUT':params['loss_function']},
                  metrics=['accuracy'])

    # Fit #
    out = model.fit({'IN':x_train},
                    {'OUT':y_train},
                    sample_weight=None,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    verbose=0,
                    validation_data=({'IN':x_val},{'OUT':y_val}),
                    callbacks=Callback_list
                    )

    return out,model

#################################################################################################
# HyperScan #
#################################################################################################
def HyperScan(x_train,y_train,x_test,y_test,scaler,name):
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
            'lr' : [0.1,0.2,0.3,0.4,0.5],
            'first_neuron' : [25,50,75,100,150,200,250,300],
            'activation' : [relu,selu,tanh],
            'dropout' : [0,0.05,0.1,0.15,0.2],
            'hidden_layers' : [6],
            'output_activation' : [relu,selu,tanh],
            'l2' : [0,0.05,0.1],
            'optimizer' : [Adam],
            'epochs' : [10000],
            'batch_size' : [1],
            'loss_function' : [mean_squared_error]
        }
    #p = {
    #        'lr' : 0.001,
    #        'first_neuron' : 20,
    #        'activation' : tanh,
    #        'dropout' : 0.5,
    #        'hidden_layers' : 1,
    #        'output_activation' : relu,
    #        'epochs' : 50,
    #        'batch_size' : 5,
    #        'loss_function' : binary_crossentropy,
    #        'optimizer': RMSprop
    #    }
    #out, model = InterpolationModel(x_train,y_train,x_train,y_train,p)
    #sys.exit()
    no = 1
    while os.path.exists(os.path.join(os.getcwd(),name+'_'+str(no)+'.csv')):
        no +=1

    parallel_gpu_jobs(0.5)
    h = Scan(  x=scaler.transform(x_train),
               y=y_train,
               params=p,
               dataset_name=name,
               experiment_no=str(no),
               model=InterpolationModel,
               reduction_metric='val_loss',
               reduction_method = 'correlation',
               reduce_loss = True,
               debug = True,
               last_epoch_value=True,
               print_params=True
            )
    h_with_eval = Autom8(scan_object = h, 
                         x_val = scaler.transform(x_test), 
                         y_val = y_test,
                         n = -1,
                         folds = 1,
                         metric = 'val_loss',
                         asc = True,
                         shuffle = True,
                         average = None)

    h_with_eval.data.to_csv(name+'.csv')

    # returns the experiment configuration details
    print
    print ('='*80,end='\n\n')
    print ('Details',end='\n\n')
    print (h.details)

    return h_with_eval

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
            input testing values (aka : mA,mH), not used during learning
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
    sorted_scores = sorted(scores,key=lambda x : x[0])
    idx_best_eval = sorted_scores[0][2]

    # Print ordered scores #
    count = 0
    for m_err,std_err, idx in sorted_scores:
        count += 1
        if count == 10:
            print ('...')
        if count >= 10 and n_rounds-count>5: # avoids printing intermediate useless states
            continue
        # Print model and error in order #
        print ('Model index %d -> Error = %0.5f (+/- %0.5f))'%(idx,m_err,std_err))
        if idx==idx_best_model:
            print ('\t-> Best model from val_loss')

    print
    print ('='*80,end='\n\n')

    # Prints best model accordind to cross-validation and val_loss #

    print ('Best model from val_loss -> id ',idx_best_model)
    print ('Eval error : %0.5f (+/- %0.5f))'%(scores[idx_best_model][0],scores[idx_best_model][1]))
    print (h.data.iloc[idx_best_model,:])
    print ('-'*80,end='\n\n')

    print ('Best model from cross validation -> id ',idx_best_eval)
    if idx_best_eval==idx_best_model:
        print ('Same model')
    else:
        print ('Eval error : %0.5f (+/- %0.5f))'%(scores[idx_best_eval][0],scores[idx_best_eval][1]))
        print (h.data.iloc[idx_best_eval,:])
    print ('-'*80,end='\n\n')

    # WARNING : model id's starts with 0 BUT on panda dataframe h.data, models start at 1

    return idx_best_eval

#################################################################################################
# HyperDeploy #
#################################################################################################
def HyperDeploy(h,name):
    """
    Performs the cross-validation of the different models
    Inputs :
        - h = Class Scan() object
            object from class Scan coming from HyperScan
        - name : str
            Name of the model package to be saved on disk
    Reference :
        /home/ucl/cp3/fbury/.local/lib/python3.6/site-packages/talos/commands/deploy.py
    """
    Deploy(h,model_name=name,metric='eval_f1score_mean',asc=True)


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
    print ('Lowest eval_error = ',r.low('eval_f1score_mean'),' obtained after ',r.rounds2high('eval_f1score_mean'))

    # Best params #
    print
    print ('='*80)
    print ('Best parameters sets')
    sorted_data = r.data.sort_values('eval_f1score_mean',ascending=True)
    for i in range(0,5):
        print ('-'*80)
        print ('Best params n°',i+1)
        print (sorted_data.iloc[i])

    print ('='*80)

    # Few plots #
    path = os.path.join(os.getcwd(),name+'/report')
    if not os.path.isdir(path):
        os.makedirs(path)

    print ('[INFO] Starting plot section')
    PlotScans(r.data,path,)

#################################################################################################
# HyperRestore #
#################################################################################################
def HyperRestore(inputs,path,scaler,fft=False):
    """
    Retrieve a zip containing the best model, parameters, x and y data, ... and restores it
    Produces an output from the input numpy array
    Inputs :
        - inputs :  numpy array [:,2]
            Inputs to be evaluated
        - path : str
            path to the model archive
        - fft : bool
            Wether or not to apply the fft to the network

    Outputs

    Reference :
        /home/ucl/cp3/fbury/.local/lib/python3.6/site-packages/talos/commands/restore.py
    """
    # Restore model #
    a = Restore(path)

    # Output of the model #
    outputs = a.model.predict(scaler.transform(inputs))
    out_dict = {}
    for i in range(0,outputs.shape[0]):
        out_dict[(inputs[i,0],inputs[i,1])] = outputs[i,:]

    if fft:
        # Build grid #
        n = 100 # number of bins in both directions
        mlljj = np.linspace(0,1000,n)
        mjj = np.linspace(0,1000,n)
        X,Y = np.meshgrid(mjj,mlljj)
        # Rescale and DNN output #
        inputs_grid = scaler.transform(np.c_[np.c_[X.ravel(),Y.ravel()]])
        out_grid = a.model.predict(inputs_grid)
        # FFT in 2D #
        # Plot the different bins outputs #
        fig = plt.figure(figsize=(15,8))
        ax1=plt.subplot(231)
        ax2=plt.subplot(232)
        ax3=plt.subplot(233)
        ax4=plt.subplot(234)
        ax5=plt.subplot(235)
        ax6=plt.subplot(236)

        Z1 = out_grid[:,0].reshape(n,n)
        Z2 = out_grid[:,1].reshape(n,n)
        Z3 = out_grid[:,2].reshape(n,n)
        Z4 = out_grid[:,3].reshape(n,n)
        Z5 = out_grid[:,4].reshape(n,n)
        Z6 = out_grid[:,5].reshape(n,n)

        im = ax1.hexbin(X.ravel(),Y.ravel(),Z1.ravel(),gridsize=30)
        ax1.plot([0, 1000], [0, 1000], ls="--", c=".3")
        ax1.set_xlabel('$m_{jj}$')
        ax1.set_ylabel('$m_{lljj}$')
        ax1.set_title('Bin 1')

        im = ax2.hexbin(X.ravel(),Y.ravel(),Z2.ravel(),gridsize=30)
        ax2.plot([0, 1000], [0, 1000], ls="--", c=".3")
        ax2.set_xlabel('$m_{jj}$')
        ax2.set_ylabel('$m_{lljj}$')
        ax2.set_title('Bin 2')

        im = ax3.hexbin(X.ravel(),Y.ravel(),Z3.ravel(),gridsize=30)
        ax3.plot([0, 1000], [0, 1000], ls="--", c=".3")
        ax3.set_xlabel('$m_{jj}$')
        ax3.set_ylabel('$m_{lljj}$')
        ax3.set_title('Bin 3')

        im = ax4.hexbin(X.ravel(),Y.ravel(),Z4.ravel(),gridsize=30)
        ax4.plot([0, 1000], [0, 1000], ls="--", c=".3")
        ax4.set_xlabel('$m_{jj}$')
        ax4.set_ylabel('$m_{lljj}$')
        ax4.set_title('Bin 4')

        im = ax5.hexbin(X.ravel(),Y.ravel(),Z5.ravel(),gridsize=30)
        ax5.plot([0, 1000], [0, 1000], ls="--", c=".3")
        ax5.set_xlabel('$m_{jj}$')
        ax5.set_ylabel('$m_{lljj}$')
        ax5.set_title('Bin 5')

        im = ax6.hexbin(X.ravel(),Y.ravel(),Z6.ravel(),gridsize=30)
        ax6.plot([0, 1000], [0, 1000], ls="--", c=".3")
        ax6.set_xlabel('$m_{jj}$')
        ax6.set_ylabel('$m_{lljj}$')
        ax6.set_title('Bin 6')

        fig.subplots_adjust(right=0.85, wspace = 0.3, hspace=0.3, left=0.05, bottom=0.1)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        fig.suptitle('Interpolation of the different bins with the NN', fontsize=16)

        #plt.show()
        fig.savefig(path.replace('.zip','')+'/bin_interpolation.png')
        plt.close()

        # Fourier Transform #
        z = Z1.ravel()
        #np.random.shuffle(z)
        sp = np.fft.fft(z) 
        freq = np.fft.fftfreq(z.shape[-1])
        fig = plt.figure(figsize=(15,8))
        plt.plot(freq, sp.real,label='real', color='b')
        plt.plot(freq, sp.imag,label='imaginary', color='r')
        plt.legend()
        #plt.show()
        plt.close()

        fig = plt.figure(figsize=(15,8))
        ax1=plt.subplot(231)
        ax1.set_title('Bin 1')
        ax2=plt.subplot(232)
        ax2.set_title('Bin 2')
        ax3=plt.subplot(233)
        ax3.set_title('Bin 3')
        ax4=plt.subplot(234)
        ax4.set_title('Bin 4')
        ax5=plt.subplot(235)
        ax5.set_title('Bin 5')
        ax6=plt.subplot(236)
        ax6.set_title('Bin 6')

        # Tests for Z #
        test = False
        if test:
            Z1 = np.random.random(size=(100,100))
            mean = np.zeros(100)
            cov = np.dot(Z1,Z1.transpose()) # random covariance matrix : positively defined
            Z2 = np.random.multivariate_normal(mean=mean,cov=cov,size=100)
            cov_id = np.identity(100)
            Z3 = np.random.multivariate_normal(mean=mean,cov=cov_id,size=100)
            Z4 = np.random.multivariate_normal(mean=np.random.random(size=100),cov=cov_id,size=100)
            X,Y = np.meshgrid(np.linspace(0,3.14,100),np.linspace(0,3.14,100))
            Z5 = np.cos(X+Y)+np.sin(Y+X)
            Z6 = np.cos(X+Y)+np.sin(Y+X)
            for i in range(0,50):
                Z6 += np.cos(3.14/6*i*(X+Y))+np.sin(3.14/6*i*(X+Y))
            print (Z1.shape)
            print (Z2.shape)
            print (Z3.shape)
            print (Z4.shape)
            print (Z5.shape)
            print (Z6.shape)

        FS1 = np.fft.fftn(Z1)
        FS2 = np.fft.fftn(Z2)
        FS3 = np.fft.fftn(Z3)
        FS4 = np.fft.fftn(Z4)
        FS5 = np.fft.fftn(Z5)
        FS6 = np.fft.fftn(Z6)
        # abs + ** => a² + b² 
        ims = ax1.imshow(np.log(np.abs(np.fft.fftshift(FS1))**2))
        ims = ax2.imshow(np.log(np.abs(np.fft.fftshift(FS2))**2))
        ims = ax3.imshow(np.log(np.abs(np.fft.fftshift(FS3))**2))
        ims = ax4.imshow(np.log(np.abs(np.fft.fftshift(FS4))**2))
        ims = ax5.imshow(np.log(np.abs(np.fft.fftshift(FS5))**2))
        ims = ax6.imshow(np.log(np.abs(np.fft.fftshift(FS6))**2))

        fig.subplots_adjust(right=0.85, wspace = 0.3, hspace=0.3, left=0.05, bottom=0.1)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(ims, cax=cbar_ax)
        cbar.set_label('$log(a²+b²)$',rotation=90)
        #plt.show()
        if not test:
            fig.suptitle('Fourier Analysis of the different bins', fontsize=16)
            fig.savefig(path.replace('.zip','')+'/bin_fourier.png')
        else:
            fig.suptitle('Fourier tests', fontsize=16)
            ax1.set_title('random')
            ax2.set_title('multivariate normal (mean = 0, random cov)')
            ax3.set_title('multivariate normal (mean = 0, identity cov)')
            ax4.set_title('multivariate normal (mean = random, identity cov)')
            ax5.set_title('cos(X)+sin(X)')
            ax6.set_title('cos(X)+sin(X) + $\sum_{i=0}^{i<50} cos(\pi/6 * i * X) + sin(\pi/6 * i * X)$')
            fig.savefig(path.replace('.zip','')+'/fourier_test.png')
        plt.close()

        


    return out_dict

#################################################################################################
# HyperVerif #
#################################################################################################
def HyperVerif(hist_dict,scaler,path):
    """
    Peforms the DNN interpolation for know points as a cross-check
    Produce comparison plots
    Inputs :
        - hist_dict : dict
            points where rho distribution is know
                -> key = ('mA','mH') tuple
                -> value = np.array of six bins
        - path : str
            name of the zip file to be used in the HyperRestore function
    Outputs :
        - output_dict : dict
            Result of the interpolation for each mass point
                -> key = ('mA','mH') tuple
                -> value = np.array of six bins
    """
    # Get evaluation array input from hist #
    eval_arr = np.empty((0,2))
    for key in hist_dict.keys():
        eval_arr = np.append(eval_arr,np.asarray(key).reshape(-1,2),axis=0)
    
    # Performs the evaluation by the network #
    output = HyperRestore(inputs=eval_arr,path=path,scaler=scaler)

    return output
