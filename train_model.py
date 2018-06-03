import tensorflow as tf
import numpy as np
import scipy as sp
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.utils import np_utils
from model_load import evaluate_lstm, evaluate_flexible, late_DNN2, early_DNN2, late_DNN3, early_DNN3
from load import load
from load_lstm import load_lstm
from utils import sample_lstm, reshape_seqlist
from sklearn.decomposition import PCA
import pandas as pd

eq=True
train_label = ''
simple = True
depth_label = True
shape = 1
if train_label=='lstm':
    """range0, range1, range2 = sample_lstm( gt_train[1,:], shape )
    trange0, trange1, trange2 = sample_lstm( gt_test[1,:], shape )

    train, depth_train, gt_train, test, depth_test, gt_test = reshape_seqlist(range0,range1,range2,
                                                                              trange0,trange1,trange2,
                                                                              train,depth_train,test,
                                                                              depth_test,shape)"""
    test, train, gt_test, gt_train, depth_train, depth_test = load_lstm(train_label)

    X_train = train[:,:]
    X_train = np.reshape(X_train,[X_train.shape[0],shape,X_train.shape[1]])
    X_depth_train = depth_train[:,:]
    X_depth_train = np.reshape(X_depth_train,[X_depth_train.shape[0],shape,X_depth_train.shape[1]])
    Y_train = gt_train[1,:]
    
    X_test = test[1][0:test[1].shape[0]-1,:]
    X_test = model1.transform(X_test)
    X_test = np.reshape(X_test,[X_test.shape[0],shape,X_test.shape[1]])
    X_depth_test = depth_test[1][0:test[1].shape[0]-1,:]
    X_depth_test = np.reshape(X_depth_test,[X_depth_test.shape[0],shape,X_depth_test.shape[1]])
    Y_test = gt_test[1]
else:
    test, train, gt_test, gt_train, depth_train, depth_test = load(train_label)   
    X_train = train[:,:]
    X_depth_train = depth_train[:,:]
    Y_train = gt_train
    X_test = test[0:test.shape[0]-1,:]
    X_depth_test = depth_test[0:test.shape[0]-1,:]
    Y_test = gt_test

shape0 = 40
shape1 = 24
shape2 = 6


modelshape = 2

model = early_DNN3(shape0,shape1,shape2)

if train_label=='lstm':
    """history, pred, cnf_matrix = evaluate_lstm(model, train, gt_train, test, 
                            gt_test, depth_train, depth_test, depth_label, simple)"""

    history, pred, cnf_matrix = evaluate_lstm(model, X_train, Y_train, X_test, 
                            Y_test, X_depth_train, X_depth_test, depth_label,simple)
else:
    history, pred, cnf_matrix = evaluate_flexible(model, X_train, Y_train, X_test, 
                            Y_test, X_depth_train, X_depth_test, modelshape)