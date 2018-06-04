import argparse
import os
import numpy as np
from os.path import join, expanduser
from model_load import evaluate_flexible, late_DNN2, early_DNN2, late_DNN3, early_DNN3

def run(f_in):
    X_train, X_depth_train, Y_train, X_test, X_depth_test, Y_test = load_data(f_in)

    shape0 = 40
    shape1 = 24
    shape2 = 6

    if args.model==2:
        if args.fusiontype==1 or args.fusiontype==0:
            model = early_DNN2(shape0,shape1,args.fusiontype)
        elif args.fusiontype==2:
            model = late_DNN2(shape0,shape1,args.type)
    elif args.model==3:
        if args.fusiontype==1 or args.fusiontype==0:
            model = early_DNN3(shape0,shape1,shape2,args.fusiontype)
        elif args.fusiontype==2:
            model = late_DNN3(shape0,shape1,shape1,args.type)

    history, pred, cnf_matrix, model = evaluate_flexible(model, X_train, Y_train, X_test, 
                                                    Y_test, X_depth_train, X_depth_test, args.model,args.bs,args.ep)
    model.save(join(os.path.dirname(__file__), 'models\\'))

def load_data(f_in):
    if args.oversample==False:
        """Training data"""
        X_train = np.load(join(f_in,'train.npy'))
        X_depth_train = np.load(join(f_in,'depth_train.npy'))
        Y_train = np.load(join(f_in,'labels_train.npy'))
        Y_train = Y_train.astype(int)
        """Validation data"""
        X_test = np.load(join(f_in,'test.npy'))
        X_depth_test = np.load(join(f_in,'depth_test.npy'))
        Y_test = np.load(join(f_in,'labels_test.npy'))
        Y_test = Y_test.astype(int)
        print('Using ',X_train.shape[0], ' samples for training and ', X_test.shape[0],' samples for validation.\n')
    else:
        """Training data"""
        X_train = np.load(join(f_in,'train_oversampled_ADASYN.npy'))
        X_depth_train = np.load(join(f_in,'depth_train_oversampled_ADASYN.npy'))
        Y_train = np.load(join(f_in,'labels_train_oversampled_ADASYN.npy'))
        Y_train = Y_train.astype(int)
        """Validation data"""
        X_test = np.load(join(f_in,'test_for_ADASYN.npy'))
        X_depth_test = np.load(join(f_in,'depth_test_for_ADASYN.npy'))
        Y_test = np.load(join(f_in,'labels_test_for_ADASYN.npy'))
        Y_test = Y_test.astype(int)
        print('Using ',X_train.shape[0], ' samples for training (with over-sampled data) and ', X_test.shape[0],' samples for validation (without over-sampled data).\n')

    return X_train, X_depth_train, Y_train, X_test, X_depth_test, Y_test

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--oversample', type=bool, default=False, help='Wether or not to use oversampled data for training. Uses the ADASYN data by default.')
    parser.add_argument('--model', type=int, default=3, help='Two (2) or three (3) stream DNN model.',choices=[2,3])
    parser.add_argument('--fusiontype', type=int, default=1, help='Use early (0), fully connected (1) or late (2) fusion.',choices=[0,1,2])
    parser.add_argument('--type', type=int, default=0, help='Use average (0), max (1) or WSLF-LW (2) fusion.',choices=[0,1,2])
    parser.add_argument('--bs', type=int, default=32, help='Batch size.')
    parser.add_argument('--ep', type=int, default=175, help='Epochs.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    f_in = join(os.path.dirname(__file__), 'PANDORA_features\\')
    run(f_in)