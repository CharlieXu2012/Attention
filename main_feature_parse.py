from json_parser_train import parse_feats
from json_parser_granade import parse_feats_granade
import argparse
import os
from oversample import oversample
import numpy as np
from os.path import join, expanduser
from utils import cross_validation, sample, norm_feats

"""
Main method for feature extraction and crafting.

Input(s):
    - f_in: input folder of .JSON files (String)
    - f_in_d: input folder of depth images (String)
    - f_out: output folder of dataset. (String)

Output(s):
    - train: pose features for training split (Numpy array(,66))
    - depth_train: depth features for training split (Numpy array(,6))
    - labels_train: labels for all data samples for training split (Numpy array(,1))
    - test: pose features for validation split (Numpy array(,66))
    - depth_test: depth features for validation split (Numpy array(,6))
    - labels_test: labels for all data samples for validation split (Numpy array(,1))

    ...OR the oversampled versions for training.
"""

def run(f_in,f_out,f_in_d):
    if args.dataset==0:
        pose_feats, d_list, labels = parse_feats(f_in,f_out,f_in_d,args.depth,args.oversampling)
        if args.oversampling==False:
            #pose_feats, d_list, labels = sample(pose_feats, d_list, labels)
            pose_feats, d_list, labels = sample(pose_feats, d_list, labels)
            test, train, gt_test, gt_train, depth_train, depth_test = cross_validation( pose_feats, d_list, labels)
            np.save(f_out + 'train',train)
            np.save(f_out + 'labels_train',gt_train)
            np.save(f_out + 'depth_train',depth_train)
            np.save(f_out + 'test',test)
            np.save(f_out + 'labels_test',gt_test)
            np.save(f_out + 'depth_test',depth_test)
            print('Training and validation splits were saved in: ', f_out,'.')

        else:
            test, train, gt_test, gt_train, depth_train, depth_test = oversample(args.method,pose_feats, d_list, labels)
            if args.method==1:
                np.save(f_out + 'train_oversampled_SMOTE',train)
                np.save(f_out + 'labels_train_oversampled_SMOTE',gt_train)
                np.save(f_out + 'depth_train_oversampled_SMOTE',depth_train)
                np.save(f_out + 'test_oversampled_SMOTE',test)
                np.save(f_out + 'labels_test_oversampled_SMOTE',gt_test)
                np.save(f_out + 'depth_test_oversampled_SMOTE',depth_test)
            elif args.method==2:
                np.save(f_out + 'train_oversampled_ADASYN',train)
                np.save(f_out + 'labels_train_oversampled_ADASYN',gt_train)
                np.save(f_out + 'depth_train_oversampled_ADASYN',depth_train)
                np.save(f_out + 'test_for_ADASYN',test)
                np.save(f_out + 'labels_test_for_ADASYN',gt_test)
                np.save(f_out + 'depth_test_for__ADASYN',depth_test)
            print('Oversampled training and validation splits were saved in: ', f_out, '.')
    else:
        f_in_p = join(os.path.dirname(__file__), 'GRANADE_keypoints\\')
        #f_in_p = "E:\\GRANADE\\keypoints\\full\\"
        f_in_d_p = join(os.path.dirname(__file__), 'GRANADE_depth\\')
        #f_in_d_p = "E:\\GRANADE\\frames\\full_d\\"
        f_out_p = join(os.path.dirname(__file__), 'GRANADE_features\\')
        pose_feats, d_list = parse_feats_granade(f_in_p,f_out_p,f_in_d_p,args.depth,args.oversampling)
        pose_feats, d_list = norm_feats(pose_feats, d_list)
        np.save(f_out_p + 'test',pose_feats)
        np.save(f_out_p + 'depth_test',d_list)
        print('Granade test samples were saved in: ', f_out_p,'.')

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=int, default=0, help='Wether to use PANDORA (0) or GRANADE (1).')
    parser.add_argument('--depth', type=bool, default=True, help='Wether to perform depth feature extraction or load dataset.')
    parser.add_argument('--oversampling', type=bool, default=False, help='Wether or not to perform oversampling of minority clases.')
    parser.add_argument('--method', type=int, default=2, help='Method for oversampling: 1)None 2)SMOTE 3)ADASYN.',choices=[0,1,2])
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    f_in = join(os.path.dirname(__file__), 'PANDORA_keypoints\\')
    #f_in = "E:\\keypoints\\full\\"
    f_in_d = join(os.path.dirname(__file__), 'PANDORA_depth\\')
    f_out = join(os.path.dirname(__file__), 'PANDORA_features\\')
    run(f_in,f_out,f_in_d)