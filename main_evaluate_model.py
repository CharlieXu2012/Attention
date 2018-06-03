import keras
import argparse
import os
import numpy as np
from os.path import join, expanduser

def run(f_in,d_in):
    model = keras.models.load_model(args.modelpath)
    model.summary()

    X_test = np.load(f_in+'test.npy')
    X_depth_test = np.load(d_in+'depth_test.npy')
    pred = model.predict([np.concatenate([X_test[:,0:12], X_test[:,26:54]],1), np.concatenate([X_test[:,12:24], X_test[:,54:66]],1),X_depth_test], batch_size=32, verbose=2, steps=None)
    class_pred = pred.argmax(axis=-1)
    np.save('GRANADE_predictions',class_pred)
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--modelpath', type=str, default=join(os.path.dirname(__file__), 'models\\keypoints_distances_depth_fc.h5'), help='Path to saved model.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    f_in = join(os.path.dirname(__file__), 'GRANADE_features\\2\\')
    d_in = join(os.path.dirname(__file__), 'GRANADE_depth\\2\\')
    run(f_in,d_in)