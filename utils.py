import numpy as np
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from experiments import plot_classes


def randomize(pose_feats_final, d_list, labels):
    # Generate the permutation index array.
    permutation = np.random.permutation(pose_feats_final.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    pose_feats_final = pose_feats_final[permutation]
    labels = labels[permutation]
    d_list = d_list[permutation]
    
    return pose_feats_final, d_list, labels

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(u, v):
    """ Returns the angle in degrees between unit vectors 'u' and 'v'"""
 
    v1_u = unit_vector(u)
    v2_u = unit_vector(v)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), 
                                        -1.0, 1.0)))

def cross_validation(pose_feats_smooth, d_list, labels):
    """ Returns normalized cross validation training and test sets. """

    """ Normalize and randomise samples features """
    pose_feats_final, d_list = norm_feats(pose_feats_smooth, d_list)
    pose_feats_final, d_list, labels = randomize(pose_feats_final, d_list, labels)

    train = np.zeros([int(np.floor(len(pose_feats_final)/4)*3), 66], dtype=np.float64)
    gt_train = np.zeros([int(np.floor(len(pose_feats_final)/4)*3)])
    test = np.zeros([int(np.floor(len(pose_feats_final)/4)), 66], dtype=np.float64)
    gt_test = np.zeros([int(np.floor(len(pose_feats_final)/4))-1])
    depth_train = np.zeros([int(np.floor(len(d_list)/4)*3), 6], dtype=np.float64)
    depth_test = np.zeros([int(np.floor(len(d_list)/4)), 6], dtype=np.float64)

    """Normal train split"""
    train[:,:] = np.array(pose_feats_final[int(np.floor(len(pose_feats_final)/4)):len(pose_feats_final)-1,:])
    depth_train[:,:] = np.array(d_list[int(np.floor(len(pose_feats_final)/4)):len(pose_feats_final)-1,:])
    gt_train = np.transpose(np.array(labels[int(np.floor(len(pose_feats_final)/4)):len(pose_feats_final)-1]))

    """Normal validation split"""
    test[:,:] = np.array(pose_feats_final[0:int(np.floor(len(pose_feats_final)/4)),:])
    depth_test[:,:] = np.array(d_list[0:int(np.floor(len(pose_feats_final)/4)),:])
    gt_test = np.transpose(np.array(labels[0:int(np.floor(len(pose_feats_final)/4))]))

    return test, train, gt_test, gt_train, depth_train, depth_test

def norm_feats(pose_feats_smooth, d_list):
    """ Normalize all features, leave out all [0,0] nose coordinates. """
    
    trainsub = pose_feats_smooth[:,2:66]

    for i in range(0 , np.size(trainsub, 1)):
        """ Standard keypoints and geometric features """
        meanc = np.mean(trainsub[:,i])
        stdc = np.std(trainsub[:,i])
        trainsub[:,i] = (trainsub[:,i]-meanc)/stdc

    for j in range(0, d_list.shape[1]):
        """ Standard depth """
        meanc = np.mean(d_list[:,j])
        stdc = np.std(d_list[:,j])
        d_list[:,j] = (d_list[:,j]-meanc)/stdc

    pose_feats_smooth[:,2:66] = trainsub

    return pose_feats_smooth, d_list

def oversample():
    plot_classes(labels)
    

    fm = SMOTE(ratio='all',kind='regular',k_neighbors=5)
    n_pose_feats, n_labels = fm.fit_sample(pose_feats, labels)

    dm = SMOTE(ratio='all',kind='regular',k_neighbors=5)
    n_d_list = dm.fit_sample(d_list, labels)

    plot_classes(n_labels)

def sample(pose_feats, d_list, labels):
    #randomize!

    idx0 = np.flatnonzero(labels == 0)
    idx1 = np.flatnonzero(labels == 1)
    idx2 = np.flatnonzero(labels == 2)
    
    dom = np.min([len(idx0), len(idx1), len(idx2)])

    n_idx0 = idx0[0:dom-2]
    n_idx1 = idx1[0:dom-2]
    n_idx2 = idx2[0:dom-2]

    n_pose_feats0 = pose_feats[n_idx0]
    n_pose_feats1 = pose_feats[n_idx1]
    n_pose_feats2 = pose_feats[n_idx2]

    pose_feats = np.concatenate([n_pose_feats0, n_pose_feats1, n_pose_feats2])
    d_list = np.concatenate([d_list[n_idx0], d_list[n_idx1], d_list[n_idx2]])
    labels = np.concatenate([labels[n_idx0], labels[n_idx1], labels[n_idx2]])

    return pose_feats, d_list, labels