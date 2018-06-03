import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from utils import norm_feats

def oversample(method,pose_feats, d_list, labels):
    """Normalize data"""
    pose_feats_n, d_list_n = norm_feats(pose_feats, d_list)

    """Extract class indecies and equalize"""
    idx0 = np.flatnonzero(labels == 0)
    idx1 = np.flatnonzero(labels == 1)
    idx2 = np.flatnonzero(labels == 2)
    
    dom = np.min([len(idx0), len(idx1), len(idx2)])

    n_idx0 = idx0[0:dom-2]
    n_idx1 = idx1[0:dom-2]
    n_idx2 = idx2[0:dom-2]

    n_pose_feats0 = pose_feats_n[n_idx0]
    n_pose_feats1 = pose_feats_n[n_idx1]
    n_pose_feats2 = pose_feats_n[n_idx2]

    pose_feats_n = np.concatenate([n_pose_feats0, n_pose_feats1, n_pose_feats2])
    d_list_n = np.concatenate([d_list_n[n_idx0], d_list_n[n_idx1], d_list_n[n_idx2]])
    labels_n = np.concatenate([labels[n_idx0], labels[n_idx1], labels[n_idx2]])
    nidx = np.concatenate([n_idx0,n_idx1,n_idx2])

    """Randomize the equalized data"""
    # Generate the permutation index array.
    permutation = np.random.permutation(pose_feats_n.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    pose_feats_n = pose_feats_n[permutation]
    labels_n = labels_n[permutation]
    d_list_n = d_list_n[permutation]
    nidx = nidx[permutation]

    test = np.zeros([int(np.floor(len(pose_feats_n)/4)), 66], dtype=np.float64)
    gt_test = np.zeros([int(np.floor(len(pose_feats_n)/4))])
    depth_test = np.zeros([int(np.floor(len(d_list_n)/4)), 6], dtype=np.float64)
    nidx_test = np.zeros([int(np.floor(len(nidx)/4))])

    """Save validation data"""
    test[:,:] = np.array(pose_feats_n[0:int(np.floor(len(pose_feats_n)/4)),:])
    depth_test[:,:] = np.array(d_list_n[0:int(np.floor(len(pose_feats_n)/4)),:])
    gt_test = np.transpose(np.array(labels_n[0:int(np.floor(len(pose_feats_n)/4))]))
    nidx_test = np.transpose(np.array(nidx[0:int(np.floor(len(pose_feats_n)/4))]))

    """Extract the indecies used for validation"""
    pose_feats = np.delete(pose_feats,nidx_test,axis=0)
    d_list = np.delete(d_list,nidx_test,axis=0)
    labels = np.delete(labels,nidx_test)
    labels = labels.astype(int)

    """For SMOTE or ADASYN on training data"""
    if method == 2:
        pose_feats = np.concatenate([pose_feats,d_list],axis=1)
        fm = ADASYN(ratio='all',n_neighbors=5)
        fm = fm.fit(pose_feats, labels)
    elif method == 1:
        pose_feats = np.concatenate([pose_feats,d_list],axis=1)
        fm = SMOTE(ratio='all',kind='regular',k_neighbors=5)
        fm = fm.fit(pose_feats, labels)

    pose_feats, labels_train = fm.sample(pose_feats, labels)
    d_list = pose_feats[:,66:72]

    """Apply normalization to over-sampled training data"""
    pose_feats, d_list = norm_feats(pose_feats, d_list)

    # Generate the permutation index array.
    permutation = np.random.permutation(pose_feats.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    pose_feats = pose_feats[permutation]
    labels_train = labels_train[permutation]
    d_list = d_list[permutation]

    train = pose_feats
    gt_train = labels_train
    depth_train = d_list

    return test, train, gt_test, gt_train, depth_train, depth_test
