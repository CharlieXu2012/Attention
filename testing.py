import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from utils import norm_feats
method=2
pose_feats = np.load('pose_feats_nonenorm.npy')
d_list = np.load('d_list_nonenorm.npy')
labels = np.load('labels_n.npy')

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

pose_feats_n = np.concatenate([n_pose_feats0, n_pose_feats1, n_pose_feats2])
d_list_n = np.concatenate([d_list[n_idx0], d_list[n_idx1], d_list[n_idx2]])
labels_n = np.concatenate([labels[n_idx0], labels[n_idx1], labels[n_idx2]])
nidx = np.concatenate([n_idx0,n_idx1,n_idx2])

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

test[:,:] = np.array(pose_feats_n[0:int(np.floor(len(pose_feats_n)/4)),:])
depth_test[:,:] = np.array(d_list_n[0:int(np.floor(len(pose_feats_n)/4)),:])
gt_test = np.transpose(np.array(labels_n[0:int(np.floor(len(pose_feats_n)/4))]))
nidx_test = np.transpose(np.array(nidx[0:int(np.floor(len(pose_feats_n)/4))]))

pose_feats = np.delete(pose_feats,nidx_test,axis=0)
d_list = np.delete(d_list,nidx_test,axis=0)
labels = np.delete(labels,nidx_test)
labels = labels.astype(int)

if method == 2:
    pose_feats = np.concatenate([pose_feats,d_list],axis=1)
    fm = ADASYN(ratio='all',n_neighbors=5)
    fm = fm.fit(pose_feats, labels)

    """dm = ADASYN(ratio='all',n_neighbors=5)
    dm = dm.fit(d_list, labels)"""
elif method == 1:
    fm = SMOTE(ratio='all',kind='regular',k_neighbors=5)
    fm = fm.fit(pose_feats, labels)

    dm = SMOTE(ratio='all',kind='regular',k_neighbors=5)
    dm = dm.fit(d_list, labels)

pose_feats, labels_train = fm.sample(pose_feats, labels)
d_list = pose_feats[:,66:72]

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