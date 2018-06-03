import numpy as np
import scipy.ndimage

def get_depth(pose_feats_points, imagelist_d,f_in_d):

    """ Get corresponding depth values of neighbours at each keypoint,
        and calculate the mean. """

    depth_k = np.zeros([pose_feats_points.shape[0],1])
    for i in range(0, pose_feats_points.shape[0]):

        y, x = np.int_(pose_feats_points[i,:])
        point = np.array([[[x-1, y-1], [x-1, y], [x-1, y+1]], 
                     [[x, y-1], [x, y], [x, y+1]],
                     [[x+1, y-1], [x+1, y], [x+1, y+1]]])

        """Read image and get color intensity values"""
        folder = f_in_d
        frame_n = folder + imagelist_d[i]
        frame = scipy.ndimage.imread(frame_n)
        point_i = np.array([[[frame[point[0,0,0],point[0,0,1]],frame[point[0,1,0],point[0,1,1]],frame[point[0,2,0],point[0,2,1]]]],
                    [[frame[point[1,0,0],point[1,0,1]],frame[point[1,1,0],point[1,1,1]],frame[point[1,2,0],point[1,2,1]]]],
                    [[frame[point[2,0,0],point[2,0,1]],frame[point[2,1,0],point[2,1,1]],frame[point[2,2,0],point[2,2,1]]]]])

        depth_k[i] = np.mean(point_i).astype(int)

    return depth_k

def depthlist(pose_feats_smooth, imagelist_d,f_in_d):

    d_list = np.zeros([pose_feats_smooth.shape[0],6])
    i = 0
    for j in range(0, 12, 2):
        pose_feats_points = pose_feats_smooth[:,j:j+2]
        depth_k = get_depth(pose_feats_points, imagelist_d,f_in_d)
        d_list[:,i:i+1] = depth_k
        i=i+1

    return d_list