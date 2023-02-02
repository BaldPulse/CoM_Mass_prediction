import os
import sys
import numpy as np
import yaml
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
# print the version of tensorflow
print(tf.__version__)
TF2 = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
print(BASE_DIR)
print(ROOT_DIR)
sys.path.append(os.path.join(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, 'pointnet2', 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'pointnet2'))

import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module, pointnet_sa_module_msg
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point

# see if my gpu is working
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



def get_model(point_cloud, is_training, bn_decay):
    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    l0_points = None
    end_points = {}
    # # set abstraction layers as described in PointNet++
    # l1_xyz, l1_points = pointnet_sa_module_msg(xyz=l0_xyz, points=l0_points, 
    #     npoint=128, radius_list=[0.04,0.08], nsample_list=[64,128], mlp_list=[[32,64],[64,64]], 
    #     is_training=is_training, bn_decay=bn_decay, scope="layer1", bn=True, use_xyz=True, use_nchw=False)
    # l2_xyz, l2_points, _ = pointnet_sa_module(l1_xyz, l1_points,
    #     npoint=None, radius=None, nsample=None, mlp=[64,128,256],
    #     mlp2=None, group_all=True, is_training=is_training,
    #     bn_decay=bn_decay, scope='layer2')
    # end_points["abstraction"] = l2_points
    
    # # feature propagation layers
    # l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points,
    #     [64,64], is_training, bn_decay, scope='fa_layer1')
    # l0_points = pointnet_fp_module(l0_xyz, l1_xyz,
    #     l0_xyz, l1_points,
    #     [64,64], is_training, bn_decay, scope='fa_layer2')
    # end_points["pc_features"] = l0_points


    # Set abstraction layers
    l1_xyz, l1_points = pointnet_sa_module_msg(l0_xyz, l0_points,
        128, [0.2,0.4,0.8], [32,64,128],
        [[32,32,64], [64,64,128], [64,96,128]],
        is_training, bn_decay, scope='layer1')
    l2_xyz, l2_points = pointnet_sa_module_msg(l1_xyz, l1_points,
        32, [0.4,0.8,1.6], [64,64,128],
        [[64,64,128], [128,128,256], [128,128,256]],
        is_training, bn_decay, scope='layer2')
    l3_xyz, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points,
        npoint=None, radius=None, nsample=None, mlp=[128,256,1024],
        mlp2=None, group_all=True, is_training=is_training,
        bn_decay=bn_decay, scope='layer3')

    # Feature Propagation layers
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points,
        [128,128], is_training, bn_decay, scope='fa_layer1')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points,
        [128,128], is_training, bn_decay, scope='fa_layer2')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz,
        l0_xyz, l1_points,
        [128,128], is_training, bn_decay, scope='fa_layer3')
    
    # FC layers. Note that in this task the layers are used for regression
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True,
        is_training=is_training, scope='conv1d-fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7,
        is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, 32, 1, padding='VALID', bn=True, 
        is_training=is_training, scope='conv1d-fc2', bn_decay=bn_decay)
    end_points["fc_features"] = net
    CoMM = tf_util.conv1d(net, 3, 1, padding='VALID', 
        activation_fn=None, scope='conv1d-fc3')
    return CoMM, end_points

def get_CoM(pred, pc):
    '''
    use RANSAC to get the center of mass
    '''
    print("pred shape: ", pred.shape)
    print("pc shape: ", pc.shape)
    # first convert the predicted to predicted CoM
    pred_CoM = pc - pred
    best_num_inliers = 0
    best_inliers = None
    # then use RANSAC to get the center of mass
    for i in range(100):
        # randomly select 3 points
        idx = np.random.choice(pred_CoM.shape[0], size=3, replace=False)
        # get the center of mass
        CoM = np.mean(pred_CoM[idx], axis=0)
        # get the distance between the center of mass and the points
        dist = np.sqrt(np.sum((pred_CoM - CoM)**2, axis=1))
        # get the inliers
        inliers = pred_CoM[dist < 0.02]
        # get the number of inliers
        num_inliers = inliers.shape[0]
        # if the number of inliers is larger than 100, then we can break the loop
        if num_inliers > 1000:
            best_num_inliers = num_inliers
            best_inliers = inliers
            print("Loop break!")
            break
        else:
            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_inliers = inliers

    # get the center of mass
    CoM = np.mean(best_inliers, axis=0)
    return CoM, best_inliers

def plot(pc, pred, tv):
    import trimesh
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(pc))
    pred_CoM = np.average(pc-pred, axis=0)
    GT_CoM = np.average(pc-tv, axis=0)
    scene.add_geometry(trimesh.creation.uv_sphere(radius=0.01), transform=trimesh.transformations.translation_matrix(pred_CoM))
    scene.add_geometry(trimesh.creation.uv_sphere(radius=0.01), transform=trimesh.transformations.translation_matrix(GT_CoM))
    # also randomly draw 10 raw points
    idx = np.random.choice(pc.shape[0], 10, replace=False)
    for i in idx:
        scene.add_geometry(trimesh.creation.icosphere(radius=0.01, color=[1.0,0,0]), transform=trimesh.transformations.translation_matrix(tv[i]))
        scene.add_geometry(trimesh.creation.icosphere(radius=0.01, color=[0,1.0,0]), transform=trimesh.transformations.translation_matrix(pred[i]))

    scene.show()