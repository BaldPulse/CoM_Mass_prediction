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

# # run a simple thing on the gpu
# with tf.device('/gpu:0'):
#     a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#     b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#     c = tf.matmul(a, b)


def get_model(point_cloud, is_training, bn_decay):
    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    l0_points = None
    end_points = {}
    # set abstraction layers as described in PointNet++
    l1_xyz, l1_points = pointnet_sa_module_msg(xyz=l0_xyz, points=l0_points, 
        npoint=128, radius_list=[0.04,0.08], nsample_list=[64,128], mlp_list=[[32,64],[64,64]], 
        is_training=is_training, bn_decay=bn_decay, scope="layer1", bn=True, use_xyz=True, use_nchw=False)
    print("l1_xyz.shape", l1_xyz.shape)
    print("l1_points.shape", l1_points.shape)
    l2_xyz, l2_points, _ = pointnet_sa_module(l1_xyz, l1_points,
        npoint=None, radius=None, nsample=None, mlp=[64,128,256],
        mlp2=None, group_all=True, is_training=is_training,
        bn_decay=bn_decay, scope='layer2')
    end_points["abstraction"] = l2_points
    
    # feature propagation layers
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points,
        [32,32], is_training, bn_decay, scope='fa_layer1')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz,
        l0_xyz, l1_points,
        [32,32], is_training, bn_decay, scope='fa_layer2')
    end_points["pc_features"] = l0_points
    
    # FC layers. Note that in this task the layers are used for regression
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True,
        is_training=is_training, scope='conv1d-fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7,
        is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, 32, 1, padding='VALID', bn=True, 
        is_training=is_training, scope='conv1d-fc2', bn_decay=bn_decay)
    end_points["fc_features"] = net
    CoMM = tf_util.conv1d(net, 4, 1, padding='VALID', 
        activation_fn=None, scope='conv1d-fc3')
    return CoMM, end_points

def loss(pred, label):
    """ pred: B*N*4,
        label: B*N*4, """
    # compute the distance between the predicted and the ground truth
    dist = tf.reduce_sum((pred - label)**2, axis=2)
    # compute the loss
    loss = tf.reduce_mean(dist)
    tf.summary.scalar('loss', loss)
    return loss