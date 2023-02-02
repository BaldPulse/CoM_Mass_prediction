from model import *

DATA = np.load(os.path.join(BASE_DIR, 'data', 'data.npz'), allow_pickle=True)
# load the mean and std
mean_std = np.load(os.path.join(BASE_DIR, 'data', 'mean_std.npz'), allow_pickle=True)
pc_mean = mean_std['pc_mean']
pc_std = mean_std['pc_std']

# randomly select a point cloud
idx = np.random.randint(0, DATA['valid_count'])
pointclouds = DATA['pc'][idx].reshape(1, 20000, 3)
# normalize the point cloud
pointclouds_norm = (pointclouds - pc_mean) / pc_std


GT_CoMs = pointclouds[0,:,:3]- DATA['target_values'][idx, :, :3]
GT_CoM = np.mean(GT_CoMs, axis=0)

BATCH_SIZE = 1
NUM_POINT = 20000

# create session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.Session(config=config)

# get the model
pointclouds_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))
labels_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 4))
is_training_pl = tf.placeholder(tf.bool, shape=())

# create the model
pred, end_points = get_model(pointclouds_pl, is_training_pl, bn_decay=None)

# restore the model
saver = tf.train.import_meta_graph(os.path.join(BASE_DIR, 'log', 'model.ckpt.index'))
saver.restore(sess, os.path.join(BASE_DIR, 'log', 'model.ckpt'))

# run the model
feed_dict = {pointclouds_pl: pointclouds, is_training_pl: False}
pred_val = sess.run([pred], feed_dict=feed_dict)[0][0]
# CoM translation
pred_CoMs = pointclouds[0,:,:3]- pred_val[:,:3]
# average the predictions
pred_CoM = np.mean(pred_CoMs, axis=0)
print(pred_val.shape)
pred_mass = np.mean(pred_val[:,3])
print("Predicted CoM unorm: ", pred_CoM)
print("GT CoM unorm: ", (GT_CoM-pc_mean[:3])/pc_std[:3])
# unnormalize the CoM
pred_CoM = pred_CoM * pc_std[:3] + pc_mean[:3]
print("Mean of the point cloud: ", pc_mean)
print("Std of the point cloud: ", pc_std)
print("Predicted CoM: ", pred_CoM)
print("Ground truth CoM: ", GT_CoM)
print("L2 Loss: ", np.linalg.norm(pred_CoM-GT_CoM))

import trimesh
CoM_matrix = trimesh.transformations.translation_matrix(pred_CoM)
GT_CoM_matrix = trimesh.transformations.translation_matrix(GT_CoM)
# show the point cloud with the predicted CoM as an axis
scene = trimesh.Scene()
scene.add_geometry(trimesh.PointCloud(pointclouds[0,:, :3]))
# add the CoM to the PC scene as an axis
scene.add_geometry(trimesh.creation.axis(origin_color=[1., 0, 0], origin_size=0.01, axis_length=0.02, axis_radius=0.001), transform=CoM_matrix)
scene.add_geometry(trimesh.creation.axis(origin_color=[0., 1, 1], origin_size=0.01, axis_length=0.02, axis_radius=0.001), transform=GT_CoM_matrix)

# show the scene
scene.show()

