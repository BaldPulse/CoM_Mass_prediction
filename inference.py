from model import *
from data import *

# load the data
print(BASE_DIR)
DATA = {}
# load every file in the data folder
DATA['pc'], DATA['target_values'] = load_data(os.path.join(BASE_DIR, 'data', '0'))
# print the shape of the data
print("Shape of the point cloud: ", DATA['pc'].shape)
print("Shape of the target values: ", DATA['target_values'].shape)

# normalize the data
# pc_mean, pc_std, lbl_mean, lbl_std = get_PC_stats(os.path.join(BASE_DIR, 'data', '0'))
# Norm_PC, Norm_Target_Values = normalize_data(DATA['pc'], DATA['target_values'], pc_mean, pc_std, lbl_mean, lbl_std)

Norm_PC = DATA['pc']
Norm_Target_Values = DATA['target_values']

BATCH_SIZE = 8
NUM_POINT = 5000
# get a random point cloud
idx = np.random.randint(0, Norm_PC.shape[0], size=BATCH_SIZE)
pointclouds = Norm_PC[idx]
pointclouds_unnorm = DATA['pc'][idx]
GT_CoM_unnorms = (DATA['pc'][idx, :, :3] - DATA['target_values'][idx,:, :3])
GT_CoM_unnorm = np.mean(GT_CoM_unnorms, axis=1)
print("GT_CoM_unnorm: ", GT_CoM_unnorm.shape)
GT_mass = Norm_Target_Values[idx, 3]


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

loss = tf.reduce_mean( tf.reduce_sum(tf.squared_difference(pred[:, :, :3], labels_pl[:, :, :3]), axis=-1 ), axis=-1 )

# run the model
feed_dict = {pointclouds_pl: pointclouds, is_training_pl: True, labels_pl: Norm_Target_Values[idx]}
pred_vals, loss_vals = sess.run([pred, loss], feed_dict=feed_dict)
print("loss: ", loss_vals)
for i in range(BATCH_SIZE):
    pred_val = pred_vals[i]
    loss_val = loss_vals[i]
    # CoM translation
    pred_CoMs = pointclouds_unnorm[i,:,:3]- pred_val[:,:3]#*lbl_std[:3] + lbl_mean[:3]
    # average the predictions
    pred_CoM_unnorm = np.mean(pred_CoMs, axis=0)
    print("Loss: ", loss_val)
    print("Predicted CoM: ", pred_CoM_unnorm)
    print("Ground truth CoM: ", GT_CoM_unnorm[i])
    # also print out a random output
    print("L2 dist: ", np.sum((pred_CoM_unnorm-GT_CoM_unnorm[i])**2)**0.5)

    # RANSAC
    print("RANSAC")
    filtered_CoM, inliers = get_CoM(pred_val, pointclouds_unnorm[i])
    print("Inliers: ", inliers.shape)

    import trimesh
    CoM_matrix = trimesh.transformations.translation_matrix(pred_CoM_unnorm)
    GT_CoM_matrix = trimesh.transformations.translation_matrix(GT_CoM_unnorm[i])
    Filter_CoM_matrix = trimesh.transformations.translation_matrix(filtered_CoM)
    # show the point cloud with the predicted CoM as an axis
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(pointclouds_unnorm[i,:, :3]))
    # add the CoM to the PC scene as an axis
    scene.add_geometry(trimesh.creation.axis(origin_color=[1., 0, 0], origin_size=0.005, axis_length=0.02, axis_radius=0.001), transform=CoM_matrix)
    scene.add_geometry(trimesh.creation.axis(origin_color=[0., 1, 1], origin_size=0.005, axis_length=0.02, axis_radius=0.001), transform=GT_CoM_matrix)
    scene.add_geometry(trimesh.creation.axis(origin_color=[0., 0, 1], origin_size=0.005, axis_length=0.02, axis_radius=0.001), transform=Filter_CoM_matrix)

    # randomly plot some CoMs 
    # for i in range(10):
    #     idx = np.random.randint(0, pred_CoMs.shape[0])
    #     CoM_matrix = trimesh.transformations.translation_matrix(pred_CoMs[idx, :3])# * lbl_std[:3] + lbl_mean[:3])
    #     scene.add_geometry(trimesh.creation.axis(origin_color=[0., 1, 0], origin_size=0.01, axis_length=0.02, axis_radius=0.001), transform=CoM_matrix)

    # randomly plot some of the inliers
    for i in range(10):
        idx = np.random.randint(0, inliers.shape[0])
        CoM_matrix = trimesh.transformations.translation_matrix(inliers[idx, :3])# * lbl_std[:3] + lbl_mean[:3])
        scene.add_geometry(trimesh.creation.axis(origin_color=[0., 0, 0.5], origin_size=0.001, axis_length=0.005, axis_radius=0.0001), transform=CoM_matrix)
    # show the scene
    scene.show()

