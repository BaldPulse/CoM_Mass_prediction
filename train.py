import shutil
from model import *
from data import *


# delete the log folder
if os.path.exists(os.path.join(BASE_DIR, 'log')):
    shutil.rmtree(os.path.join(BASE_DIR, 'log'))

# load the data
print(BASE_DIR)
DATA = {}
# load every file in the data folder
DATA['pc'], DATA['target_values'] = load_data(os.path.join(BASE_DIR, 'data', '0'))
# print the shape of the data
print("Shape of the point cloud: ", DATA['pc'].shape)
print("Shape of the target values: ", DATA['target_values'].shape)



# normalize the data
# pc_mean, pc_std, lbl_mean, lbl_std  = get_PC_stats(os.path.join(BASE_DIR, 'data', '0'))
# Norm_PC, Norm_Target_Values = normalize_data(DATA['pc'], DATA['target_values'], pc_mean, pc_std, lbl_mean, lbl_std)

Norm_PC = DATA['pc']
Norm_Target_Values = DATA['target_values']

# randomize the data
idx = np.arange(Norm_PC.shape[0])
np.random.shuffle(idx)
PC_shuffled = Norm_PC[idx]
Target_Values_shuffled = Norm_Target_Values[idx]

# train eval split; first 90% for training, last 10% for evaluation
train_idx = range(int(0.9*PC_shuffled.shape[0]))
eval_idx = range(int(0.9*PC_shuffled.shape[0]), PC_shuffled.shape[0])

# training data
train_data = PC_shuffled[train_idx]
train_target_values = Target_Values_shuffled[train_idx]
print ("train_data: ", train_data.shape)
print ("train_target_values: ", train_target_values.shape)

# evaluation data
eval_data = PC_shuffled[eval_idx]
eval_target_values = Target_Values_shuffled[eval_idx]


# hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_POINT = 5000
LEARNING_RATE = 0.001

# create log folder
if not os.path.exists(os.path.join(BASE_DIR, 'log')):
    os.mkdir(os.path.join(BASE_DIR, 'log'))
LOG_FOUT = open(os.path.join(BASE_DIR, 'log', 'log_train.txt'), 'w')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def train():
    last_eval_loss = 1000000
    consecutive_eval_loss_increase = 0
    
    with tf.Graph().as_default():
        with tf.device('/gpu:'+"0"):
            pointclouds_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))
            labels_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 4))
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # create the model
            pred, end_points = get_model(pointclouds_pl, is_training_pl, bn_decay=None)
            print("pred: ", pred)
            # loss,
            loss = tf.reduce_mean( tf.reduce_sum(tf.squared_difference(pred[:, :, :3], labels_pl[:, :, :3]), axis=-1 ) )
            tf.summary.scalar('loss', loss)

            # optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, name='Adam')
            train_op = optimizer.minimize(loss)

            # add ops to save and restore all the variables
            saver = tf.train.Saver()


        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(BASE_DIR, 'log', 'train'), sess.graph) 
        test_writer = tf.summary.FileWriter(os.path.join(BASE_DIR, 'log', 'test'), sess.graph)  


        # create a global step variable
        tf.train.create_global_step(sess.graph)
        # Init variables
        init = tf.global_variables_initializer()

        # Restore variables from disk. (if possible)
        if os.path.exists(os.path.join(BASE_DIR, 'log', 'model.ckpt.index')):
            saver.restore(sess, os.path.join(BASE_DIR, 'log', 'model.ckpt'))
            print("Model restored.")
        else:
            sess.run(init)


        ops = {'pointclouds_pl': pointclouds_pl,
                'labels_pl': labels_pl,
                'is_training_pl': is_training_pl,
                'pred': pred,
                'loss': loss,
                'train_op': train_op,
                'merged': merged,
                'step': tf.train.get_global_step()
                }


        for epoch in range(NUM_EPOCHS):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            # write the loss of this epoch into the log file


            # every 10 epochs, save the model into a checkpoint file
            # the checkpoint file should be separate for each epoch
            if (epoch+1) % 5 == 0:
                # evaluate the model on the evaluation data
                eval_loss = eval_model(sess, ops, test_writer)
                save_path = saver.save(sess, os.path.join(BASE_DIR, 'log', 'model.ckpt'))
                log_string("Model saved in file: %s" % save_path)

                # if the loss if not decreasing, stop the training
                if epoch > 0 and eval_loss > last_eval_loss:
                    consecutive_eval_loss_increase += 1
                else:
                    last_eval_loss = eval_loss
                    
                if consecutive_eval_loss_increase > 3:
                    log_string("Loss is not decreasing, stop training.")
                    break
        
        LOG_FOUT.close()


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Shuffle train files
    train_file_idxs = np.arange(0, len(train_data))
    np.random.shuffle(train_file_idxs)


    # batch training
    for batch in range(len(train_data)//BATCH_SIZE):
        start_idx = batch * BATCH_SIZE
        end_idx = (batch+1) * BATCH_SIZE

        batch_data = train_data[train_file_idxs[start_idx:end_idx]]
        batch_target_values = train_target_values[train_file_idxs[start_idx:end_idx]]

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_target_values,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)

        if batch % 10 == 0:
            log_string('batch: '+ str(batch)+ ' loss: '+ str(loss_val))
            # # sanity check, manually calculate the loss
            # # manually calculate the loss
            # loss = np.sum(np.square(pred[:, :, :3] - batch_target_values[:, :, :3]))/BATCH_SIZE/NUM_POINT
            # print("loss: ", loss)
            sys.stdout.flush()

    


def eval_model(sess, ops, test_writer):
    is_training = True

    # eval on eval data
    eval_loss = 0
    for batch in range(len(eval_data)//BATCH_SIZE):
        start_idx = batch * BATCH_SIZE
        end_idx = (batch+1) * BATCH_SIZE

        batch_data = eval_data[start_idx:end_idx]
        batch_target_values = eval_target_values[start_idx:end_idx]

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_target_values,
                     ops['is_training_pl']: is_training,}
        summary, step, loss_val, pred = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        eval_loss += loss_val
        if (batch+1) %1 == 0:
            # sanity check, plot the first point cloud and the predicted center of mass
            print("tv: ", batch_target_values[0])
            plot(batch_data[0], pred[0], batch_target_values[0,:,0:3])

    eval_loss /= (len(eval_data)//BATCH_SIZE)
    print('eval loss: %f' % (eval_loss))
    return eval_loss

if __name__ == "__main__":
    train()