from model import *

# load the data
print(BASE_DIR)
DATA = np.load(os.path.join(BASE_DIR, 'data', 'data.npz'), allow_pickle=True)
print("Loaded data with headers ", DATA.files)
print("Valid line numbers: ", DATA['valid_count'])

# train eval split; first 80% for training, last 20% for evaluation
train_idx = np.arange(0, int(0.8*DATA["valid_count"]))
eval_idx = np.arange(int(0.8*DATA["valid_count"]), DATA["valid_count"])

# training data
train_data = DATA['pc'][train_idx]
train_target_values = DATA['target_values'][train_idx]


# evaluation data
eval_data = DATA['pc'][eval_idx]
eval_target_values = DATA['target_values'][eval_idx]


# hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_POINT = 20000
LEARNING_RATE = 0.001


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+"0"):
            pointclouds_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))
            labels_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 4))
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # create the model
            pred, end_points = get_model(pointclouds_pl, is_training_pl, bn_decay=None)

            # loss
            loss = tf.reduce_mean(tf.square(pred-labels_pl))
            tf.summary.scalar('loss', loss)

            # optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
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
        test_writer = tf.summary.FileWriter(os.path.join(BASE_DIR, 'log', 'test'))


        # Create a variable to hold the global_step.
        global_step_tensor  = tf.Variable(0)
        # init the global step
        global_step_tensor.initializer.run(session=sess)

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
                'step': global_step_tensor
                }

        LOG_FOUT = open(os.path.join(BASE_DIR, 'log', 'log_train.txt'), 'w')
        def log_string(out_str):
            LOG_FOUT.write(out_str+'\n')
            LOG_FOUT.flush()
            print(out_str)

        for epoch in range(NUM_EPOCHS):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(BASE_DIR, 'log', 'model.ckpt'))
                log_string("Model saved in file: %s" % save_path)
                # eval the current model on eval data
                eval_model(sess, ops, test_writer)
        
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
        # convert batch data and target values to array
        # currently the data is a list of numpy arrays

        
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_target_values,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        if batch % 10 == 0:
            print('batch: %d, loss: %f' % (batch, loss_val))
    


def eval_model(sess, ops, test_writer):
    is_training = False

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
        summary, step, loss_val = sess.run([ops['merged'], ops['step'],
            ops['loss']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        eval_loss += loss_val
    eval_loss /= (len(eval_data)//BATCH_SIZE)
    print('eval loss: %f' % (eval_loss))

if __name__ == "__main__":
    train()