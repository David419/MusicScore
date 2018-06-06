# coding=utf-8
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import time
import sys


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

RNN_train_dataset = '/home/todd/New_Rule/Data/dataset/train.tfrecords'
RNN_validation_dataset = '/home/todd/New_Rule/Data/dataset/validation.tfrecords'
RNN_test_dataset = '/home/todd/New_Rule/Data/dataset/test_dataset.tfrecords'
train_midi_num = 36452
val_midi_num = 6295
test_midi_num = 6476
rnn_event_len = 10
overlap_len = 32
rnn_midi_len = 256

threshold_h = 125
threshold_l = 3
round_num = 0.5


decay = 0.8
length = timestep_size = rnn_midi_len         # 句子长度
width = input_size = embedding_size = rnn_event_len      # 字向量长度
class_num = 10
hidden_size = 256    # 隐含层节点数
layer_num = 2        # bi-lstm 层数
save_epoch = 10  # 每10个epoch 保存一次模型

# optimize hyper paratmeters
tr_batch_size = 64
init_lr = 0.01
lr_change_rate = 10
lr_change_epoch = [0, 25, 500, 600, 700]
max_grad_norm = 1.0  # 最大梯度（超过此值的梯度将被裁剪）
max_max_epoch = 800
drop_keep_rate = 0.5

lr = tf.placeholder(tf.float32, [], name='learning_rate')
keep_prob = tf.placeholder(tf.float32, [], name='keep_prob') # dropout rate
batch_size = tf.placeholder(tf.int32, [], name='batch_size')  # 注意类型必须为 tf.int32


tensorboard_save_path = 'logs'
model_save_path = tensorboard_save_path + '/ckpt/bi-lstm.ckpt'  # 模型保存位置


# get dataset
def read_and_decode(tfrecord_file, midi_num):
    # midi_num is how many midi in this dataset
    # e.g.  the test dataset contains 500 midis, so the midi_num is 500

    filename_q = tf.train.string_input_producer([tfrecord_file], num_epochs=None)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_q)

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'events': tf.FixedLenFeature([(rnn_event_len+1) * rnn_midi_len], tf.int64),
                                           'labels': tf.FixedLenFeature([rnn_event_len * rnn_midi_len], tf.int64),
                                           'keys_mask': tf.FixedLenFeature([rnn_event_len * rnn_midi_len], tf.int64),
                                           'durations': tf.FixedLenFeature([(rnn_event_len+1) * rnn_midi_len], tf.int64),
                                       })

    label_out = features['labels']
    events_out = features['events']
    keys_mask_out = features['keys_mask']
    durations_out = features['durations']
    with tf.Session() as sess:
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)

        events_out_list = list()
        label_out_list = list()
        keys_mask_out_list = list()
        durations_out_list = list()
        # for i in range(10):  # debug
        for i in range(midi_num):
            # [label_out_single, events_out_single, keys_mask_out_single, midi_mask_out_single] = sess.run([
            #     label_out, events_out, keys_mask_out, midi_mask_out
            # ])
            [events_out_single, label_out_single, keys_mask_out_single, durations_out_single] = sess.run([
                events_out, label_out, keys_mask_out, durations_out
            ])

            # reshape
            events_out_single = events_out_single.reshape([rnn_midi_len, -1])
            label_out_single = label_out_single.reshape([rnn_midi_len, -1])
            keys_mask_out_single = keys_mask_out_single.reshape([rnn_midi_len, -1])
            durations_out_single = durations_out_single.reshape([rnn_midi_len, -1])
            # gather all midi
            events_out_list.append(events_out_single)
            label_out_list.append(label_out_single)
            keys_mask_out_list.append(keys_mask_out_single)
            durations_out_list.append(durations_out_single)
        events_out_array = np.array(events_out_list)
        label_out_array = np.array(label_out_list)
        keys_mask_out_array = np.array(keys_mask_out_list)
        durations_out_array = np.array(durations_out_list)
    return events_out_array, label_out_array, keys_mask_out_array, durations_out_array

event_val, y_val, mask_val, dur_val = read_and_decode(RNN_validation_dataset, val_midi_num)
event_test, y_test, mask_test, dur_test = read_and_decode(RNN_test_dataset, test_midi_num)
event_train, y_train, mask_train, dur_train = read_and_decode(RNN_train_dataset, train_midi_num)
X_val = list()
for i in range(len(event_val)):
    for j in range(len(event_val[0])):
        for k in range(len(event_val[0][0])):
            X_val.append([event_val[i][j][k], dur_val[i][j][k]])
X_val = np.array(X_val)
X_val = X_val.reshape([-1, rnn_midi_len, (rnn_event_len+1) * 2])
X_test = list()
for i in range(len(event_test)):
    for j in range(len(event_test[0])):
        for k in range(len(event_test[0][0])):
            X_test.append([event_test[i][j][k], dur_test[i][j][k]])
X_test = np.array(X_test)
X_test = X_test.reshape([-1, rnn_midi_len, (rnn_event_len+1) * 2])
X_train = list()
for i in range(len(event_train)):
    for j in range(len(event_train[0])):
        for k in range(len(event_train[0][0])):
            X_train.append([event_train[i][j][k], dur_train[i][j][k]])
X_train = np.array(X_train)
X_train = X_train.reshape([-1, rnn_midi_len, (rnn_event_len+1) * 2])


def set_threshold(inputs, masks, min_value=0, max_value=128):
    ones_matrix = np.ones([i for i in inputs.shape])
    max_matrix = ones_matrix * max_value
    min_matrix = ones_matrix * min_value
    m1 = np.greater_equal(inputs, min_matrix)
    m2 = np.less_equal(inputs, max_matrix)
    m = m1 * m2
    return m * masks


def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def lstm_cell():
    cell = rnn.LSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
    return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)


def bi_lstm(X_inputs):
    """build the bi-LSTMs network. Return the y_pred"""
    # X_inputs.shape = [batchsize, timestep_size]  ->  inputs.shape = [batchsize, timestep_size, embedding_size]
    inputs = X_inputs

    # ** 1.构建前向后向多层 LSTM
    cell_fw = rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple=True)
    cell_bw = rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple=True)

    # ** 2.初始状态
    initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
    initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)

    # ** 3. bi-lstm 计算（展开）
    with tf.variable_scope('bidirectional_rnn'):
        # *** 下面，两个网络是分别计算 output 和 state
        # Forward direction
        outputs_fw = list()
        state_fw = initial_state_fw
        with tf.variable_scope('fw'):
            for timestep in range(timestep_size):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                (output_fw, state_fw) = cell_fw(inputs[:, timestep, :], state_fw)
                outputs_fw.append(output_fw)

        # backward direction
        outputs_bw = list()
        state_bw = initial_state_bw
        with tf.variable_scope('bw') as bw_scope:
            inputs = tf.reverse(inputs, [1])
            for timestep in range(timestep_size):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                (output_bw, state_bw) = cell_bw(inputs[:, timestep, :], state_bw)
                outputs_bw.append(output_bw)
        # *** 然后把 output_bw 在 timestep 维度进行翻转
        # outputs_bw.shape = [timestep_size, batch_size, hidden_size]
        outputs_bw = tf.reverse(outputs_bw, [0])
        # 把两个outputs 拼成 [timestep_size, batch_size, hidden_size*2]
        output = tf.concat([outputs_fw, outputs_bw], 2)
        output = tf.transpose(output, perm=[1, 0, 2])
        output = tf.reshape(output, [-1, hidden_size * 2])
    # ***********************************************************
    return output  # [-1, hidden_size*2]


with tf.variable_scope('Inputs'):
    X_inputs = tf.placeholder(tf.float32, [None, timestep_size, (width + 1) * 2], name='x_input')
    y_inputs = tf.placeholder(tf.int32, [None, timestep_size, width], name='y_input')
    mask_inputs = tf.placeholder(tf.float32, [None, timestep_size, width], name='mask_input')


with tf.variable_scope('Outputs'):
    bilstm_output = bi_lstm(X_inputs)
    softmax_w = weight_variable([hidden_size * 2, class_num])
    softmax_b = bias_variable([class_num])
    y_pred = tf.add(tf.matmul(bilstm_output, softmax_w), softmax_b, name='y_pred')
    y_pred_sig = tf.nn.sigmoid(y_pred)
    y_pred_round = tf.cast(tf.greater_equal(y_pred_sig, round_num), tf.int32, name='y_pred_round')


with tf.variable_scope('Accuracy'):
    correct_prediction_ = tf.equal(tf.reshape(y_pred_round, [-1]), tf.reshape(y_inputs, [-1]))
    correct_prediction = tf.multiply(tf.cast(correct_prediction_, tf.float32),
                                     tf.cast(tf.reshape(mask_inputs, [-1]), tf.float32), name='correct_pred')
    accuracy = tf.divide(tf.reduce_sum(correct_prediction),
                         tf.reduce_sum(tf.cast(mask_inputs, tf.float32)), name='accuracy')


with tf.variable_scope('Cost'):
    def loss_fn(outputs, targets, input_mask):
        labels = tf.reshape(targets, [-1])  # to one vector
        labels = tf.cast(labels, tf.float32)
        masks = tf.to_float(tf.reshape(input_mask, [-1]))  # to one vector like targets
        losses = tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(outputs, [-1]), labels=labels), masks)
        loss = tf.divide(tf.reduce_sum(losses),  # loss from mask. reduce_sum before element-wise mul with mask !!
                         tf.reduce_sum(masks),
                         name="seq_loss_with_mask")
        return loss

    # Cost for Training
    cost = loss_fn(y_pred, y_inputs, mask_inputs)


with tf.variable_scope('Optimizer'):
    # ***** 优化求解 *******
    tvars = tf.trainable_variables()  # 获取模型的所有参数
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)  # 获取损失函数对于每个参数的梯度
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)   # 优化器
    # 梯度下降计算
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.contrib.framework.get_or_create_global_step())
print 'Finished creating the bi-lstm model.'


def test_epoch(data_x, data_y, data_mask, data_dur, is_val=False, _batch_size=10):
    """Testing or valid."""
    # _batch_size = 10
    fetches = [accuracy, cost]
    if is_val:
        data_size = val_midi_num
    else:
        data_size = test_midi_num
    batch_num = 0
    # start_time = time.time()
    _costs = 0.0
    _accs = 0.0

    for step in range(0, data_size, _batch_size):
        if step + _batch_size >= data_size:
            continue
        x = np.float32(data_x[step:step + _batch_size])
        y = data_y[step:step + _batch_size]
        y = y.reshape([_batch_size, length, width])
        mask = data_mask[step:step + _batch_size]
        mask = np.float32(mask.reshape([_batch_size, length, width]))
        feed_dict = {X_inputs: x, y_inputs: y, batch_size: _batch_size, mask_inputs: mask, lr: 1e-5, keep_prob: 1.0}
        _acc, _cost = sess.run(fetches, feed_dict)
        _accs += _acc
        _costs += _cost
        batch_num += 1

    mean_acc= _accs / batch_num
    mean_cost = _costs / batch_num

    return mean_acc, mean_cost


sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(tensorboard_save_path, sess.graph)  # put this command after sess.run(var_init)


saver = tf.train.Saver(max_to_keep=2)  # 最多保存的模型数量
for epoch in xrange(max_max_epoch):
    sys.stdout.flush()
    # if epoch > max_epoch:
    #     _lr = _lr * ((decay) ** (epoch - max_epoch))
    _lr = init_lr
    if epoch > lr_change_epoch[1]:
        _lr = init_lr / lr_change_rate
    if epoch > lr_change_epoch[2]:
        _lr = init_lr / (lr_change_rate ** 2)
    if epoch > lr_change_epoch[3]:
        _lr = init_lr / (lr_change_rate ** 3)
    if epoch > lr_change_epoch[4]:
        _lr = init_lr / (lr_change_rate ** 4)

    print 'EPOCH %d， lr=%g' % (epoch+1, _lr)
    start_time = time.time()
    _costs = 0.0
    _accs = 0.0
    show_accs = 0.0
    show_costs = 0.0
    tr_batch_num = 0  # 每个 epoch 中包含的 batch 数

    for step in range(0, train_midi_num, tr_batch_size):
        if step + tr_batch_size >= train_midi_num:
            continue
        fetches = [accuracy, cost, train_op]
        x = np.float32(X_train[step:step + tr_batch_size])
        y = y_train[step:step + tr_batch_size]
        y = y.reshape([tr_batch_size, length, width])
        mask = mask_train[step:step + tr_batch_size]
        mask = np.float32(mask.reshape([tr_batch_size, length, width]))
        # mask = set_threshold(x, mask, threshold_l, threshold_h)
        feed_dict = {X_inputs: x, y_inputs: y, mask_inputs: mask, batch_size: tr_batch_size, lr: _lr, keep_prob: drop_keep_rate}
        _acc, _cost, _ = sess.run(fetches, feed_dict)  # the cost is the mean cost of one batch
        _accs += _acc
        _costs += _cost
        show_accs += _acc
        show_costs += _cost
        tr_batch_num += 1

    mean_acc = _accs / tr_batch_num
    mean_cost = _costs / tr_batch_num
    valid_acc, valid_cost = test_epoch(X_val, y_val, mask_val, dur_val, is_val=True)  # valid

    summary1 = tf.Summary(value=[
        tf.Summary.Value(tag="train_loss", simple_value=mean_cost),
        tf.Summary.Value(tag="train_acc", simple_value=mean_acc),
        tf.Summary.Value(tag="learning_rate", simple_value=_lr),
        tf.Summary.Value(tag="validate_loss", simple_value=valid_cost),
        tf.Summary.Value(tag="validate_acc", simple_value=valid_acc),
    ])
    summary_writer.add_summary(summary1, epoch + 1)
    summary_writer.flush()

    if (epoch + 1) % save_epoch == 0:  # 每 save_epoch 个 epoch 保存一次模型
        save_path = saver.save(sess, model_save_path, global_step=(epoch+1))
        print 'the save path is ', save_path
    print '\ttraining, acc=%g, cost=%g;  valid acc= %g, cost=%g' % (mean_acc, mean_cost, valid_acc, valid_cost)
    print 'Epoch training %d, speed=%g s/epoch' % (train_midi_num, time.time()-start_time)
# testing
print '**TEST RESULT:'
test_acc, test_cost = test_epoch(X_test, y_test, mask_test, dur_test)
print '**Test %d, acc=%g, cost=%g' % (test_midi_num, test_acc, test_cost)
summary_writer.close()
sess.close()
