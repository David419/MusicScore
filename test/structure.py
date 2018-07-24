# coding=utf-8
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import sys

hidden_size = 256


layer_num = 2
class_num = 10
length = timestep_size = 256
width = input_size = embedding_size = 10
lasting_padded_len = 10
round_num = 0.5

lr = tf.placeholder(tf.float32, [], name='learning_rate')
keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')  # dropout rate
batch_size = tf.placeholder(tf.int32, [], name='batch_size')


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
    return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob, state_keep_prob=keep_prob)


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
    X_inputs = tf.placeholder(tf.float32, [None, timestep_size, width + lasting_padded_len], name='x_input')
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


def get_output(checkpoint_file_path, events, masks, seq_num, labels=None, midi_num=None):
    no_labels = False
    if labels is None:
        no_labels = True
        labels = np.zeros(np.shape(masks))
    pred_right_hand = tf.multiply(tf.cast(tf.reshape(y_pred_round, [-1, timestep_size, width]), tf.float32),
                                  tf.cast(mask_inputs, tf.float32))
    if midi_num is None:
        midi_num = 1
        events = [events]
        masks = [masks]
        seq_num = [seq_num]
        labels = [labels]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # saver = tf.train.import_meta_graph(meta_path)
    saver = tf.train.Saver(max_to_keep=2)
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_file_path)
    saver.restore(sess, checkpoint_path)
    graph = tf.get_default_graph()
    pred_list = list()
    acc_list = list()
    for i in range(midi_num):
        print 'processing midi', i
        sys.stdout.flush()
        ev = events[i]
        la = labels[i]
        b_s = seq_num[i]
        m_i = masks[i]
        feed_dict = {X_inputs: ev, y_inputs: la, batch_size: b_s, mask_inputs: m_i, lr: 1e-5, keep_prob: 1.0}
        if no_labels:
            pred = sess.run(pred_right_hand, feed_dict=feed_dict)
            # print 'func_pred:', np.shape(pred)
            pred_list.append(pred)
        else:
            pred, acc = sess.run([pred_right_hand, accuracy], feed_dict=feed_dict)
            pred_list.append(pred)
            acc_list.append(acc)
    if midi_num == 1:
        if no_labels:
            return pred_list[0]
        else:
            return pred_list[0], acc_list[0]
    else:
        if no_labels:
            return pred_list
        else:
            return pred_list, acc_list

