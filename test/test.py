# coding=utf-8
import tensorflow as tf
import structure as st
import numpy as np
import sys


ckpt_path = '/home/todd/final_exp/on_d6/logs/ckpt'
RNN_test_dataset = '/home/todd/New_Rule/Data/dataset/new_lasting_test.tfrecords'
test_midi_num = 6476

rnn_event_len = 10
rnn_midi_len = 256
lasting_padded_len = 10


# get dataset
def read_and_decode(tfrecord_file, midi_num):
    # midi_num is how many midi in this dataset
    # e.g.  the test dataset contains 500 midis, so the midi_num is 500

    filename_q = tf.train.string_input_producer([tfrecord_file], num_epochs=None)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_q)

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'events': tf.FixedLenFeature([(rnn_event_len+lasting_padded_len) * rnn_midi_len], tf.int64),
                                           'labels': tf.FixedLenFeature([rnn_event_len * rnn_midi_len], tf.int64),
                                           'keys_mask': tf.FixedLenFeature([rnn_event_len * rnn_midi_len], tf.int64),
                                       })

    label_out = features['labels']
    events_out = features['events']
    keys_mask_out = features['keys_mask']
    with tf.Session() as sess:
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)

        events_out_list = list()
        label_out_list = list()
        keys_mask_out_list = list()
        # for i in range(10):  # debug
        for i in range(midi_num):
            [events_out_single, label_out_single, keys_mask_out_single] = sess.run([
                events_out, label_out, keys_mask_out
            ])
            # reshape
            events_out_single = events_out_single.reshape([rnn_midi_len, -1])
            label_out_single = label_out_single.reshape([rnn_midi_len, -1])
            keys_mask_out_single = keys_mask_out_single.reshape([rnn_midi_len, -1])
            # gather all midi
            events_out_list.append(events_out_single)
            label_out_list.append(label_out_single)
            keys_mask_out_list.append(keys_mask_out_single)
        events_out_array = np.array(events_out_list)
        label_out_array = np.array(label_out_list)
        keys_mask_out_array = np.array(keys_mask_out_list)
    return events_out_array, label_out_array, keys_mask_out_array


events, labels, masks = read_and_decode(RNN_test_dataset, test_midi_num)
for i in range(6,7):
    ckpt_path = '/home/todd/final_exp/on_d6/logs/ckpt'
    _, acc = st.get_output(ckpt_path, events, masks, test_midi_num, labels=labels)
    print "on_d%d Test acc:" % (i+1), acc
# _, acc = st.get_output(ckpt_path, events, masks, test_midi_num, labels=labels)
# print "Test acc:", acc
