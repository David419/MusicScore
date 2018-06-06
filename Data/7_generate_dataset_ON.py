#!/usr/bin/python
# -*- coding:utf8 -*-
import copy
import random
import tensorflow as tf
import process_func as pf
import mido as mi
import numpy as np
import os
import sys
from compiler.ast import flatten


rnn_event_len = 10
lasting_padded_len = 10
overlap_len = 32
rnn_midi_len = 256

# to place your own the whole selected(old) MIDI document
old_midi_dir_path = '/home/todd/Data/old_midi'

if not os.path.exists('./dataset'):
    os.mkdir('./dataset')

lasting_train_dataset_new = './dataset/new_lasting_train.tfrecords'
lasting_validation_dataset_new = './dataset/new_lasting_validation.tfrecords'
lasting_test_dataset_new = './dataset/new_lasting_test.tfrecords'

test_midi = './path/test_midi_new_rule.txt'
val_midi = './path/val_midi_new_rule.txt'
train_midi = './path/train_midi_new_rule.txt'


def generate_dataset(tf_record, midi_txt):

    def get_lasting_keys_on(_events, _durations, _key_on_intervals):
        def remove_redundance(keys_on_info, on_time_info, off_time_info):
            new_keys_on_info = [i for i in keys_on_info]
            new_on_time_info = [i for i in on_time_info]
            new_off_time_info = [i for i in off_time_info]
            while True:
                if len(new_keys_on_info) <= 10:
                    break
                first_on_pos = list()
                for i, on_time in enumerate(new_on_time_info):
                    if on_time == new_on_time_info[0]:
                        first_on_pos.append(i)
                pop_pos = list()
                first_off_time = new_off_time_info[0]
                for i, pos in enumerate(first_on_pos):
                    if new_off_time_info[pos] < first_off_time:
                        pop_pos = [pos]
                    elif new_off_time_info[pos] == first_off_time:
                        pop_pos.append(pos)
                    else:
                        continue
                for i, pos in enumerate(pop_pos):
                    new_keys_on_info.pop(pos - i)
                    new_on_time_info.pop(pos - i)
                    new_off_time_info.pop(pos - i)
            return new_keys_on_info, new_on_time_info, new_off_time_info

        keys_on = list()
        keys_on_time = list()
        keys_off_time = list()
        current_time = 0

        lasting_keys = list()

        for i, event in enumerate(_events):
            current_time += _key_on_intervals[i]
            if keys_on:  # keys_on list is not empty
                new_keys_on_time = [temp for temp in keys_on_time]
                new_keys_on = [temp for temp in keys_on]
                new_keys_off_time = [temp for temp in keys_off_time]
                for j, key_on in enumerate(keys_on):
                    if keys_off_time[j] <= current_time:
                        a = new_keys_on.index(keys_on[j])
                        new_keys_on.pop(a)
                        new_keys_off_time.pop(a)
                        new_keys_on_time.pop(a)
            else:  # keys_on list is empty
                new_keys_on_time = list()
                new_keys_on = list()
                new_keys_off_time = list()
            new_keys_on, new_keys_on_time, new_keys_off_time = remove_redundance(new_keys_on, new_keys_on_time, new_keys_off_time)
            if len(new_keys_on) > 10:
                print 'The new keys_on length is larger than 10, which is:', len(new_keys_on)
            lasting_keys.append(new_keys_on)

            keys_on_time = [temp for temp in new_keys_on_time]
            keys_on = [temp for temp in new_keys_on]
            keys_off_time = [temp for temp in new_keys_off_time]
            for j, key in enumerate(event):
                if key in keys_on:
                    pos = keys_on.index(key)
                    keys_on.pop(pos)
                    keys_off_time.pop(pos)
                    keys_on_time.pop(pos)
                keys_on.append(key)
                keys_on_time.append(current_time)
                keys_off_time.append(_durations[i][j] + current_time)

        return lasting_keys

    def get_dataset_on_and_lasting(_midi_path):
        invalid_num = 0
        paths = open(_midi_path)
        midi_list = list()
        while True:
            path = paths.readline()
            if not path:
                break
            path = path.replace('\n', '')
            midi_list.append(path)
        paths.close()

        split_events = list()
        split_labels = list()
        split_mask = list()

        for _count, _path in enumerate(midi_list):
            print _count, pf.find_file_name(_path)
            sys.stdout.flush()
            old_midi = mi.MidiFile(_path)
            is_invalid_midi = False
            _labels, _events, _durations, _key_on_intervals = pf.get_info(old_midi)

            _lasting_keys_on = get_lasting_keys_on(_events, _durations, _key_on_intervals)

            _events_len = len(_events)
            _interval_len = len(_key_on_intervals)
            if _events_len != _interval_len:
                print pf.find_file_name(_path), 'is invalid because the length of events and intervals are not same!'
                continue

            keys_mask = list()  # to confirm whether each key in each event is a padding value
            padded_events = list()
            padded_labels = _labels

            pad_event_num = 0
            pad_label_num = 0
            pad_lasting_num = 0

            null_event = [pad_event_num for n in range(rnn_event_len)]
            null_label = [pad_label_num for n in range(rnn_event_len)]
            null_lasting = [pad_lasting_num for n in range(lasting_padded_len)]
            null_key_mask = [0 for n in range(rnn_event_len)]

            # padding keys
            for _i, event in enumerate(_events):
                a_event_key_mask = list()
                a_event = list()
                a_lasting = [n for n in _lasting_keys_on[_i]]

                key_n = len(event)
                if key_n < rnn_event_len:
                    for n in range(key_n):
                        a_event_key_mask.append(1)
                        a_event.append(_events[_i][n])
                    for j in range(rnn_event_len - key_n):
                        a_event.append(pad_event_num)
                        a_event_key_mask.append(0)
                        padded_labels[_i].append(pad_label_num)
                elif key_n == rnn_event_len:
                    for n in range(key_n):
                        a_event_key_mask.append(1)
                        a_event.append(_events[_i][n])
                else:
                    is_invalid_midi = True
                    print 'the number of pressed key at once is larger than 10!'
                    sys.stdout.flush()
                    break

                # pad a lasting keys on event
                last_n = len(_lasting_keys_on[_i])
                if last_n < lasting_padded_len:
                    for j in range(lasting_padded_len - last_n):
                        a_lasting.append(pad_lasting_num)

                keys_mask.append(a_event_key_mask)
                padded_events.append(a_event + a_lasting)

            if len(np.shape(padded_events)) != 2:
                is_invalid_midi = True

            if is_invalid_midi:
                print 'The midi is invalid!'
                invalid_num += 1
                continue
            # print "shape event:", np.shape(padded_events)
            # pad top of the sequence with length of overlap_len
            for _i in range(overlap_len):
                padded_events.insert(0, null_event + null_lasting)
                padded_labels.insert(0, null_label)
                keys_mask.insert(0, null_key_mask)

            seq_len = len(padded_events)
            for i in range(0, seq_len, rnn_midi_len - overlap_len):
                # padding split sequence
                if (seq_len - i) < rnn_midi_len:
                    temp_events = copy.deepcopy(padded_events[i:seq_len])
                    temp_labels = copy.deepcopy(padded_labels[i:seq_len])
                    # temp_lasting = padded_lasting[i:seq_len]
                    temp_keys_mask = copy.deepcopy(keys_mask[i:seq_len])
                    for j in range(rnn_midi_len - (seq_len - i)):
                        temp_events.append(null_event + null_lasting)
                        temp_labels.append(null_label)
                        temp_keys_mask.append(null_key_mask)
                    for jj in range(overlap_len):
                        for kk in range(rnn_event_len):
                            temp_keys_mask[jj][kk] = 0
                    split_events.append(temp_events)
                    split_labels.append(temp_labels)
                    split_mask.append(temp_keys_mask)
                else:
                    temp_keys_mask = copy.deepcopy(keys_mask[i:i + rnn_midi_len])
                    for jj in range(overlap_len):
                        for kk in range(rnn_event_len):
                            temp_keys_mask[jj][kk] = 0
                    split_events.append(padded_events[i:i + rnn_midi_len])
                    split_labels.append(padded_labels[i:i + rnn_midi_len])
                    split_mask.append(temp_keys_mask)
            # print 'split event shape:', np.shape(split_events)
        print 'Invalid midi num:', invalid_num

        index_nums = [_i for _i in range(len(split_events))]
        random.shuffle(index_nums)
        _shuffled_events = [split_events[_i] for _i in index_nums]
        _shuffled_labels = [split_labels[_i] for _i in index_nums]
        _shuffled_masks = [split_mask[_i] for _i in index_nums]
        print np.shape(_shuffled_events), np.shape(_shuffled_labels), np.shape(_shuffled_masks)
        return _shuffled_events, _shuffled_labels, _shuffled_masks

    train_writer = tf.python_io.TFRecordWriter(tf_record[0])
    validation_writer = tf.python_io.TFRecordWriter(tf_record[1])
    test_writer = tf.python_io.TFRecordWriter(tf_record[2])

    def write_data(_writer, _events, _labels, _masks):
        current_split_midi_num = len(_events)
        for k in range(current_split_midi_num):
            padded_events_flat = flatten(_events[k])
            padded_labels_flat = flatten(_labels[k])
            keys_mask_flat = flatten(_masks[k])

            # write TFrecord    each label and events sampled
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'events': tf.train.Feature(int64_list=tf.train.Int64List(value=padded_events_flat)),
                        'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=padded_labels_flat)),
                        'keys_mask': tf.train.Feature(int64_list=tf.train.Int64List(value=keys_mask_flat)),
                    }))
            serialized = example.SerializeToString()
            _writer.write(serialized)

    print 'train data:'
    sys.stdout.flush()
    train_events, train_labels, train_masks = get_dataset_on_and_lasting(midi_txt[0])
    print 'The number of train sequence:', len(train_labels)
    write_data(train_writer, train_events, train_labels, train_masks)

    print 'validation data:'
    sys.stdout.flush()
    val_events, val_labels, val_masks = get_dataset_on_and_lasting(midi_txt[1])
    print 'The number of validation sequence:', len(val_labels)
    write_data(validation_writer, val_events, val_labels, val_masks)

    print 'test data:'
    sys.stdout.flush()
    test_events, test_labels, test_masks = get_dataset_on_and_lasting(midi_txt[2])
    print 'The number of test sequence:', len(test_labels)
    write_data(test_writer, test_events, test_labels, test_masks)

    train_writer.close()
    validation_writer.close()
    test_writer.close()

    return


if __name__ == '__main__':
    generate_dataset([lasting_train_dataset_new, lasting_validation_dataset_new, lasting_test_dataset_new],
                            [train_midi, val_midi, test_midi])
