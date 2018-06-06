#!/usr/bin/python
# -*- coding:utf8 -*-

import process_func as pf


RNN_train_dataset = './dataset/train.tfrecords'
RNN_validation_dataset = './dataset/validation.tfrecords'
RNN_test_dataset = './dataset/test_dataset.tfrecords'

test_midi = './path/test_midi_new_rule.txt'
val_midi = './path/val_midi_new_rule.txt'
train_midi = './path/train_midi_new_rule.txt'

if __name__ == '__main__':
    pf.generate_dataset([RNN_train_dataset, RNN_validation_dataset, RNN_test_dataset], [train_midi, val_midi, test_midi])

