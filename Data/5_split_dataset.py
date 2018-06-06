#!/usr/bin/python
# -*- coding:utf8 -*-
import csv
import random
import process_func as pf
import os
import sys

# set your own test/validation MIDI number
test_num = 2000
val_num = 2000

# to place your own the whole selected(old) MIDI document
old_midi_dir_path = '/home/todd/Data/old_midi'

if not os.path.exists('./dataset'):
    os.mkdir('./dataset')

test_midi = './path/test_midi_new_rule.txt'
val_midi = './path/val_midi_new_rule.txt'
train_midi = './path/train_midi_new_rule.txt'


def get_similar_midi(cmb_csv_name, names):

    csvfile = open(cmb_csv_name, "r")
    content = csv.reader(csvfile)
    similar_midi = list()
    for line in content:
        if line[1] == 'search' or line[1] == 'needle':
            continue
        else:
            n1 = int(line[1])
            n2 = int(line[2])
            if n1 not in similar_midi:
                similar_midi.append(n1)
            if n2 not in similar_midi:
                similar_midi.append(n2)
    similar_midi.sort()

    similar_midi_name = [names[i] for i in similar_midi]
    return similar_midi_name, similar_midi


def divide_midis(name_csv_name):
    # get all names of midis
    csvfile = open(name_csv_name, "r")
    content = csv.reader(csvfile)
    whole_names = list()
    for line in content:
        whole_names.append(line[1])
    csvfile.close()

    print len(whole_names)

    # get names and ids of similar midis
    similar_midi_name, similar_midi_id = get_similar_midi('similar2_comb.csv', whole_names)

    not_similar_name = list()
    for i in range(len(whole_names)):
        if i not in similar_midi_id:
            not_similar_name.append(whole_names[i])
    print 'number of not similar:', len(not_similar_name)
    print 'number of similar:', len(similar_midi_name)
    print 'total number:', len(whole_names)
    sys.stdout.flush()
    random.shuffle(not_similar_name)

    f = open(test_midi, 'w')
    f.close()
    f = open(val_midi, 'w')
    f.close()
    f = open(train_midi, 'w')
    f.close()

    path_file = open(test_midi, 'a')
    for i in range(test_num):
        name = not_similar_name.pop(0)
        path_file.write(old_midi_dir_path + '/' + name + '\n')
    path_file.close()

    path_file = open(val_midi, 'a')
    for i in range(val_num):
        name = not_similar_name.pop(0)
        path_file.write(old_midi_dir_path + '/' + name + '\n')
    path_file.close()

    path_file = open(train_midi, 'a')
    train_list = not_similar_name + similar_midi_name
    random.shuffle(train_list)
    for i in range(len(train_list)):
        name = train_list.pop(0)
        path_file.write(old_midi_dir_path + '/' + name + '\n')
    path_file.close()

if __name__ == '__main__':
    divide_midis('similar.csv')