#!/usr/bin/python
# -*- coding:utf8 -*-
import mido as mi
import os
import process_func as pf

# path for midi file
if not os.path.exists('./mixed_midi'):
    os.mkdir('./mixed_midi')
new_midi_dir_path = '/home/todd/Data/mixed_midi'
old_midi_dir_path = '/home/todd/Data/old_midi'
# if not os.path.exists('/Users/david/Documents/mixed_midi'):
#     os.makedirs('/Users/david/Documents/mixed_midi')
# new_midi_dir_path = '/Users/david/Documents/Select_MIDI/mixed_midi'
# old_midi_dir_path = '/Users/david/Documents/Select_MIDI/old_midi'

if not os.path.exists('./path'):
    os.mkdir('./path')
# generated midi path file
new_midi_path_txt = './path/new_midi_path.txt'
old_midi_path_txt = './path/old_midi_path.txt'


# clear the path file
f = open(old_midi_path_txt, 'w')
f.close()
f = open(new_midi_path_txt, 'w')
f.close()

# get old midi paths
pf.printPath(1, old_midi_dir_path, old_midi_path_txt)

midi_paths = open(old_midi_path_txt)

count = 0

# for i in range(100):
while True:
    # get each midi path in the path file
    path = midi_paths.readline()
    if not path:
        break
    path = path.replace('\n', '')
    file_name = pf.find_file_name(path)

    try:
        old_mid = mi.MidiFile(path)
        new_mid = mi.MidiFile()
        new_mid = pf.mix_midi_tracks(old_mid)

        new_mid.save(new_midi_dir_path + '/' + file_name)
        count += 1
        print file_name, count

    except:
        continue

midi_paths.close()
pf.printPath(1, new_midi_dir_path, new_midi_path_txt)



