#!/usr/bin/python
# -*- coding:utf8 -*-
import mido as mi
import os
import process_func as pf

if not os.path.exists('./old_midi'):
    os.mkdir('./old_midi')
if not os.path.exists('./path'):
    os.mkdir('./path')
midi_paths_path = './path/midi_paths.txt'
all_original_file_path = '/home/todd/Data/original_midi'
old_midi_dir_path = '/home/todd/Data/old_midi'
# all_original_file_path = '/Users/david/Documents/Select_MIDI/original_midi'
# old_midi_dir_path = '/Users/david/Documents/Select_MIDI/old_midi'
midi_tracks_num = 3


# clear the path file
f = open(midi_paths_path, 'w')
f.close()

# get midi paths
pf.printPath(1, all_original_file_path, midi_paths_path)

midi_paths = open(midi_paths_path)

count = 0
while True:
    # get each midi path in the path file
    path = midi_paths.readline()
    if not path:
        break
    path = path.replace('\n', '')
    file_name = pf.find_file_name(path)

    try:
        mid = mi.MidiFile(path)
        is_invalid_midi = False
        if mid.tracks.__len__() == midi_tracks_num:
            is_invalid_midi = pf.remove_invalid_midi(mid)
            if is_invalid_midi:
                continue

            old_file = pf.get_off_txt_msg(mid)
            old_file.save(old_midi_dir_path + '/' + file_name)
            count += 1
            print file_name, count
    except:
        continue

midi_paths.close()



