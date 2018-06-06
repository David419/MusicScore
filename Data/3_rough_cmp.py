#!/usr/bin/python
# -*- coding:utf8 -*-

import ctypes
import numpy as np
import process_func as pf
import mido as mi
import csv
import codecs
import sys

# to place your own the whole selected(old) MIDI path file
old_midi_path_txt = './path/old_midi_path.txt'


def compare_midi(midi_path):
    func = ctypes.CDLL('./find_seq.so')
    find_sub_seq = func.find_seq
    find_sub_seq.argtypes = [ctypes.POINTER(ctypes.c_int), ]

    info = list()

    for i in midi_path:
        midi = mi.MidiFile(i)
        _, events, _, _ = pf.get_info(midi)
        info.append(np.array(pf.flatten(events)))
    d = len(info)
    identity = np.zeros([d, d])
    for i in range(d):
        print i
        sys.stdout.flush()
        for j in range(d):
            if i == j:
                continue
            pyarr = info[i][100:160]
            sub_seq = (ctypes.c_int * len(pyarr))(*pyarr)
            pyarr = info[j]
            seq = (ctypes.c_int * len(pyarr))(*pyarr)
            ctypes.cast(sub_seq, ctypes.POINTER(ctypes.c_int))
            ctypes.cast(seq, ctypes.POINTER(ctypes.c_int))
            identity[i][j] = find_sub_seq(sub_seq, len(sub_seq), seq, len(seq), int(0.55 * len(sub_seq)))
    return identity


if __name__ == '__main__':
    csvfile = open('similar.csv', 'wb')
    csvfile.write(codecs.BOM_UTF8)
    writer = csv.writer(csvfile)

    midi_paths = open(old_midi_path_txt)
    paths = list()

    # for i in range(300):
    while True:
        # get each midi path in the path file
        path = midi_paths.readline()
        if not path:
            break
        path = path.replace('\n', '')
        paths.append(path)

    is_similar = compare_midi(paths)

    for i, path in enumerate(paths):
        name = pf.find_file_name(path)
        similar_midi = list()
        for j in range(len(is_similar[i])):
            if is_similar[i][j]:
                similar_midi.append(j)
        content = tuple([i, name] + similar_midi)
        writer.writerow(content)

    midi_paths.close()
    csvfile.close()