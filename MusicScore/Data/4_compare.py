#!/usr/bin/python
# -*- coding:utf8 -*-
from __future__ import division
import numpy as np
import process_func as pf
import mido as mi
from compiler.ast import flatten
import sys
import csv
import ctypes
import codecs

# to place your own the whole selected(old) MIDI document
old_midi_dir_path = '/home/todd/Data/old_midi'
pt = {'match': 1, 'mismatch': -1, 'gap': -1}


def mch(alpha, beta):
    if alpha == beta:
        return pt['match']
    elif alpha == '-' or beta == '-':
        return pt['gap']
    else:
        return pt['mismatch']


def needle(s1, s2):
    m, n = len(s1), len(s2)
    score = np.zeros((m + 1, n + 1))

    # Initialization
    for i in range(m + 1):
        score[i][0] = pt['gap'] * i
    for j in range(n + 1):
        score[0][j] = pt['gap'] * j

    # Fill
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diag = score[i - 1][j - 1] + mch(s1[i - 1], s2[j - 1])
            delete = score[i - 1][j] + pt['gap']
            insert = score[i][j - 1] + pt['gap']
            score[i][j] = max(diag, delete, insert)

    # print('score matrix = \n%s\n' % score)
    align1, align2 = list(), list()
    i, j = m, n

    # Traceback
    while i > 0 and j > 0:
        score_current = score[i][j]
        score_diag = score[i - 1][j - 1]
        score_left = score[i][j - 1]
        score_up = score[i - 1][j]

        # print('score_current: ', score_current)
        # print('score_diag: ', score_diag)
        # print('score_left: ', score_left)
        # print('score_up: ', score_up)
        a1 = a2 = ''
        if score_current == score_diag + mch(s1[i - 1], s2[j - 1]):
            # print('diag')
            a1, a2 = s1[i - 1], s2[j - 1]
            i, j = i - 1, j - 1
        elif score_current == score_up + pt['gap']:
            # print('up')
            a1, a2 = s1[i - 1], ''
            i -= 1
        elif score_current == score_left + pt['gap']:
            # print('left')
            a1, a2 = '', s2[j - 1]
            j -= 1
        # print('%s ---> a1 = %s\t a2 = %s\n' % ('Add', a1, a2))
        align1.append(a1)
        align2.append(a2)

    while i > 0:
        a1, a2 = s1[i - 1], ''
        # print('%s ---> a1 = %s\t a2 = %s\n' % ('Add', a1, a2))
        align1.append(a1)
        align2.append(a2)
        i -= 1

    while j > 0:
        a1, a2 = '', s2[j - 1]
        # print('%s --> a1 = %s\t a2 = %s\n' % ('Add', a1, a2))
        align1.append(a1)
        align2.append(a2)
        j -= 1

    align1 = align1[::-1]
    align2 = align2[::-1]
    seqN = len(align1)
    sym = list()
    seq_score = 0
    ident = 0
    for i in range(seqN):
        a1 = align1[i]
        a2 = align2[i]
        if a1 == a2:
            sym.append(a1)
            ident += 1
            seq_score += mch(a1, a2)

        else:
            seq_score += mch(a1, a2)
            sym.append('')

    ident = ident / seqN * 100

    # print('Identity = %2.1f percent' % ident)
    sys.stdout.flush()
    # print('Score = %d\n' % seq_score)
    # print align1
    # print sym
    # print align2
    return ident


def compare_midi_with_needleman(midi_path):
    info = list()

    for i in midi_path:
        midi = mi.MidiFile(i)
        _, events, _, _ = pf.get_info(midi)
        info.append(np.array(flatten(events)))
    d = len(info)
    identity = np.zeros([d, d])
    for i in range(d):
        for j in range(1+i, d):
            len_i = len(info[i])
            len_j = len(info[j])
            if min(len_i, len_j) * 1.0 / max(len_i, len_j) >= 0.9:
                pre_identity = needle(info[i][50:200], info[j][50:200])
                print('pre_identity = %2.1f percent' % pre_identity)
                if pre_identity > 50:
                    identity[i][j] = int(needle(info[i], info[j]))
                    print('Identity = %2.1f percent' % identity[i][j])
    return identity


def read_csv(filename):
    csvfile = open(filename, "r")
    content = csv.reader(csvfile)
    data = list()
    for line in content:
        data.append(line)

    similar_id = list()
    midi_name_dict = dict()
    cmp_list = list()
    for line in data:
        if len(line) > 2:
            temp = [i for i in line]
            temp.remove(temp[1])
            similar_id.append([int(i.encode("utf-8")) for i in temp])
    for i, line in enumerate(similar_id):
        for j, num in enumerate(line):
            if num not in midi_name_dict:
                midi_name_dict[num] = data[num][1]
            if j != 0:
                temp = [line[0], num]
                temp.sort()
                if temp not in cmp_list:
                    cmp_list.append(temp)
    csvfile.close()
    return cmp_list, midi_name_dict


def find_similar_midi(cmp_list, midi_name_dict):
    func = ctypes.CDLL('./find_seq.so')
    find_sub_seq = func.find_seq
    find_sub_seq.argtypes = [ctypes.POINTER(ctypes.c_int), ]

    keys = midi_name_dict.keys()
    keys.sort()
    names = [midi_name_dict[i] for i in keys]
    info = list()
    similar_comb = list()
    similar_comb_needle = list()
    # path = old_midi_dir_path + '/' + names[0]
    # midi = mi.MidiFile(path)
    # _, events, _, _ = pf.get_info(midi)

    for name in names:
        path = old_midi_dir_path + '/' + name
        print name
        sys.stdout.flush()
        midi = mi.MidiFile(path)
        _, events, _, _ = pf.get_info(midi)
        info.append(flatten(events))

    for i, combination in enumerate(cmp_list):
        print i
        sys.stdout.flush()
        index1 = keys.index(combination[0])
        index2 = keys.index(combination[1])
        seq1 = info[index1]
        seq2 = info[index2]
        len1 = len(seq1)
        len2 = len(seq2)
        if len1 < len2:
            sub_seq = (ctypes.c_int * len(seq1))(*seq1)
            seq = (ctypes.c_int * len(seq2))(*seq2)
        else:
            sub_seq = (ctypes.c_int * len(seq2))(*seq2)
            seq = (ctypes.c_int * len(seq1))(*seq1)
        ctypes.cast(sub_seq, ctypes.POINTER(ctypes.c_int))
        ctypes.cast(seq, ctypes.POINTER(ctypes.c_int))
        is_similar = find_sub_seq(sub_seq, len1, seq, len2, int(0.2 * len(sub_seq)))
        if is_similar:
            similar_comb.append(combination)
            continue
        else:
            if min(len1, len2) * 1.0 / max(len1, len2) >= 0.9:
                identity = needle(seq1, seq2)
                if identity > 80:
                    similar_comb_needle.append(combination)

    return similar_comb, similar_comb_needle


if __name__ == '__main__':
    # path = ['./test_midi/m1.mid', './test_midi/m2.mid', './test_midi/m3.mid', './test_midi/m4.mid', './test_midi/m5.mid',
    #         './test_midi/y1.mid', './test_midi/y2.mid', './test_midi/y3.mid', './test_midi/y4.mid', './test_midi/y5.mid',
    #         './test_midi/c1.mid', './test_midi/c2.mid', './test_midi/c3.mid', './test_midi/c4.mid', './test_midi/c5.mid',
    #         './test_midi/c6.mid']
    #
    # print compare_midi(path)
    # read_csv('similar.csv')

    cmp_combine, name_dict = read_csv('similar.csv')
    s_c, s_c_n = find_similar_midi(cmp_combine, name_dict)

    csvfile = open('similar2_comb.csv', 'wb')
    csvfile.write(codecs.BOM_UTF8)
    writer = csv.writer(csvfile)
    writer.writerow((0, 'search'))
    for i, b in enumerate(s_c):
        print i, b
        sys.stdout.flush()
        writer.writerow((i, b[0], b[1]))
    writer.writerow((1, 'needle'))
    for i, b in enumerate(s_c_n):
        print i, b
        sys.stdout.flush()
        writer.writerow((i, b[0], b[1]))

    csvfile.close()




