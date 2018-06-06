#!/usr/bin/python
# -*- coding:utf8 -*-
import copy
import os
import mido as mi
from compiler.ast import flatten
import tensorflow as tf
import random
import numpy as np
import sys

rnn_event_len = 10
lasting_padded_len = 22
overlap_len = 32
rnn_midi_len = 256


def printPath(level, path, path_file_path):
    path_file = open(path_file_path, 'a')
    dirList = []
    fileList = []
    files = os.listdir(path)
    dirList.append(str(level))
    for f in files:
        if os.path.isdir(path + '/' + f):
            if f[0] == '.':
                pass
            else:
                dirList.append(f)
        if os.path.isfile(path + '/' + f):
            fileList.append(f)
    i_dl = 0
    for dl in dirList:
        if i_dl == 0:
            i_dl = i_dl + 1
        else:
            print '-' * (int(dirList[0])), dl
            printPath((int(dirList[0]) + 1), path + '/' + dl, path_file_path)
    fileList.sort()
    for fl in fileList:
        if '.mid' in fl:
            path_file.write(path + '/' + fl + '\n')
            path_file.flush()
    path_file.close()


def find_min_note(tracks, pos, is_finish):
    # 'tracks' is a list of several tracks which contain the information about note
    # 'pos' is a list of position which stand for the position of notes need to be filled in each track
    # The dimension of 'pos' is the number of the tracks in 'tracks'. If there are 3 tracks needed to be mixed,
    # the 'pos' should be a list of 1*3
    # 'is_finish' is a bool list. The dimension of 'is_finish' is the number of the tracks in 'tracks'.
    # True in 'is_finish' stands for the track of 'True''s index finish filling into the new track.

    # init min valuables
    min_track_i = 0
    min_total_time = 0

    # unfinished track number is the list of which track unfinishes filling the new track
    unfinish_track_num = list()
    # get unfinished track num
    for i in range(len(is_finish)):
        if not is_finish[i]:
            unfinish_track_num.append(i)

    # from unfinished track to find the earliest(minimum) note
    for i in unfinish_track_num:
        # init the min valuables by the first unfinished track's note of current position
        if i == unfinish_track_num[0]:
            min_total_time = tracks[i][pos[i]].time
            min_track_i = i

        # find the min note
        else:
            if tracks[i][pos[i]].time < min_total_time:
                min_track_i = i
                min_total_time = tracks[i][pos[i]].time

    # get the min note
    min_note = tracks[min_track_i][pos[min_track_i]]

    # update position list
    new_pos = [i for i in pos]
    new_pos[min_track_i] += 1

    # update is_finish list
    if new_pos[min_track_i] == len(tracks[min_track_i]):
        is_finish[min_track_i] = True
    return min_note, new_pos, is_finish


def find_file_name(file_path):
    file_name = ''
    for ch in file_path:
        if ch == '/':
            file_name = ''
        else:
            file_name += ch
    return file_name


def get_info(src_midi):
    # src_midi is a unmixed midi, or the returning labels will be all 0
    copy_midi = copy.deepcopy(src_midi)

    # get musical tracks
    tracks = list()
    for i, track in enumerate(copy_midi.tracks):
        if i == 0:
            continue
        else:
            tracks.append(track)

    # transform Message's time(interval) to real time
    for i, track in enumerate(tracks):
        total_time = 0
        for msg in track:
            if msg.is_meta:
                if msg.type == 'key_signature':
                    interval = msg.time
                    msg.time += total_time
                    total_time += interval
            else:
                interval = msg.time
                msg.time += total_time
                total_time += interval

    # init variables
    tracks_num = len(tracks)
    pos = [0 for i in range(tracks_num)]
    is_finish = [0 for i in range(tracks_num)]
    all_finish = [1 for i in range(tracks_num)]

    labels = list()
    types = list()
    times = list()
    notes = list()
    is_ctrl_change = list()
    is_prog_change = list()
    is_note_on = list()
    is_note_off = list()

    # get_label
    # only get message labels
    while is_finish != all_finish:
        min_note, new_pos, is_finish = find_min_note(tracks, pos, is_finish)
        if min_note.is_meta:
            pos = new_pos
            continue

        for i in range(tracks_num):
            if new_pos[i] != pos[i]:
                labels.append((i+1) % 2)
                types.append(min_note.type)
                times.append(min_note.time)
                if min_note.type == 'note_on':
                    notes.append(min_note.note)
                    is_note_on.append(1)
                    is_note_off.append(0)
                    is_ctrl_change.append(0)
                    is_prog_change.append(0)
                elif min_note.type == 'note_off':
                    notes.append(min_note.note)
                    is_note_on.append(0)
                    is_note_off.append(1)
                    is_ctrl_change.append(0)
                    is_prog_change.append(0)
                elif min_note.type == 'control_change':
                    notes.append(0)
                    is_note_on.append(0)
                    is_note_off.append(0)
                    is_ctrl_change.append(1)
                    is_prog_change.append(0)
                elif min_note.type == 'program_change':
                    notes.append(0)
                    is_note_on.append(0)
                    is_note_off.append(0)
                    is_ctrl_change.append(0)
                    is_prog_change.append(1)
                else:
                    print '【Error!!!!!!!】'
                    print min_note.type
        pos = new_pos

    note_labels_on = list()
    note_times_on = list()
    note_notes_on = list()
    note_duration = list()

    note_labels_off = list()
    note_times_off = list()
    note_notes_off = list()

    for i, msg_type in enumerate(types):
        if msg_type == 'note_on':
            note_labels_on.append(labels[i])
            note_times_on.append(times[i])
            note_notes_on.append(notes[i])
        elif msg_type == 'note_off':
            note_labels_off.append(labels[i])
            note_times_off.append(times[i])
            note_notes_off.append(notes[i])
        else:
            continue
    for i in range(len(note_notes_on)):
        key_on = note_notes_on[i]
        key_time_on = note_times_on[i]
        key_label_on = note_labels_on[i]
        find_key_off = False
        for j in range(len(note_notes_off)):
            key_off = note_notes_off[j]
            key_time_off = note_times_off[j]
            key_label_off = note_labels_off[j]
            if key_label_on == key_label_off and key_on == key_off:
                note_duration.append(key_time_off - key_time_on)
                if note_duration[i] < 0:
                    print "Error!! The duration of key is less than 0!"
                    return -1
                note_notes_off.pop(j)
                note_times_off.pop(j)
                note_labels_off.pop(j)
                find_key_off = True
                break
            else:
                continue
        if not find_key_off:
            print "Error!! Not find key off when get info!"
            return -1

    # generate a transformed midi format

    note_times_on_set = [i for i in sorted(set(note_times_on))]
    rnn_key_on_interval = list()
    rnn_key_on_interval.append(0)
    for i in range(1, len(note_times_on_set)):
        rnn_key_on_interval.append(note_times_on_set[i] - note_times_on_set[i - 1])

    rnn_events = list()
    rnn_labels = list()
    rnn_durations = list()
    rnn_event = list()
    rnn_label = list()
    rnn_duration = list()

    last_time = note_times_on[0]
    for i, time in enumerate(note_times_on):
        if time != last_time:
            if rnn_label.__len__() > 1:
                # sort events and labels
                l_e_zip = zip(rnn_label, rnn_event, rnn_duration)
                l_e_zip = sorted(l_e_zip)
                l_e_zip = map(list, zip(*l_e_zip))
                rnn_label = l_e_zip[0]
                rnn_event = l_e_zip[1]
                rnn_duration = l_e_zip[2]

            # add events and labels to new_midi_form
            rnn_events.append(rnn_event)
            rnn_labels.append(rnn_label)
            rnn_durations.append(rnn_duration)

            rnn_event = list()  # set trans_event to null
            rnn_label = list()
            rnn_duration = list()
        rnn_event.append(note_notes_on[i])
        rnn_label.append(note_labels_on[i])
        rnn_duration.append(note_duration[i])

        last_time = time
    rnn_events.append(rnn_event)  # add last event
    rnn_labels.append(rnn_label)
    rnn_durations.append(rnn_duration)

    return rnn_labels, rnn_events, rnn_durations, rnn_key_on_interval


def remove_invalid_midi(midi_file):
    is_remove = False
    for i, track in enumerate(midi_file.tracks):
        for msg in track:
            if msg.type == 'sysex':
                is_remove = True
                print 'Midi is invalid! Sysex type error.'
                return is_remove
            elif msg.type == 'polytouch':
                is_remove = True
                print 'Midi is invalid! Polytouch type error.'
                return is_remove
            elif msg.type == 'pitchwheel':
                is_remove = True
                print 'Midi is invalid! Pitchwheel type error.'
                return is_remove
            elif msg.type == 'aftertouch':
                is_remove = True
                print 'Midi is invalid! Aftertouch type error.'
                return is_remove
            elif msg.type == 'program_change':
                if msg.program > 8:  # remove the midis that are not played by piano
                    is_remove = True
                    print 'Midi is invalid! Program_change type error.'
                    return is_remove
    try:
        lab, eve, dur, intv = get_info(midi_file)  # remove the midi that cannot be extracted information

        flat_lab = flatten(lab)  # remove the midi that only has one track actually.
        if np.all(np.array(flat_lab) > 0) or np.all(np.array(flat_lab) < 1):
            is_remove = True
            print 'Midi is invalid! midi only has one track actually.'
            return is_remove

        for i, label in enumerate(lab):  # remove the midi that one hand presses over 5 keys at a moment
            right_hand_count = 0
            for j in label:
                if j == 1:
                    right_hand_count += 1
            if right_hand_count > 5:
                is_remove = True
                # print eve[i], label
                print 'Midi is invalid! right hand presses over 5 keys at a moment.'
                return is_remove
            if len(eve[i]) - right_hand_count > 5:
                is_remove = True
                # print eve[i], label
                print 'Midi is invalid! left hand presses over 5 keys at a moment.'
                return is_remove

        flat_events = flatten(eve)  # remove the total number of pressed keys is less than 200
        if len(flat_events) < 200:
            is_remove = True
            print 'Midi is invalid! Total number of pressed keys is less than 200.'
            return is_remove

        for event in eve:
            if len(event) > 10:  # remove the midi that the pressed keys at once is larger than 10
                is_remove = True
                print 'Midi is invalid! The pressed keys at once is larger than 10.'
                return is_remove
    except:
        is_remove = True
    return is_remove


def get_off_txt_msg(origin_midi):

    new_midi = mi.MidiFile()

    for i, track in enumerate(origin_midi.tracks):
        new_track = mi.MidiTrack()
        have_text = False
        t = 0
        if i == 0:
            for msg in track:
                if msg.type == 'text':
                    continue
                if msg.type == 'copyright':
                    continue
                if msg.type == 'lyrics':
                    continue
                new_track.append(msg)
        else:
            for msg in track:
                if msg.type == 'text':
                    t = msg.time + t
                    have_text = True
                elif msg.type == 'lyrics':
                    t = msg.time + t
                    have_text = True
                    continue
                else:
                    if have_text:
                        new_msg = msg
                        new_msg.time = msg.time + t
                        t = 0
                        have_text = False
                        new_track.append(new_msg)
                    else:
                        new_track.append(msg)

        new_midi.tracks.append(new_track)
    return new_midi


def mix_tracks(tracks, new_track):

    # transform Message's time(interval) to realtime
    for i, track in enumerate(tracks):
        total_time = 0
        for msg in track:
            if msg.is_meta:
                if msg.type == 'key_signature':
                    interval = msg.time
                    msg.time += total_time
                    total_time += interval
            else:
                interval = msg.time
                msg.time += total_time
                total_time += interval

    # append MetaMessage to new track and record the end of track
    for msg in tracks[0]:
        if msg.is_meta:
            if msg.type == 'end_of_track':
                end_note = msg
            elif msg.type == 'key_signature':
                continue
            else:
                new_track.append(msg)

    # init variables
    tracks_num = len(tracks)
    pos = [0 for i in range(tracks_num)]
    is_finish = [0 for i in range(tracks_num)]
    all_finish = [1 for i in range(tracks_num)]
    current_time = 0

    # mix tracks
    while is_finish != all_finish:
        min_note, pos, is_finish = find_min_note(tracks, pos, is_finish)
        if min_note.is_meta is True and min_note.type != 'key_signature':
            continue

        # add the min note to new track
        temp_note = copy.deepcopy(min_note)
        temp_note.time -= current_time # change real time to interval
        new_track.append(temp_note)

        current_time = min_note.time

    # add end note to new track
    new_track.append(end_note)

    return new_track


def mix_midi_tracks(src_file):
    dst_file = mi.MidiFile()
    dst_track_1 = mi.MidiTrack()
    dst_track_2 = mi.MidiTrack()
    dst_file.tracks.append(dst_track_1)
    dst_file.tracks.append(dst_track_2)

    multi_tracks = list()
    for i, track in enumerate(src_file.tracks):
        if i == 0:
            for msg in track:
                dst_track_1.append(msg)
        else:
            multi_tracks.append(track)

    dst_track_2 = mix_tracks(multi_tracks, dst_track_2)

    return dst_file


def generate_dataset(tf_record, midi_txt, is_lasting=False):

    def get_lasting_keys_on(_events, _durations, _key_on_intervals):
        keys_on = list()
        keys_on_time = list()
        current_time = 0
        lasting_keys = list()
        for i, event in enumerate(_events):
            current_time += _key_on_intervals[i]
            if keys_on:  # if keys_on list is empty?
                new_keys_on = [temp for temp in keys_on]
                new_keys_on_time = [temp for temp in keys_on_time]
                for j, key_on in enumerate(keys_on):
                    if keys_on_time[j] <= current_time:
                        new_keys_on.pop(new_keys_on.index(keys_on[j]))
                        new_keys_on_time.pop(new_keys_on_time.index(keys_on_time[j]))
            else:  # keys_on list is empty
                new_keys_on = list()
                new_keys_on_time = list()

            lasting_keys.append(new_keys_on)

            keys_on = [temp for temp in new_keys_on]
            keys_on_time = [temp for temp in new_keys_on_time]
            for j, key in enumerate(event):
                keys_on.append(key)
                keys_on_time.append(_durations[i][j] + current_time)

        return lasting_keys

    def get_dataset(_midi_path):
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
        split_durations = list()
        split_mask = list()

        for _count, _path in enumerate(midi_list):
            old_midi = mi.MidiFile(_path)
            is_invalid_midi = False
            _labels, _events, _durations, _key_on_intervals = get_info(old_midi)
            _events_len = len(_events)
            _interval_len = len(_key_on_intervals)

            if _events_len != _interval_len:
                print find_file_name(_path), 'is invalid because the length of events and intervals are not same!'
                continue

            print _count, find_file_name(_path)
            sys.stdout.flush()
            keys_mask = list()  # to confirm whether each key in each event is a padding value
            padded_events = list()
            padded_durations = list()
            padded_labels = _labels

            pad_event_num = 0
            pad_label_num = 0
            pad_duration_num = 0

            null_event = [pad_event_num for n in range(rnn_event_len + 1)]
            null_label = [pad_label_num for n in range(rnn_event_len)]
            null_duration = [pad_duration_num for n in range(rnn_event_len + 1)]
            null_key_mask = [0 for n in range(rnn_event_len)]

            # padding keys
            for _i, event in enumerate(_events):
                a_event_key_mask = list()
                a_event = list()
                a_duration = list()

                key_n = len(event)
                a_event.append(0)
                a_duration.append(_key_on_intervals[_i])
                if key_n < rnn_event_len:
                    for n in range(key_n):
                        a_event_key_mask.append(1)
                        a_event.append(_events[_i][n])
                        a_duration.append(_durations[_i][n])
                    for j in range(rnn_event_len - key_n):
                        a_event.append(pad_event_num)
                        a_duration.append(pad_duration_num)
                        a_event_key_mask.append(0)
                        padded_labels[_i].append(pad_label_num)
                elif key_n == rnn_event_len:
                    for n in range(key_n):
                        a_event_key_mask.append(1)
                        a_event.append(_events[_i][n])
                        a_duration.append(_durations[_i][n])
                else:
                    is_invalid_midi = True
                    print 'the number of pressed key at once is larger than 10!'
                    sys.stdout.flush()
                    break

                keys_mask.append(a_event_key_mask)
                padded_events.append(a_event)
                padded_durations.append(a_duration)

            if is_invalid_midi:
                continue
            # print "shape event:", np.shape(padded_events), "shape dur:", np.shape(padded_durations)
            # pad top of the sequence with length of overlap_len
            for _i in range(overlap_len):
                padded_events.insert(0, null_event)
                padded_labels.insert(0, null_label)
                keys_mask.insert(0, null_key_mask)
                padded_durations.insert(0, null_duration)

            # print "shape event:", np.shape(padded_events), "shape dur:", np.shape(padded_durations)
            seq_len = len(padded_events)
            for i in range(0, seq_len, rnn_midi_len - overlap_len):
                # padding split sequence
                if (seq_len - i) < rnn_midi_len:
                    temp_events = copy.deepcopy(padded_events[i:seq_len])
                    temp_labels = copy.deepcopy(padded_labels[i:seq_len])
                    temp_duriations = copy.deepcopy(padded_durations[i:seq_len])
                    temp_keys_mask = copy.deepcopy(keys_mask[i:seq_len])
                    for j in range(rnn_midi_len - (seq_len - i)):
                        temp_events.append(null_event)
                        temp_labels.append(null_label)
                        temp_duriations.append(null_duration)
                        temp_keys_mask.append(null_key_mask)
                    for jj in range(overlap_len):
                        for kk in range(rnn_event_len):
                            temp_keys_mask[jj][kk] = 0
                    split_events.append(temp_events)
                    split_labels.append(temp_labels)
                    split_durations.append(temp_duriations)
                    split_mask.append(temp_keys_mask)
                else:
                    temp_keys_mask = copy.deepcopy(keys_mask[i:i + rnn_midi_len])
                    for jj in range(overlap_len):
                        for kk in range(rnn_event_len):
                            temp_keys_mask[jj][kk] = 0
                    split_events.append(padded_events[i:i + rnn_midi_len])
                    split_labels.append(padded_labels[i:i + rnn_midi_len])
                    split_durations.append(padded_durations[i:i + rnn_midi_len])
                    split_mask.append(temp_keys_mask)

        index_nums = [_i for _i in range(len(split_events))]
        random.shuffle(index_nums)
        _shuffled_events = [split_events[_i] for _i in index_nums]
        _shuffled_labels = [split_labels[_i] for _i in index_nums]
        _shuffled_masks = [split_mask[_i] for _i in index_nums]
        _shuffled_durations = [split_durations[_i] for _i in index_nums]
        return _shuffled_labels, _shuffled_events, _shuffled_masks, _shuffled_durations

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
            old_midi = mi.MidiFile(_path)
            is_invalid_midi = False
            _labels, _events, _durations, _key_on_intervals = get_info(old_midi)

            _lasting_keys_on = get_lasting_keys_on(_events, _durations, _key_on_intervals)

            _events_len = len(_events)
            _interval_len = len(_key_on_intervals)
            if _events_len != _interval_len:
                print find_file_name(_path), 'is invalid because the length of events and intervals are not same!'
                continue

            print _count, find_file_name(_path)
            sys.stdout.flush()
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
            # print 'invalid midi num', invalid_num

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

    if is_lasting:

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

    else:

        def write_data(_writer, _events, _labels, _masks, _durations):
            current_split_midi_num = len(_events)
            for k in range(current_split_midi_num):
                padded_events_flat = flatten(_events[k])
                padded_labels_flat = flatten(_labels[k])
                keys_mask_flat = flatten(_masks[k])
                durations_flat = flatten(_durations[k])

                # write TFrecord    each label and events sampled
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'events': tf.train.Feature(int64_list=tf.train.Int64List(value=padded_events_flat)),
                            'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=padded_labels_flat)),
                            'keys_mask': tf.train.Feature(int64_list=tf.train.Int64List(value=keys_mask_flat)),
                            'durations': tf.train.Feature(int64_list=tf.train.Int64List(value=durations_flat)),
                        }))
                serialized = example.SerializeToString()
                _writer.write(serialized)

        print 'train data:'
        sys.stdout.flush()
        train_labels, train_events, train_masks, train_durations = get_dataset(midi_txt[0])
        print 'The number of train sequence:', len(train_labels)
        write_data(train_writer, train_events, train_labels, train_masks, train_durations)

        print 'validation data:'
        sys.stdout.flush()
        val_labels, val_events, val_masks, val_durations = get_dataset(midi_txt[1])
        print 'The number of validation sequence:', len(val_labels)
        write_data(validation_writer, val_events, val_labels, val_masks, val_durations)

        print 'test data:'
        sys.stdout.flush()
        test_labels, test_events, test_masks, test_durations = get_dataset(midi_txt[2])
        print 'The number of test sequence:', len(test_labels)
        write_data(test_writer, test_events, test_labels, test_masks, test_durations)

    train_writer.close()
    validation_writer.close()
    test_writer.close()

    return


def read_and_decode(tfrecord_file, midi_num, is_lasting=False):
    # midi_num is how many midi in this dataset
    # e.g.  the test dataset contains 500 midis, so the midi_num is 500
    if is_lasting:
        filename_q = tf.train.string_input_producer([tfrecord_file], num_epochs=None)

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_q)

        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'events': tf.FixedLenFeature([rnn_event_len * 2 * rnn_midi_len], tf.int64),
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
            # for i in range(1):  # debug
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

    else:
        filename_q = tf.train.string_input_producer([tfrecord_file], num_epochs=None)

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_q)

        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'events': tf.FixedLenFeature([(rnn_event_len + 1) * rnn_midi_len], tf.int64),
                                               'labels': tf.FixedLenFeature([rnn_event_len * rnn_midi_len], tf.int64),
                                               'keys_mask': tf.FixedLenFeature([rnn_event_len * rnn_midi_len], tf.int64),
                                               'durations': tf.FixedLenFeature([(rnn_event_len + 1) * rnn_midi_len], tf.int64),
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
