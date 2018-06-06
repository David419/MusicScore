# coding=utf-8
import process_func as pf
import mido as mi
import sys
is_local = False

# to place your own MIDI index file path
# sever_xxx_txt is your own three split MIDI index file(train, validation and test)
# new_sever_xxx_txt is the index file that is filtered MIDIs with another a set of rules
server_test_txt = '/home/todd/New_Rule/Data/path/test_midi.txt'
server_val_txt = '/home/todd/New_Rule/Data/path/val_midi.txt'
server_train_txt = '/home/todd/New_Rule/Data/path/train_midi.txt'
new_server_test_txt = '/home/todd/New_Rule/Data/path/test_midi_new_rule.txt'
new_server_val_txt = '/home/todd/New_Rule/Data/path/val_midi_new_rule.txt'
new_server_train_txt = '/home/todd/New_Rule/Data/path/train_midi_new_rule.txt'


def filter_midi_with_new_rules(original_path_txt, new_path_txt):

    path_list = list()
    f = open(original_path_txt)
    while True:
        # get each midi path in the path file
        path = f.readline()
        if not path:
            break
        path = path.replace('\n', '')
        path_list.append(path)
    f.close()

    f = open(new_path_txt, 'w')
    f.close()
    write_f = open(new_path_txt, 'a')
    count = 0
    for i, path in enumerate(path_list):
        name = pf.find_file_name(path)
        mid = mi.MidiFile(path)
        is_invalid_mid = pf.remove_invalid_midi(mid)
        if is_invalid_mid:
            print i, name
            sys.stdout.flush()
            count += 1
        else:
            write_f.write(path + '\n')
    print 'Total new invalid mid:', count
    write_f.close()
    return


if __name__ == '__main__':

    print('Train:')
    sys.stdout.flush()
    filter_midi_with_new_rules(server_train_txt, new_server_train_txt)
    print('Validation:')
    sys.stdout.flush()
    filter_midi_with_new_rules(server_val_txt, new_server_val_txt)
    print('Test:')
    sys.stdout.flush()
    filter_midi_with_new_rules(server_test_txt, new_server_test_txt)
