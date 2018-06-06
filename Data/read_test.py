# coding=utf-8


import process_func as pf
import numpy as np

RNN_train_dataset = '/home/todd/New_Rule/Data/dataset/train.tfrecords'
RNN_validation_dataset = '/home/todd/New_Rule/Data/dataset/validation.tfrecords'
RNN_test_dataset = '/home/todd/New_Rule/Data/dataset/test.tfrecords'

lasting_train_dataset = './dataset/new_lasting_train.tfrecords'
lasting_validation_dataset = './dataset/new_lasting_validation.tfrecords'
lasting_test_dataset = './dataset/new_lasting_test.tfrecords'


rnn_event_len = 10
overlap_len = 32
rnn_midi_len = 256

n = 1
event_val, label_val, mask, dur = pf.read_and_decode(RNN_validation_dataset, n, is_lasting=False)
print np.shape(event_val), np.shape(label_val), np.shape(mask), np.shape(dur)
print dur



#
# for i in range(len(event_val[2])):
#     print event_val[2][i]
#     print label_val[2][i]
#     print '_______________'

# import mix_midi as mm
# import my_constant as mc
# import numpy as np
# n = 3261
# X_val, y_val, mask1, mask2 = mm.read_and_decode(mc.RNN_train_dataset, n, is_rnn_dataset=True)
# y_sum = np.sum(y_val)
# mask1_sum = np.sum(mask1)
# p = float(y_sum)/float(mask1_sum)
# print "y_sum:", y_sum
# print "mask1_sum:", mask1_sum
# print "p:", p



# #以中音do进行划分的正确率
#
# import process_func as pf
# import numpy as np
#
# lasting_train_dataset = './dataset/new_lasting_train.tfrecords'
# lasting_validation_dataset = './dataset/new_lasting_validation.tfrecords'
# lasting_test_dataset = './dataset/new_lasting_test.tfrecords'
#
# mask_sum = 0
# X_val, y_val, mask1 = pf.read_and_decode(lasting_train_dataset, 38368, is_lasting=True)
# X_val = X_val[:, :, :10]
# y_val = y_val * mask1
# print X_val.shape
# # for j in range(55, 70):
# #     do_matrix = np.ones([i for i in X_val[0].shape])*j
# #     pred = 0
# #     for i in range(3260):
# #         x = X_val[i]
# #         right = np.greater_equal(x, do_matrix)
# #         pred += np.sum(y_val*right)
# #         left = np.less(x, do_matrix)
# #         pred += np.sum((mask1[i] - y_val) * left)
# #         mask_sum += np.sum(mask1[i])
# #     print "key:", j, "acc:", float(pred)/float(mask_sum)
#
# mask_sum += np.sum(mask1)
# for j in range(55, 70):
#     do_matrix = np.ones([i for i in X_val.shape]) * j
#     pred = 0
#     right = np.greater_equal(X_val, do_matrix)
#     pred += np.sum(y_val*right)
#     left = np.less(X_val, do_matrix)
#     pred += np.sum((mask1 - y_val) * left)
#     print "key:", j, "acc:", float(pred)/float(mask_sum)



# # 统计左右手的音值
# import process_func as pf
# import numpy as np
# import csv
# from collections import Counter
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt
# csvfile = open('sta_hand_note.csv', 'wb')
# writer = csv.writer(csvfile)
# lasting_train_dataset = './dataset/new_lasting_train.tfrecords'
# X_val, y_val, mask1 = pf.read_and_decode(lasting_train_dataset, 38368, is_lasting=True)
# X_val = X_val[:, :, :10]
# y_val = y_val * mask1
# left_key = (mask1 - y_val)*X_val
# left_key = np.reshape(left_key, [-1])
# right_key = y_val*X_val
# right_key = np.reshape(right_key, [-1])
# l = Counter(left_key)
# r = Counter(right_key)
# lx = [i for i in l]
# lx = lx[1:]
# print "lx"
# print lx
# ly = [l[i] for i in l]
# ly = ly[1:]
# print "ly"
# print ly
# rx = [i for i in r]
# rx = rx[1:]
# ry = [r[i] for i in r]
# ry = ry[1:]
# print "rx"
# print rx
# print "ry"
# print ry
# writer.writerow(('l_key', 'l_num', 'r_key', 'r_num'))
# for i in range(max(len(rx), len(lx))):
#     if i >= len(lx):
#         writer.writerow(('', '', rx[i], ry[i]))
#     elif i >= len(rx):
#         writer.writerow((lx[i], ly[i], '', ''))
#     else:
#         writer.writerow((lx[i], ly[i], rx[i], ry[i]))
# csvfile.close()
# plt.figure(1)
# plt.plot(lx, ly)
# plt.plot(rx, ry)
# plt.savefig('plot.png', format='png')



# # 统计有多少个重复的MIDI
# import csv
#
# def get_similar_midi(cmb_csv_name, names):
#
#     csvfile = open(cmb_csv_name, "r")
#     content = csv.reader(csvfile)
#     similar_midi = list()
#     for line in content:
#         if line[1] == 'search' or line[1] == 'needle':
#             continue
#         else:
#             n1 = int(line[1])
#             n2 = int(line[2])
#             if n1 not in similar_midi:
#                 similar_midi.append(n1)
#             if n2 not in similar_midi:
#                 similar_midi.append(n2)
#     similar_midi.sort()
#
#     similar_midi_name = [names[i] for i in similar_midi]
#     return similar_midi_name, similar_midi
#
#
# def divide_midis(name_csv_name):
#     # get all names of midis
#     csvfile = open(name_csv_name, "r")
#     content = csv.reader(csvfile)
#     whole_names = list()
#     for line in content:
#         whole_names.append(line[1])
#     csvfile.close()
#
#     print len(whole_names)
#
#     # get names and ids of similar midis
#     similar_midi_name, similar_midi_id = get_similar_midi('similar2_comb.csv', whole_names)
#
#     not_similar_name = list()
#     for i in range(len(whole_names)):
#         if i not in similar_midi_id:
#             not_similar_name.append(whole_names[i])
#     print 'number of not similar:', len(not_similar_name)
#     print 'number of similar:', len(similar_midi_name)
#     print 'total number:', len(whole_names)
#
# divide_midis('similar.csv')

