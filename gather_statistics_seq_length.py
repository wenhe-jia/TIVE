# -*- coding: UTF-8 -*-
'''
@Project ：code 
@File ：gather_statistics_seq_length.py
@Author ：jzl
@Date ：2022/4/25 9:34 
'''

import os, sys
import numpy as np
import json
import matplotlib.pylab as plt
from collections import Counter

if __name__ == '__main__':
    path = r'G:\code\ytvis2022\mini360relate\valid_mini.json'

    json_file = json.load(open(path, 'r'))
    txt = open('./seq.txt', '+a')

    seq_length = []

    for ann in json_file['annotations']:
        sl = 0
        for seg in ann['segmentations']:
            if seg != None:
                sl += 1
        # if sl!=len(ann['segmentations']):
        # # print('video_length', ann['length'])
        #     print('seq_length', len(ann['segmentations']))
        #     print('------',sl)
        txt.write(str(sl) + '\n')
        seq_length.append(sl)

    count = dict(Counter(seq_length))
    print(count)
    len_list = list(count.keys())
    len_value = list(count.values())

    print('short:', np.sum(np.array(seq_length) < 16))
    print('long:', np.sum(np.array(seq_length) > 32))
    print('medium:', np.sum(np.logical_and(np.array(seq_length) < 32, np.array(seq_length) > 16)))

    # plt.bar(len_list, len_value)
    # plt.savefig('./length.png')
