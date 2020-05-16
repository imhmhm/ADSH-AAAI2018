import pickle
import numpy as np
import os
import sys
import torch
import utils.calc_hr as calc_hr

def load_label_cifar10(filename, DATA_DIR):
    path = os.path.join(DATA_DIR, filename)
    fp = open(path, 'r')
    labels = [x.strip() for x in fp]
    fp.close()
    return torch.LongTensor(list(map(int, labels)))

def encoding_onehot(target, nclasses=10):
    target_onehot = torch.FloatTensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot

def load_label(filename, DATA_DIR):
    label_filepath = os.path.join(DATA_DIR, filename)
    label = np.loadtxt(label_filepath, dtype=np.int64)
    return torch.from_numpy(label)

## path
code_length = 32
logdir = ''
filename = os.path.join(logdir, str(code_length) + 'bits-record.pkl')
## load pkl
fp = open(filename, 'rb')
record = pickle.load(fp)
rB = record['rB']
qB = record['qB']
## top5k

## read labels
test_labels = load_label('test_label.txt', 'data/NUS-WIDE')
database_labels = load_label('database_label.txt', 'data/NUS-WIDE')

# test_labels = encoding_onehot(test_labels)
# database_labels = encoding_onehot(database_labels)

map, top5k = calc_hr.calc_map(qB, rB, test_labels.numpy(), database_labels.numpy())
topkmap, top5k_5k = calc_hr.calc_topMap(qB, rB, test_labels.numpy(), database_labels.numpy(), 5000)
print('[Evaluation: mAP: %.4f, top-%d mAP: %.4f]', map, 5000, topkmap)
print('[top5k: %.4f]', top5k_5k)
fp.close()
