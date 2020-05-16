import sys
import copy
import random
import numpy as np
from collections import defaultdict

import torch
from torch.utils.data.sampler import Sampler

class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class ClassSampler(Sampler):

    def __init__(self, indices, data_source, batch_size, num_instances):
        self.indices = indices
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances

        self.index_dic = defaultdict(list)
        for index, (_, label, idx_spl) in enumerate(map(self.data_source.__getitem__, self.indices)):
        # for index, (_, label, idx_spl) in enumerate(self.data_source):
            # for id_spl, binary in enumerate(reversed(label.numpy())):
            #     if binary == 1:
            #         self.index_dic[id_spl].append(idx_spl)
            #         break

            ### cifar10
            self.index_dic[label[0].item()].append(idx_spl)
        self.pids = list(self.index_dic.keys())
        # for i in self.pids:
        #     print(len(self.index_dic[i]))
        # sys.exit()

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

        self.batch_idxs_dict = defaultdict(list)

        for pid in self.pids:

            idxs = copy.deepcopy(self.index_dic[pid])
            # remaining = len(idxs) % self.num_instances
            # if remaining != 0:
            #     plus = np.random.choice(
            #         idxs, size=(self.num_instances-remaining), replace=True)
            #     idxs.extend(plus)
            # random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    self.batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        self.final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs_select = self.batch_idxs_dict[pid].pop(0)
                self.final_idxs.extend(batch_idxs_select)
                if len(self.batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

    def __iter__(self):
        return iter(self.final_idxs)

    def __len__(self):
        return self.length
