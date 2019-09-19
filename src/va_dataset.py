import os
from os.path import join

import copy
import random

import torch
import torch.utils.data.dataset as dataset


class EmbeddingDataset(dataset.Dataset):
    def __init__(self, ds_dir, sample_length=16, activities=""):
        self.ds_dir = ds_dir
        self.sample_length = sample_length
        self.f_cache = self._init_cache(activities)

    def _init_cache(self, activities):
        cache = []
        if activities:
            activities = sorted([join(self.ds_dir, line.rstrip('\n')) for line in open(activities)])
        else:
            activities = sorted([join(self.ds_dir,activity) for activity in os.listdir(self.ds_dir) if os.path.isdir(join(self.ds_dir,activity))])
        for lbl, activity in enumerate(activities):
            samples = sorted([join(activity, vidname) for vidname in os.listdir(activity)])
            for sample in samples:
                for i in range(len(os.listdir(sample))//2):
                    embedding_name = f"{sample}/embedding_{i * 16}"
                    if not os.path.exists(embedding_name+"_video.pth"):
                        # os.remove(embedding_name+"_audio.pth")
                        continue
                    elif not os.path.exists(embedding_name+"_audio.pth"):
                        # os.remove(embedding_name + "_video.pth")
                        continue
                    cache.append((((embedding_name+"_video.pth"),(embedding_name+"_audio.pth")),lbl))
        return cache

    def __getitem__(self, idx):
        (v, a), lbl = self.f_cache[idx]
        v = torch.load(v)
        v = (v - v.min())/(v.max() - v.min())
        a = torch.load(a)
        a = (a - a.min()) / (a.max() - a.min())
        t = torch.zeros(1,v.shape[1] + a.shape[1])
        # t = torch.zeros(1,v.shape[1] + v.shape[1])
        t[0,:v.shape[1]] = v.data
        t[0,a.shape[1]:] = a.data
        return t

    def __len__(self):
        return len(self.f_cache)
