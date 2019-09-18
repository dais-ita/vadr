import os
from os.path import join

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
                for i in range(len(os.listdir(sample))/2):
                    embedding_name = f"{sample}_{i*16}"
                    cache.append(((embedding_name+"_video.pth"),(embedding_name+"_audio.pth")),lbl)
        return cache
    def __getitem__(self, idx):
        (v, a), lbl = self.f_cache[idx]
        v = torch.load(v)
        a = torch.load(a)
        t = torch.zeros(1,v.shape[1] + a.shape[1])
        t[0,:v.shape[1]].data = v.data
        t[a.shape[1]:].data = a.data
        return t



