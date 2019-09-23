import os
from os.path import join

import cv2

import numpy as np
import torch
import torch.nn.functional as F

from matplotlib import pyplot as plt
from tqdm import tqdm

from src import train

def get_video_in(sample_name="", shape=(112,112), mean=[0,0,0]):
    video_path = sample_name.split("/")
    video_path[3] = "Video"
    video_path = "/".join(video_path)
    frame_paths = [join(video_path,f) for f in os.listdir(video_path) if not ("horiz" in f or "vert" in f) and ".jpg" in f]
    frame_paths = sorted(frame_paths, key=lambda x: int(x.split("_")[-1][:-4]))

    sample = torch.zeros((1,3,16)+shape)
    mean_sample = torch.zeros_like(sample)
    for i, m in enumerate(mean):
        mean_sample[:,i,...] += m
    samples = []
    i = 0
    for fpath in frame_paths:
        frame = cv2.imread(fpath)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(cv2.resize(frame, shape).transpose(2,0,1)).float()
        sample[0,:,i,...] = frame
        i += 1
        if i == 16:
            sample -= mean_sample
            samples.append(sample)
            sample = torch.zeros((1, 3, 16) + shape)
            i = 0
    if i > 0:
        last = i
        while i < 16:
            sample[...,i,:,:] = sample[...,last,:,:]
            i += 1
        sample -= mean_sample
        samples.append(sample)
    return samples, mean_sample

def get_audio_in(sample_name=""):
    chunks = [join(sample_name,f) for f in os.listdir(sample_name) if ".npy" in f]
    samples = []
    for chunk in chunks:
        samples.append(torch.from_numpy(np.load(chunk)).unsqueeze(0).float())
    return samples

def get_mdl(weights="fusion_init.pth"):
    return train.build_mdl(51,False,weights=weights)

def explain(mdl, input, lbl):
    mdl.eval()
    vid_samples, aud_samples = input
    assert len(vid_samples) == len(aud_samples)
    grads = []
    for video, audio in tqdm(zip(vid_samples,aud_samples)):

        video.requires_grad_()
        audio.requires_grad_()

        out = mdl(video,audio)
        grad_out = torch.zeros_like(out)
        grad_out[0,lbl] += 1
        grad = torch.autograd.grad(out, (video,audio), grad_out)
        grads.append(grad)
    return grads

def visualise(grads,cmap="seismic"):
    cmap = plt.get_cmap(cmap)
    video = []
    audio = []
    std = 1

    sob_t = torch.tensor([[[1, 2, 1], [2, 4, 2], [1, 2, 1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                          [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]])
    sob_t = sob_t.reshape((1, 1, 3, 3, 3))

    # sob_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # sob_x = sob_x.reshape((1, 1, 3, 3))
    sob_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sob_y = sob_y.reshape((1, 1, 3, 3))

    for grad in grads:
        video_grad, audio_grad = grad
        # video
        video_grad = video_grad.sum(dim=(0,1))
        dtd = video_grad / abs(video_grad).max()

        deriv_t = F.conv3d(dtd[None][None], sob_t.float(), padding=(1, 1, 1))[0, 0, ...]
        dtd = dtd * (deriv_t[0,0,...] > (deriv_t.std() * std)).float()
        dtd = (dtd - -1) / (1 - -1)
        for fr in dtd:
            fr = cmap(fr)
            fr *= 255
            fr = fr.astype(np.uint8)
            fr = cv2.cvtColor(fr, cv2.COLOR_RGBA2BGR)
            video.append(fr)
        # audio
        audio_grad = audio_grad.sum(dim=(0, 1))
        dtd = audio_grad / abs(audio_grad).max()

        deriv_x = F.conv2d(dtd[None][None], sob_y.float(), padding=(1, 1))[0,0,...]
        dtd = dtd * (deriv_x[0,0,...] > (deriv_x.std() * std)).float()
        dtd = (dtd - -1) / (1 - -1)
        fr = cmap(dtd)
        fr *= 255
        fr = fr.astype(np.uint8)
        fr = cv2.cvtColor(fr, cv2.COLOR_RGBA2BGR)
        cv2.imshow("windowname", fr)
        cv2.waitKey(0)
        # press any key
        cv2.destroyAllWindows()
        audio.append(fr)
    return video, audio


def get_class_dict(activities="audio_classes.txt"):
    activities = sorted([line.rstrip('\n') for line in open(activities)])
    class_dict = {}
    idx_dict = {}
    for c_idx, activity in enumerate(activities):
        class_dict[c_idx] = activity
    return class_dict, activities

def get_sample(dir="/media/datasets/VA/UCF-101", sample_name=""):
    activity = sample_name.split("_")[1]
    return (join(dir, activity, sample_name),activity)


if __name__ == "__main__":
    mdl = get_mdl("results/embedding_2019-09-19-21:31_92.576.pth")
    class_dict, activities = get_class_dict()
    sample, lbl = get_sample(sample_name="v_PlayingDhol_g01_c01")
    lbl = activities.index(lbl)
    vid, mean = get_video_in(sample,mean=[90.0, 98.0, 102.0])
    aud = get_audio_in(sample)
    grads = explain(mdl, (vid,aud), lbl)
    video, audio = visualise(grads)




