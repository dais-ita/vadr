import os
from os.path import join
import subprocess
import torch
import soundfile as sf
from vggish import vggish
import vggish_input
import numpy as np
from tqdm import tqdm
import mel_features

DATA_DIR = "/media/datasets/Video/UCF-101/"
TARGET_DIR = "/media/datasets/VA/UCF-101/"
extract_audio = False

activities = sorted([join(DATA_DIR, line.rstrip('\n')) for line in open("audio_classes.txt")])
model = vggish()

if extract_audio:
    for name in activities:
        samples = sorted([join(name, sample) for sample in os.listdir(name) if sample.endswith(".avi")])
        for sample in samples:
            # Extract audio track
            subprocess.call(["ffmpeg", "-i", sample, "-y", "-vn", "-acodec", "pcm_s16le", (str(sample.split(".")[0]) + ".wav")])

for name in activities:
    print(f"Processing {name}")
    samples = sorted([join(name, sample) for sample in os.listdir(name) if sample.endswith(".wav")])
    for sample in tqdm(samples):
        d, fs = sf.read(sample)
        chunk_window_len = fs/2
        remainder = chunk_window_len - np.shape(d)[0] % chunk_window_len
        if len(np.shape(d)) > 1:
            pad = np.zeros((int(remainder), np.shape(d)[1]))
        else:
            pad = np.zeros((int(remainder)))

        d = np.concatenate((d, pad))

        def get_embedding(module, in_, out_):
            global embedding
            embedding = out_.data
            return None

        model = vggish()
        model.eval()
        hook = model.embeddings[3].register_forward_hook(get_embedding)

        for i in range(0, np.shape(d)[0], int(chunk_window_len)):
            # for compat with liam
            frame = int(i / chunk_window_len) * 16

            # chunk file & pad up to 1 sec for vggish to be happy
            wave = np.concatenate((d[i:i + int(chunk_window_len)],
                                     np.zeros(np.shape(d[i:i + int(chunk_window_len)]))))

            # preprocess for VGGish input
            x = vggish_input.waveform_to_examples(wave, fs)
            out = model(torch.tensor(x[None, ...]).float())

            embedding_name = f"{name}/{sample.split('/')[-1][:-4]}/embedding_{frame}_audio.pth"
            target_name = embedding_name.split("/")
            target_name[3] = "VA"
            embedding_name = "/".join(target_name)
            torch.save(embedding, embedding_name)


