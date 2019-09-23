import torch
import c3d
import cv2
import os
import shutil
from os.path import join
from tqdm import tqdm

model = c3d.C3D(101)
model = torch.nn.DataParallel(model)
save = torch.load("mdl.pth")
model.load_state_dict(save["state_dict"])
model = model.cuda()
model.eval()

inp = torch.zeros(1,3,16,112,112)

embedding = 0

def get_embedding(module, in_, out_):
    global embedding
    embedding = out_.data
    return None

hook = model.module.fc7.register_forward_hook(get_embedding)

ds_mean = [90.0, 98.0, 102.0]
mean = torch.zeros_like(inp)
for c in range(3):
    mean[:,c,...] += ds_mean[c]

ds_dir = "/media/datasets/Video/UCF-101"
target_dir = "/media/datasets/VA/UCF-101"
activities = sorted([join(ds_dir,line.rstrip('\n')) for line in open("audio_classes.txt")])



for activity in activities:
    print(f"Processing embeddings for {activity}")
    samples = sorted([join(activity,vidname) for vidname in os.listdir(activity) if not (vidname.endswith(".avi") or vidname.endswith(".wav"))])
    for sample in tqdm(samples):
        cap = cv2.VideoCapture(sample + ".avi")
        print(sample, cap.get(cv2.CAP_PROP_FPS))
        continue
        frames = sorted([join(sample,frame) for frame in os.listdir(sample) if not(frame.endswith("vert.jpg") or frame.endswith("horiz.jpg") or frame.endswith(".pth"))], key=lambda x: int(x.split("_")[-1][:-4]))
        while (len(frames) % 16) != 0:
            frames.append(frames[-1])
        for i in range(0,len(frames),16):
            embedding_name = f"{sample}/embedding_{i}_video.pth"
            target_name = embedding_name.split("/")
            target_name[3] = "VA"
            for fldr in range(1,len(target_name)-1):
                dest = "/".join(target_name[:fldr+1])
                if not os.path.exists(dest):
                    os.mkdir(dest)
            target_name = "/".join(target_name)
            if not os.path.exists(target_name):
                if os.path.exists(embedding_name):
                    shutil.move(embedding_name,target_name)
                    continue
            else:
                continue
            sample_frames = frames[i:i+16]
            for fidx, frame in enumerate(sample_frames):
                frame = cv2.imread(frame)
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame,(112,112)).transpose(2,0,1)
                inp[0,:,fidx,:,:] = torch.from_numpy(frame).cuda()
            inp -= mean
            out = model(inp)
            torch.save(embedding,target_name)


