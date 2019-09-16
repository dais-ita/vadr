import sys
import numpy as np
import cv2
import torch
import torchvision as tv
import matplotlib.pyplot as plt
# from audio import resnet, layers
sys.path.append('../torchexplain')
import torch.nn.functional as F

norm_value = 1
# ds_mean = [114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value]
ds_mean = [90.0, 98.0, 102.0]

from audio import c3d
import collections
import time

mdl = c3d.C3D(101, range=(-max(ds_mean), 255-min(ds_mean)))
if hasattr(mdl, "module"):
     module = mdl.module
else:
    module = mdl

state = torch.load('save_20.pth')
n_state = []
for k, v in state['state_dict'].items():
    if 'module' in k:
        n_state.append((k[7:], v))
if n_state == []:
    n_state = state['state_dict']
else:
    n_state = collections.OrderedDict(n_state)
module.load_state_dict(n_state)

# In[6]:

module.eval()
vid_id = None
samples = []
sample_out = []
mdl = mdl.cuda()

cap = cv2.VideoCapture(0)
i = 0
cap.set(cv2.CAP_PROP_FPS,30)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
vid = torch.zeros((1, 3, 16, 112, 112)).requires_grad_()
cap.release()
print(vid.shape)

recording = [0]

def capture(recording, vid):
    cap = cv2.VideoCapture(0)
    while recording[0] < 16:
        # Capture frame-by-frame
        ret, f = cap.read()
        if not ret:
            break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = cv2.resize(f, (112,112))
        f = f.transpose(2, 0, 1)
        f = torch.from_numpy(f)
        vid[0,:,recording[0],...] = f
        recording[0] += 1

def explain(recording, sample):
    seismic = plt.get_cmap("seismic")
    while recording[0] < 16:
        # print(f.max())
        img = sample[0,:,recording[0],...]
        img = tv.transforms.Normalize(ds_mean, [1,1,1])(img)
        sample[0, :, recording[0], :, :] = img
        recording[0] += 1


    st = time.time()
    i = 0
    sample = sample.cuda()
    out = mdl(sample)
    sft = F.softmax(out)
    fc, vid_label = sft.topk(1, 1)
    vid_label = vid_label.item()
    filter_out = torch.zeros_like(out)
    filter_out[:, vid_label] = 1
    pos_evidence = torch.autograd.grad(out, sample, grad_outputs=filter_out)[0]
    pos_vis = pos_evidence.sum(dim=(0, 1))
    pos_vis /= abs(pos_vis.max())
    sob_t = torch.tensor([[[1, 2, 1], [2, 4, 2], [1, 2, 1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                          [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]])
    sob_t = sob_t.reshape((1, 1, 3, 3, 3))
    deriv_t = F.conv3d(pos_vis[None][None], sob_t.cuda().float(), padding=(1, 1, 1))[0, 0, ...]
    # pos_vis = (pos_vis - pos_vis.min()) / (pos_vis.max() - pos_vis.min())
    output = []
    for j in range(16):
        fr = deriv_t[j,...].detach().cpu().numpy()
        fr = seismic(fr).astype(np.float32)
        fr = cv2.cvtColor(fr, cv2.COLOR_RGBA2BGR)
        fr = cv2.resize(fr, (896,896))
        output.append(fr)
        # if i % 2 == 0:
    return output

def put(recording,output):
    for f in output:
        time.sleep(1/30)
        cv2.imshow('frame', f)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            quit()


print("capturing")
capture(recording,vid)
print("explain")
recording[0] = 0
output = explain(recording,vid)
print("displaying")
put(recording,output)

# c.join()
# e.join()
# p.join()

# When everything done, release the capture

cv2.destroyAllWindows()
cap.release()
