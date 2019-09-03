import sys
import queue
import threading
import pickle
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
cudnn.enabled = True
cudnn.benchmark = True
import torchvision as tv
import matplotlib.pyplot as plt
# from audio import resnet, layers
sys.path.append('/home/c1435690/Projects/DAIS-ITA/torchexplain')
import lrp
import torch.nn.functional as F

norm_value = 1
# ds_mean = [114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value]
ds_mean = [90.0, 98.0, 102.0]
from audio import c3d
import collections
import time
import explain
mdl = c3d.C3D(101, range=(-max(ds_mean), 255-min(ds_mean)))
if hasattr(mdl, "module"):
     module = mdl.module
else:
    module = mdl

state = torch.load('save_22.pth')
n_state = []
for k, v in state['state_dict'].items():
    if 'module' in k:
        n_state.append((k[7:], v))
if n_state == []:
    n_state = state['state_dict']
else:
    n_state = collections.OrderedDict(n_state)
state = None
module.load_state_dict(n_state)
n_state = None

with open("class_dict.bin","rb") as f:
    class_names = pickle.load(f)

# In[6]:

module.eval()
mdl = mdl.cuda().eval()

recording = [0]

font = cv2.FONT_HERSHEY_SIMPLEX
text_x, text_y = (448,880)
fontScale = 1
fontColor = (0,0,0)
lineType = 2

def capture(recording):
    cap = cv2.VideoCapture(0)
    while recording:
        print(recording)
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            in_.put(None)
            recording.pop()
        f = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
        in_.put(f)
    print("in done")

def explain(recording):
    i = 0
    mean = torch.zeros(3, 112, 112)
    sample = torch.zeros(1, 3, 16, 112, 112).requires_grad_()
    seismic = plt.get_cmap("seismic")
    while recording:
        print(recording)
        f = in_.get()
        # print(f.max())
        img = tv.transforms.ToPILImage()(f)
        img = tv.transforms.Scale((112, 112))(img)
        img = tv.transforms.ToTensor()(img) * 255
        img = tv.transforms.Normalize(ds_mean, [1,1,1])(img)
        sample[0, :, i, :, :] = img
        i += 1

        if i == 16:
            st = time.time()
            i = 0
            sample = sample.cuda()
            sample_out = mdl(sample)
            vid_out = sample_out  # torch.cat(sample_out,0).mean(0, keepdim=True)
            sft = F.softmax(vid_out)
            fc, vid_label = sft.topk(1, 1)
            vid_label = vid_label.item()
            filter_out = torch.zeros_like(vid_out)
            filter_out[:, vid_label] = 1
            pos_evidence = torch.autograd.grad(vid_out, sample, grad_outputs=filter_out)[0]
            pos_vis = pos_evidence.sum(dim=(0, 1))
            pos_vis /= abs(pos_vis.max())
            # sob_t = torch.tensor([[[1, 2, 1], [2, 4, 2], [1, 2, 1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            #                       [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]])
            # sob_t = sob_t.reshape((1, 1, 3, 3, 3))
            # deriv_t = F.conv3d(pos_vis[None][None], sob_t.cuda().float(), padding=(1, 1, 1))[0, 0, ...]
            # pos_vis = (pos_vis - pos_vis.min()) / (pos_vis.max() - pos_vis.min())
            for j in range(16):
                fr = pos_vis[j,...].detach().cpu().numpy()
                fr = ((fr - fr.min() / (fr.max() - fr.min())) * (1.0 - 0.5) + 0.5)
                fr = seismic(fr).astype(np.float32)
                fr = cv2.cvtColor(fr, cv2.COLOR_RGBA2BGR)
                fr = cv2.resize(fr, (896,896))
                out_.put((fr,class_names[vid_label]))
        in_.task_done()
    print("exp done")
        # if i % 2 == 0:


def put(recording):
    name = None
    while recording:
        time.sleep(1/30)
        f, lbl = out_.get(block=True,timeout=90)#.type(torch.uint8)
        text_width, text_height = cv2.getTextSize(f"{lbl}", font, fontScale, lineType)[0]
        cv2.putText(f,f'{lbl}',
                (text_x-text_width//2,text_y),
                font,
                fontScale,
                fontColor,
                lineType)
        cv2.imshow('Discriminative Relevance', f)
        name = lbl
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            while recording:
                recording.pop()
            print(recording)
            break
    out_.task_done()
    print("out done")

in_ = queue.Queue(16)
out_ = queue.Queue(16)
c = threading.Thread(target=capture,args=(recording,))
e = threading.Thread(target=explain,args=(recording,))
p = threading.Thread(target=put,args=(recording,))
c.start()
e.start()
p.start()
c.join()
e.join()
p.join()

# When everything done, release the capture

cv2.destroyAllWindows()
cap.release()
