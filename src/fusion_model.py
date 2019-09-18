from collections import OrderedDict

import torch
from torch import nn
import torchexplain

import c3d
import vggish


class FusionModel(nn.Module):
    def __init__(self, nclasses, embedding=False,train=True):
        self.embedding = embedding
        self.lib = nn if train else torchexplain
        self.video = c3d.C3D(nclasses, False, range=(0,255),embedding=embedding,train=train)
        self.audio = vggish.VGGish()
        self.scaler = lambda x: (x - min(x)/(max(x)-min(x))) if train else self.lib.MinMaxScaler()
        self.classifier = nn.Sequential(
            self.lib.Linear(8192,4096),
            self.lib.ReLU(),
            self.lib.Linear(4096,1024),
            self.lib.ReLU(),
            self.lib.Linear(1024,512),
            self.lib.ReLU(),
            self.lib.Linear(512,nclasses)
        )
    def forward(self, *input):
        if not self.embedding:
            vid_embeddings = self.scaler(self.video(input[0]))
            aud_embeddings = self.scaler(self.audio(input[1]))
            embeddings = torch.cat([vid_embeddings,aud_embeddings],1)
        else:
            embeddings = input
        out = self.classifier(embeddings)
        return out

    def load_state_dict(self, state_dict, strict=True):
        vid_sd = OrderedDict()
        aud_sd = OrderedDict()
        class_sd = OrderedDict()
        for k, v in state_dict:
            if "video" in state_dict:
                vid_sd[k] = v
            if "audio" in state_dict:
                aud_sd[k] = v
            else:
                class_sd[k] = v
        self.video.load_state_dict(vid_sd,strict)
        self.audio.load_state_dict(aud_sd,strict)
        for layer in self.classifier:
            if hasattr(layer,"weight"):
                layer.bias = class_sd.popitem(last=False)
                layer.weight = class_sd.popitem(last=False)

