import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from audtorch.datasets import SpeechCommands
import audtorch.transforms as transforms

import pytorch_lightning as pl


class VGGish(pl.LightningModule):
    """
    Input:      224x96 amplitude log-spectrogram
    Output:     confidence for class in commands
    """
    def __init__(self, num_classes):
        super(VGGish, self).__init__()

        self.command_list = [
            "yes", "no", "up", "down",
            "left", "right", "stop", "go"]

        self.features = nn.Sequential(
            nn.Conv2d(1,  64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512*14*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, len(self.command_list)))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'loss': F.cross_entropy(y_hat, y)}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.02)]

    @pl.data_loader
    def tng_dataloader(self):
        transform = transforms.Compose([
            transforms.RandomCrop(4096 * 4, method='replicate'),
            transforms.Normalize(),
            # out size of [224, 96]. 447,172 if 4096*3. 447,130 if *4
            transforms.Spectrogram(window_size=447, hop_size=130),
            transforms.Log(),
            ToTensor()])
        dataset = SpeechCommands(root="~/Workspace/Datasets/Audio/Speech/speech_commands_v0.02", train=True,
                                 include=self.command_list, download=False, silence=False, transform=transform)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True
        )
        return loader

    @pl.data_loader
    def test_dataloader(self):
        transform = transforms.Compose([
            transforms.RandomCrop(4096 * 4, method='pad'),
            transforms.Normalize(),
            # out size of [224, 96]. 447,172 if 4096*3. 447,130 if *4
            transforms.Spectrogram(window_size=447, hop_size=130),
            transforms.Log(),
            ToTensor()])
        dataset = SpeechCommands(root="~/Workspace/Datasets/Audio/Speech/speech_commands_v0.02", train=False,
                                 include=self.command_list, download=False, silence=False, transform=transform)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True
        )
        return loader


# audtorch doesn't include this yet
class ToTensor(object):
    def __call__(self, x):
        return torch.tensor(x)
