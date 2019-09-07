import os
from collections import OrderedDict
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import optim
from audtorch.datasets import SpeechCommands
from test_tube import HyperOptArgumentParser
import audtorch.transforms as transforms
from torch.utils.data.distributed import DistributedSampler

import librosa
import numpy as np

import pytorch_lightning as pl


class VGGish(pl.LightningModule):
    """
    VGGish architecture for Speech Commands
    Input:      64x96 Mel log spectrogram
    Output:     confidence for class in commands
    """

    def __init__(self, hparams):
        super(VGGish, self).__init__()
        self.hparams = hparams

        self.batch_size = hparams.batch_size

        # hack to set n_logits before building the model
        self.n_logits = len(set(SpeechCommands(root=self.hparams.data_root, include=self.hparams.data_partition,
                                               download=True).add_silence().targets))

        self.__build_model()

    def __build_model(self):
        """
        Layout model
        :return:
        """
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
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
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams.drop_prob),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams.drop_prob),
            nn.Linear(4096, self.n_logits))


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        logits = F.softmax(x, dim=1)
        return logits

    def loss(self, labels, logits):
        ce = F.cross_entropy(logits, labels)
        return ce

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)

        loss_val = self.loss(y, y_hat)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)

        output = OrderedDict({
            'loss': loss_val
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)

        loss_val = self.loss(y, y_hat)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc': val_acc,
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output['val_acc']
            if self.trainer.use_dp:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dic = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        return tqdm_dic

    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return [optimizer]

    def __dataloader(self, train):

        # --------------------------------------------------------
        # Custom data transforms that audtorch doesn't include yet
        # --------------------------------------------------------
        class ToTensor(object):
            def __call__(self, x):
                return torch.tensor(x, dtype=torch.float)

        class Trim(object):
            # trim off 7 cols to fit nicely into network shape (64,96)
            def __call__(self, x):
                return x[..., 3:-4]

        class Spectrogram(object):
            # Return a log-scaled melspectrogram as per honk
            def __call__(self, x):
                return librosa.power_to_db(librosa.feature.melspectrogram(
                    y=np.squeeze(x), sr=16000, n_mels=64, fmax=4000, fmin=20,
                    n_fft=480, hop_length=16000 // 1000 * 10), ref=np.max)

        class Unsqueeze(object):
            def __call__(self, x):
                return x.unsqueeze(0)

        transform = transforms.Compose([
            transforms.RandomCrop(4096 * 4, method='replicate'),
            transforms.Normalize(),
            Spectrogram(),
            Trim(),
            ToTensor(),
            Unsqueeze()])

        dataset = SpeechCommands(root=self.hparams.data_root, train=train, include=self.hparams.data_partition,
                                 transform=transform, download=True, silence=False)

        # when using multi-node (ddp) we need to add the datasampler
        train_sampler = None
        batch_size = self.hparams.batch_size

        if self.use_ddp:
            train_sampler = DistributedSampler(dataset, rank=self.trainer.proc_rank)
            batch_size = batch_size // self.trainer.world_size  # scale batch size

        should_shuffle = train_sampler is None
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler
        )

        return loader

    @pl.data_loader
    def tng_dataloader(self):
        print('tng data loader called')
        return self.__dataloader(train=True)

    @pl.data_loader
    def val_dataloader(self):
        print('val data loader called')
        return self.__dataloader(train=False)

    @pl.data_loader
    def test_dataloader(self):
        print('test data loader called')
        return self.__dataloader(train=False)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams/
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip=5.0)

        parser.add_argument('--data_partition', default="10cmd", type=str)

        # network params
        parser.opt_list('--drop_prob', default=0.2, options=[0.2, 0.5], type=float, tunable=False)

        # data
        parser.add_argument('--data_root', default=os.path.join('~/Datasets/Audio', 'speech_commands_v0.02'), type=str)

        # training params (opt)
        parser.opt_list('--learning_rate', default=0.001 * 8, type=float,
                        options=[0.0001, 0.0005, 0.001, 0.005],
                        tunable=False)
        parser.opt_list('--optimizer_name', default='adam', type=str,
                        options=['adam'], tunable=False)

        # if using 2 nodes with 4 gpus each the batch size here
        #  (256) will be 256 / (2*8) = 16 per gpu
        parser.opt_list('--batch_size', default=64, type=int,
                        options=[32, 64, 128, 256], tunable=False,
                        help='batch size will be divided over all gpus being used across all nodes')
        return parser
