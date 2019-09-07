"""
Runs a model on a CPU
"""
import os
import numpy as np
import torch

from test_tube import HyperOptArgumentParser, Experiment
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from argparse import ArgumentParser

from audio.speech_commands.vggish import VGGish


def main(hparams):
    # init module
    print('loading model...')
    model = VGGish(hparams)
    print('model built')

    #init experiment
    exp = Experiment(
        name=hparams.experiment_name,
        save_dir=hparams.test_tube_save_path,
        autosave=False,
        description='test training speech commands'
    )

    exp.argparse(hparams)
    exp.save()

    # define callbacks
    model_save_path = '{}/{}/{}'.format(hparams.model_save_path, exp.name, exp.version)
    early_stop = EarlyStopping(
        monitor='val_acc',
        patience=3,
        verbose=True,
        mode='max'
    )
    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )

    # init trainer
    trainer = Trainer(
        experiment=exp,
        checkpoint_callback=checkpoint,
        early_stop_callback=early_stop,
        gpus=[0],
    )

    # start training
    trainer.fit(model)


if __name__ == '__main__':

    # dirs
    root_dir = os.path.dirname(os.path.realpath(__file__))
    demo_log_dir = os.path.join(root_dir, 'pt_lightning_demo_logs')
    checkpoint_dir = os.path.join(demo_log_dir, 'model_weights')
    test_tube_dir = os.path.join(demo_log_dir, 'test_tube_data')

    # although we user hyperOptParser, we are using it only as argparse right now
    parent_parser = HyperOptArgumentParser(strategy='grid_search', add_help=False)

    # gpu args
    parent_parser.add_argument('--test_tube_save_path', type=str,
                               default=test_tube_dir, help='where to save logs')
    parent_parser.add_argument('--model_save_path', type=str,
                               default=checkpoint_dir, help='where to save model')
    parent_parser.add_argument('--experiment_name', type=str,
                               default='pt_lightning_exp_a', help='test tube exp name')

    # allow mo/del to overwrite or extend args
    parser = VGGish.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    # run on HPC cluster
    main(hyperparams)
