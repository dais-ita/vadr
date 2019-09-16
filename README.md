# Visual & Acoustic Discriminative Relevance for Activity Recognition
[![Conference](http://img.shields.io/badge/BMVA_Symposium_on_Video_Understanding-2019-blue.svg?style=flat-square)](https://dimadamen.github.io/bmva_symposium_2019/)

This repository holds experiment code for Visual & Acoustic Discriminative Relevance (VADR) demonstrated on the UCF-101 action recognition dataset.

The model architecture used is a concatenated fusion of VGGish and C3D at the layer before the bottleneck, with a simple MLP for classification of actions.

## UCF-101
https://www.crcv.ucf.edu/papers/UCF101_CRCV-TR-12-01.pdf

Unfortunately the original release of UCF-50 included videos without corresponding audio tracks, therefore for the visual-audio aspect of this work, we are using a subset of UCF-101 that only contains classes with audio tracks.

## Architecture
We use a C3D model pretrained on UCF-101 as the video subnetwork, and VGGish, trained on AudioSet for the audio subnetwork. Both architectures are based off VGG. They are concatenated at the feature embedding layer.

## Explainability
We use discriminative relevance (https://arxiv.org/abs/1908.01536), a modification of layerwise relevance backpropagation, to show relevant components in a four tuple consisting of <img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;<spatial_v,temporal_v,spectral_a,&space;temporal_a>" title="<spatial_v,temporal_v,spectral_a, temporal_a>" />. 