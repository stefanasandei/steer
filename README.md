
<p align="center">
  <img src="./assets/logo.png" width="400"/>
</p>

<p align="center">
    open-source autonomous vehicle software&nbsp | <a href="https://asandei.com"> Website</a>&nbsp
<br>

This repository contains training and inference code for the Steer family of models. These are end-to-end neural networks for self driving.

## dependencies

Most notably:
- pytorch >= 2.0
- numpy
- opencv
- wandb

```
pip3 install -r requirements.txt
```

## quick start

First of all, read and configure the option from `./config`. The `data.json` file has options for processing the dataset, turn on the `debug` option to only download a few samples instead of the full 80gb dataset.

To download & preprocess the dataset, run:

```
python3 ./src/prepare.py
```

To start training the model:

```
python3 ./src/train.py
```

## dataset

The dataset used is comma2k19, by Comma AI. It consists of 2019 segments of recorded driving information across a highway in California. The preprocessing script extracts the relevant information, such as steering angles (radians), speed (m/s) and frame location data (position, orientation - converted to local frame reference).

![debug picture](./assets/debug0.png)

Debug information projected into a sample frame from the dataset.

## training

**model** | **loss** | **steps** | **batch size** | **frames ctx** | **GPU**
:--------:|:--------:|:---------:|:--------------:|:--------------:|:------:
 PilotNet |  62.44   |    500    |      128       |      3/30      |  A100
 Seq2Seq  |    -     |    500    |      128       |      3/30      |  A100
  Steer   |    -     |    500    |      128       |      3/30      |  A100

The first model is based on the PilotNet architecture. It has two main components: a series of convolutional layers and a series of feed forward layers, acting as a controller. The frames are concatanated and feed into the conv layers, after that the features are concatenated with the past path. The resulted tensor is then passed thru the linear layers, later branching into multiple linear output layers to create predictions. This model proved effective in roughly estimating the steering angle and the speed of the vehicle, however it presented poor results in predicting a future path. However, the model did understand to predict the vehicle will go (about) in the forward direction. This inability to predict the path can be caused by the lack of temporal features.

## license

[Apache 2](LICENSE) © 2024 [Asandei Stefan-Alexandru](https://asandei.com). All rights reserved.
