# steer

<p align="center">
  <img src="https://media.discordapp.net/attachments/1035221936680222853/1271394129951330335/SPOILER_image.png?ex=66cd970d&is=66cc458d&hm=5333d7d3edc4b357993418f95eb554cb6e40f5504c0b760200f907219f3b2d9f&=&format=webp&quality=lossless&width=1424&height=608" width="400"/>
</p>

<p align="center">
    open-source autonomous vehicle software&nbsp | <a href="https://asandei.com"> Website</a>&nbsp
<br>

This repository contains training and inference code for the Steer family of models. These are end-to-end neural networks for self driving.

## dependencies

- pytorch >= 2.0

## quick start

First of all, read and configure the option from `./config`. The `data.json` file has options for processing the dataset, turn on the `debug` option to only download a few samples instead of the full 80gb dataset.

To download & preprocess the dataset, run:

```
python3 ./src/setup.py
```

To start training the model:

```
python3 ./src/train.py
```

~pics here~

## dataset

comma2k19 (todo: the actual text)

## training

trained on an h100

## license

[Apache 2](LICENSE) © 2024 [Asandei Stefan-Alexandru](https://asandei.com). All rights reserved.
