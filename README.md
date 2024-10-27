
<p align="center">
  <img src="./assets/logo.png" width="400"/>
</p>

<p align="center">
    open-source autonomous vehicle software&nbsp | <a href="https://asandei.com"> website</a>&nbsp
<br>

Proof of concept work for utilizing Mamba based models for self driving. 

Read the [technical report](./notebooks/report.pdf) for more details regarding the research process, architecture details and findings.

Note: this project is for educational purposes only, the models are undertrained due to hardware restraints.

## dependencies

Most notably:
- pytorch >= 2.0
- numpy
- opencv
- wandb
- mamba-ssm
- causal-conv1d

```
pip3 install -r requirements.txt
```

## quick start

First of all, read and configure the option from `./config`. The `data.json` file has options for processing the dataset, turn on the `debug` option to only download a few samples instead of the full 80gb dataset.

To download & preprocess the dataset, run:

```
python3 ./src/prepare.py
```

To start training the model (edit the parameters within the script file):

```
python3 ./src/train.py
```

To generate a video using model predictions:

```
python3 --model steer.pt --route 2018-07-29--16-37-17 --out out.mp4
```

[steer-15-09-2024-r3.webm](https://github.com/user-attachments/assets/f9702535-d440-406e-81ac-6f1424419517)

## dataset

The dataset used is comma2k19, by Comma AI. It consists of 2019 segments of recorded driving information across a highway in California. The preprocessing script extracts the relevant information, such as steering angles (radians), speed (m/s) and frame location data (position, orientation - converted to local frame reference).

<!-- ![debug picture](./assets/debug0.png) -->

<!-- Debug information projected into a sample frame from the dataset. -->

## training

**model** | **loss** | **parameters** | **MFLOPs** | **backbone**
:--------:|:--------:|:--------------:|:----------:|:-----------:
 PilotNet |  29.70   |      0.8M      |    119     |     CNN
 Seq2Seq  |  151.80  |      5.9M      |    9417    | RegNet + GRU
**Steer** | **7.16** |    **6.4M**    |  **5130**  |     ViM

The presented model is **Steer**, however for comparison, also PilotNet and Seq2Seq models have been trained. Note: the small number of epochs (~4) and the batch size (4) is due to the hardware limitations.

Steer is an end-to-end neural network based on the Mamba architecture. It takes as input a sequence of N past frames and N past xyz positions (in local frame coordonates). The frames are converted into patches and processed through a series of Mamba bidirectional blocks. These features are then fed into a video encoder to compute a hidden state. Simultaneously, the past positions are passed through a path encoder, also utilizing Mamba blocks to generate another hidden state. The two hidden states are combined and passed through a final MLP head to produce the final features.

The first comparison model is based on the PilotNet architecture, comprising two main components: a series of convolutional layers and feed-forward layers functioning as a controller. The frames are concatenated and processed by the convolutional layers, after which the features are combined with the past path data. The resulting tensor is passed through linear layers, branching into multiple output layers to generate predictions. This model effectively estimated steering angles and vehicle speed but struggled with accurate path prediction, likely due to its lack of temporal feature integration. Nevertheless, it understood that the vehicle would move primarily in a forward direction.

The Seq2Seq model consists of an encoder and a decoder. The encoder, built around a RegNet, extracts image features from each frame in the sequence. These features, after being concatenated with past path data, are passed to the decoder. The decoder, utilizing a GRU network, processes the sequence and generates predictions. Its sequential structure enables it to capture spatial information and make more accurate predictions.

All models were trained on 4 epochs on only 2% of the first chunk of the dataset. The main model is able to determine when to drive forward, stop at a semaphore and speed up again. This proves its potential, however my available hardware limits the amount of training as well as other hyperparameters (batch size, dataset usage, epochs, model size). Because of this, I focused on the educational value of this repository. 

## citation

If you use this material, please cite this repository with the following:

```bibtex
@misc{asandei_efficient_2024,
  author       = {Asandei Ștefan-Alexandru},
  title        = {Efficient temporal awareness for autonomous vehicles using State Space Models},
  year         = 2024,
  publisher    = {Github},
  url          = {https://github.com/stefanasandei/steer},
}
```

## license

[Apache 2](LICENSE) © 2024 [Asandei Stefan-Alexandru](https://asandei.com). All rights reserved.
