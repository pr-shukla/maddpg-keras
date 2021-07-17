# MADDPG KERAS Implementation
Implementation Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm in keras with very simple customization. Link to the paper https://arxiv.org/pdf/1706.02275.pdf

## Project Description
There are many repositories available with multi-agent RL implementation either in older version of tensorflow or in pytorch. Those repositories have lots of dependecies and need spend good amount of time to figure out code structure. Lots of dependencies and complicated code structure can make customization difficult for someone new in RL  with not much knowledge about different RL tools present in python and could be time consuming.

This work is a part of my MS project where I implemented MADDPG algorithm in keras. My efforts was to make code structure less complicated and implement it using very basic deep learning libraries. For customization of the code you just need basic knowledge of keras and understanding of Deep Reinforcement Learning and you will be good to go.

The project was build-up by getting motivation from keras implementation of DDPG algorithm on https://keras.io/examples/rl/ddpg_pendulum/

Their are many possible improvements possible in this work. With addition of new agent code needs bit of customization. Although it can be generalized. Project is still open and any contributions are most welcome.

I have added trained model for demosntration 

https://user-images.githubusercontent.com/50385421/114270020-b837d180-9a27-11eb-89ac-635e01092d96.mp4

## Requirements:
* Python >= 3.5.6
* tensorflow >= 2.3.0
* numpy
* matplotlib
* math

## Installation
Run this command on command prompt
`git clone https://github.com/pr-shukla/maddpg-keras.git`

## Running code
To train the models run on command line
* `python train.py`

For test run on command line
* `python prediction.py`

## Code Structure
* `train.py` --> To start training of models
* `buffer.py` --> Contains code for replay buffer and training of models using MADDPG algorithm
* `noise.py` --> generates noise for exploration
* `env.py` --> code for environment
* `predict.py` --> for prediction of models
* `\saved_models\.h5` --> All trained models

## Contact Information
Please feel free to contact me if any help needed

Email: *prshukla.ind@gmail.com*






