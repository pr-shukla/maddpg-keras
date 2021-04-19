# maddpg-keras
Implementation Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm in keras

I have already added trained model for demosntration 

https://user-images.githubusercontent.com/50385421/114270020-b837d180-9a27-11eb-89ac-635e01092d96.mp4


## Requirements:
* tensorflow >= 2.3.0
* numpy
* matplotlib
* math

## Running code
To train the models run on command line
* python train.py

For prediction run on command line
* python prediction.py

## Code Structure
* train.py --> To start training of models
* buffer.py --> Contains code for replay buffer and training of models using MADDPG algorithm
* noise.py --> generates noise for exploration
* env.py --> code for environment
* predict.py --> for prediction of models
* .h5 --> All trained models

## Contact Information
Please feel free to contact me if any help needed

Email: *shuklaprashant652@gmail.com*






