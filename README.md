# TSNet
This repository provides code and scripts for reproducing the experimental results of [Tensor Switching Networks](https://arxiv.org/abs/1610.10087) by Chuan-Yung Tsai, Andrew Saxe, and David Cox.

TSNet is a novel neural network algorithm, which generalizes the Rectified Linear Unit (ReLU) nonlinearity to tensor-valued hidden units, and avoids the vanishing gradient problem by construction.
Our experimental results show that the TSNet is not only more expressive, but also consistently learns faster than standard ReLU networks.

## Requirements

[Keras](https://github.com/fchollet/keras), [Kerosene](https://github.com/dribnet/kerosene), [Blessings](https://github.com/erikrose/blessings), and [libsvm-compact](http://www.di.ens.fr/data/software/).

## Usage

Use `bash nips06/run.sh` if you wish to run all of our experiments.
Otherwise, use `python tsnet_cli.py` to run single models.
For example, you can use the following commands to compare a simple single-hidden-layer ReLU network (first line) and its TS counterpart using the inverted backpropagation learning (second line):
```
python tsnet_cli.py -d mnist -n conv:0/16 relu:0 flat:0 sfmx:0/10 -e 10 -lrnparam 1e-3 1e-3 0.9 -v 1
python tsnet_cli.py -d mnist -n conv:2/16 relu:2 flat:0 sfmx:0/10 -e 10 -lrnparam 1e-3 1e-3 0.9 -v 1
```
By default, the [Numpy backend](tsnet/core_numpy) supporting all learning algorithms is used, but you can also switch to the simpler and faster [Keras backend](tsnet/core_keras) by using `-k`.
Please refer to [cmp_mlp.py](nips16/experiments/cmp_mlp.py) and [cmp_cnn.py](nips16/experiments/cmp_cnn.py) for more examples of how to 
define networks.
