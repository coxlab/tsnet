# TSNet
Code and scripts for reproducing the experimental results of [Tensor Switching Networks](https://arxiv.org/) by Chuan-Yung Tsai, Andrew Saxe, and David Cox.

TSNet is a novel neural network algorithm, which generalizes the Rectified Linear Unit (ReLU) nonlinearity to tensor-valued hidden units, and avoids the vanishing gradient problem by construction.
Our results demonstrate that the TSNet is more expressive and consistently learns faster than standard ReLU networks.

## Requirements

[Keras](https://github.com/fchollet/keras), [Kerosene](https://github.com/dribnet/kerosene), [Blessings](https://github.com/erikrose/blessings), and [libsvm-compact](http://www.di.ens.fr/data/software/).

## Usage

Use `bash nips06/run.sh` if you want to run all of our experiments.
Otherwise, simply use `python tsnet_cli.py` to train, validate and test single models.
For example, you can use the following 2 lines of commands to compare a standard single-hidden-layer ReLU network (first line) and a TS-ReLU network with inverted backpropagation (second line) using the [Numpy implementation](tsnet/core_numpy) (add `-k` to switch to using the [Keras implementation](tsnet/core_keras)):
```
python tsnet_cli.py -d mnist -n conv:0/16 relu:0 flat:0 sfmx:0/10 -e 10 -lrnparam 1e-3 1e-3 0.9 -v 1
python tsnet_cli.py -d mnist -n conv:2/16 relu:2 flat:0 sfmx:0/10 -e 10 -lrnparam 1e-3 1e-3 0.9 -v 1
```

Please see [cmp_mlp.py](nips16/experiments/cmp_mlp.py) and [cmp_cnn.py](nips16/experiments/cmp_cnn.py) for more examples of how to define a network.
