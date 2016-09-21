from tsnet.launcher import run

run('-d mnist -n conv:2/16 relu:2 flat:0 sfmx:0/10 -e 100 -b 128 -lrnalg sgd -lrnparam 1e-3 1e-3 0.9 -v 1 -save mnist_2.mat')
