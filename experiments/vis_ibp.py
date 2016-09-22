import sys; sys.path.append('./')

import numpy as np
from tsnet.launcher import run

hst = run('-d mnist   -n conv:2/16 relu:2 flat:0 sfmx:0/10 -e 100 -b 128 -lrnalg sgd -lrnparam 1e-3 1e-3 0.9 -v 2 -save   mnist_ibp.mat'); print np.argmax(hst['val_acc']); print '-' * 80
hst = run('-d cifar10 -n conv:2/16 relu:2 flat:0 sfmx:0/10 -e 100 -b 128 -lrnalg sgd -lrnparam 1e-3 1e-3 0.9 -v 2 -save cifar10_ibp.mat'); print np.argmax(hst['val_acc']); print '-' * 80
hst = run('-d svhn2   -n conv:2/16 relu:2 flat:0 sfmx:0/10 -e 100 -b 128 -lrnalg sgd -lrnparam 1e-3 1e-3 0.9 -v 2 -save   svhn2_ibp.mat'); print np.argmax(hst['val_acc']); print '-' * 80

hst = run('-d mnist   -n conv:0/16 relu:0 flat:0 sfmx:0/10 -e 100 -b 128 -lrnalg sgd -lrnparam 1e-3 1e-3 0.9 -v 2 -save   mnist_bp.mat'); print np.argmax(hst['val_acc']); print '-' * 80
hst = run('-d cifar10 -n conv:0/16 relu:0 flat:0 sfmx:0/10 -e 100 -b 128 -lrnalg sgd -lrnparam 1e-3 1e-3 0.9 -v 2 -save cifar10_bp.mat'); print np.argmax(hst['val_acc']); print '-' * 80
hst = run('-d svhn2   -n conv:0/16 relu:0 flat:0 sfmx:0/10 -e 100 -b 128 -lrnalg sgd -lrnparam 1e-3 1e-3 0.9 -v 2 -save   svhn2_bp.mat'); print np.argmax(hst['val_acc']); print '-' * 80
