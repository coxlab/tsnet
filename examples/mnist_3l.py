## Network Example

import numpy as np
from scipy.io import loadmat

W1 = loadmat('../Wfull.mat')['W1'].astype('float32').transpose(3,2,0,1)
#B1 = None
W2 = loadmat('../Wfull.mat')['W2'].astype('float32').transpose(3,2,0,1)
#B2 = None
#W3 = loadmat('../Wfull.mat')['W3'].astype('float32').transpose(3,2,0,1)[:10]
#B3 = None

net = []

net += [[]]; net[-1] += ['CONV']; net[-1] += [True]; net[-1] += [W1];    net[-1] += [None];
net += [[]]; net[-1] += ['MPOL']; net[-1] += [True]; net[-1] += [[2,2]]; net[-1] += [[2,2]];
net += [[]]; net[-1] += ['CONV']; net[-1] += [True]; net[-1] += [W2];    net[-1] += [None];
net += [[]]; net[-1] += ['MPOL']; net[-1] += [True]; net[-1] += [[2,2]]; net[-1] += [[2,2]];
#net += [[]]; net[-1] += ['CONV']; net[-1] += [True]; net[-1] += [W3];    net[-1] += [None];
#net += [[]]; net[-1] += ['RELU']; net[-1] += [True];

