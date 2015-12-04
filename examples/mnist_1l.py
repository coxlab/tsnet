## Network Example

import numpy as np
from scipy.io import loadmat

V = loadmat('../V.mat')['V'].astype('float32').reshape(1,7,7,49)[...,:25].transpose(0,2,1,3)
P = loadmat('../W.mat')['W'].astype('float32').reshape(55,25)
#P = np.random.randn(400, 25).astype('float32')
W = np.tensordot(P, V, [(1,),(3,)])
#B = None

#W = loadmat('WB50.mat')['W'].astype('float32').transpose(3,2,0,1)
#B = loadmat('WB50.mat')['B'].astype('float32')

#W = np.random.randn(55, 1, 7, 7).astype('float32')
#B = None

#W /= np.linalg.norm(W)
V = V[:,:,:,None,None,:]

net = []

net += [[]]; net[-1] += ['PADD']; net[-1] += [True]; net[-1] += [[3,3,3,3]];
net += [[]]; net[-1] += ['CONV']; net[-1] += [True]; net[-1] += [W];         net[-1] += [None];
net += [[]]; net[-1] += ['MPOL']; net[-1] += [True]; net[-1] += [[7,7]];     net[-1] += [[7,7]];
net += [[]]; net[-1] += ['DRED']; net[-1] += [True]; net[-1] += [V];

