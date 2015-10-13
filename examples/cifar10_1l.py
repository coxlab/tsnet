## Network Example

import numpy as np
#rom scipy.io import loadmat

# = loadmat('../V.mat')['V'].astype('float32').reshape(1,7,7,49)[...,:25].transpose(0,2,1,3)
# = loadmat('../W.mat')['W'].astype('float32').reshape(55,25)
#P = np.random.randn(400, 25).astype('float32')
# = np.tensordot(P, V[(1,),(3,)])
# = None

#W = loadmat('WB50.mat')['W'].astype('float32').transpose(3,2,0,1)
#B = loadmat('WB50.mat')['B'].astype('float32')

W = np.random.randn(100, 3, 5, 5).astype('float32')
B = None

#W /= np.linalg.norm(W)

net = []

#net += [[]]; net[-1] += ['pad'];     net[-1] += [True]; net[-1] += [[2,2,2,2]];
net += [[]]; net[-1] += ['conv'];    net[-1] += [True]; net[-1] += [W];         net[-1] += [B];     net[-1] += [[3,3]];
net += [[]]; net[-1] += ['maxpool']; net[-1] += [True]; net[-1] += [[3,3]];     net[-1] += [[2,2]];
#net += [[]]; net[-1] += ['dimredc']; net[-1] += [True]; net[-1] += [V];

