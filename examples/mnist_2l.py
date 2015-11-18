## Network Example

import numpy as np
from scipy.io import loadmat

#W1 = loadmat('WB-deep2.mat')['W1'].astype('float32').transpose(3,2,0,1)
#B1 = loadmat('WB-deep2.mat')['B1'].astype('float32')
#W2 = loadmat('WB-deep2.mat')['W2'].astype('float32').transpose(3,2,0,1)
#B2 = loadmat('WB-deep2.mat')['B2'].astype('float32')
V  = loadmat('../V.mat')['V'].astype('float32').reshape(1,7,7,49)[...,:25].transpose(0,2,1,3)
V  = V[:,:,:,None,None,:]

W1 = np.random.randn( 80, 1,7,7).astype('float32')
B1 = None
W2 = np.random.randn(160, 80,3,3).astype('float32')
B2 = None

net = []

net += [[]]; net[-1] += ['CONV']; net[-1] += [True]; net[-1] += [W1];    net[-1] += [B1];    net[-1] += [[2,2]];
net += [[]]; net[-1] += ['MPOL']; net[-1] += [True]; net[-1] += [[3,3]]; net[-1] += [[2,2]];
net += [[]]; net[-1] += ['DRED']; net[-1] += [True]; net[-1] += [V];
net += [[]]; net[-1] += ['CONV']; net[-1] += [True]; net[-1] += [W2];    net[-1] += [B2];    net[-1] += [None];
net += [[]]; net[-1] += ['MPOL']; net[-1] += [True]; net[-1] += [[3,3]]; net[-1] += [None];

