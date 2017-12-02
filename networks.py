# CSE 546 Homework 1
# Problem 2
#
# Networksx

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def weights_init(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        init.xavier_uniform(layer.weight)
        init.constant(layer.bias, 0)

class CNNExample(nn.Module):

    def __init__(self):
        super(CNNExample, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Net(nn.Module):
    """Some neural network"""
    def __init__(self, config={}):
        super(Net, self).__init__()
        self.config = config
        
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
# 2a
class FCN(Net):
    """Fully connected single layer network"""
    def __init__(self):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 10, bias=True)
        self.apply(weights_init)

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = self.fc1(x)
        return x


# 2b
class FCN_FCH(Net):
    """Fully connected single layer network with one fully connected
    hidden layer nonlinearized by ReLu."""
    def __init__(self, config):
        if "M" not in config:
            raise ValueError("M must be configured!")
        super(FCN_FCH, self).__init__(config)
        M = self.config['M']
        self.fc_h = nn.Linear(3*32*32, M, bias=True)
        self.fc_o = nn.Linear(M, 10, bias=True)
        self.apply(weights_init)

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = F.relu(self.fc_h(x))
        x = self.fc_o(x)
        return x

    
# 2c
class CNN_Basic(Net):
    """Fully connected output + one convolutional layer with max pool"""
    def __init__(self, config):
        if "M" not in config or "p" not in config or "N" not in config:
            raise ValueError("M, p, N must be configured!")
        super(CNN_Basic, self).__init__(config)
        M = self.config['M']
        p = self.config['p']
        N = self.config['N']  # Pool size (i.e. dimension of the pool of cells (think about swimming pool))
        self.conv1 = nn.Conv2d(3, M, p)
        self.pool = nn.MaxPool2d(N, N)
        dim = int((33 - p) / N)  # side dimension of layer after pool
        self.fc1 = nn.Linear(dim*dim*M, 10, bias=True)
        self.apply(weights_init)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        dim = int((33 - self.config['p']) / self.config['N'])
        x = x.view(-1, dim*dim*self.config['M'])
        x = self.fc1(x)
        return x


# 2d
class CNN_Complex(Net):
    """Fully connected output + one convolutional layer with max pool"""
    def __init__(self, config):
        """How to config:

        'input_dim': integer  # Image dimension
        'convs': integer # Nubmer of convolution layers
        'depths': [...q...]  # List of integers q for depth of each convolution
                             # layer. 
        'kernels': [...k...]  # List of integers k for kernel size of each
                              # convolution layer.
        'pools': [(METHOD, N, i), ...] # List of tuples (METHOD, N) to specify
                                      # pooling method (Max or Avg), with pool
                                      # size N by N. The pool is performed at
                                      # conv layer i.
        'fcs': integer # Number of fully connected layers
        'fcdims': [...d...] # List of integers d for dimensions of each
                            # fully connected layer.
        """
        super(CNN_Complex, self).__init__(config)

        # Convolution layers, and pools
        layer_dim = self.config['input_dim']
        depth_prev = 3  # R, G, B
        

        self.pool_info = {}
        for method, size, layer in self.config['pools']:
            self.pool_info[layer] = (method, size)

        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(self.config['convs']):
            self.convs.append(nn.Conv2d(depth_prev, self.config['depths'][i], self.config['kernels'][i]))
            depth_prev = self.config['depths'][i]

            layer_dim = int(layer_dim - self.config['kernels'][i])
            if i in self.pool_info:
                pool_method, pool_size = self.pool_info[i]
                if pool_method == "Max":
                    self.pools.append(nn.MaxPool2d(pool_size, pool_size))
                elif pool_method == "Avg":
                    self.pools.append(nn.AvgPool2d(pool_size, pool_size))
                else:
                    raise ValueError("Unrecognized pool method %s" % pool_method)
                layer_dim = layer_dim // pool_size
            else:
                self.pools.append(None)
                 
        # Fully connected layers
        self.fcs = nn.ModuleList()
        self.last_conv_size = layer_dim * layer_dim * self.config['depths'][-1]
        fc_prev_size = self.last_conv_size
        for j in range(self.config['fcs']):
            self.fcs.append(nn.Linear(fc_prev_size, self.config['fcdims'][j], bias=True))
            fc_prev_size = self.config['fcdims'][j]
        self.apply(weights_init)

    def forward(self, x):
        # Push through convolutions with pools and ReLu
        for i in range(self.config['convs']):
            x = F.relu(self.convs[i](x))
            if i in self.pool_info:
                x = self.pools[i](x)
        x = x.view(-1, self.last_conv_size)
        # Push through the fully connected layers
        for j in range(self.config['fcs']):
            x = self.fcs[j](x)
        return x
