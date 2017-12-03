# Networks

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def weights_init(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        init.xavier_uniform(layer.weight)
        init.constant(layer.bias, 0)


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


class CNN(Net):
    """Convolutional neural network"""
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
        'dropout': float  # If non-zero, apply a dropout layer this percentage
                          # after each convolution layer. If there is pooling after
                          # convolution layer, then apply dropout after pooling.
        'fcs': integer # Number of fully connected layers
        'fcdims': [...d...] # List of integers d for dimensions of each
                            # fully connected layer.
        """
        super(CNN, self).__init__(config)

        # Convolution layers, and pools
        layer_dim = self.config['input_dim']
        depth_prev = 3  # R, G, B
        

        self.pool_info = {}
        for method, size, layer in self.config['pools']:
            self.pool_info[layer] = (method, size)

        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        if self.config['dropout'] > 0:
            self.dropouts = nn.ModuleList()
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
                layer_dim = int(np.ceil(layer_dim / pool_size))
            else:
                self.pools.append(None)

            # Dropout layer
            if self.config['dropout'] > 0:
                self.dropouts.append(nn.Dropout(p=self.config['dropout']))
                 
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
            # Dropout, if specified
            if self.config['dropout'] > 0:
                x = self.dropouts[i](x)
        x = x.view(-1, self.last_conv_size)
        # Push through the fully connected layers
        for j in range(self.config['fcs']):
            x = self.fcs[j](x)
        return x
