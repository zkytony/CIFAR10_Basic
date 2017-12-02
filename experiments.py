# CSE 546 Homework 1
# Problem 2
#
# Networks

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.init as init

import numpy as np
import matplotlib.pyplot as plt
import yaml
from pprint import pprint

from data import CIFAR10
import networks as nets

def accuracy(net, loader):
    correct = 0
    total = 0
    for data in loader:
        images, labels = data
        outputs = net(Variable(images).cuda())
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels.cuda()).squeeze()
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()
    return 100 * correct / total


def save_plot(name, training_acc, testing_acc, setting):  # setting: assignments of params
    iters = np.arange(len(training_acc))
    plt.plot(iters, training_acc, label="train", color="green")
    plt.plot(iters, testing_acc, label="test", color="orange")
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend(loc="lower right", fontsize=10)
    plt.savefig("plots/%s_plot.png" % name, dpi=300, bbox_inches='tight')
    with open("plots/%s_config.yaml" % name, "w") as f:
        yaml.dump(setting, f)
    plt.clf()



def run_experiment(name, net, dataset, lr_range, momentum_range, rounds=5, num_epochs=10):
    """Train, test, cross validation (random search)"""
    print("=============== Experiment %s ===============" % name)
    best_setting = {}
    best_training_acc = []
    best_testing_acc = []
    
    try:
        criterion = nn.CrossEntropyLoss()   #?
        for r in range(rounds):
            print("~~Round %d~~" % r)
            setting = {'lr': np.random.uniform(low=lr_range[0], high=lr_range[1]),
                      'momentum': np.random.uniform(low=momentum_range[0], high=momentum_range[1])}
            pprint(setting)

            optimizer = optim.SGD(net.parameters(), lr=setting['lr'], momentum=setting['momentum'])

            training_acc = []
            testing_acc = []
            for epoch in range(num_epochs):
                running_loss = 0.0
                for i, data in enumerate(dataset.trainloader, 0):
                    inputs, labels = data
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                    optimizer.zero_grad()

                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.data[0]
                    if i % 2000 == 1999:
                        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                        running_loss = 0.0
                training_acc.append(accuracy(net, dataset.trainloader))
                testing_acc.append(accuracy(net, dataset.testloader))
                
            # Done experiment for this configuration.
            # For sake of comparison, plot each one.
            save_plot("%s_%d" % (name, r), training_acc, testing_acc, setting)
            
            # Check if we have a better result
            if len(best_testing_acc) == 0 \
               or max(testing_acc) > max(best_testing_acc):
                best_testing_acc = testing_acc
                best_training_acc = training_acc
                best_setting = setting
            # Reinitialize weights
            net.apply(nets.weights_init)
        
    except KeyboardInterrupt:
        print("Terminating...")
    finally:
        save_plot("%s_best" % (name), best_training_acc,
                  best_testing_acc, best_setting)

if __name__ == "__main__":

    dataset = CIFAR10()

    net = nets.CNN({'input_dim': 33,
                    'convs': 2,
                    'depths': [128, 412],
                    'kernels': [4, 3],
                    'pools': [('Max', 3, 0), ('Avg', 3, 1)],
                    'fcs': 2,
                    'fcdims': [888, 10]}).cuda()
    run_experiment("CNN-C%d_F%d" % (2, 2),
                   net,
                   dataset,
                   [1e-5, 1e-3],  # lr range
                   [0.6, 0.9],    # momentum range
                   rounds=1,
                   num_epochs=2)
