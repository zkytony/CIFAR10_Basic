# CSE 546 Homework 1
# Problem 2
#
# CIFAR10 dataset

import torch
import torchvision
import torchvision.transforms as transforms


class CIFAR10:
    def __init__(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                      download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=4,
                                                        shuffle=True, num_workers=2)
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                     download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=4,
                                                       shuffle=False, num_workers=2)
        self.classes = ('plane', 'car', 'bird', 'cat',
                         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    def show(self, img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


    def get_training(self, show=False):
        # get some random training images
        dataiter = iter(self.trainloader)
        images, labels = dataiter.next()

        if show:
            # show images
            imshow(torchvision.utils.make_grid(images))
            # print labels
            print(' '.join('%5s' % self.classes[labels[j]] for j in range(4)))
        return images, labels
