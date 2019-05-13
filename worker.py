from multiprocessing import Queue, Process
import cv2
import numpy as np
import os
import net_builder
import torch_models 
from torchvision import datasets, transforms
import torch.utils as utils

# helper class for scheduling workers
class Scheduler:
    def __init__(self, use_cuda):
        self._queue = []
        self._results = []
        self.use_cuda = use_cuda
        batch_size = 64
        # setup our dataloaders
        self.train_loader = utils.data.DataLoader(
            datasets.FashionMNIST('./data', train=True, download=False,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])),
            batch_size=batch_size, shuffle=True)

        self.test_loader = utils.data.DataLoader(
            datasets.FashionMNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])),
            batch_size=batch_size, shuffle=True)

    def start(self, xlist):

        # put all of models into queue
        for model_info in xlist:
            xnet  = torch_models.CustomModel(model_info, self.use_cuda)
            ret = xnet.train(self.train_loader)
            score = -1
            if ret ==1:
                score = xnet.test(self.test_loader)
                print('score:{}'.format(score))
            self._results.append(score)
            self._queue.append(model_info)
        
        print("All workers are done")
        return self._queue, self._results


if __name__ == '__main__':
    workerpool = Scheduler(False )
    xlist = list()
    for i in range(5):
        xlist.append(net_builder.randomize_network())
    print(xlist)
    population, returns = workerpool.start(xlist)
    print(population)
    print(returns)
    # xlist = list()
    # for i in range(5):
    #     xlist.append(net_builder.randomize_network())
    # que = Queue()
    # r = Queue()
    
    # for model_info in xlist:
    #     que.put(model_info)

    # cw = CustomWorker(1, que, r, False)
    # cw.run()

