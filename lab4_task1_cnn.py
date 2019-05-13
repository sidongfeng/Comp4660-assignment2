from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Fasion MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0, metavar='M',
                    help='SGD momentum (default: 0.0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        """
        define a cnn with two conv layers (experiment with the number of channels and kernal sizes) 
        followed by doing dropout, then two fully connected layers.
        """
        self.conv1 = nn.Conv2d(in_channels = 1,out_channels = 16,kernel_size = 5,stride=1,padding=2)
        self.relu1 = nn.ReLU()
        self.conv1_bn = nn.BatchNorm2d(16)
        self.MaxPool1 = nn.MaxPool2d(kernel_size=2)
        
        #Conv 2
        self.conv2 = nn.Conv2d(in_channels= 16, out_channels= 32, kernel_size=5,stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.conv2_bn = nn.BatchNorm2d(32)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size=5,stride=1, padding=2)
        self.relu3 = nn.ReLU()
        self.conv3_bn = nn.BatchNorm2d(64)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(576, 10)

    def forward(self, x):
        """
	define the forward pass of the cnn with a relu activation function for each hidden layer
        and dropout after the first fully connected layer.
        """
        out = self.conv1(x)
        out= self.relu1(out)
        out = self.conv1_bn(out)
        out= self.MaxPool1(out)
        
        out= self.conv2(out)
        out = self.relu2(out)
        out = self.conv2_bn(out)
        out = self.MaxPool2(out)
        
        out= self.conv3(out)
        out = self.relu3(out)
        out = self.conv3_bn(out)
        out = self.MaxPool3(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        
        return out

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
criterion = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    correct = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader)
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    print(100. * correct / len(train_loader.dataset))
    return train_loss, 100. * correct / len(train_loader.dataset)

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, 100. * correct / len(test_loader.dataset)

losses = []
test_losses = []
accuracys = []
test_accuracys = []
for epoch in range(1, args.epochs + 1):
    loss, accuracy = train(epoch)
    losses.append(loss)
    accuracys.append(accuracy)

    test_loss, test_accuracy = test()
    test_losses.append(test_loss)
    test_accuracys.append(test_accuracy)

plt.figure()
plt.plot(losses)
plt.plot(accuracys)
plt.plot(test_losses)
plt.plot(test_accuracys)
plt.gca().legend(('loss','acc','test_loss','test_acc'))
plt.show()
