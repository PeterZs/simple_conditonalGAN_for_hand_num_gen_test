import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(110, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, 784)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(784)

    def forward(self,x,y): # y: batch_size*10(one-hot vector)
        x=torch.cat((x,y),1)
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.tanh(self.fc3(x)))
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(794, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self,x,y): # y:batch_size*10(one-hot vector)
        x = x.view(-1, 784)
        x=torch.cat((x,y),1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = self.fc3(x)
        return x


def sample_Z(size):
    return torch.from_numpy(np.random.uniform(-1, 1, size=size)).float()


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):  # [i,samples[i]] imax=16
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

generator = Generator()
discriminator = Discriminator()

generator.to(device)
discriminator.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer_gen = optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))
optimizer_dis = optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))

for epoch in range(20):  # loop over the dataset multiple times
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        #print(labels)
        batch_size = inputs.size(0)
        labels.view(batch_size,1)
        #print(labels.size())
        labels_one_hot=torch.FloatTensor(batch_size,10)
        labels_one_hot.zero_()
        #print(labels_one_hot.size())
        for j in range(batch_size):  
             labels_one_hot[j][labels[j]]=1
        #labels_one_hot.scatter_(1,labels,1)
        ones_labels = torch.ones((batch_size, 1)).to(device)
        zeros_labels = torch.zeros((batch_size, 1)).to(device)
        inputs_Z = sample_Z((batch_size, 100)).to(device)
        inputs = inputs.to(device)
        labels_one_hot=labels_one_hot.to(device)

        # dis
        optimizer_dis.zero_grad()
        for p in discriminator.parameters():
            p.requires_grad = True
        for p in generator.parameters():
            p.requires_grad = False

        loss_real = criterion(discriminator(inputs,labels_one_hot), ones_labels)
        loss_fake = criterion(discriminator(generator(inputs_Z,labels_one_hot),labels_one_hot), zeros_labels)
        loss_dis = loss_real + loss_fake
        loss_dis.backward()
        optimizer_dis.step()

        # gen
        optimizer_gen.zero_grad()
        for p in discriminator.parameters():
            p.requires_grad = False
        for p in generator.parameters():
            p.requires_grad = True

        inputs_Z = sample_Z((batch_size, 100)).to(device)
        loss_gen = criterion(discriminator(generator(inputs_Z,labels_one_hot),labels_one_hot), ones_labels)
        loss_gen.backward()
        optimizer_gen.step()

        # print statistics
        if i % 200 == 199:
            print('[%d, %5d] loss: %.3f %.3f' % (epoch + 1, i + 1, loss_dis.item(), loss_gen.item()))

    with torch.no_grad():
        inputs_Z = sample_Z((10, 100)).to(device)
        labels_one_hot = torch.cuda.FloatTensor(10,10)
        labels_one_hot.zero_()
        for j in range(10):  
             labels_one_hot[j][j]=1
    
        result = generator(inputs_Z,labels_one_hot).cpu()
        fig = plot(result.numpy())
        plt.savefig('results/result_%d.png' % epoch, bbox_inches='tight')
        plt.close(fig)

print('Finished Training')

with torch.no_grad():
    generator.eval()
    inputs_Z = sample_Z((10, 100)).to(device)
    
    labels_one_hot = torch.cuda.FloatTensor(10,10)
    labels_one_hot.zero_()
    for j in range(10):  
             labels_one_hot[j][j]=1
    
    result = generator(inputs_Z,labels_one_hot).cpu()
    fig = plot(result.numpy())
    plt.savefig('results/result_final.png', bbox_inches='tight')
    plt.close(fig)
