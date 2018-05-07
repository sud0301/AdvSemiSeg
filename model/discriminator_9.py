#import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
#from torch.nn import Parameter
#import numpy as np
#import torch.nn.init as nn_init


class GaussianNoise(nn.Module):
    def __init__(self, sigma):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, input):
        if self.training:
            noise = Variable(input.data.new(input.size()).normal_(std=self.sigma))
            return input + noise
        else:
            return input

class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func

    def forward(self, input):
        return self.func(input)


class FCDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf = 64):
        super(FCDiscriminator, self).__init__()


        self.core_net = nn.Sequential(

            nn.Sequential(GaussianNoise(0.05), nn.Dropout2d(0.15)),

            nn.Conv2d(num_classes, ndf, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(       ndf,  ndf, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(       ndf,  ndf, 3, 2, 1), nn.LeakyReLU(0.2),

            nn.Dropout2d(0.5),

            nn.Conv2d(       ndf,  ndf, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(       ndf,  ndf*2, 3, 2, 1), nn.LeakyReLU(0.2),

            nn.Dropout2d(0.5),

            nn.Conv2d(       ndf*2,  ndf*2, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(       ndf*2,  ndf*4, 3, 2, 1), nn.LeakyReLU(0.2),

            nn.Dropout2d(0.5),

            nn.Conv2d(       ndf*4,  ndf*4, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(       ndf*4,  ndf*8, 3, 2, 1), nn.LeakyReLU(0.2),

            nn.Dropout2d(0.5),

            nn.Conv2d(       ndf*8,  ndf*8, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(       ndf*8,  ndf*8, 3, 1, 1), nn.LeakyReLU(0.2),

            Expression(lambda tensor: tensor.mean(3).mean(2).squeeze()),
        )

        #self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        #self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        #self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        #self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        #self.conv5 = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)
        #self.classifier = nn.Sequential(
        self.fc = nn.Linear(512, 1)
        self.out = nn.Sigmoid()
        #)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop = nn.Dropout2d(0.5)
        #self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        '''
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        '''
        y = self.core_net(x)
        #y = x.mean(3).mean(2).squeeze()
        #y = x.view(-1, 512)
        z = self.fc(y)
        z = self.out(z)
        return z, y
