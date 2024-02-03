import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import os

model_path = os.getcwd().replace(os.sep, '/')


class DownBlock(nn.Module):
    def __init__(self, inputs, outputs, kernel=4, stride=2, padding=1, norm=True, z_pad=False, is_last=False):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(inputs, outputs, kernel, stride, padding)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.instance_norm = nn.InstanceNorm2d(outputs)
        self.z_padding = nn.ZeroPad2d((1 ,0 ,1 ,0))
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        self.norm = norm
        self.z_pad = z_pad
        self.is_last = is_last

    def forward(self, x):
        down_block = nn.Sequential(self.conv)
        if self.is_last:
            down_block.add_module('flatten', self.flatten)
            down_block.add_module('sigmoid', self.sigmoid)
        else:
            if self.norm:
                down_block.add_module('norm', self.instance_norm)
            down_block.add_module('leaky_relu', self.leaky_relu)
            if self.z_pad:
                down_block.add_module('z_padding', self.z_padding)
        return down_block(x)


class UpBlock(nn.Module):
    def __init__(self, inputs, outputs, kernel=4, stride=2, padding=1, norm=True, is_drop=False, is_last=False):
        super(UpBlock, self).__init__()
        self.conv_trans = nn.ConvTranspose2d(inputs, outputs, kernel, stride, padding)
        self.instance_norm = nn.InstanceNorm2d(outputs)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.tanh = nn.Tanh()
        self.norm = norm
        self.is_drop = is_drop
        self.is_last = is_last

    def forward(self, x):
        up_block = nn.Sequential(self.conv_trans)
        if self.is_last:
            up_block.add_module('tanh', self.tanh)
        else:
            if self.norm:
                up_block.add_module('norm', self.instance_norm)
            if self.is_drop:
                up_block.add_module('dropout', self.dropout)
            up_block.add_module('relu', self.relu)
        return up_block(x)


class ResidualBlock(nn.Module):
    def __init__(self, inputs, kernel=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.refl = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(inputs, inputs, kernel, stride, padding)
        self.norm = nn.InstanceNorm2d(inputs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual_block = nn.Sequential(self.refl,
                                       self.conv,
                                       self.norm,
                                       self.relu,
                                       self.refl,
                                       self.conv,
                                       self.norm)
        return x + residual_block(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.down_blocks = nn.ModuleList([DownBlock(3, 32, norm=False),
                                          DownBlock(32, 64),
                                          DownBlock(64, 128),
                                          DownBlock(128, 256)])

        self.residual_block = nn.ModuleList([ResidualBlock(256),
                                             ResidualBlock(256),
                                             ResidualBlock(256),
                                             ResidualBlock(256),
                                             ResidualBlock(256),
                                             ResidualBlock(256),
                                             ResidualBlock(256)])

        self.up_blocks = nn.ModuleList([UpBlock(256, 128),
            UpBlock(256, 64),
            UpBlock(128, 32),
            UpBlock(64, 3)])

    def forward(self, x):
        skip_connections = []
        for block in self.down_blocks:
            x = block(x)
            skip_connections.append(x)
        skip_connections = reversed(skip_connections[:-1])

        for block, skip in zip(self.up_blocks[:-1], skip_connections):
            x = block(x)
            x = torch.cat([x, skip], axis=1)
        x = self.up_blocks[-1](x)
        return x


model_gan = Generator()
model_gan.load_state_dict(torch.load(f'{model_path}/models/style_generator.pth',
                                     map_location=torch.device('cpu')))
model_gan.eval()

