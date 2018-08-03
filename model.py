import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    
    def __init__(self, ni, no, ks=3, stride=1, reflection_pad=1, pad=0, add_relu=True, add_ref=True, add_norm=True):
        super(ConvBlock, self).__init__()
        self.pad = nn.ReflectionPad2d(reflection_pad)
        self.conv = nn.Conv2d(ni, no, ks, stride=stride, padding=pad)
        self.norm = nn.InstanceNorm2d(no)
        self.relu = nn.ReLU(inplace=True)
        self.add_relu = add_relu
        self.add_ref = add_ref
        self.add_norm = add_norm

    def forward(self, x):
        if self.add_ref:
            activations = self.norm(self.conv(self.pad(x))) if self.add_norm else self.conv(self.pad(x))
        else:
            activations = self.norm(self.conv(x)) if self.add_norm else self.conv(x)
        return self.relu(activations) if self.add_relu else activations


class Upsample(nn.Module):
    
    def __init__(self, ni, no, ks, stride=2, padding=1, out_pad=1):
        super(Upsample, self).__init__()
        self.conv = nn.ConvTranspose2d(ni, no, ks, stride=stride, padding=padding, output_padding=out_pad)
        self.norm = nn.InstanceNorm2d(no)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    
    def __init__(self, ni):
        super(ResBlock, self).__init__()

        conv = [ConvBlock(ni, ni),
                ConvBlock(ni, ni, add_relu=False)]

        self.res_fwd = nn.Sequential(*conv)

    def forward(self, x):
        return x + self.res_fwd(x)


class Generator(nn.Module):
    
    def __init__(self, ni, no, num_layers=7):
        super(Generator, self).__init__()

        # Initial Convolution Block
        model = [ConvBlock(ni, 64, ks=7, reflection_pad=3, add_relu=True)]
        # Downsampling
        ni = 64
        no_ = ni * 2
        for _ in range(2):
            model += [ConvBlock(ni, no_, ks=3, stride=2, pad=1, add_ref=False)]
            ni = no_
            no_ = ni * 2
        # Adding ResBlocks
        for _ in range(num_layers):
            model += [ResBlock(ni)]
        # Upsampling
        no_ = ni // 2
        for _ in range(2):
            model += [Upsample(ni, no_, 3)]
            ni = no_
            no_ = ni // 2
        # Output Layer
        model += [ConvBlock(ni, no, ks=7, reflection_pad=3, add_norm=False),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):

    def __init__(self, ni):
        super(Discriminator, self).__init__()
        
        model = []
        no_ = 64
        # Conv Pyramid
        for _ in range(3):
            model += [ConvBlock(ni, no_, 4, stride=2, pad=1, add_ref=False)]
            ni = no_
            no_ *= 2
        model += [ConvBlock(ni, no_, ks=4, stride=1, pad=1, add_ref=False),
                  ConvBlock(no_, 1, 4, pad=1, add_norm=False, add_ref=False, add_relu=False)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


