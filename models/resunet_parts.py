import torch
from torch import nn



###########################################################
###                  ResUNet original                   ###
###########################################################

class ResidualBlock(nn.Module):
    """
    Residual block with options for downsampling (stride=2) or first convolution block (first=True)
    """
    def __init__(self, in_channels, out_channels, stride=1, first=False):
        """
        Constructor
        :param in_channels: Number of input channels of the input.
        :param out_channels: Number of input channels of the input.
        :param stride: Stride to apply for first convolutional layer. Stride=1 -> no change in resolution. Stride=2 -> downsamling. Default: 1
        :param first: Indicates first convolution block in encoding process. Default: False.
        """
        super(ResidualBlock, self).__init__()
        self.first = first
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.conv_block1 = ConvBlock(in_channels, out_channels, stride=stride)
        self.conv_block2 = ConvBlock(out_channels, out_channels, stride=1)
        self.shortcut_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride),
            # nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        if self.first:
            res = self.conv(x)
            res = self.conv_block2(res)
        else:
            res = self.conv_block1(x)
            res = self.conv_block2(res)
        shortcut = self.shortcut_block(x)
        x = torch.add(shortcut, res)
        return x 


class ConvBlock(nn.Module):
    """
    Convolution block consisting of batch normalisation layer, ReLU lyer and concolution layer
    """
    def __init__(self, in_channels, out_channels, stride=1):
        '''
        Constructor
        :param in_channels: Number of input channels of the input.
        :param out_channels: Number of input channels of the input.
        :param stride: Stride to apply. Default 1.
        '''
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        # self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)

    def forward(self, x):
        # x = self.bn(x)
        x = self.act(x)
        x = self.conv(x)
        return x


class UpsampleConcatBlock(nn.Module):
    """
    Block to upsample layer and concatenate it with skip connection.
    """
    def __init__(self, n_channels, stride=2):
        """
        Constructor
        :param n_channels: Number of channels for input and output. Number of channels for skip connection must be half of n_channels.
        :param stride: Stride to apply for upsampling layer. Default: 2.
        """
        super(UpsampleConcatBlock, self).__init__()
        self.up = nn.ConvTranspose2d(n_channels, n_channels // 2, kernel_size=2, stride=stride)
        
    def forward(self, x, xskip):
        x = self.up(x) # number of channels is halved, size doubled
        x = torch.cat((x, xskip), dim=1) # number of channels is number of input channels again
        return x

class OutConv(nn.Module):
    """
    Block for last convolution
    """
    def __init__(self, in_channels, out_channels):
        """
        Constructor
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        return self.conv(x)
    

###########################################################
###          ResUNet with Batch Normalization           ###
###########################################################


class ResidualBlock_bn(nn.Module):
    """
    Residual block with options for downsampling (stride=2) or first convolution block (first=True)
    """
    def __init__(self, in_channels, out_channels, stride=1, first=False):
        """
        Constructor
        :param in_channels: Number of input channels of the input.
        :param out_channels: Number of input channels of the input.
        :param stride: Stride to apply for first convolutional layer. Stride=1 -> no change in resolution. Stride=2 -> downsamling. Default: 1
        :param first: Indicates first convolution block in encoding process. Default: False.
        """
        super(ResidualBlock_bn, self).__init__()
        self.first = first
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.conv_block1 = ConvBlock_bn(in_channels, out_channels, stride=stride)
        self.conv_block2 = ConvBlock_bn(out_channels, out_channels, stride=1)
        self.shortcut_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride),
            # nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        if self.first:
            res = self.conv(x)
            res = self.conv_block2(res)
        else:
            res = self.conv_block1(x)
            res = self.conv_block2(res)
        shortcut = self.shortcut_block(x)
        x = torch.add(shortcut, res)
        return x 


class ConvBlock_bn(nn.Module):
    """
    Convolution block consisting of batch normalisation layer, ReLU lyer and concolution layer
    """
    def __init__(self, in_channels, out_channels, stride=1):
        '''
        Constructor
        :param in_channels: Number of input channels of the input.
        :param out_channels: Number of input channels of the input.
        :param stride: Stride to apply. Default 1.
        '''
        super(ConvBlock_bn, self).__init__()
        self.in_channels = in_channels
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)

    def forward(self, x):
        # x = self.bn(x)
        x = self.act(x)
        x = self.conv(x)
        return x
    
    

###########################################################
###             ResUNet with Kernel Size 5              ###
###########################################################

class ResidualBlock_KS5(nn.Module):
    """
    Residual block with options for downsampling (stride=2) or first convolution block (first=True)
    """
    def __init__(self, in_channels, out_channels, stride=1, first=False):
        """
        Constructor
        :param in_channels: Number of input channels of the input.
        :param out_channels: Number of input channels of the input.
        :param stride: Stride to apply for first convolutional layer. Stride=1 -> no change in resolution. Stride=2 -> downsamling. Default: 1
        :param first: Indicates first convolution block in encoding process. Default: False.
        """
        super(ResidualBlock_KS5, self).__init__()
        self.first = first
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, stride=1)
        self.conv_block1 = ConvBlock_KS5(in_channels, out_channels, stride=stride)
        self.conv_block2 = ConvBlock_KS5(out_channels, out_channels, stride=1)
        self.shortcut_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride),
            # nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        if self.first:
            res = self.conv(x)
            res = self.conv_block2(res)
        else:
            res = self.conv_block1(x)
            res = self.conv_block2(res)
        shortcut = self.shortcut_block(x)
        x = torch.add(shortcut, res)
        return x 


class ConvBlock_KS5(nn.Module):
    """
    Convolution block consisting of batch normalisation layer, ReLU lyer and concolution layer
    """
    def __init__(self, in_channels, out_channels, stride=1):
        '''
        Constructor
        :param in_channels: Number of input channels of the input.
        :param out_channels: Number of input channels of the input.
        :param stride: Stride to apply. Default 1.
        '''
        super(ConvBlock_KS5, self).__init__()
        self.in_channels = in_channels
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.LeakyReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, stride=stride)

    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        x = self.conv(x)
        return x


###########################################################
###             ResUNet with Kernel Size 3              ###
###########################################################

class ResidualBlock_KS3(nn.Module):
    """
    Residual block with options for downsampling (stride=2) or first convolution block (first=True)
    """
    def __init__(self, in_channels, out_channels, stride=1, first=False):
        """
        Constructor
        :param in_channels: Number of input channels of the input.
        :param out_channels: Number of input channels of the input.
        :param stride: Stride to apply for first convolutional layer. Stride=1 -> no change in resolution. Stride=2 -> downsamling. Default: 1
        :param first: Indicates first convolution block in encoding process. Default: False.
        """
        super(ResidualBlock_KS3, self).__init__()
        self.first = first
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.conv_block1 = ConvBlock_KS3(in_channels, out_channels, stride=stride)
        self.conv_block2 = ConvBlock_KS3(out_channels, out_channels, stride=1)
        self.shortcut_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride),
            # nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        if self.first:
            res = self.conv(x)
            res = self.conv_block2(res)
        else:
            res = self.conv_block1(x)
            res = self.conv_block2(res)
        shortcut = self.shortcut_block(x)
        x = torch.add(shortcut, res)
        return x 


class ConvBlock_KS3(nn.Module):
    """
    Convolution block consisting of batch normalisation layer, ReLU lyer and concolution layer
    """
    def __init__(self, in_channels, out_channels, stride=1):
        '''
        Constructor
        :param in_channels: Number of input channels of the input.
        :param out_channels: Number of input channels of the input.
        :param stride: Stride to apply. Default 1.
        '''
        super(ConvBlock_KS3, self).__init__()
        self.in_channels = in_channels
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.LeakyReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)

    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        x = self.conv(x)
        return x