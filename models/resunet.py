from torch import nn
from models.resunet_parts import *



#################################################################
###          ResUNet with Kernel Size 5 and 3 Layers          ###
#################################################################
# ex ResUNet_2.0

class ResUNet_KS5_L3(nn.Module):
    def __init__(self, n_channels, n_classes):
        """
        constructor
        :param n_channels: number of channels.
        :param n_classes: number of classes.
        """

        super(ResUNet_KS5_L3, self).__init__()
        
        self.constr_command = f"ResUNet(n_channels={n_channels}, n_classes={n_classes})"
        self.n_channels = n_channels
        self.n_classes = n_classes

        # encoder
        self.inc = ResidualBlock_KS5(n_channels, 32, stride=1, first=True)

        self.encoder1 = ResidualBlock_KS5(32, 64, stride=2)
        self.encoder2 = ResidualBlock_KS5(64, 128, stride=2)
        self.encoder3 = ResidualBlock_KS5(128, 256, stride=2)

        # bridge
        self.bridge1 = ConvBlock_KS5(256, 256, stride=1)
        self.bridge2 = ConvBlock_KS5(256, 256, stride=1)

        # decoder
        self.up_con1 = UpsampleConcatBlock(256, stride=2)
        self.decoder1 = ResidualBlock_KS5(256, 128, stride=1)

        self.up_con2 = UpsampleConcatBlock(128, stride=2)
        self.decoder2 = ResidualBlock_KS5(128, 64, stride=1)

        self.up_con3 = UpsampleConcatBlock(64, stride=2)
        self.decoder3 = ResidualBlock_KS5(64, 32, stride=1)

        self.outc = OutConv(32, n_classes)

        
    def forward(self, x):
        # Encoder
        e0 = self.inc(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        # Bridge
        b1 = self.bridge1(e3)
        b2 = self.bridge2(b1)

        # Decoder
        u1 = self.up_con1(b2, e2) 
        d1 = self.decoder1(u1)
        
        u2 = self.up_con2(d1, e1)
        d2 = self.decoder2(u2)
        
        u3 = self.up_con3(d2, e0)
        d3 = self.decoder3(u3)

        x = self.outc(d3)

        return x



#################################################################
###     ResUNet with Kernel Size 5, 3 Layers, channels 64     ###
#################################################################

class ResUNet_KS5_L3_CH64(nn.Module):
    def __init__(self, n_channels, n_classes):
        """
        constructor
        :param n_channels: number of channels.
        :param n_classes: number of classes.
        """

        super(ResUNet_KS5_L3_CH64, self).__init__()
        
        self.constr_command = f"ResUNet(n_channels={n_channels}, n_classes={n_classes})"
        self.n_channels = n_channels
        self.n_classes = n_classes

        # encoder
        self.inc = ResidualBlock_KS5(n_channels, 64, stride=1, first=True)

        self.encoder1 = ResidualBlock_KS5(64, 128, stride=2)
        self.encoder2 = ResidualBlock_KS5(128, 256, stride=2)
        self.encoder3 = ResidualBlock_KS5(256, 512, stride=2)

        # bridge
        self.bridge1 = ConvBlock_KS5(512, 512, stride=1)
        self.bridge2 = ConvBlock_KS5(512, 512, stride=1)

        # decoder
        self.up_con1 = UpsampleConcatBlock(512, stride=2)
        self.decoder1 = ResidualBlock_KS5(512, 256, stride=1)

        self.up_con2 = UpsampleConcatBlock(256, stride=2)
        self.decoder2 = ResidualBlock_KS5(256, 128, stride=1)

        self.up_con3 = UpsampleConcatBlock(128, stride=2)
        self.decoder3 = ResidualBlock_KS5(128, 64, stride=1)

        self.outc = OutConv(64, n_classes)

        
    def forward(self, x):
        # Encoder
        e0 = self.inc(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        # Bridge
        b1 = self.bridge1(e3)
        b2 = self.bridge2(b1)

        # Decoder
        u1 = self.up_con1(b2, e2) 
        d1 = self.decoder1(u1)
        
        u2 = self.up_con2(d1, e1)
        d2 = self.decoder2(u2)
        
        u3 = self.up_con3(d2, e0)
        d3 = self.decoder3(u3)

        x = self.outc(d3)

        return x



#################################################################
###          ResUNet with Kernel Size 5 and 4 Layers          ###
#################################################################

class ResUNet_KS5_L4(nn.Module):
    def __init__(self, n_channels, n_classes):
        """
        constructor
        :param n_channels: number of channels.
        :param n_classes: number of classes.
        """

        super(ResUNet_KS5_L4, self).__init__()
        
        self.constr_command = f"ResUNet(n_channels={n_channels}, n_classes={n_classes})"
        self.n_channels = n_channels
        self.n_classes = n_classes

        # encoder
        self.inc = ResidualBlock_KS5(n_channels, 32, stride=1, first=True)

        self.encoder1 = ResidualBlock_KS5(32, 64, stride=2)
        self.encoder2 = ResidualBlock_KS5(64, 128, stride=2)
        self.encoder3 = ResidualBlock_KS5(128, 256, stride=2)
        self.encoder4 = ResidualBlock_KS5(256, 512, stride=2)

        # bridge
        self.bridge1 = ConvBlock_KS5(512, 512, stride=1)
        self.bridge2 = ConvBlock_KS5(512, 512, stride=1)

        # decoder
        self.up_con1 = UpsampleConcatBlock(512, stride=2)
        self.decoder1 = ResidualBlock_KS5(512, 256, stride=1)

        self.up_con2 = UpsampleConcatBlock(256, stride=2)
        self.decoder2 = ResidualBlock_KS5(256, 128, stride=1)

        self.up_con3 = UpsampleConcatBlock(128, stride=2)
        self.decoder3 = ResidualBlock_KS5(128, 64, stride=1)

        self.up_con4 = UpsampleConcatBlock(64, stride=2)
        self.decoder4 = ResidualBlock_KS5(64, 32, stride=1)

        self.outc = OutConv(32, n_classes)

        
    def forward(self, x):
        # Encoder
        e0 = self.inc(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Bridge
        b1 = self.bridge1(e4)
        b2 = self.bridge2(b1)

        # Decoder
        u1 = self.up_con1(b2, e3) 
        d1 = self.decoder1(u1)
        
        u2 = self.up_con2(d1, e2)
        d2 = self.decoder2(u2)
        
        u3 = self.up_con3(d2, e1)
        d3 = self.decoder3(u3)

        u4 = self.up_con4(d3, e0)
        d4 = self.decoder4(u4)

        x = self.outc(d4)

        return x
    


#################################################################
###          ResUNet with Kernel Size 3 and 3 Layers          ###
#################################################################

class ResUNet_KS3_L3(nn.Module):
    def __init__(self, n_channels, n_classes):
        """
        constructor
        :param n_channels: number of channels.
        :param n_classes: number of classes.
        """

        super(ResUNet_KS3_L3, self).__init__()
        
        self.constr_command = f"ResUNet(n_channels={n_channels}, n_classes={n_classes})"
        self.n_channels = n_channels
        self.n_classes = n_classes

        # encoder
        self.inc = ResidualBlock_KS3(n_channels, 32, stride=1, first=True)

        self.encoder1 = ResidualBlock_KS3(32, 64, stride=2)
        self.encoder2 = ResidualBlock_KS3(64, 128, stride=2)
        self.encoder3 = ResidualBlock_KS3(128, 256, stride=2)

        # bridge
        self.bridge1 = ConvBlock_KS3(256, 256, stride=1)
        self.bridge2 = ConvBlock_KS3(256, 256, stride=1)

        # decoder
        self.up_con1 = UpsampleConcatBlock(256, stride=2)
        self.decoder1 = ResidualBlock_KS3(256, 128, stride=1)

        self.up_con2 = UpsampleConcatBlock(128, stride=2)
        self.decoder2 = ResidualBlock_KS3(128, 64, stride=1)

        self.up_con3 = UpsampleConcatBlock(64, stride=2)
        self.decoder3 = ResidualBlock_KS3(64, 32, stride=1)

        self.outc = OutConv(32, n_classes)

        
    def forward(self, x):
        # Encoder
        e0 = self.inc(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        # Bridge
        b1 = self.bridge1(e3)
        b2 = self.bridge2(b1)

        # Decoder
        u1 = self.up_con1(b2, e2) 
        d1 = self.decoder1(u1)
        
        u2 = self.up_con2(d1, e1)
        d2 = self.decoder2(u2)
        
        u3 = self.up_con3(d2, e0)
        d3 = self.decoder3(u3)

        x = self.outc(d3)

        return x
    


#################################################################
###          ResUNet with Kernel Size 3 and 4 Layers          ###
#################################################################

class ResUNet_KS3_L4(nn.Module):
    def __init__(self, n_channels, n_classes):
        """
        constructor
        :param n_channels: number of channels.
        :param n_classes: number of classes.
        """

        super(ResUNet_KS3_L4, self).__init__()
        
        self.constr_command = f"ResUNet(n_channels={n_channels}, n_classes={n_classes})"
        self.n_channels = n_channels
        self.n_classes = n_classes

        # encoder
        self.inc = ResidualBlock_KS3(n_channels, 32, stride=1, first=True)

        self.encoder1 = ResidualBlock_KS3(32, 64, stride=2)
        self.encoder2 = ResidualBlock_KS3(64, 128, stride=2)
        self.encoder3 = ResidualBlock_KS3(128, 256, stride=2)
        self.encoder4 = ResidualBlock_KS3(256, 512, stride=2)

        # bridge
        self.bridge1 = ConvBlock_KS3(512, 512, stride=1)
        self.bridge2 = ConvBlock_KS3(512, 512, stride=1)

        # decoder
        self.up_con1 = UpsampleConcatBlock(512, stride=2)
        self.decoder1 = ResidualBlock_KS3(512, 256, stride=1)

        self.up_con2 = UpsampleConcatBlock(256, stride=2)
        self.decoder2 = ResidualBlock_KS3(256, 128, stride=1)

        self.up_con3 = UpsampleConcatBlock(128, stride=2)
        self.decoder3 = ResidualBlock_KS3(128, 64, stride=1)

        self.up_con4 = UpsampleConcatBlock(64, stride=2)
        self.decoder4 = ResidualBlock_KS3(64, 32, stride=1)

        self.outc = OutConv(32, n_classes)

        
    def forward(self, x):
        # Encoder
        e0 = self.inc(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Bridge
        b1 = self.bridge1(e4)
        b2 = self.bridge2(b1)

        # Decoder
        u1 = self.up_con1(b2, e3) 
        d1 = self.decoder1(u1)
        
        u2 = self.up_con2(d1, e2)
        d2 = self.decoder2(u2)
        
        u3 = self.up_con3(d2, e1)
        d3 = self.decoder3(u3)

        u4 = self.up_con4(d3, e0)
        d4 = self.decoder4(u4)

        x = self.outc(d4)

        return x