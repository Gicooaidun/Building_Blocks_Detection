import torch.nn as nn


class dice_loss(nn.Module):
    # https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss
    def __init__(self, weight=None, size_average=True):
        super(dice_loss, self).__init__()


    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        sigmoid = nn.Sigmoid()
        inputs = sigmoid(inputs)     
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice