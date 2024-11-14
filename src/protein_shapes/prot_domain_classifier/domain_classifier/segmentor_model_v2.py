import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Loss Function
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)

# Weight Initialization
def initializeWeights(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)
        m.bias.data.fill_(0)

    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Model
class SegmentorModel(nn.Module):
    def __init__(self, nc1, nc2, nc3, arch_classes):
        # nc is the number of input channels
        super(SegmentorModel, self).__init__()
        self.main = nn.Sequential(
            # input is (nc1) x 512 x 512
            # arguments: input channels, output_channels, kernel_size, stride, padding
            nn.Conv2d(nc1, nc2, 4, 2, 1, bias=True), 
            nn.BatchNorm2d(nc2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),

            # state size. (nc2) x 256 x 256
            nn.Conv2d(nc2, nc2 * 2, 4, 2, 1, bias=True), 
            nn.BatchNorm2d(nc2 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),

            # state size. (nc2*2) x 128 x 128
            nn.Conv2d(nc2 * 2, nc2 * 4, 4, 4, 0, bias=True),
            nn.BatchNorm2d(nc2 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),

            # state size. (nc2*4) x 32 x 32
            nn.Conv2d(nc2 * 4, nc2 * 8, 4, 4, 0, bias=True),
            nn.BatchNorm2d(nc2 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),

            # state size. (nc2*8) x 8 x 8
            nn.Conv2d(nc2 * 8, nc2 * 16, 4, 2, 1, bias=True),
            nn.BatchNorm2d(nc2 * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),

            # state size. (nc2*16) x 4 x 4
            nn.Conv2d(nc2 * 16, nc2 * 32, 4, 2, 1, bias=True),
            nn.BatchNorm2d(nc2 * 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),

            # Upsampling 
            # state size. (nc2 * 32) x 2 x 2. 
            nn.Conv2d(nc2 * 32, (nc2 * 32) ** 2, 1, 1, 0, bias=True), # Extend the number of channels to be d^2 = 256^2. -> Expand to 512x512.
            nn.BatchNorm2d((nc2 * 32) **2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            
            # state_size 256^2 x 2 x 2. 
            nn.PixelShuffle(256),

            # state size. 1x512x512 
            nn.Conv2d(1, nc3, (4,1), (2,1), (1,0), bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),

            # state size. 16x256x512
            # nn.Conv2d(nc3, arch_classes, (256,1), (1,1), (0,0))
        )

    # Forward Pass.
    def forward(self, input):
        return self.main(input) # 38 x 1 x 512 output size.

