import torch.nn as nn


class MnistNetV1(nn.Module):
    def __init__(self, debug):
        super(MnistNetV1, self).__init__()
        self.debug = debug
        self.firstTime = True

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.out = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        if (self.debug and self.firstTime):
            print('Shape before conv1 {}'.format(x.shape))

        x = self.conv1(x)
        if (self.debug and self.firstTime):
            print('Shape before conv2 {}'.format(x.shape))

        x = self.conv2(x)
        if (self.debug and self.firstTime):
            print('Shape before view {}'.format(x.shape))

        x = x.view(x.size(0), -1)
        if (self.debug and self.firstTime):
            print('Shape before linear {}'.format(x.shape))

        output = self.out(x)
        if (self.debug and self.firstTime):
            print('Shape before return {}'.format(x.shape))

        self.firstTime = False
        return output
