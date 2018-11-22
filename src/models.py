import torch.nn as nn

# Simple CNN taken from https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/401_CNN.py


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 224, 224)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
                padding=2,
            ),                              # output shape (16, 224, 224)
            nn.ReLU(),                      # activation
            # choose max value in 2x2 area, output shape (16, 112, 112)
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(         # input shape (16, 112, 112)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 112, 112)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 56, 56)
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 56 * 56, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 56 * 56)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x    # return x for visualization
