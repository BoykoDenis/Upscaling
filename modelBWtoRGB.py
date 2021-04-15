import torch
import torch.nn as nn

class UpscaleBWtoRGB(nn.Module):

    def __init__(self, output_chanels, n_features):
        super().__init__()

        self.t_conv_1 = nn.ConvTranspose2d(1, n_features, kernel_size = 5, stride = 2, padding = 2, output_padding = 1)
        self.conv_2 = nn.Conv2d(n_features, n_features*20, kernel_size = 5, padding = 2)
        self.conv_3 = nn.Conv2d(n_features*20, n_features*40, kernel_size = 5, padding = 2)
        self.conv_4 = nn.Conv2d(n_features*40, output_chanels, kernel_size = 5, padding = 2)


    def forward(self, x):

        x = self.t_conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)

        return x


