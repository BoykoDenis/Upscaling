import torch
import torch.nn as nn

class UpscaleUUD(nn.Module):

    def __init__(self, input_chanels, n_features):
        super().__init__()

        self.n_features = n_features
        self.t_conv_1 = nn.ConvTranspose2d(input_chanels, n_features, kernel_size = 11, stride = 2, padding = 5, output_padding = 1)
        #self.lin1 = nn.Linear(input_chanels*120*120, n_features*240*240)
        self.conv_2 =  nn.Conv2d(n_features, n_features*10, kernel_size = 11, padding = 5)
        self.conv_3 = nn.Conv2d(n_features*10, n_features*20, kernel_size = 11, padding = 5)
        self.conv_4 = nn .Conv2d(n_features*20, n_features*10, kernel_size = 11, padding = 5)
        self.conv_5 = nn .Conv2d(n_features*10, input_chanels, kernel_size = 11, padding = 5)


    def forward(self, x):

        x = self.t_conv_1(x)
        #print(x.shape)
        #x = self.lin1(x)
        #x = x.reshape((-1, self.n_features, 240, 240))
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        #print(x.shape)
        x = self.conv_5(x)

        return x