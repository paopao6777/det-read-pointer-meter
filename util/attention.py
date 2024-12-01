import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchinfo import summary

class MLCA(nn.Module):
    def __init__(self, in_size, local_size=5, gamma = 2, b = 1,local_weight=0.5):
        super(MLCA, self).__init__()

        # ECA 计算方法
        self.local_size=local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)   # eca  gamma=2
        k = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        self.local_weight=local_weight

        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool=nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        local_arv=self.local_arv_pool(x)
        global_arv=self.global_arv_pool(local_arv)

        b,c,m,n = x.shape
        b_local, c_local, m_local, n_local = local_arv.shape

        # (b,c,local_size,local_size) -> (b,c,local_size*local_size)-> (b,local_size*local_size,c)-> (b,1,local_size*local_size*c)
        temp_local= local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_global = global_arv.view(b, c, -1).transpose(-1, -2)

        y_local = self.conv_local(temp_local)
        y_global = self.conv(temp_global)


        # (b,c,local_size,local_size) <- (b,c,local_size*local_size)<-(b,local_size*local_size,c) <- (b,1,local_size*local_size*c)
        y_local_transpose=y_local.reshape(b, self.local_size * self.local_size,c).transpose(-1,-2).view(b,c, self.local_size , self.local_size)
        y_global_transpose = y_global.view(b, -1).transpose(-1, -2).unsqueeze(-1)

        # 反池化
        att_local = y_local_transpose.sigmoid()
        att_global = F.adaptive_avg_pool2d(y_global_transpose.sigmoid(),[self.local_size, self.local_size])
        att_all = F.adaptive_avg_pool2d(att_global*(1-self.local_weight)+(att_local*self.local_weight), [m, n])

        x = x * att_all
        return x

class CDCA_ChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(CDCA_ChannelAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, padding=0, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.conv1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.conv2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.conv1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.conv2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x

class CDCA(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()

        self.ca = CDCA_ChannelAttention(channel, channel // reduction)
        self.dconv3_3 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel)
        self.dconv1_3 = nn.Conv2d(channel, channel, kernel_size=(1, 3), padding=(0, 1), groups=channel)
        self.dconv3_1 = nn.Conv2d(channel, channel, kernel_size=(3, 1), padding=(1, 0), groups=channel)
        self.conv = nn.Conv2d(channel, channel, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        #   Global Perceptron
        inputs = self.conv(inputs)
        inputs = self.act(inputs)

        inputs = self.ca(inputs)

        x_0 = self.dconv3_3(inputs)
        x = self.dconv1_3(x_0)
        x = self.dconv3_1(x)

        x = x + x_0
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        return out
