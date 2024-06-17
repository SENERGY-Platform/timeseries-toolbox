from torch import nn 
import torch.nn.functional as F
import math 
import torch

class Encoder(nn.Module):
    def __init__(self, latent_dims, window_length, kernel_size):
        super().__init__()
        self.window_length = window_length
        self.stride = 3
        self.conv1 = nn.Conv1d(1, 16, kernel_size, stride=self.stride)
        con1_output_size = int(math.floor((window_length + 2*self.conv1.padding[0] - kernel_size)/self.stride + 1))
        self.conv2 = nn.Conv1d(16, 32, kernel_size, stride=self.stride)
        conv2_output_size = int((con1_output_size-kernel_size)/self.stride + 1)
        self.fc1 = nn.Linear(conv2_output_size, 672)
        self.fc2 = nn.Linear(672, latent_dims)
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x):
        x = torch.unsqueeze(x,1)
        x = F.relu(self.dropout(self.conv1(x)))
        x = F.relu(self.dropout(self.conv2(x)))        
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dims, window_length, kernel_size):
        super().__init__()
        self.stride = 3
        self.padding = 0
        self.window_length = window_length
        hidden_size = 672
        self.fc1 = nn.Linear(latent_dims, hidden_size)
        
        self.convt1 = nn.ConvTranspose1d(32, 16, kernel_size=kernel_size, stride=self.stride)
        conv1_output_size = self.calc_output_shape_conv_trans(hidden_size, self.stride, self.convt1.padding[0], self.convt1.dilation[0], kernel_size, self.convt1.output_padding[0])
        
        self.convt2 = nn.ConvTranspose1d(16, 1, kernel_size=kernel_size, stride=self.stride)
        conv2_output_size = self.calc_output_shape_conv_trans(conv1_output_size, self.stride, self.convt2.padding[0], self.convt2.dilation[0], kernel_size, self.convt2.output_padding[0])
        
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(conv2_output_size, self.window_length)
        
    def calc_output_shape_conv_trans(self, input_size, stride, padding, dialtion, kernel_size, out_padding):
        return (input_size-1) * stride - 2 * padding + dialtion * (kernel_size - 1) + out_padding + 1

    def forward(self, z):
        z = F.relu(self.dropout(self.fc1(z)))
        z = F.relu(self.convt1(z))
        z = self.convt2(z)
        z = torch.squeeze(z)
        z = self.fc2(z)
        return z

class Encoder2(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 7, stride=3) # Size of each channel: (205-7)/3+1=67
        self.conv2 = nn.Conv1d(16, 32, 7, stride=3)# Size of each channel: (67-7)/3+1=21
        
        self.fc1 = nn.Linear(672, latent_dims)
        
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x):
        print(x.shape)
        x = x.view(1,1,205)
        x = F.relu(self.dropout(self.conv1(x)))
        x = F.relu(self.dropout(self.conv2(x)))
        
        x = x.view(-1,672)
        
        x = self.fc1(x)
        
        return x

class Decoder2(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.fc1 = nn.Linear(latent_dims, 672)
        self.convt1 = nn.ConvTranspose1d(32, 16, kernel_size=7, stride=3)
        self.convt2 = nn.ConvTranspose1d(16, 1, kernel_size=7, stride=3)
        
        self.dropout = nn.Dropout(p=0.4)
        

    def forward(self, z):
        print(z.shape)
        z = F.relu(self.dropout(self.fc1(z)))
        
        z = z.view(-1,32,21)
        print(z.shape)
        z = F.relu(self.convt1(z))
        z = self.convt2(z)
        print(z.shape)
        z = z.view(-1,205)
        
        return z

class Autoencoder2(nn.Module):
    def __init__(self, latent_dims, a, b):
        super().__init__()
        self.encoder = Encoder2(latent_dims)
        self.decoder = Decoder2(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class Autoencoder(nn.Module):
    def __init__(self, latent_dims, window_length, kernel_size=7):
        super().__init__()
        self.window_length = window_length
        self.encoder = Encoder(latent_dims, window_length, kernel_size)
        self.decoder = Decoder(latent_dims, window_length, kernel_size)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)