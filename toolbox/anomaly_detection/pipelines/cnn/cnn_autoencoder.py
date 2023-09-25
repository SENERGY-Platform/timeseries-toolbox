from torch import nn 
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dims, window_length, kernel_size):
        super().__init__()
        self.window_length = window_length
        self.stride = 3
        self.conv1 = nn.Conv1d(1, 16, kernel_size, stride=self.stride) # Size of each channel: (205-7)/3+1=67
        self.conv1.padding
        print(self.conv1.padding)
        con1_output_size = (window_length + 2*self.conv1.padding[0] - kernel_size)/self.stride + 1
        print(con1_output_size)
        self.conv2 = nn.Conv1d(16, 32, kernel_size, stride=self.stride)# Size of each channel: (67-7)/3+1=21
        conv2_output_size = (con1_output_size-kernel_size)/self.stride + 1
        print(conv2_output_size)
        self.fc1 = nn.Linear(conv2_output_size, 672)
        self.fc2 = nn.Linear(672, latent_dims)
        
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x):
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
        self.fc1 = nn.Linear(latent_dims, 672)
        self.convt1 = nn.ConvTranspose1d(32, 16, kernel_size=kernel_size, stride=self.stride)
        conv1_output_size = (672 - 1) * self.stride + kernel_size - 2 * self.padding
        self.convt2 = nn.ConvTranspose1d(16, 1, kernel_size=kernel_size, stride=self.stride)
        conv2_output_size = (conv1_output_size - 1) * self.stride + kernel_size - 2 * self.padding
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(conv2_output_size, self.window_length)
        
    def forward(self, z):
        z = F.relu(self.dropout(self.fc1(z)))
        z = z.view(-1,32,21)
        z = F.relu(self.convt1(z))
        z = self.convt2(z)
        z = self.fc2(z)
        return z

class Autoencoder(nn.Module):
    def __init__(self, latent_dims, window_length, kernel_size=7):
        super().__init__()
        self.window_length = window_length
        self.encoder = Encoder(latent_dims, window_length, kernel_size)
        self.decoder = Decoder(latent_dims, window_length, kernel_size)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)