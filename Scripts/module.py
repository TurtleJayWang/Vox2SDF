import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np

class VAEBottleNeck(nn.Module):
    def __init__(self, latent_dim=256):
        super(VAEBottleNeck, self).__init__()

        self.latent_dim = latent_dim
        self.gaussian_nums = nn.Parameter(torch.randn(latent_dim))
        
    def forward(self, x):
        self.means = x[0 : self.latent_dim]
        self.stddev = x[self.latent_dim : self.latent_dim * 2]
        return self.means + self.stddev * self.gaussian_nums

class VQVAEBottleNeck(nn.Module):
    def __init__(self, latent_dim=256):
        super(VQVAEBottleNeck, self).__init__()

class VoxelCNNEncoder(nn.Module):
    def __init__(self, input_size=(128, 128, 128), latent_size=256):
        super().__init__()
        
        self.input_size = input_size

        # Add batch normalization for better training stability
        self.encoder = nn.Sequential(
            # First block
            nn.Conv3d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=4, stride=2),
            nn.Dropout3d(0.3),
            
            # Second block
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=4, stride=2),
            nn.Dropout3d(0.3),
            
            # Third block
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=4, stride=2),
            nn.Dropout3d(0.3),
            
            nn.Flatten()
        )
        
        # Calculate the size of the flattened features
        # Input: 128x128x128
        # After first block: 128x128x128 -> 64x64x64
        # After second block: 16x16x16 -> 8x8x8
        # After third block: 2x2x2 -> 1x1x1
        flattened_size = 64 * 1 * 1 * 1
        
        # Add final linear layers with skip connection
        self.fc1 = nn.Linear(flattened_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, latent_size)
        
    def forward(self, x):
        # Main encoder path
        features = self.encoder(x)
        
        # Final fully connected layers with skip connection
        x = self.fc1(features)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc2(x)
        
        return x

class SDFDecoder(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=512, num_layers=10):
        super(SDFDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.hiddem_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = [nn.Linear(latent_dim + 3, hidden_dim)]
        self.dropout03 = nn.Dropout(0.3)
        self.hidden_activation = nn.ReLU()
        self.output_activation = nn.Tanh()

        for i in range(1, num_layers - 1):
            if i % 4 == 0:
                self.layers.append(nn.Linear(hidden_dim + latent_dim + 3, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers.append(nn.Linear(512, 1))

    def forward(self, latent_code, position):
        x = self.layers[0](torch.cat((latent_code, position)))
        for i in range(1, self.num_layers - 1):
            if i % 4 == 0:
                x = self.layers[i](torch.cat((latent_code, position, x)))
                x = self.dropout03(x)
                x = self.hidden_activation(x)
            else:
                x = self.layers[i](x)
                x = self.dropout03(x)
                x = self.hidden_activation(x)
        x = self.layers[-1](x)
        x = self.output_activation(x)
        return x

class FullNetwork(nn.Module):
    def __init__(self,
        latent_folder,
        input_size=(128, 128, 128), 
        latent_dim=256,
        num_decoder_layer=10,
        load_latent=False
    ):
        super(FullNetwork, self).__init__()

        self.load_latent = load_latent
        self.latent_folder = latent_folder

        self.encoder = VoxelCNNEncoder(input_size=input_size, latent_size=latent_dim)
        self.decoder = SDFDecoder(latent_dim=latent_dim, num_layers=num_decoder_layer)

    def forward(self, voxel_grid, xyz):
        if self.load_latent:
            latent_file = os.listdir(self.latent_folder)[-1]
            self.latent_code : torch.Tensor = self.load_latent_code(latent_file)
            self.latent_code.requires_grad = False
        else:
            self.latent_code = self.encoder(voxel_grid)
            self.latent_code.requires_grad = False

        self.save_latent_code()
        return self.decoder(self.latent_code, xyz)

    def to_train(self):
        self.load_latent = False

    def to_use(self):
        self.load_latent = True

    def save_latent_code(self):
        np.save(os.path.join(self.latent_folder, "latent_code.pkl"), self.latent_code.numpy())

    def load_latent_code(self, file=""):
        return torch.from_numpy(
            np.load(os.path.join(self.latent_folder, "latent_code.pkl"))
        )
    
if __name__ == "__main__":
    print(os.path.join("../LatentCode", "latent_code.pkl"))