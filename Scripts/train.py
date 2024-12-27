import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from module import FullNetwork, VoxelCNNEncoder, SDFDecoder
import data
import config
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, train_dataloader : DataLoader, config : config.Config):
        self.epoch = config.train_epoch
        self.dataloader = train_dataloader
        self.network = FullNetwork("latent_code", config=config)
        self.device = config.device
        self.checkpoint_filename = config.check_point_filename

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)

    def train(self):
        for _ in tqdm(range(self.epoch), desc="Epoch", position=3):
            for i, (voxel_tensor, point, sdf) in tqdm(enumerate(self.dataloader), desc="Batch", position=4):
                voxel_tensor = voxel_tensor.to(device=self.device)
                point = point.to(device=self.device)
                sdf = torch.tensor([sdf], device="cuda")
                
                sdf_pred = self.network(voxel_tensor, point)
                loss = self.criterion(sdf_pred, sdf)

                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
                
    def save_parameters(self):
        with open(self.checkpoint_filename, "b+r") as cp_f:
            torch.save(self.network.state_dict(), cp_f)
        
    def load_parameters(self):
        with open(self.checkpoint_filename, "b+a") as cp_f:
            self.network.load_state_dict(torch.load(cp_f))

if __name__ == "__main__":
    cfg = config.Config()
    train_dataloader, validation_loader = data.create_test_validation_loader(config=cfg)
    model_trainer = ModelTrainer(train_dataloader=train_dataloader, config=cfg)
    model_trainer.train()
    model_trainer.save_parameters()
