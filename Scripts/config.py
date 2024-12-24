import os
import json
import logging

class Config:
    def __init__(self, config_file="Scripts/config.json"):
        with open(config_file) as cfile:
            self.config = json.load(cfile)
        
        self.input_voxel_grid_size = self.config["input_voxel_grid_size"]
        self.decoder_hidden_dimension = self.config["decoder_hidden_dimension"]
        self.decoder_num_hidden_layers = self.config["decoder_num_hidden_layers"]
        self.latent_dimension = self.config["latent_dimension"]
        self.num_sdf_samples = self.config["num_sdf_samples"]
        self.shapenet_path = self.config["shapenet_path"]
        self.cuda_voxelizer_path = self.config["cuda_voxelizer_path"]
        self.shapenet_pickle_name = self.config["shapenet_pickle_name"]

    def log(self): 
        logging.info(self.config)

if __name__ == "__main__":
    config = Config()
    logging.basicConfig(level=logging.DEBUG)
    config.log()