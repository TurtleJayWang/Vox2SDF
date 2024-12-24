import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import trimesh
import mesh_to_sdf as mts
from voxypy.models import Entity, Voxel
import os
import subprocess

from tqdm import tqdm
import logging
import random
import pickle
import config

def load_category(category_name, load_pickle=False, num_sdf_samples=250000, shapenet_dir="Assets/ShapeNet"):
    if load_pickle:
        category_models = pickle.load(os.path.join(shapenet_dir, category_name))
        return category_models

    category_models = []
    models_directories = os.listdir(os.path.join(shapenet_dir, category_name))
    i = 1
    for model in tqdm(models_directories, desc="Processing through models...", position=1):
        model_dir = os.path.join(models_directories, model)
        i += 1
        points, sdfs, voxel_tensor = process_model(model_dir, num_sdf_samples)
        category_models.append({
            "positions" : points, 
            "sdfs" : sdfs, 
            "voxel_tensor" : voxel_tensor,
            "latent_code" : torch.zeros(256)
        })
        random.shuffle(category_models)
    with open(os.path.join(models_directories, "models_data.pkl")) as models_data_file:
        pickle.dump(category_models, file=models_data_file)
    return category_models        

def load_shapenet(num_sdf_samples=250000, load_pickle=False, shapenet_dir = "Assets/ShapeNet"):
    results = {}
    shapenet_pkl_filename = config.Config().shapenet_pickle_name
    if load_pickle:
        results = pickle.load(file=os.path.join(shapenet_dir, shapenet_pkl_filename))
        return results
    
    category_directories = os.listdir(shapenet_dir)
    for category in tqdm(category_directories, desc="Processing through categories...", position=0):
        results[category] = load_category(
            category_name=category, 
            load_pickle=load_pickle, 
            num_sdf_samples=num_sdf_samples, 
            shapenet_dir=shapenet_dir
        )
    with open(os.path.join(shapenet_dir, shapenet_pkl_filename)) as shapenet_pkl:
        pickle.dump(results, file=shapenet_pkl)
    
    return results

def process_model(model_dir, num_of_sdf_points = 250000):
    contents = os.listdir(model_dir)

    # Check if the directory arrangement is right
    if not "models" in contents:
        logging.info("Wrong directory arrangement!, directory:", model_dir)
        return
    
    model_file = os.path.join(model_dir, "models", "model_normalized.obj")

    logging.info(f"Loading \"{model_file}\" ...")
    mesh = trimesh.load(model_file, force="mesh")
    logging.info(mesh)
    logging.info("Done")
    # Sample SDFs
    logging.info("Sampling SDF...")
    points, sdfs = mts.sample_sdf_near_surface(mesh, number_of_points=num_of_sdf_points)
    logging.info("Done")
    # Get the voxelized object
    logging.info("Voxelizing...")
    voxel_tensor = voxelize_model_cuda_voxelizer(model_file)
    logging.info("Done")
    logging.info("-" * 50)

    return (points, sdfs, voxel_tensor)

def voxelize_model_cuda_voxelizer(mesh_file_path, resolution=128, cuda_voxelizer_path="thirdparty/cuda_voxelizer/cuda_voxelizer"):
    subprocess.run(
        [cuda_voxelizer_path, "-f", mesh_file_path, "-s", str(resolution)], 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    entity = Entity().from_file(mesh_file_path + "_128.vox")
    voxel_array = entity.get_dense()
    voxel_array = np.pad(voxel_array, pad_width=1, mode="constant", constant_values=0)
    logging.info(type(voxel_array))
    logging.info(entity)
    voxel_tensor = torch.from_numpy(voxel_array.copy())
    logging.info(voxel_tensor.shape)
    return voxel_tensor

class ModelData(Dataset):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    def __init__(self, category_id=""):
        super(ModelData, self).__init__()
        
        self.shapenet_dir = config.Config().shapenet_path

        self.num_sdf_samples = config.Config().num_sdf_samples
        if category_id == "":
            self.datas = load_shapenet(shapenet_dir=self.shapenet_dir)
        else:
            self.datas = load_category(category_name=category_id)

    def __getitem__(self, index):
        data = self.datas[index // self.num_sdf_samples]
        i = index % self.num_sdf_samples
        return data["voxel_tensor"], (data["positions"][i], data["sdfs"[i]])

    def __len__(self):
        return len(sum(self.get_batch_sizes()))

    def get_batch_sizes(self):
        return self.num_sdf_samples

def create_test_and_validation_loader(dataset : ModelData, lengths=(0.8, 0.2)):
    generator = torch.Generator().manual_seed(42)
    train_loader, validation_loader = random_split(dataset=dataset, lengths=lengths, generator=generator)
    return train_loader, validation_loader

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    test_dataset = ModelData()
    logging.basicConfig(level=logging.DEBUG)
    logging.info("How many data?", len(test_dataset.datas))
    logging.info("Batch size: ", test_dataset.get_batch_sizes())
    train_dataloader, test_dataloader = create_test_and_validation_loader(test_dataset)
