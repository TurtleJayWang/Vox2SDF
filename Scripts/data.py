import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import trimesh
import mesh_to_sdf as mts
import os
import subprocess

from tqdm import tqdm

def load_shapenet(num_sdf_samples=250000, shapenet_dir = "Assets/ShapeNet"):
    category_directories = os.listdir(shapenet_dir)
    results = []
    #for category in tqdm(category_directories, desc="Processing through categories...", position=0):
    for category in category_directories:
        models_directories = os.listdir(os.path.join(shapenet_dir, category))
        i = 1
        #for model in tqdm(models_directories[0:2], desc="Processing through models...", position=1):
        for model in models_directories[0:2]:
            model_dir = os.path.join(shapenet_dir, category, model)
            i += 1
            points, sdfs, voxel_tensor = process_model(model_dir, num_sdf_samples)
            results.append({
                "positions" : points, 
                "sdfs" : sdfs, 
                "voxel_tensor" : voxel_tensor
            })
    return results

def process_model(model_dir, num_of_sdf_points = 250000):
    contents = os.listdir(model_dir)

    # Check if the directory arrangement is right
    if not "models" in contents:
        print("Wrong directory arrangement!, directory:", model_dir)
        return
    
    model_file = os.path.join(model_dir, "models", "model_normalized.obj")

    print(f"Loading \"{model_file}\" ...")
    mesh = trimesh.load(model_file, force="mesh")
    print(mesh)
    print("Done")
    # Sample SDFs
    print("Sampling SDF...")
    points, sdfs = mts.sample_sdf_near_surface(mesh, number_of_points=num_of_sdf_points)
    print("Done")
    # Get the voxelized object
    print("Voxelizing...")
    voxel_tensor = voxelize_model(mesh=mesh)
    print("Done")
    print("-" * 50)

    return (points, sdfs, voxel_tensor)

def voxelize_model(mesh : trimesh.Geometry, resolution=128):
    # Normalize the mesh
    mesh_size = max(map(lambda x, y : abs(x - y), mesh.bounds[1], mesh.bounds[0]))  
    mesh.apply_scale(1 / mesh_size)
    print(f"The size of the mesh: {mesh_size}")

    # Voxelize the mesh
    print("Creating voxel grid")
    voxel_grid = trimesh.voxel.creation.voxelize(mesh, pitch= mesh_size / resolution)

    # Convert the generated voxel grid to tensor (voxel grid -> numpy array -> torch tensor)
    print("Converting to numpy array")
    voxel_np_array = voxel_grid.matrix.astype(np.float32)
    print("Converting to torch tensor")
    voxel_tensor = torch.from_numpy(voxel_np_array)
    return voxel_tensor

class ModelData(Dataset):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    def __init__(self, num_sdf_points=250000, use_shapenet=True):
        super(ModelData, self).__init__()
        self.num_sdf_samples = num_sdf_points
        if use_shapenet:
            self.datas = load_shapenet()

    def __getitem__(self, index):
        data = self.datas[index // self.num_sdf_samples]
        i = index % self.num_sdf_samples
        return data["voxel_tensor"], (data["positions"][i], data["sdfs"[i]])

    def __len__(self):
        return len(sum(self.get_batch_sizes()))

    def get_batch_sizes(self):
        return self.num_sdf_samples
    
    def create_data_loader(self):
        dataloader = DataLoader(self, batch_size=self.get_batch_sizes(), shuffle=False)
        return dataloader

if __name__ == "__main__":
    test_dataset = ModelData()
    print("How many data?", len(test_dataset.datas))
    print("Batch size: ", test_dataset.get_batch_sizes())
    dataloader = test_dataset.create_data_loader()
