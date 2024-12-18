import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import trimesh
import mesh_to_sdf as mts
import os

from tqdm import tqdm

def load_shapenet(shapenet_dir = "Assets/ShapeNet"):
    category_directories = os.listdir(shapenet_dir)
    print(category_directories)
    results = []
    for category in tqdm(category_directories, desc="Processing through categories...", position=0):
        models_directories = os.listdir(os.path.join(shapenet_dir, category))
        i = 1
        for model in tqdm(models_directories[0:2], desc="Processing through models...", position=1):
            model_dir = os.path.join(shapenet_dir, category, model)
            i += 1
            print(model_dir)
            points, sdfs, voxel_tensor, num_sdf_samples = process_model(model_dir)
            results.append({
                "positions" : points, 
                "sdfs" : sdfs, 
                "voxel_tensor" : voxel_tensor,
                "num_sdf_samples" : num_sdf_samples
            })
    return results

def process_model(model_dir, num_of_sdf_points = 250000):
    contents = os.listdir(model_dir)

    # Check if the directory arrangement is right
    if not "models" in contents:
        print("Wrong directory arrangement!, directory:", model_dir)
        return
    
    model_file = os.path.join(model_dir, "models", "model_normalized.obj")

    mesh = trimesh.load(model_file, force="mesh")
    # Sample SDFs
    points, sdfs = mts.sample_sdf_near_surface(mesh, number_of_points=num_of_sdf_points)
    # Get the voxelized object
    voxel_tensor = voxelize_model(mesh=mesh)

    num_sdf_samples = num_of_sdf_points
    return (points, sdfs, voxel_tensor, num_sdf_samples)

def voxelize_model(mesh : trimesh.Geometry, resolution=128):
    # Normalize the mesh
    mesh_size = map(lambda x, y : abs(x - y), mesh.bounds[1], mesh.bounds[0])
    mesh.apply_scale(1 / mesh_size)

    # Voxelize the mesh
    voxel_grid = trimesh.voxel.creation.voxelize(mesh, pitch= mesh_size / resolution)

    # Convert the generated voxel grid to tensor (voxel grid -> numpy array -> torch tensor)
    voxel_np_array = voxel_grid.matrix.astype(np.float32)
    voxel_tensor = torch.from_numpy(voxel_np_array)
    return voxel_tensor

class ModelData(Dataset):
    def __init__(self, use_shapenet=True):
        super(ModelData, self).__init__()
        if use_shapenet:
            self.datas = load_shapenet()

    def __getitem__(self, index):
        temp = index
        for data in self.datas:
            if temp <= data["num_sdf_samples"]:
                return data["voxel_tensor"], (data["positions"][temp], data["sdfs"[temp]])
            temp -= data["num_sdf_samples"]

    def __len__(self):
        return len(sum(self.get_batch_sizes()))

    def get_batch_sizes(self):
        result = []
        for data in self.datas:
            result.append(data["num_sdf_samples"])
        return result
    
    def create_data_loader(self):
        dataloader = DataLoader(self, batch_size=self.get_batch_sizes(), shuffle=False)
        return dataloader

if __name__ == "__main__":
    test_dataset = ModelData()
    print("How many data? =>", len(test_dataset.datas))
    print("Batch sizes =>", test_dataset.get_batch_sizes())
    dataloader = test_dataset.create_data_loader()
