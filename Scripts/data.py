import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import trimesh
import mesh_to_sdf as mts
from voxypy.models import Entity, Voxel
import subprocess

from tqdm import tqdm
import logging
import random
import pickle
import config
import os
from tempfile import NamedTemporaryFile

def generate_random_rotation_matrix():
    q = np.random.normal(0, 1, 4)
    q = q / np.linalg.norm(q)  # Normalize to get a unit quaternion
    
    w, x, y, z = q
    rotation_matrix = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y, 0],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x, 0],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y, 0],
        [0, 0, 0, 0]
    ])
    
    return rotation_matrix

def normalize_mesh(mesh, unit_sphere=True):
    """
    Normalize a mesh by centering it at origin and scaling to fit within a unit sphere or cube.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        The input mesh to normalize
    unit_sphere : bool, optional
        If True, normalize to unit sphere. If False, normalize to unit cube
        Default is True
        
    Returns:
    --------
    trimesh.Trimesh
        Normalized copy of the input mesh
    """
    # Create a copy of the mesh to avoid modifying the original
    normalized = mesh.copy()
    
    # Center the mesh at origin
    center = normalized.vertices.mean(axis=0)
    normalized.vertices -= center
    
    # Scale the mesh
    if unit_sphere:
        # Scale to fit in unit sphere (radius = 1)
        scale = np.max(np.linalg.norm(normalized.vertices, axis=1))
    else:
        # Scale to fit in unit cube (max dimension = 1)
        scale = np.max(np.abs(normalized.vertices))
    
    if scale > 0:
        normalized.vertices /= scale
        
    return normalized

def process_model(normalized_mesh, i, config : config.Config):
    if i != 0:
        normalized_mesh.apply_scale(random.random() * 2)
        normalized_mesh.apply_transform(generate_random_rotation_matrix())

    tempf = NamedTemporaryFile(suffix=".obj")
    normalized_mesh.export(file_obj=tempf.name)

    logging.info("Sampling SDF...")
    points, sdfs = mts.sample_sdf_near_surface(normalized_mesh, number_of_points=config.num_sdf_samples)
    logging.info("Done")

    # Get the voxelized object
    logging.info("Voxelizing...")
    voxel_tensor = voxelize_model_cuda_voxelizer(tempf.name, config=config)
    logging.info("Done")
    logging.info("-" * 50)

    tempf.close()

    return points, sdfs, voxel_tensor

def voxelize_model_cuda_voxelizer(mesh_file_path, config : config.Config):
    cuda_voxelizer_path = config.cuda_voxelizer_path
    resolution = config.input_voxel_grid_size

    subprocess.run(
        [cuda_voxelizer_path, "-f", mesh_file_path, "-s", str(resolution)], 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    entity = Entity().from_file(mesh_file_path + f"_{config.input_voxel_grid_size}.vox")
    voxel_array = entity.get_dense()
    voxel_array = np.pad(voxel_array, pad_width=1, mode="constant", constant_values=0)
    logging.info(type(voxel_array))
    logging.info(entity)
    voxel_tensor = torch.from_numpy(voxel_array.copy())
    logging.info(voxel_tensor.shape)
    os.remove(mesh_file_path + f"_{config.input_voxel_grid_size}.vox")
    return voxel_tensor

"""
Yield all sdfs, points, and voxel_grid in all models in the provided category folder and its augmented one
"""
def load_category(category_name, config : config.Config):
    shapenet_pkl_name, shapenet_pkl_fmt= os.path.splitext(config.shapenet_pickle_name)
    shapenet_dir = config.shapenet_path
    category_dir = os.path.join(shapenet_dir, category_name)
    models_directories = os.listdir(category_dir)

    category_models = []
    for model in tqdm(models_directories, desc="Processing through models...", position=1):
        model_dir = os.path.join(category_dir, model)
        normalized_mesh = 0

        model_pkl_name = lambda n : f"{shapenet_pkl_name}_{category_name}_{model}_{n}{shapenet_pkl_fmt}"
        model_pkl_path = lambda n : os.path.join(model_dir, model_pkl_name(n))
        for i in range(config.num_augment_data):
            if os.path.exists(model_pkl_path(i)):
                with open(model_pkl_path(i), "b+r") as f:
                    category_models.append(pickle.load(f))
            else:
                if normalized_mesh == 0:
                    model_file = os.path.join(model_dir, "models", "model_normalized.obj")
                    normalized_mesh = normalize_mesh(trimesh.load(model_file, force="mesh"))

                points, sdfs, voxel_tensor = process_model(normalized_mesh, i, config=config)
                category_models.append({ 
                    "positions" : points, 
                    "sdfs" : sdfs, 
                    "voxel_tensor" : voxel_tensor 
                })
                with open(model_pkl_path(i), "b+w") as f:
                    pickle.dump(category_models[-1], f)

    return category_models

class ModelData(Dataset):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    def __init__(self, data, config : config.Config):
        super(ModelData, self).__init__()
        
        self.num_sdf_samples = config.num_sdf_samples
        self.num_points_per_iter = config.num_points_per_iter

        self.config = config

        self.datas = data

    def __getitem__(self, index):
        data = self.datas[index // (self.num_sdf_samples // self.num_points_per_iter)]
        i = index % (self.num_sdf_samples // self.num_points_per_iter)
        start = i * self.num_points_per_iter
        end = (i + 1) * self.num_points_per_iter
        position = torch.from_numpy(data["positions"][start:end])
        sdf = torch.from_numpy(data["sdfs"][start:end])
        return data["voxel_tensor"].float(), position, sdf

    def __len__(self):
        return len(self.datas) * (self.config.num_sdf_samples // self.num_points_per_iter)

    def get_batch_sizes(self):
        return 32

def load_shapenet(config : config.Config):
    results = {}
    shapenet_categories = config.shapenet_categories
    shapenet_dir = config.shapenet_path
    
    if "All" in shapenet_categories:
        category_directories = os.listdir(shapenet_dir)
    else:
        category_directories = shapenet_categories

    for category in tqdm(category_directories, desc="Processing through categories...", position=0):
        results[category] = load_category(
            category_name=category, 
            config=config
        )
    
    return results

def load_train_validation_seperated_data(config : config.Config):
    ratio = config.seperate_ratio
    data = load_shapenet(config=config)

    train_data, validation_data = [], []
    for category in data.values():
        random.seed(42)
        random.shuffle(category)
        n = len(category)
        n_test = int(n * ratio[0])
        train_data.extend(category[0:n_test])
        validation_data.extend(category[n_test:n])

    return train_data, validation_data

def create_test_validation_loader(config : config.Config):
    train_data, validation_data = load_train_validation_seperated_data(config=config)
    train_dataset = ModelData(train_data, config=config)
    validation_dataset = ModelData(validation_data, config=config)
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=32, 
        shuffle=True
    )
    validation_loader = DataLoader(
        dataset=validation_dataset, 
        batch_size=32,
        shuffle=False
    )

    return train_loader, validation_loader

if __name__ == "__main__":
    cfg = config.Config()
    logging.basicConfig(level=logging.WARNING)
    train_loader, validation_loader = create_test_validation_loader(config=cfg)
