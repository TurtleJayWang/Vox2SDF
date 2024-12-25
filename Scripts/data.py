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

def process_model(model_dir, config : config.Config):
    contents = os.listdir(model_dir)

    # Check if the directory arrangement is right
    if not "models" in contents:
        logging.warning("Wrong directory arrangement!, directory:", model_dir)
        return
    
    model_file = os.path.join(model_dir, "models", "model_normalized.obj")

    logging.info(f"Loading \"{model_file}\" ...")
    mesh = trimesh.load(model_file, force="mesh")
    logging.info(mesh)
    logging.info("Done")

    meshes = [mesh] * config.num_augment_data
    for i in range(1, config.num_augment_data):
        meshes[i].apply_scale(random.random() * 2)
        meshes[i].apply_transform(generate_random_rotation_matrix())

        tempf = NamedTemporaryFile(suffix=".obj")
        mesh.export(file_obj=tempf.name, file_type="obj")

        logging.info("Sampling SDF...")
        points, sdfs = mts.sample_sdf_near_surface(meshes[i], number_of_points=config.num_sdf_samples)
        logging.info("Done")

        # Get the voxelized object
        logging.info("Voxelizing...")
        voxel_tensor = voxelize_model_cuda_voxelizer(tempf.name, config=config)
        logging.info("Done")
        logging.info("-" * 50)

        tempf.close()

        yield (points, sdfs, voxel_tensor)

def voxelize_model_cuda_voxelizer(mesh_file_path, config : config.Config):
    cuda_voxelizer_path = config.cuda_voxelizer_path
    resolution = config.input_voxel_grid_size

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

def load_category(category_name, config : config.Config):
    shapenet_pkl_name, shapenet_pkl_fmt= os.path.splitext(config.shapenet_pickle_name)
    models_pickle_data_filename = shapenet_pkl_name + "_" + category_name + shapenet_pkl_fmt
    shapenet_dir = config.shapenet_path
    category_dir = os.path.join(shapenet_dir, category_name)
    models_directories = os.listdir(category_dir)

    load_pickle = models_pickle_data_filename in os.listdir(category_dir)
    if load_pickle:
        category_models = pickle.load(os.path.join(shapenet_dir, category_name))
        return category_models

    category_models = []
    i = 1
    for model in tqdm(models_directories, desc="Processing through models...", position=1):
        model_dir = os.path.join(category_dir, model)
        i += 1
        for points, sdfs, voxel_tensor in process_model(model_dir, config=config):
            category_models.append({
                "positions" : points,
                "sdfs" : sdfs,
                "voxel_tensor" : voxel_tensor
            })

    with open(os.path.join(category_dir, models_pickle_data_filename)) as models_data_file:
        pickle.dump(category_models, file=models_data_file)

    return category_models        

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
        random.shuffle(category)
        n = len(category)
        n_test = int(n * ratio[0])
        train_data.extend(category[0:n_test])
        validation_data.extend(category[n_test:n])

    return train_data, validation_data

class ModelData(Dataset):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    def __init__(self, data, config : config.Config):
        super(ModelData, self).__init__()
        
        self.num_sdf_samples = config.num_sdf_samples

        self.datas = data

    def __getitem__(self, index):
        data = self.datas[index // self.num_sdf_samples]
        i = index % self.num_sdf_samples
        return data["voxel_tensor"], data["positions"][i], data["sdfs"[i]]

    def __len__(self):
        return self.get_batch_sizes * len(self.datas)

    def get_batch_sizes(self):
        return self.num_sdf_samples

def create_test_validation_loader(config : config.Config):
    train_data, validation_data = load_train_validation_seperated_data(config=config)
    train_dataset = ModelData(train_data, config=config)
    validation_dataset = ModelData(validation_data, config=config)
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=train_dataset.get_batch_sizes(), 
        shuffle=True
    )
    validation_loader = DataLoader(
        dataset=validation_dataset, 
        batch_size=validation_dataset.get_batch_sizes(),
        shuffle=False
    )

    return train_loader, validation_loader

if __name__ == "__main__":
    cfg = config.Config()
    logging.basicConfig(level=logging.WARNING)
    train_loader, validation_loader = create_test_validation_loader(config=cfg)
