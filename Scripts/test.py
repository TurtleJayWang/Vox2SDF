import torch

import numpy as np
import mesh_to_sdf as mts
import trimesh

import glob
import config
import os
import random
from tempfile import NamedTemporaryFile
import subprocess
import logging
from voxypy import Entity

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

def process_model(model_path, config : config.Config):
    mesh = trimesh.load(model_path, force="mesh")
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

def load_models(directory, config : config.Config):
    model_paths = glob.iglob(os.path.join(directory, "/**/*.obj"), recursive=True)

    if not os.path.exists(os.path.join(directory, "processed_data")):
        os.mkdir(os.path.join(directory, "processed_data"))

    for i, model_path in enumerate(model_paths):
        for j, model_data in enumerate(process_model(model_path, config)):
            pkl_path = os.path.join(directory, "processed_data", f"model_data_{i}_{j}.pkl")
            if os.path.exists(pkl_path):
                with open(os.path.join(directory, "processed_data", f"model_data_{i}_{j}.pkl")):
                    pass
    
if __name__ == "__main__":
    load_models()
