# Vox2SDF
## Description
This is a model aiming to convert a voxel model into a neural SDF representation.

## How to setup the environment
### Dependencies
* PyTorch
* CUDA (this project has been tested on CUDA 12.2)
* Trimesh
* Voxypy
* Open3D
* tqdm
#### Install Script
```
conda create -n Vox2SDF
conda activate Vox2SDF
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
pip install trimesh
pip install voxypy
pip install open3d
conda install tqdm
```
### Tryout
Without installing the dependencies, you can try this project in Vox2SDF.ipynb with google colab(this notebook has been tested in Kaggle)

## Build