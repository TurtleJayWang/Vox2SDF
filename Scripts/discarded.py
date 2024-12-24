# discarded
def voxelize_model(mesh : trimesh.Geometry, resolution=128):
    # Normalize the mesh
    mesh_size = max(map(lambda x, y : abs(x - y), mesh.bounds[1], mesh.bounds[0]))  
    mesh.apply_scale(1 / mesh_size)
    print(f"The size of the mesh: {mesh_size}")

    # Voxelize the mesh
    print("Creating voxel grid")
    voxel_grid = trimesh.geometry.VoxelGrid.create_from_triangle_mesh(mesh, mesh_size / resolution)

    # Convert the generated voxel grid to tensor (voxel grid -> numpy array -> torch tensor)
    print("Converting to numpy array")
    voxel_np_array = voxel_grid.matrix.astype(np.float32)
    print("Converting to torch tensor")
    voxel_tensor = torch.from_numpy(voxel_np_array)
    return voxel_tensor
