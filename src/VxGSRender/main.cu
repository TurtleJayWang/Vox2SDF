#include <iostream>
#include <vector>

#include <torch/extension.h>

void render_voxel_gaussians(
    const std::vector<torch::Tensor>& gaussian_parameters,
    const torch::Tensor& voxel_grid,
    const int grid_size[],
    torch::Tensor* output
) {
    
}
