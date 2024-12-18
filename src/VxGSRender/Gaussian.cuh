#pragma once

#include <torch/extension.h>

class Gaussian {
public:

    Gaussian(
        const torch::Tensor& position, 
        const torch::Tensor& rotation,
        const torch::Tensor& scaling,
        const torch::Tensor& color,
        const torch::Tensor& alpha
    );

    __host__ void to_device();

private:

    __host__ torch::Tensor input_tensor;

    __host__ __device__ float* position, *covariance, *color, *alpha;

};

class GaussianVoxel {
public:

    GaussianVoxel(
        int level, int position, 
        const std::vector<torch::Tensor>& gaussian_parameters
    );

private:

};
