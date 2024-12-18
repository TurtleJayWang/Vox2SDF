#include <vector>

#include <torch/extension.h>

struct Ray {
    float3 origin;
    float3 direction;
};

class MarchingTree {
public:

    struct Node {
        float norm_position[3];
        int children_idx[8];
        int parent_idx;
        float norm_size;
    };

    MarchingTree();

    __host__ static MarchingTree* from_grid(const torch::Tensor& voxel_grid);

    __host__ void update(const torch::Tensor& norm_pos);

    __host__ void to_device();

    __device__ float3 march(Ray ray);

private:

    __host__ std::vector<Node> node_list;

    __host__ __device__ size_t num_nodes;
    __host__ __device__ Node* node_array;

};
