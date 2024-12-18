// Voxelizer with GPU acceleration and texture support using C++ and CUDA
#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <stb_image.h>
#include <tinyobjloader/tiny_obj_loader.h>

// CUDA Kernel for voxelization with texture support
__global__ void voxelize(const float* vertices, const int* faces, const float* uvs, int num_faces, int resolution,
                         glm::vec3 min_bounds, glm::vec3 grid_size, bool* voxel_grid,
                         uchar3* voxel_colors, const unsigned char* texture, int tex_width, int tex_height) {
    int face_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_idx >= num_faces) return;

    int face_offset = face_idx * 3;
    glm::vec3 v0 = glm::make_vec3(&vertices[faces[face_offset] * 3]);
    glm::vec3 v1 = glm::make_vec3(&vertices[faces[face_offset + 1] * 3]);
    glm::vec3 v2 = glm::make_vec3(&vertices[faces[face_offset + 2] * 3]);

    glm::vec2 uv0 = glm::make_vec2(&uvs[faces[face_offset] * 2]);
    glm::vec2 uv1 = glm::make_vec2(&uvs[faces[face_offset + 1] * 2]);
    glm::vec2 uv2 = glm::make_vec2(&uvs[faces[face_offset + 2] * 2]);

    // Compute voxel bounding box for this triangle
    glm::ivec3 min_voxel = glm::floor((glm::min(v0, glm::min(v1, v2)) - min_bounds) / grid_size);
    glm::ivec3 max_voxel = glm::ceil((glm::max(v0, glm::max(v1, v2)) - min_bounds) / grid_size);

    for (int x = min_voxel.x; x <= max_voxel.x; ++x) {
        for (int y = min_voxel.y; y <= max_voxel.y; ++y) {
            for (int z = min_voxel.z; z <= max_voxel.z; ++z) {
                int voxel_idx = x + resolution * (y + resolution * z);
                voxel_grid[voxel_idx] = true;

                // Compute barycentric coordinates to interpolate UVs
                glm::vec3 voxel_center = min_bounds + glm::vec3(x, y, z) * grid_size;
                glm::vec3 bary = glm::inverse(glm::mat3(v0 - v2, v1 - v2, v2 - voxel_center)) * (voxel_center - v2);

                if (bary.x >= 0 && bary.y >= 0 && bary.x + bary.y <= 1.0f) {
                    glm::vec2 uv = bary.x * uv0 + bary.y * uv1 + (1.0f - bary.x - bary.y) * uv2;
                    int tex_x = static_cast<int>(uv.x * tex_width) % tex_width;
                    int tex_y = static_cast<int>(uv.y * tex_height) % tex_height;

                    int tex_idx = (tex_y * tex_width + tex_x) * 3;
                    voxel_colors[voxel_idx] = make_uchar3(texture[tex_idx], texture[tex_idx + 1], texture[tex_idx + 2]);
                }
            }
        }
    }
}

void voxelizeMeshWithTexture(const std::string& obj_file, const std::string& texture_file, int resolution) {
    // Load the mesh using TinyOBJLoader
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, obj_file.c_str())) {
        std::cerr << "Error loading OBJ: " << warn << err << std::endl;
        return;
    }

    // Load the texture
    int tex_width, tex_height, tex_channels;
    unsigned char* texture = stbi_load(texture_file.c_str(), &tex_width, &tex_height, &tex_channels, 3);
    if (!texture) {
        std::cerr << "Error loading texture." << std::endl;
        return;
    }

    // Compute mesh bounds
    glm::vec3 min_bounds(FLT_MAX), max_bounds(-FLT_MAX);
    for (size_t i = 0; i < attrib.vertices.size() / 3; ++i) {
        glm::vec3 vertex = glm::make_vec3(&attrib.vertices[i * 3]);
        min_bounds = glm::min(min_bounds, vertex);
        max_bounds = glm::max(max_bounds, vertex);
    }

    glm::vec3 grid_size = (max_bounds - min_bounds) / static_cast<float>(resolution);

    // Prepare data for GPU
    int num_faces = static_cast<int>(shapes[0].mesh.num_face_vertices.size());
    std::vector<int> faces;
    for (size_t i = 0; i < shapes[0].mesh.indices.size(); ++i) {
        faces.push_back(shapes[0].mesh.indices[i].vertex_index);
    }

    float* d_vertices;
    int* d_faces;
    float* d_uvs;
    bool* d_voxel_grid;
    uchar3* d_voxel_colors;
    unsigned char* d_texture;

    cudaMalloc(&d_vertices, attrib.vertices.size() * sizeof(float));
    cudaMalloc(&d_faces, faces.size() * sizeof(int));
    cudaMalloc(&d_uvs, attrib.texcoords.size() * sizeof(float));
    cudaMalloc(&d_voxel_grid, resolution * resolution * resolution * sizeof(bool));
    cudaMalloc(&d_voxel_colors, resolution * resolution * resolution * sizeof(uchar3));
    cudaMalloc(&d_texture, tex_width * tex_height * 3 * sizeof(unsigned char));

    cudaMemcpy(d_vertices, attrib.vertices.data(), attrib.vertices.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_faces, faces.data(), faces.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uvs, attrib.texcoords.data(), attrib.texcoords.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_texture, texture, tex_width * tex_height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    int threads_per_block = 128;
    int blocks_per_grid = (num_faces + threads_per_block - 1) / threads_per_block;
    voxelize<<<blocks_per_grid, threads_per_block>>>(d_vertices, d_faces, d_uvs, num_faces, resolution,
                                                     min_bounds, grid_size, d_voxel_grid,
                                                     d_voxel_colors, d_texture, tex_width, tex_height);

    // Retrieve results
    std::vector<bool> voxel_grid(resolution * resolution * resolution);
    std::vector<uchar3> voxel_colors(resolution * resolution * resolution);
    cudaMemcpy(voxel_grid.data(), d_voxel_grid, voxel_grid.size() * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(voxel_colors.data(), d_voxel_colors, voxel_colors.size() * sizeof(uchar3), cudaMemcpyDeviceToHost);

    cudaFree(d_vertices);
    cudaFree(d_faces);
    cudaFree(d_uvs);
    cudaFree(d_voxel_grid);
    cudaFree(d_voxel_colors);
    cudaFree(d_texture);
    stbi_image_free(texture);

    // Save voxel grid and colors (implementation left as exercise)
    std::cout << "Voxelization with texture complete." << std::endl;
}

int main() {
    std::string obj_file = "path_to_mesh.obj";  // Replace with your OBJ file
    std::string texture_file = "path_to_texture.jpg";  // Replace with your texture file
    int resolution = 64;  // Define the voxel resolution

    voxelizeMeshWithTexture(obj_file, texture_file, resolution);

    return 0;
}
