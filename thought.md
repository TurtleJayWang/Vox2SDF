# Thought
I want to make an architecture which contains an encoder and a decoder, this architecture is used to convert a voxel model from a particular dataset into a DeepSDF representation with higher resolution. 

Below are the detail of this architecture:
1. Encoder: Use a 1024*1024*1024 voxel model as input, and then use 3D CNN or 3D vision transformer to encode the model into a latent code.
2. Decoder(like DeepSDF): Use the latent code and a point in 3D space as input, and then pass the value through several fully-connected layers. After passing the fully-connected layers, it outputs an SDF value of the input point. 

How I train it:
1. Preprocess the model from dataset(ABC dataset or ShapeNet):
   * Voxelize the  model
   * Create the SDF point cloud
2. Use the method proposed in DeepSDF to train the decoder part(using the SDF point cloud)
3. Use the voxelized part to train the encoder part by using MSE loss between the latent codes generated from the voxel encoder and the code from the DeepSDF auto-decoder

Here is my question:
1. Which fits my purpose better, 3D vision transformer or 3D CNN?
2. Should I use ABC dataset or ShapeNet to train the model?
