export HF_TOKEN=$1

if [ ! -d Data ]; then
    mkdir Data
fi
if [ ! -d Data/ShapeNet ]; then
    mkdir Data/ShapeNet
fi
cd Data/ShapeNet
wget --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/ShapeNet/ShapeNetCore/resolve/main/02880940.zip
wget --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/ShapeNet/ShapeNetCore/resolve/main/02747177.zip
wget --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/ShapeNet/ShapeNetCore/resolve/main/02773838.zip
wget --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/ShapeNet/ShapeNetCore/resolve/main/02801938.zip
wget --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/ShapeNet/ShapeNetCore/resolve/main/02808440.zip
unzip  -o '*.zip'
rm '*.zip'
wget --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/ShapeNet/ShapeNetCore/resolve/main/02828884.zip
wget --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/ShapeNet/ShapeNetCore/resolve/main/02843684.zip
wget --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/ShapeNet/ShapeNetCore/resolve/main/02871439.zip
wget --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/ShapeNet/ShapeNetCore/resolve/main/02876657.zip
unzip  -o '*.zip' 
rm '*.zip'
