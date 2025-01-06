export HF_TOKEN=$1

if [ ! -d Data ]; then
    mkdir Data
fi
if [ ! -d Data/ShapeNet ]; then
    mkdir Data/ShapeNet
fi
cd Data/ShapeNet
wget --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/ShapeNet/ShapeNetCore/resolve/main/02958343.zip
wget --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/ShapeNet/ShapeNetCore/resolve/main/02880940.zip
wget --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/ShapeNet/ShapeNetCore/resolve/main/02747177.zip
unzip '*.zip'
rm '*.zip'