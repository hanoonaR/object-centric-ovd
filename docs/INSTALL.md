# Installation
The code is tested with PyTorch 1.10.0 and CUDA 11.3. After cloning the repository, follow the below steps for installation,

## Requirements
- Python ≥ 3.7
- PyTorch ≥ 1.8
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).

Our project uses a submodule `mavl` for generating psuedo proposals from MViT model MAVL. The proposals have been
provided with the datasets and detailed in [DATASETS.md](DATASETS.md). The submodule is required to generate the pseudo-proposals.
If you forget to add --recurse-submodules when you clone the repository, do:
1. Update the submodules [optional]
```shell
git clone https://github.com/hanoonaR/object-centric-ovd.git
git submodule init
git submodule update
```

2. Install PyTorch and torchvision

Installation with pip:
```shell
pip install torch==1.10.0 torchvision==0.11.0 torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```
or install with conda: 
```shell
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
3. Install Detectron2
Install with [pre-build Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html#install-pre-built-detectron2-linux-only):
```shell
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```
or install from source:
```shell
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
```
4. Install other dependencies
```shell
pip install -r requirements.txt
```