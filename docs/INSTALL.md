# Installation
The code is tested with PyTorch 1.10.0 and CUDA 11.3. After cloning the repository, follow the below steps for installation,

## Requirements
- Python ≥ 3.7
- PyTorch ≥ 1.8
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).

1. Install PyTorch and torchvision

Installation with pip:
```shell
pip install torch==1.10.0 torchvision==0.11.0 torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```
or install with conda: 
```shell
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
2. Install Detectron2
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
3. Install other dependencies
```shell
pip install -r requirements.txt
```