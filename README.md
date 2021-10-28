# Omni-GAN-DGP


This repository contains the code for the paper, [Omni-GAN: On the Secrets of cGANs and Beyond](https://arxiv.org/abs/2011.13074). </br >
In particular, it contains the code for the ImageNet and DGP experiments. </br >
We recommend that you refer to another project, [Omni-GAN-PyTorch](https://github.com/PeterouZh/Omni-GAN-PyTorch), to learn Omni-GAN quickly.



## My tasks

✔️ Training code for ImageNet experiments. To do by the end of this month (2021-10-19)  
⬜️ The inversion code for DGP with Omni-INR-GAN experiments.


===========================================================
<p float="left">
<img src=.github/truncation_curve.png width="600" />
</p>

## DGP experiments
- Colorization

<p float="left">
<img src=.github/colorization.png width="800" />
</p>

- Super-resolution x60+

<p float="left">
<img src=.github/SR60.png width="800" />
</p>

## Envs

```bash
git clone --recursive https://github.com/PeterouZh/Omni-GAN-DGP.git
cd Omni-GAN-DGP

# Create virtual environment
conda create -y --name omnigan python=3.6.7
conda activate omnigan

pip install torch==1.8.2+cu102 torchvision==0.9.2+cu102 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html

pip install --no-cache-dir tl2==0.0.3
pip install --no-cache-dir -r requirements.txt


```

## Prepare dataset

- Make hdf5 file
```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=./BigGAN_Pytorch_lib:./
python scripts/make_hdf5.py \
  --tl_config_file configs/make_hdf5.yaml \
  --tl_command make_hdf5_ImageNet128 \
  --tl_outdir results/make_hdf5_ImageNet128 \
  --tl_opts data_root datasets/ImageNet/train \
    index_filename datasets/ImageNet_hdf5/I128_index.npz \
    saved_hdf5_file datasets/ImageNet_hdf5/ILSVRC128.hdf5

```

Note: `data_root`: the path of ImageNet training images.

- Prepare inception moment file for evaluation of FID
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=./BigGAN_Pytorch_lib:./
python scripts/calculate_inception_moments.py \
  --tl_config_file configs/make_hdf5.yaml \
  --tl_command calculate_inception_moments_ImageNet128 \
  --tl_outdir results/calculate_inception_moments_ImageNet128 \
  --tl_opts data_root datasets/ImageNet_hdf5/ILSVRC128.hdf5 \
    saved_inception_file datasets/ImageNet_hdf5/I128_inception_moments.npz

```

- For 256x256
```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=./BigGAN_Pytorch_lib:./
python scripts/make_hdf5.py \
  --tl_config_file configs/make_hdf5.yaml \
  --tl_command make_hdf5_ImageNet256 \
  --tl_outdir results/make_hdf5_ImageNet256 \
  --tl_opts data_root datasets/ImageNet/train \
    index_filename datasets/ImageNet_hdf5/I256_index.npz \
    saved_hdf5_file datasets/ImageNet_hdf5/ILSVRC256.hdf5


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=./BigGAN_Pytorch_lib:./
python scripts/calculate_inception_moments.py \
  --tl_config_file configs/make_hdf5.yaml \
  --tl_command calculate_inception_moments_ImageNet256 \
  --tl_outdir results/calculate_inception_moments_ImageNet256 \
  --tl_opts data_root datasets/ImageNet_hdf5/ILSVRC256.hdf5 \
    saved_inception_file datasets/ImageNet_hdf5/I256_inception_moments.npz

```

## Evaluation (Omni-GAN)

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=./BigGAN_Pytorch_lib:./
python scripts/train.py \
  --tl_config_file configs/omnigan_imagenet128.yaml \
  --tl_command eval_ImageNet128 \
  --tl_outdir results/eval_ImageNet128 \
  --tl_opts inception_file  datasets/ImageNet_hdf5/I128_inception_moments.npz \
    evaluation.G_ema_model datasets/pretrained/omnigan_r128_G_ema.pth


```

## Train (Omni-GAN)
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=./BigGAN_Pytorch_lib:./
python scripts/train.py \
  --tl_config_file configs/omnigan_imagenet128.yaml \
  --tl_command train_ImageNet128 \
  --tl_outdir results/train_ImageNet128 \
  --tl_opts args.data_root datasets/ImageNet_hdf5/ILSVRC128.hdf5 \
    inception_file  datasets/ImageNet_hdf5/I128_inception_moments.npz

```

## For 256x256 (Omni-GAN)
```bash


```

## Omni-INR-GAN



## Acknowledgments

- BigGAN implemented from [https://github.com/ajbrock/BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch).
- Multi-label classification loss derived by [Jianlin Su](https://kexue.fm/archives/7359).
- Detectron2 library [https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2).


