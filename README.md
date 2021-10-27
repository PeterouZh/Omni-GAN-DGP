# Omni-GAN-DGP


This repository contains the code for the paper, [Omni-GAN: On the Secrets of cGANs and Beyond](https://arxiv.org/abs/2011.13074). </br >
In particular, it contains the code for the ImageNet and DGP experiments.



### TODO

⬜️ Release training code for ImageNet experiments. To do by the end of this month (2021-10-19)  
✔️ DGP experiments with pretrained Omni-INR-GAN models  
✔️ Resutls on ImageNet 256x256 dataset  
✔️ Results on ImageNet 128x128 dataset  



===========================================================
### Updates

#### **2021-03-30**: Omni-INR-GAN

We invented Omni-INR-GAN, which is more friendly to GAN inversion tasks. Please see our paper [Omni-GAN: On the Secrets of cGANs and Beyond](https://arxiv.org/abs/2011.13074).

<p float="left">
<img src=.github/truncation_curve.png width="600" />
</p>

- Colorization

[https://www.bilibili.com/video/BV1nZ4y1A7H8?share_source=copy_web](https://www.bilibili.com/video/BV1nZ4y1A7H8?share_source=copy_web)
<p float="left">
<img src=.github/colorization.png width="800" />
</p>

- Generating images of arbitrary resolution

[https://www.bilibili.com/video/BV1SZ4y1w7gu?share_source=copy_web](https://www.bilibili.com/video/BV1SZ4y1w7gu?share_source=copy_web)

- Super-resolution x60+

[https://www.bilibili.com/video/BV1Rh411S7Eg?share_source=copy_web](https://www.bilibili.com/video/BV1Rh411S7Eg?share_source=copy_web)
<p float="left">
<img src=.github/SR60.png width="800" />
</p>

#### **2021-02-17**: Results on ImageNet (256x256 resolution)

The ImageNet 256x256 experiment requires much longer training time. For example, it took about thirty days to train the BigGAN using eight v100 GPUs, and then the BigGAN began to collapse. Omni-GAN enjoys faster convergence and superior performance than BigGAN in terms of both IS and FID. To see if the Omni-GAN will collapse, we trained Omni-GAN for more epochs and no mode collapse is observed.

<p float="left">
<img src=.github/save_OmniGAN_ImageNet256_IS.pdf.png width="400" />
<img src=.github/save_OmniGAN_ImageNet256_FID.pdf.png width="400" />
</p>

#### **2020-12-23**: Results on ImageNet (128x128 resolution)

Since I recently acquired GPU resources, I trained Omni-GAN on ImageNet datasets (with 128x128 resolution). Omni-GAN only needs one day to reach the IS score of BigGAN which is trained for two weeks!  Experiments were conducted on 8xV100 (32GB VRAM each). Below are the IS and FID curves. We will release the trained models to benefit the research of the community.

<p float="left">
<img src=.github/save_OmniGAN_ImageNet128_IS.pdf.png width="400" />
<img src=.github/save_OmniGAN_ImageNet128_FID.pdf.png width="400" />
</p>



## Acknowledgments

- BigGAN implemented from [https://github.com/ajbrock/BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch).
- Multi-label classification loss derived by [Jianlin Su](https://kexue.fm/archives/7359).
- Detectron2 library [https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2).


