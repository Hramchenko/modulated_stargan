# StarGAN V2 with Modulated Convolutions

![logo](data/compose.jpg)

Experiments on replacing Adaptive Instance Normalization(AdaIN) layers in [__StarGAN V2__](https://github.com/clovaai/stargan-v2) model.

The presented results was obtained in a 24 hours on a single Colab GPU.

For more information see [Modifying StarGAN V2 using Modulated Convolutions](https://v-hramchenko.medium.com/modifying-stargan-v2-using-modulated-convolutions-13dc5796cd6e).

## Images generation
For images generation:
1. Unpack pretrained [weights](https://cloud.mail.ru/public/7fKF/oPW7FDLro) in repository root.
2. Run generate_images.ipynb file.

## Training the model

For images generation see train_colab.ipynb file.

## Requirements
* python;
* pytorch;
* torchvision;
* opencv;
* munch;
* kornia;
* ffmpeg.


