# patchGAN

[![PyPI version](https://badge.fury.io/py/patchGAN.svg)](https://badge.fury.io/py/patchGAN)

UNet-based GAN model for image segmentation using a patch-wise discriminator.
Based on the [pix2pix](https://phillipi.github.io/pix2pix/) model.

## Installation

Install the package with pip:
```
pip install patchgan
```

Upgrading existing install:
```
pip install -U patchgan
```

Get the current development branch:
```
pip install -U git+https://github.com/ramanakumars/patchGAN.git
```

## Training
You can train the patchGAN model with a config file and the `patchgan_train` command:
```
patchgan_train --config_file train_coco.yaml --n_epochs 100 --batch_size 16
```
See `examples/train_coco.yaml` for the corresponding config for the COCO stuff dataset.
