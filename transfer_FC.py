import numpy as np
from torchinfo import summary
from patchgan.unet import UNet, Discriminator, get_norm_layer
from patchgan.io import MmapDataGenerator
from patchgan.trainer import Trainer, device

# nc_file = './data/FloatingForest/data/trainval.nc'
mmap_imgs = '../shuffled_data_b_cropped/train_aug_imgs.npy'
mmap_mask = '../shuffled_data_b_cropped/train_aug_mask.npy'
batch_size = 48
traindata = MmapDataGenerator(mmap_imgs, mmap_mask, batch_size)

# nc_file_val = './data/FloatingForest/data/test.nc'
mmap_imgs_val = '../shuffled_data_b_cropped/valid_aug_imgs.npy'
mmap_mask_val = '../shuffled_data_b_cropped/valid_aug_mask.npy'
batch_size = 48
val_dl = MmapDataGenerator(mmap_imgs_val, mmap_mask_val, batch_size)

GEN_FILTS = 32
DISC_FILTS = 16
ACTIV = 'relu'

IN_NC = 4
OUT_NC = 1

norm_layer = get_norm_layer()

# create the generator
generator = UNet(IN_NC, OUT_NC, GEN_FILTS, norm_layer=norm_layer,
                 use_dropout=False, activation=ACTIV).to(device)

# create the discriminator
discriminator = Discriminator(IN_NC + OUT_NC, DISC_FILTS, n_layers=3, norm_layer=norm_layer).to(device)

summary(generator, [1, 4, 256, 256])


# create the training object and start training
trainer = Trainer(generator, discriminator,
                  f'checkpoints-{GEN_FILTS}-{DISC_FILTS}-{ACTIV}/')
generator.load_transfer_data(
    '/home/fortson/manth145/codes/patchGAN_FF_ImageNet/patchGAN/FC_checkpoints-64-32-relu/generator_epoch_50.pth'
)
discriminator.load_transfer_data(
    '/home/fortson/manth145/codes/patchGAN_FF_ImageNet/patchGAN/FC_checkpoints-64-32-relu/discriminator_epoch_50.pth'
)

G_loss, D_loss = trainer.train(traindata, val_dl, 200, gen_learning_rate=5.e-4,
                               dsc_learning_rate=1.e-4, lr_decay=0.95)

# save the loss history
np.savez('loss_history.npz', D_loss=D_loss, G_loss=G_loss)
