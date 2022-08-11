import numpy as np
import torch
from torch import optim
from torch import nn
from torchinfo import summary
import tqdm
from patchgan import *

# nc_file = './data/FloatingForest/data/trainval.nc'
mmap_imgs = '/dev/shm/shuffled_data_b_cropped/train_aug_imgs.npy'
mmap_mask = '/dev/shm/shuffled_data_b_cropped/train_aug_mask.npy'
batch_size= 256
traindata = MmapDataGenerator(mmap_imgs, mmap_mask, batch_size)

# nc_file_val = './data/FloatingForest/data/test.nc'
mmap_imgs_val = '/dev/shm/shuffled_data_b_cropped/valid_aug_imgs.npy'
mmap_mask_val = '/dev/shm/shuffled_data_b_cropped/valid_aug_mask.npy'
batch_size= 256
val_dl = MmapDataGenerator(mmap_imgs_val, mmap_mask_val, batch_size)

GEN_FILTS  = 64
DISC_FILTS = 32
ACTIV      = 'relu'

# create the generator
norm_layer = get_norm_layer()
generator = UnetGenerator(4, 1, GEN_FILTS, norm_layer=norm_layer, 
                          use_dropout=False, activation=ACTIV).cuda()
generator.apply(weights_init)
#generator = generator.cuda()

generator.load_transfer_data(torch.load(\
    '/home/fortson/manth145/codes/patchGAN_FF_ImageNet/patchGAN/FC_checkpoints-64-32-relu/generator_epoch_50.pth')
    )

# create the discriminator
discriminator = Discriminator(5, DISC_FILTS, n_layers=3, norm_layer=norm_layer).cuda()
discriminator.apply(weights_init)
generator.load_transfer_data(torch.load('/home/fortson/manth145/codes/patchGAN_FF_ImageNet/patchGAN/FC_checkpoints-64-32-relu/discriminator_epoch_50.pth'))

#print(generator)

# create the training object and start training
trainer = Trainer(generator, discriminator, 
                  f'transferFC_checkpoints-{GEN_FILTS}-{DISC_FILTS}-{ACTIV}/', crop=False)

G_loss_plot, D_loss_plot = trainer.train(traindata, val_dl, 50, gen_learning_rate=5.e-4, 
                                        dsc_learning_rate=1.e-4, lr_decay=0.95)
        
# save the loss history
np.savez('loss_history.npz', D_loss = D_loss_plot ,G_loss = G_loss_plot)

