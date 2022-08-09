import numpy as np
import torch
from torch import optim
from torch import nn
from torchinfo import summary
import tqdm
from patchgan import *

nc_file = './data/FloatingForest/data/trainval.nc'
batch_size= 48
traindata = DataGenerator(nc_file, batch_size)

nc_file_val = './data/FloatingForest/data/test.nc'
batch_size= 48
val_dl = DataGenerator(nc_file, batch_size)


GEN_FILTS  = 32
DISC_FILTS = 32
ACTIV      = 'tanh'

# create the generator
norm_layer = get_norm_layer()
generator = UnetGenerator(4, 1, GEN_FILTS, norm_layer=norm_layer, 
                          use_dropout=False, activation=ACTIV)
generator.apply(weights_init)
generator = generator.cuda()

# create the discriminator
discriminator = Discriminator(5, DISC_FILTS, n_layers=3, norm_layer=norm_layer).cuda()
discriminator.apply(weights_init)

summary(generator, [1, 4, 256, 256])

# create the training object and start training
trainer = Trainer(generator, discriminator, 
                  f'checkpoints-{GEN_FILTS}-{DISC_FILTS}-{ACTIV}/', crop=False)

try:
    trainer.load_last_checkpoint()
except Exception as e:
    raise(e)

G_loss_plot, D_loss_plot = trainer.train(traindata, val_dl, 200, learning_rate=1.e-3)
        
# save the loss history
np.savez('loss_history.npz', D_loss = D_loss_plot ,G_loss = G_loss_plot)

