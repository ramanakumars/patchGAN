import torch
import os
import tqdm
import glob
import numpy as np
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from .losses import fc_tversky, bce_loss
from torch.nn.functional import binary_cross_entropy
from collections import defaultdict


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Trainer:
    '''
        Trainer module which contains both the full training driver
        which calls the train_batch method
    '''

    seg_alpha = 200
    loss_type = 'tversky'
    tversky_beta = 0.75
    tversky_gamma = 0.75

    neptune_config = None

    def __init__(self, generator, discriminator, savefolder, device='cuda'):
        '''
            Store the generator and discriminator info
        '''

        generator.apply(weights_init)
        discriminator.apply(weights_init)

        self.generator = generator
        self.discriminator = discriminator
        self.device = device

        if savefolder[-1] != '/':
            savefolder += '/'

        self.savefolder = savefolder
        if not os.path.exists(savefolder):
            os.mkdir(savefolder)

        self.start = 1

    def batch(self, x, y, train=False):
        '''
            Train the generator and discriminator on a single batch
        '''

        if not isinstance(x, torch.Tensor):
            input_tensor = torch.as_tensor(x, dtype=torch.float).to(self.device)
            target_tensor = torch.as_tensor(y, dtype=torch.float).to(self.device)
        else:
            input_tensor = x.to(self.device, non_blocking=True)
            target_tensor = y.to(self.device, non_blocking=True)

        # train the generator
        gen_img = self.generator(input_tensor)

        disc_inp_fake = torch.cat((input_tensor, gen_img), 1)
        disc_fake = self.discriminator(disc_inp_fake)

        labels_real = torch.full(disc_fake.shape, 1, dtype=torch.float, device=device)
        labels_fake = torch.full(disc_fake.shape, 0, dtype=torch.float, device=device)

        if self.loss_type == 'tversky':
            gen_loss = fc_tversky(target_tensor, gen_img,
                                  beta=self.tversky_beta,
                                  gamma=self.tversky_gamma) * self.seg_alpha
        elif self.loss_type == 'weighted_bce':
            if gen_img.shape[1] > 1:
                weight = 1 - torch.sum(target_tensor, dim=(2, 3), keepdim=True) / torch.sum(target_tensor)
            else:
                weight = torch.ones_like(target_tensor)
            gen_loss = binary_cross_entropy(gen_img, target_tensor, weight=weight) * self.seg_alpha

        gen_loss_disc = bce_loss(disc_fake, labels_real)
        gen_loss = gen_loss + gen_loss_disc

        if train:
            self.generator.zero_grad()
            gen_loss.backward()
            self.gen_optimizer.step()

        # Train the discriminator
        if train:
            self.discriminator.zero_grad()

        disc_inp_real = torch.cat((input_tensor, target_tensor), 1)
        disc_real = self.discriminator(disc_inp_real)
        disc_inp_fake = torch.cat((input_tensor, gen_img.detach()), 1)
        disc_fake = self.discriminator(disc_inp_fake)

        loss_real = bce_loss(disc_real, labels_real.detach())
        loss_fake = bce_loss(disc_fake, labels_fake)
        disc_loss = (loss_fake + loss_real) / 2.

        if train:
            disc_loss.backward()
            self.disc_optimizer.step()

        keys = ['gen', 'gen_loss', 'gdisc', 'discr', 'discf', 'disc']
        mean_loss_i = [gen_loss.item(), gen_loss.item(), gen_loss_disc.item(),
                       loss_real.item(), loss_fake.item(), disc_loss.item()]

        loss = {key: val for key, val in zip(keys, mean_loss_i)}

        return loss

    def train(self, train_data, val_data, epochs, dsc_learning_rate=1.e-3,
              gen_learning_rate=1.e-3, save_freq=10, lr_decay=None, decay_freq=5,
              reduce_on_plateau=False):
        '''
            Training driver which loads the optimizer and calls the
            `train_batch` method. Also handles checkpoint saving
            Inputs
            ------
            train_data : DataLoader object
                Training data that is mapped using the DataLoader or
                MmapDataLoader object defined in io.py
            val_data : DataLoader object
                Validation data loaded in using the DataLoader or
                MmapDataLoader object
            epochs : int
                Number of epochs to run the model
            dsc_learning_rate : float [default: 1e-4]
                Initial learning rate for the discriminator
            gen_learning_rate : float [default: 1e-3]
                Initial learning rate for the generator
            save_freq : int [default: 10]
                Frequency at which to save checkpoints to the save folder
            lr_decay : float [default: None]
                Learning rate decay rate (ratio of new learning rate
                to previous). A value of 0.95, for example, would set the
                new LR to 95% of the previous value
            decay_freq : int [default: 5]
                Frequency at which to decay the learning rate. For example,
                a value of for decay_freq and 0.95 for lr_decay would decay
                the learning to 95% of the current value every 5 epochs.
            Outputs
            -------
            G_loss_plot : numpy.ndarray
                Generator loss history as a function of the epochs
            D_loss_plot : numpy.ndarray
                Discriminator loss history as a function of the epochs
        '''

        if (lr_decay is not None) and not reduce_on_plateau:
            gen_lr = gen_learning_rate * (lr_decay)**((self.start - 1) / (decay_freq))
            dsc_lr = dsc_learning_rate * (lr_decay)**((self.start - 1) / (decay_freq))
        else:
            gen_lr = gen_learning_rate
            dsc_lr = dsc_learning_rate

        if self.neptune_config is not None:
            self.neptune_config['model/parameters/gen_learning_rate'] = gen_lr
            self.neptune_config['model/parameters/dsc_learning_rate'] = dsc_lr
            self.neptune_config['model/parameters/start'] = self.start
            self.neptune_config['model/parameters/n_epochs'] = epochs

        # create the Adam optimzers
        self.gen_optimizer = optim.NAdam(
            self.generator.parameters(), lr=gen_lr, betas=(0.9, 0.999))
        self.disc_optimizer = optim.NAdam(
            self.discriminator.parameters(), lr=dsc_lr, betas=(0.9, 0.999))

        # set up the learning rate scheduler with exponential lr decay
        if reduce_on_plateau:
            gen_scheduler = ReduceLROnPlateau(self.gen_optimizer, verbose=True)
            dsc_scheduler = ReduceLROnPlateau(self.disc_optimizer, verbose=True)
            self.neptune_config['model/parameters/scheduler'] = 'ReduceLROnPlateau'
        elif lr_decay is not None:
            gen_scheduler = ExponentialLR(self.gen_optimizer, gamma=lr_decay)
            dsc_scheduler = ExponentialLR(self.disc_optimizer, gamma=lr_decay)
            if self.neptune_config is not None:
                self.neptune_config['model/parameters/scheduler'] = 'ExponentialLR'
                self.neptune_config['model/parameters/decay_freq'] = decay_freq
                self.neptune_config['model/parameters/lr_decay'] = lr_decay
        else:
            gen_scheduler = None
            dsc_scheduler = None

        # empty lists for storing epoch loss data
        D_loss_ep, G_loss_ep = [], []
        for epoch in range(self.start, epochs + 1):
            if isinstance(gen_scheduler, ExponentialLR):
                gen_lr = gen_scheduler.get_last_lr()[0]
                dsc_lr = dsc_scheduler.get_last_lr()[0]
            else:
                gen_lr = gen_learning_rate
                dsc_lr = dsc_learning_rate

            print(f"Epoch {epoch} -- lr: {gen_lr:5.3e}, {dsc_lr:5.3e}")
            print("-------------------------------------------------------")

            # batch loss data
            pbar = tqdm.tqdm(train_data, desc='Training: ', dynamic_ncols=True)

            if hasattr(train_data, 'shuffle'):
                train_data.shuffle()

            # set to training mode
            self.generator.train()
            self.discriminator.train()

            losses = defaultdict(list)
            # loop through the training data
            for i, (input_img, target_mask) in enumerate(pbar):

                # train on this batch
                batch_loss = self.batch(input_img, target_mask, train=True)

                # append the current batch loss
                loss_mean = {}
                for key, value in batch_loss.items():
                    losses[key].append(value)
                    loss_mean[key] = np.mean(losses[key], axis=0)

                loss_str = " ".join([f"{key}: {value:.2e}" for key, value in loss_mean.items()])

                pbar.set_postfix_str(loss_str)

            # update the epoch loss
            D_loss_ep.append(loss_mean['disc'])
            G_loss_ep.append(loss_mean['gen'])

            if self.neptune_config is not None:
                self.neptune_config['train/gen_loss'].append(loss_mean['gen'])
                self.neptune_config['train/disc_loss'].append(loss_mean['disc'])

            # validate every `validation_freq` epochs
            self.discriminator.eval()
            self.generator.eval()
            pbar = tqdm.tqdm(val_data, desc='Validation: ')

            if hasattr(val_data, 'shuffle'):
                val_data.shuffle()

            losses = defaultdict(list)
            # loop through the training data
            for i, (input_img, target_mask) in enumerate(pbar):
                # validate on this batch
                batch_loss = self.batch(input_img, target_mask, train=False)

                loss_mean = {}
                for key, value in batch_loss.items():
                    losses[key].append(value)
                    loss_mean[key] = np.mean(losses[key], axis=0)

                loss_str = " ".join([f"{key}: {value:.2e}" for key, value in loss_mean.items()])

                pbar.set_postfix_str(loss_str)

            if self.neptune_config is not None:
                self.neptune_config['eval/gen_loss'].append(loss_mean['gen'])
                self.neptune_config['eval/disc_loss'].append(loss_mean['disc'])

            # apply learning rate decay
            if (gen_scheduler is not None) & (dsc_scheduler is not None):
                if isinstance(gen_scheduler, ExponentialLR):
                    if epoch % decay_freq == 0:
                        gen_scheduler.step()
                        dsc_scheduler.step()
                else:
                    gen_scheduler.step(loss_mean['gen'])
                    dsc_scheduler.step(loss_mean['disc'])

            # save checkpoints
            if epoch % save_freq == 0:
                self.save(epoch)

        return G_loss_ep, D_loss_ep

    def save(self, epoch):
        gen_savefile = f'{self.savefolder}/generator_ep_{epoch:03d}.pth'
        disc_savefile = f'{self.savefolder}/discriminator_ep_{epoch:03d}.pth'

        print(f"Saving to {gen_savefile} and {disc_savefile}")
        torch.save(self.generator.state_dict(), gen_savefile)
        torch.save(self.discriminator.state_dict(), disc_savefile)

    def load_last_checkpoint(self):
        gen_checkpoints = sorted(
            glob.glob(self.savefolder + "generator_ep*.pth"))
        disc_checkpoints = sorted(
            glob.glob(self.savefolder + "discriminator_ep*.pth"))

        gen_epochs = set([int(ch.split(
            '/')[-1].replace('generator_ep_', '')[:-4]) for
            ch in gen_checkpoints])
        dsc_epochs = set([int(ch.split(
            '/')[-1].replace('discriminator_ep_', '')[:-4]) for
            ch in disc_checkpoints])

        try:
            assert len(gen_epochs) > 0, "No checkpoints found!"

            start = max(gen_epochs.union(dsc_epochs))
            self.load(f"{self.savefolder}/generator_ep_{start:03d}.pth",
                      f"{self.savefolder}/discriminator_ep_{start:03d}.pth")
            self.start = start + 1
        except Exception as e:
            print(e)
            print("Checkpoints not loaded")

    def load(self, generator_save, discriminator_save):
        print(generator_save, discriminator_save)
        self.generator.load_state_dict(torch.load(generator_save))
        self.discriminator.load_state_dict(torch.load(discriminator_save))

        gfname = generator_save.split('/')[-1]
        dfname = discriminator_save.split('/')[-1]
        print(
            f"Loaded checkpoints from {gfname} and {dfname}")

# custom weights initialization called on generator and discriminator
# scaling here means std


def weights_init(net, init_type='normal', scaling=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')) != -1:
            torch.nn.init.xavier_uniform_(m.weight.data)
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.xavier_uniform_(m.weight.data, 1.0)
            torch.nn.init.constant_(m.bias.data, 0.0)
