import torch
from torch import nn
import numpy as np
import os
import tqdm
import glob
from torch import optim
from torch.autograd import Variable
from .losses import *
from .utils import crop_images_batch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Trainer:
    '''
        Trainer module which contains both the full training driver
        which calls the train_batch method
    '''

    def __init__(self, generator, discriminator, savefolder, crop=True):
        '''
            Store the generator and discriminator info
        '''
        self.generator     = generator
        self.discriminator = discriminator

        self.savefolder = savefolder
        if not os.path.exists(savefolder):
            os.mkdir(savefolder)

        # flag for cropping the data to the right size
        self.crop = crop

        self.start = 1
        


    def train_batch(self, x, y):
        '''
            Train the generator and discriminator on a single batch
        '''

        # crop the batch randomly to 256x256
        if self.crop:
            input_img, target_img = crop_images_batch(x, y)
        else:
            input_img, target_img = x, y

        # conver the input image and mask to tensors
        input_img = torch.as_tensor(input_img, dtype=torch.float).to(device)
        target_img = torch.as_tensor(target_img, dtype=torch.float).to(device)
        
        # generate the image mask
        generated_image = self.generator(input_img)
        
        
        # Train generator with real labels
        fake_gen = torch.cat((input_img, generated_image), 1)
        
        # get the generator (tversky) loss
        G_loss = generator_loss(generated_image, target_img)                                 

        # train the generator one batch
        self.gen_optimizer.zero_grad()
        G_loss.backward()
        self.gen_optimizer.step()
        
        # Train the discriminator
        disc_inp_fake = torch.cat((input_img, generated_image), 1)
        disc_inp_real = torch.cat((input_img, target_img), 1)

        D_real = self.discriminator(disc_inp_real)
        D_fake = self.discriminator(disc_inp_fake.detach())
 
        try:
            D_fake_loss  = discriminator_loss(D_fake, self.fake_target_train)
            D_real_loss  = discriminator_loss(D_real,  self.real_target_train)
        except Exception as e: 
            D_fake_loss  = discriminator_loss(D_fake, self.fake_target_train[:input_img.shape[0]])
            D_real_loss  = discriminator_loss(D_real,  self.real_target_train[:input_img.shape[0]])

        D_total_loss = D_real_loss + D_fake_loss
        
        self.disc_optimizer.zero_grad()
        D_total_loss.backward()
        self.disc_optimizer.step()

        return G_loss, D_total_loss
    
    def test_batch(self, x, y):
        '''
            Train the generator and discriminator on a single batch
        '''

        # crop the batch randomly to 256x256
        if self.crop:
            input_img, target_img = crop_images_batch(x, y)
        else:
            input_img, target_img = x, y

        # conver the input image and mask to tensors
        input_img = torch.Tensor(input_img).to(device)
        target_img = torch.Tensor(target_img).to(device)
        
        # generate the image mask
        generated_image = self.generator(input_img)
        
        # Train generator with real labels
        fake_gen = torch.cat((input_img, generated_image), 1)
        
        # get the generator (tversky) loss
        G_loss = generator_loss(generated_image, target_img)                                 
        
        # Train the discriminator
        disc_inp_fake = torch.cat((input_img, generated_image), 1)
        disc_inp_real = torch.cat((input_img, target_img), 1)

        D_real = self.discriminator(disc_inp_real)
        D_fake = self.discriminator(disc_inp_fake.detach())
        
        try:
            D_fake_loss  = discriminator_loss(D_fake, self.fake_target_val)
            D_real_loss  = discriminator_loss(D_real,  self.real_target_val)
        except Exception as e: 
            D_fake_loss  = discriminator_loss(D_fake, self.fake_target_val[:input_img.shape[0]])
            D_real_loss  = discriminator_loss(D_real,  self.real_target_val[:input_img.shape[0]])
        
        D_total_loss = D_real_loss + D_fake_loss

        return G_loss.cpu().item(), D_total_loss.cpu().item()
            

    def train(self, train_data, val_data, epochs, learning_rate=2.e-4, validation_freq=5):
        '''
            Training driver which loads the optimizer and calls the `train_batch`
            method. Also handles checkpoint saving
        '''

        # create the Adam optimzers
        self.gen_optimizer  = optim.Adam(self.generator.parameters(), lr = learning_rate)#, betas=(0.5, 0.999))
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr = learning_rate)#, betas=(0.5, 0.999))

        # create the output data for the discriminator
        self.real_target_train = torch.ones(train_data.batch_size, 1, 30, 30).to(device)
        self.fake_target_train = torch.zeros(train_data.batch_size, 1, 30, 30).to(device)
        
        self.real_target_val = torch.ones(val_data.batch_size, 1, 30, 30).to(device)
        self.fake_target_val = torch.zeros(val_data.batch_size, 1, 30, 30).to(device)

        # empty lists for storing epoch loss data
        D_loss_plot, G_loss_plot = [], []
        for epoch in range(self.start, epochs+1): 
          
            # batch loss data
            pbar = tqdm.tqdm(train_data, desc=f'Epoch {epoch}/{epochs}')

            train_data.shuffle()
           
            # set to training mode
            self.generator.train()
            self.discriminator.train()

            D_loss_list = torch.zeros(len(train_data)+1).to(device)
            G_loss_list = torch.zeros(len(train_data)+1).to(device)
            # loop through the training data
            for i, (input_img, target_img) in enumerate(pbar): 
                
                # train on this batch
                gen_loss, disc_loss = self.train_batch(input_img, target_img)

                # append the current batch loss
                D_loss_list[i] = disc_loss.item()
                G_loss_list[i] = gen_loss.item()

                pbar.set_postfix_str(f'gen: {torch.mean(G_loss_list[:i]):.3e} disc {torch.mean(D_loss_list[:i]):.3e}')
                
            # update the epoch loss
            D_loss_plot.append(torch.mean(D_loss_list).cpu().item())
            G_loss_plot.append(torch.mean(G_loss_list).cpu().item())


            if epoch%validation_freq==0:
                # validate every `validation_freq` epochs
                self.discriminator.eval()
                self.generator.eval()
                pbar = tqdm.tqdm(val_data, desc=f'Epoch {epoch} validation')

                val_data.shuffle()
               
                D_loss_list, G_loss_list = [], []
                # loop through the training data
                for (input_img, target_img) in pbar: 
                    
                    # train on this batch
                    gen_loss, disc_loss = self.train_batch(input_img, target_img)

                    # append the current batch loss
                    D_loss_list.append(disc_loss)
                    G_loss_list.append(gen_loss)

                    pbar.set_postfix_str(f'gen: {np.mean(G_loss_list):.3e} disc {np.mean(D_loss_list):.3e}')
             
            # save checkpoints
            torch.save(self.generator.state_dict(), f'{self.savefolder}/generator_epoch_{epoch}.pth')
            torch.save(self.discriminator.state_dict(), f'{self.savefolder}/discriminator_epoch_{epoch}.pth')

        return G_loss_plot, D_loss_plot

    def load_last_checkpoint(self):
        gen_checkpoints  = sorted(glob.glob(self.savefolder+"generator_epoch*.pth"))
        disc_checkpoints = sorted(glob.glob(self.savefolder+"discriminator_epoch*.pth"))

        gen_epochs = set([int(ch.split('/')[-1].replace('generator_epoch_','')[:-4]) for ch in gen_checkpoints])
        dsc_epochs = set([int(ch.split('/')[-1].replace('discriminator_epoch_','')[:-4]) for ch in disc_checkpoints])

        self.start = max(gen_epochs.union(dsc_epochs))

        assert len(gen_epochs) > 0, "No checkpoints found!"

        self.load(f"{self.savefolder}/generator_epoch_{self.start}.pth", f"{self.savefolder}/discriminator_epoch_{self.start}.pth")

    def load(self, generator_save, discriminator_save):
        print(generator_save, discriminator_save)
        self.generator.load_state_dict(torch.load(generator_save))
        self.discriminator.load_state_dict(torch.load(discriminator_save))

        print(f"Loaded checkpoints from {generator_save.split('/')[-1]} and {discriminator_save.split('/')[-1]}")
