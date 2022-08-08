import torch
from torch import nn
import numpy as np
import os
import tqdm
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

    def __init__(self, generator, discriminator, savefolder):
        '''
            Store the generator and discriminator info
        '''
        self.generator     = generator
        self.discriminator = discriminator

        self.savefolder = savefolder
        if not os.path.exists(savefolder):
            os.mkdir(savefolder)


    def train_batch(self, x, y):
        '''
            Train the generator and discriminator on a single batch
        '''

        # crop the batch randomly to 256x256
        input_img, target_img = crop_images_batch(x, y)

        # conver the input image and mask to tensors
        input_img = torch.Tensor(input_img).to(device).float()
        target_img = torch.Tensor(target_img).to(device).float()
        
        # generate the image mask
        generated_image = self.generator(input_img)
        
        # create the output data for the discriminator
        real_target = Variable(torch.ones(input_img.size(0), 1, 30, 30).to(device))
        fake_target = Variable(torch.zeros(input_img.size(0), 1, 30, 30).to(device))
        
        # Train generator with real labels
        fake_gen = torch.cat((input_img, generated_image), 1)
        
        # get the generator (tversky) loss
        G_loss = generator_loss(generated_image, target_img)                                 

        # train the generator one batch
        self.gen_optimizer.zero_grad()
        G_loss.backward()
        self.gen_optimizer.step()
        
        # Train the discriminator
        disc_inp_fake = torch.cat((input_img, generated_image), 1).to(device).float()
        disc_inp_real = torch.cat((input_img, target_img), 1)

        output = self.discriminator(disc_inp_real)
        D_fake = self.discriminator(disc_inp_fake.detach())
        
        D_fake_loss  = discriminator_loss(D_fake, fake_target)
        D_real_loss  = discriminator_loss(output,  real_target)

        D_total_loss = (D_real_loss + D_fake_loss) / 2
        
        self.disc_optimizer.zero_grad()
        D_total_loss.backward()
        self.disc_optimizer.step()

        return G_loss.cpu().item(), D_total_loss.cpu().item()
    
    def test_batch(self, x, y):
        '''
            Train the generator and discriminator on a single batch
        '''

        # crop the batch randomly to 256x256
        input_img, target_img = crop_images_batch(x, y)

        # conver the input image and mask to tensors
        input_img = torch.Tensor(input_img).to(device).float()
        target_img = torch.Tensor(target_img).to(device).float()
        
        # generate the image mask
        generated_image = self.generator(input_img)
        
        # create the output data for the discriminator
        real_target = Variable(torch.ones(input_img.size(0), 1, 30, 30).to(device))
        fake_target = Variable(torch.zeros(input_img.size(0), 1, 30, 30).to(device))
        
        # Train generator with real labels
        fake_gen = torch.cat((input_img, generated_image), 1)
        
        # get the generator (tversky) loss
        G_loss = generator_loss(generated_image, target_img)                                 
        
        # Train the discriminator
        disc_inp_fake = torch.cat((input_img, generated_image), 1).to(device).float()
        disc_inp_real = torch.cat((input_img, target_img), 1)

        output = self.discriminator(disc_inp_real)
        D_fake = self.discriminator(disc_inp_fake.detach())
        
        D_fake_loss  = discriminator_loss(D_fake, fake_target)
        D_real_loss  = discriminator_loss(output,  real_target)

        return G_loss.cpu().item(), D_total_loss.cpu().item()
            

    def train(self, train_data, val_data, epochs, learning_rate=2.e-4, validation_freq=5):
        '''
            Training driver which loads the optimizer and calls the `train_batch`
            method. Also handles checkpoint saving
        '''

        # create the Adam optimzers
        self.gen_optimizer  = optim.Adam(self.generator.parameters(), lr = learning_rate)#, betas=(0.5, 0.999))
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr = learning_rate)#, betas=(0.5, 0.999))

        # empty lists for storing epoch loss data
        D_loss_plot, G_loss_plot = [], []
        for epoch in range(1, epochs+1): 
          
            # batch loss data

            pbar = tqdm.tqdm(train_data, desc=f'Epoch {epoch}/{epochs}')

            train_data.shuffle()
           
            # set to training mode
            self.generator.train()
            self.discriminator.train()

            D_loss_list, G_loss_list = [], []
            # loop through the training data
            for (input_img, target_img) in pbar: 
                
                # train on this batch
                gen_loss, disc_loss = self.train_batch(input_img, target_img)

                # append the current batch loss
                D_loss_list.append(disc_loss)
                G_loss_list.append(gen_loss)

                pbar.set_postfix_str(f'gen: {np.mean(G_loss_list):.3e} disc {np.mean(D_loss_list):.3e}')
                
            # update the epoch loss
            D_loss_plot.append(np.mean(D_loss_list))
            G_loss_plot.append(np.mean(G_loss_list))


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
