import torch
from torch.optim.lr_scheduler import ExponentialLR
from .losses import fc_tversky, bce_loss, MAE_loss
from torch.nn.functional import binary_cross_entropy
from .unet import UNet
from .disc import Discriminator
from typing import Union, Optional
import lightning as L


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PatchGAN(L.LightningModule):
    def __init__(self, input_channels: int, output_channels: int, gen_filts: int, disc_filts: int, final_activation: str,
                 n_disc_layers: int = 5, use_gen_dropout: bool = True, gen_activation: str = 'leakyrelu',
                 disc_norm: bool = False, gen_lr: float = 1.e-3, dsc_lr: float = 1.e-3, lr_decay: float = 0.98,
                 decay_freq: int = 5, adam_b1: float = 0.5, adam_b2: float = 0.999, seg_alpha: float = 200,
                 loss_type: str = 'tversky', tversky_beta: float = 0.75, tversky_gamma: float = 0.75):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.generator = UNet(input_channels, output_channels, gen_filts, use_dropout=use_gen_dropout,
                              activation=gen_activation, final_act=final_activation)
        self.discriminator = Discriminator(input_channels + output_channels, disc_filts,
                                           norm=disc_norm, n_layers=n_disc_layers)

    def forward(self, img, return_hidden=False):
        return self.generator(img, return_hidden)

    def training_step(self, batch):
        '''
            Train the generator and discriminator on a single batch
        '''
        optimizer_g, optimizer_d = self.optimizers()

        mean_loss = self.batch_step(batch, True, optimizer_g, optimizer_d)

        sch_g, sch_d = self.lr_schedulers()
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % self.hparams.decay_freq == 0:
            sch_g.step()
            sch_d.step()

        for key, val in mean_loss.items():
            self.log(key, val, prog_bar=True, on_epoch=True, reduce_fx=torch.mean)

    def validation_step(self, batch):
        mean_loss = self.batch_step(batch, False)

        for key, val in mean_loss.items():
            self.log(key, val, prog_bar=True, on_epoch=True, reduce_fx=torch.mean)

    def batch_step(self, batch: Union[torch.Tensor, tuple[torch.Tensor]], train: bool,
                   optimizer_g: Optional[torch.optim.Optimizer] = None,
                   optimizer_d: Optional[torch.optim.Optimizer] = None):
        input_tensor, target_tensor = batch

        # train the generator
        gen_img = self.generator(input_tensor)

        disc_inp_fake = torch.cat((input_tensor, gen_img), 1)
        disc_fake = self.discriminator(disc_inp_fake)

        labels_real = torch.full(disc_fake.shape, 1, dtype=torch.float, device=device)
        labels_fake = torch.full(disc_fake.shape, 0, dtype=torch.float, device=device)

        if self.hparams.loss_type == 'tversky':
            gen_loss = fc_tversky(target_tensor, gen_img,
                                  beta=self.hparams.tversky_beta,
                                  gamma=self.hparams.tversky_gamma) * self.hparams.seg_alpha
        elif self.hparams.loss_type == 'weighted_bce':
            if gen_img.shape[1] > 1:
                weight = 1 - torch.sum(target_tensor, dim=(2, 3), keepdim=True) / torch.sum(target_tensor)
            else:
                weight = torch.ones_like(target_tensor)
            gen_loss = binary_cross_entropy(gen_img, target_tensor, weight=weight) * self.hparams.seg_alpha
        elif self.hparams.loss_type == 'MAE':
            gen_loss = MAE_loss(gen_img, target_tensor) * self.hparams.seg_alpha

        gen_loss_disc = bce_loss(disc_fake, labels_real)
        gen_loss = gen_loss + gen_loss_disc

        if train:
            self.toggle_optimizer(optimizer_g)
            optimizer_g.zero_grad()
            self.manual_backward(gen_loss)
            optimizer_g.step()
            self.untoggle_optimizer(optimizer_g)

        # Train the discriminator
        if train:
            self.toggle_optimizer(optimizer_d)

        disc_inp_real = torch.cat((input_tensor, target_tensor), 1)
        disc_real = self.discriminator(disc_inp_real)
        disc_inp_fake = torch.cat((input_tensor, gen_img.detach()), 1)
        disc_fake = self.discriminator(disc_inp_fake)

        loss_real = bce_loss(disc_real, labels_real.detach())
        loss_fake = bce_loss(disc_fake, labels_fake)
        disc_loss = (loss_fake + loss_real) / 2.

        if train:
            optimizer_d.zero_grad()
            self.manual_backward(disc_loss)
            optimizer_d.step()
            self.untoggle_optimizer(optimizer_d)

        keys = ['gen', 'gen_loss', 'gdisc', 'discr', 'discf', 'disc']
        mean_loss_i = [gen_loss.item(), gen_loss.item(), gen_loss_disc.item(),
                       loss_real.item(), loss_fake.item(), disc_loss.item()]

        loss = {key: val for key, val in zip(keys, mean_loss_i)}

        return loss

    def configure_optimizers(self):
        gen_lr = self.hparams.gen_lr
        dsc_lr = self.hparams.dsc_lr

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=gen_lr, betas=(self.hparams.adam_b1, self.hparams.adam_b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=dsc_lr, betas=(self.hparams.adam_b1, self.hparams.adam_b2))

        gen_lr_scheduler = ExponentialLR(opt_g, gamma=self.hparams.lr_decay)
        dsc_lr_scheduler = ExponentialLR(opt_d, gamma=self.hparams.lr_decay)

        gen_lr_scheduler_config = {"scheduler": gen_lr_scheduler,
                                   "interval": "epoch",
                                   "frequency": self.hparams.decay_freq}

        dsc_lr_scheduler_config = {"scheduler": dsc_lr_scheduler,
                                   "interval": "epoch",
                                   "frequency": self.hparams.decay_freq}

        return [{"optimizer": opt_g, "lr_scheduler": gen_lr_scheduler_config},
                {"optimizer": opt_d, "lr_scheduler": dsc_lr_scheduler_config}]
