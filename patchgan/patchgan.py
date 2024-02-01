import torch
from typing import Iterable
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from .losses import fc_tversky, bce_loss, MAE_loss
from torch.nn.functional import binary_cross_entropy, one_hot
from .disc import Discriminator
from .point_encoder import PointEncoder
from .conv_layers import Encoder, Decoder
from typing import Union, Optional
import lightning as L


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PatchGAN(L.LightningModule):
    def __init__(self, input_channels: int, output_channels: int, gen_filts: int, disc_filts: int, final_activation: str,
                 n_disc_layers: int = 5, use_gen_dropout: bool = True, gen_activation: str = 'leakyrelu',
                 disc_norm: bool = False, gen_lr: float = 1.e-3, dsc_lr: float = 1.e-3, lr_decay: float = 0.98,
                 decay_freq: int = 5, adam_b1: float = 0.9, adam_b2: float = 0.999, seg_alpha: float = 200,
                 loss_type: str = 'tversky', tversky_beta: float = 0.75, tversky_gamma: float = 0.75):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.encoder = Encoder(input_channels, gen_filts, gen_activation, use_gen_dropout)
        self.decoder = Decoder(output_channels, gen_filts, gen_activation, final_activation, use_gen_dropout)

        self.discriminator = Discriminator(input_channels + output_channels, disc_filts,
                                           norm=disc_norm, n_layers=n_disc_layers)

    @classmethod
    def load_transfer_data(cls, checkpoint_path: str, input_channels: int, output_channels: int):
        checkpoint = torch.load(checkpoint_path)
        model_kwargs = checkpoint['hyperparameters']
        model_kwargs['input_channels'] = input_channels
        model_kwargs['output_channels'] = output_channels
        obj = cls(**model_kwargs)

        raise ValueError("weights loading not implemented!")

        return obj

    def forward(self, x):
        xencs = []

        for i, layer in enumerate(self.encoder):
            x = layer(x)
            xencs.append(x)

        hidden = xencs[-1]

        xencs = xencs[::-1]

        for i, layer in enumerate(self.decoder):
            if i == 0:
                xinp = hidden
            else:
                xinp = torch.cat([x, xencs[i]], dim=1)

            x = layer(xinp)

        return x

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

    def forward_batch(self, batch):
        input_tensor, target_tensor = batch
        return self(input_tensor), input_tensor, target_tensor

    def batch_step(self, batch: Union[torch.Tensor, tuple[torch.Tensor]], train: bool,
                   optimizer_g: Optional[torch.optim.Optimizer] = None,
                   optimizer_d: Optional[torch.optim.Optimizer] = None):
        # train the generator
        gen_img, input_tensor, target_tensor = self.forward_batch(batch)

        disc_inp_fake = torch.cat((input_tensor, gen_img), 1)
        disc_fake = self.discriminator(disc_inp_fake)

        labels_real = torch.full(disc_fake.shape, 1, dtype=torch.float, device=device)
        labels_fake = torch.full(disc_fake.shape, 0, dtype=torch.float, device=device)

        target_tensor_full = one_hot(target_tensor, self.hparams.output_channels).permute(0, 3, 1, 2).to(torch.float)

        if self.hparams.loss_type == 'tversky':
            gen_loss = fc_tversky(target_tensor_full, gen_img,
                                  beta=self.hparams.tversky_beta,
                                  gamma=self.hparams.tversky_gamma) * self.hparams.seg_alpha
        elif self.hparams.loss_type == 'weighted_bce':
            if gen_img.shape[1] > 1:
                weight = 1 - torch.sum(target_tensor_full, dim=(2, 3), keepdim=True) / (torch.sum(target_tensor_full) + 1.e-6)
            else:
                weight = torch.ones_like(target_tensor_full)
            gen_loss = binary_cross_entropy(gen_img, target_tensor_full, weight=weight) * self.hparams.seg_alpha
        elif self.hparams.loss_type == 'MAE':
            gen_loss = MAE_loss(gen_img, target_tensor_full) * self.hparams.seg_alpha

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

        disc_inp_real = torch.cat((input_tensor, target_tensor_full), 1)
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

    def get_parameters(self) -> tuple[Iterable[nn.Parameter]]:
        return list(self.encoder.parameters()) + list(self.decoder.parameters()), self.discriminator.parameters()

    def configure_optimizers(self):
        gen_lr = self.hparams.gen_lr
        dsc_lr = self.hparams.dsc_lr

        generator_params, discriminator_params = self.get_parameters()

        opt_g = torch.optim.Adam(generator_params, lr=gen_lr, betas=(self.hparams.adam_b1, self.hparams.adam_b2))
        opt_d = torch.optim.Adam(discriminator_params, lr=dsc_lr, betas=(self.hparams.adam_b1, self.hparams.adam_b2))

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


class PatchGANPoint(PatchGAN):
    def __init__(self, *patchgan_args, **patchgan_kwargs):
        super().__init__(*patchgan_args, **patchgan_kwargs)

        # create the point encoder to attach to the latent block
        # by default use the final number of filters in the UNet (hard-coded to gen_filts * 8)
        self.point_encoder = PointEncoder(self.hparams.gen_filts * 8)

    def get_parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.point_encoder.parameters()), self.discriminator.parameters()

    def forward(self, x, point):
        xencs = []

        for i, layer in enumerate(self.encoder):
            x = layer(x)
            xencs.append(x)

        hidden = xencs[-1]

        _, _, h, w = hidden.shape
        point_mask = self.point_encoder(point).unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w)

        hidden = hidden * point_mask

        xencs = xencs[::-1]

        for i, layer in enumerate(self.decoder):
            if i == 0:
                xinp = hidden
            else:
                xinp = torch.cat([x, xencs[i]], dim=1)

            x = layer(xinp)

        return x

    def forward_batch(self, batch):
        input_tensor, point, target_tensor = batch
        return self(input_tensor, point), input_tensor, target_tensor
