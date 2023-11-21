import torch
from torchinfo import summary
from patchgan.unet import UNet
from patchgan.disc import Discriminator
from patchgan.io import COCOStuffDataset
from patchgan.trainer import PatchGAN
from torch.utils.data import DataLoader, random_split
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import yaml
import importlib.machinery
import argparse


def patchgan_train():
    parser = argparse.ArgumentParser(
        prog='PatchGAN',
        description='Train the PatchGAN architecture'
    )

    parser.add_argument('-c', '--config_file', required=True, type=str, help='Location of the config YAML file')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='Number of images per batch')
    parser.add_argument('--dataloader_workers', default=4, type=int, help='Number of workers to use with dataloader (set to 0 to disable multithreading)')
    parser.add_argument('-n', '--n_epochs', required=True, type=int, help='Number of epochs to train the model')
    parser.add_argument('-d', '--device', default='auto', help='Device to use to train the model (CUDA=GPU)')
    parser.add_argument('--summary', default=True, action='store_true', help="Print summary of the models")

    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif args.device in ['cuda', 'cpu']:
        device = args.device

    with open(args.config_file, 'r') as infile:
        config = yaml.safe_load(infile)

    dataset_params = config['dataset']
    if ('train_data' in dataset_params) and ('validation_data' in dataset_params):
        train_data_paths = dataset_params['train_data']
        val_data_paths = dataset_params['validation_data']
        train_val_split = None
    elif ('data' in dataset_params) and ('train_val_split' in dataset_params):
        data_paths = dataset_params['data']
        train_val_split = dataset_params['train_val_split']
    else:
        raise AttributeError("Please provide either the training and validation data paths or a train/val split!")

    size = dataset_params.get('size', 256)
    augmentation = dataset_params.get('augmentation', 'randomcrop')

    dataset_kwargs = {}
    if dataset_params['type'] == 'COCOStuff':
        Dataset = COCOStuffDataset
        in_channels = 3
        labels = dataset_params.get('labels', [1])
        out_channels = len(labels)
        dataset_kwargs['labels'] = labels
    else:
        try:
            spec = importlib.machinery.SourceFileLoader('io', 'io.py')
            Dataset = spec.load_module().__getattribute__(dataset_params['type'])
        except FileNotFoundError:
            print("Make sure io.py is in the working directory!")
            raise
        except (ImportError, ModuleNotFoundError):
            print(f"io.py does not contain {dataset_params['type']}")
            raise
        in_channels = dataset_params.get('in_channels', 3)
        out_channels = dataset_params.get('out_channels', 1)

    if train_val_split is None:
        train_datagen = Dataset(train_data_paths['images'], train_data_paths['masks'], size=size, augmentation=augmentation, **dataset_kwargs)
        val_datagen = Dataset(val_data_paths['images'], val_data_paths['masks'], size=size, augmentation=augmentation, **dataset_kwargs)
    else:
        datagen = Dataset(data_paths['images'], data_paths['masks'], size=size, augmentation=augmentation, **dataset_kwargs)
        train_datagen, val_datagen = random_split(datagen, train_val_split)

    dloader_kwargs = {}
    if args.dataloader_workers > 0:
        dloader_kwargs['num_workers'] = args.dataloader_workers
        dloader_kwargs['persistent_workers'] = True

    train_data = DataLoader(train_datagen, batch_size=args.batch_size, shuffle=True, pin_memory=True, **dloader_kwargs)
    val_data = DataLoader(val_datagen, batch_size=args.batch_size, pin_memory=True, **dloader_kwargs)

    checkpoint_path = config.get('checkpoint_path', './checkpoints/')
    model = None
    if config.get('load_last_checkpoint', False):
        model = PatchGAN.load_last_checkpoint(checkpoint_path)

    if model is None:
        model_params = config['model_params']
        generator_config = model_params['generator']
        discriminator_config = model_params['discriminator']

        # get the discriminator and generator configs
        gen_filts = generator_config['filters']
        activation = generator_config['activation']
        use_dropout = generator_config.get('use_dropout', True)
        final_activation = generator_config.get('final_activation', 'sigmoid')
        disc_filts = discriminator_config['filters']
        disc_norm = discriminator_config.get('norm', False)
        n_disc_layers = discriminator_config['n_layers']

        # and the training parameters
        train_params = config['train_params']
        loss_type = train_params['loss_type']
        seg_alpha = train_params['seg_alpha']
        dsc_learning_rate = train_params['disc_learning_rate']
        gen_learning_rate = train_params['gen_learning_rate']
        lr_decay = train_params.get('decay_rate', 0.98)
        decay_freq = train_params.get('decay_freq', 5)
        save_freq = train_params.get('save_freq', 10)
        model = PatchGAN(in_channels, out_channels, gen_filts, disc_filts, final_activation, n_disc_layers, use_dropout,
                         activation, disc_norm, gen_learning_rate, dsc_learning_rate, lr_decay, decay_freq,
                         loss_type=loss_type, seg_alpha=seg_alpha)

    if config.get('transfer_learn', {}).get('checkpoint', None) is not None:
        checkpoint = torch.load(config['transfer_learn']['checkpoint'], map_location=device)
        model.generator.load_transfer_data({key: value for key, value in checkpoint['state_dict'].items() if 'generator' in key})
        model.discriminator.load_transfer_data({key: value for key, value in checkpoint['state_dict'].items() if 'discriminator' in key})

    if args.summary:
        summary(model.generator, [1, in_channels, size, size], depth=4)
        summary(model.discriminator, [1, in_channels + out_channels, size, size])

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path,
                                          filename='patchgan_{epoch:03d}',
                                          save_top_k=-1,
                                          every_n_epochs=save_freq,
                                          verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(accelerator=device, max_epochs=args.n_epochs, callbacks=[checkpoint_callback, lr_monitor])

    trainer.fit(model, train_data, val_data)
