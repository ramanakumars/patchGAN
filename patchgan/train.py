import torch
from torchinfo import summary
from patchgan.unet import UNet
from patchgan.disc import Discriminator
from patchgan.io import COCOStuffDataset
from patchgan.trainer import Trainer
from torch.utils.data import DataLoader, random_split
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

    model_params = config['model_params']
    gen_filts = model_params['gen_filts']
    disc_filts = model_params['disc_filts']
    n_disc_layers = model_params['n_disc_layers']
    activation = model_params['activation']
    use_dropout = model_params.get('use_dropout', True)
    final_activation = model_params.get('final_activation', 'sigmoid')

    dloader_kwargs = {}
    if args.dataloader_workers > 0:
        dloader_kwargs['num_workers'] = args.dataloader_workers
        dloader_kwargs['persistent_workers'] = True

    train_data = DataLoader(train_datagen, batch_size=args.batch_size, shuffle=True, pin_memory=True, **dloader_kwargs)
    val_data = DataLoader(val_datagen, batch_size=args.batch_size, shuffle=True, pin_memory=True, **dloader_kwargs)

    # create the generator
    generator = UNet(in_channels, out_channels, gen_filts, use_dropout=use_dropout, activation=activation, final_act=final_activation).to(device)

    # create the discriminator
    discriminator = Discriminator(in_channels + out_channels, disc_filts, n_layers=n_disc_layers).to(device)

    if args.summary:
        summary(generator, [1, in_channels, size, size])
        summary(discriminator, [1, in_channels + out_channels, size, size])

    checkpoint_path = config.get('checkpoint_path', './checkpoints/')

    trainer = Trainer(generator, discriminator, savefolder=checkpoint_path)

    if config.get('load_last_checkpoint', False):
        trainer.load_last_checkpoint()
    elif config.get('transfer_learn', {}).get('generator_checkpoint', None) is not None:
        gen_checkpoint = config['transfer_learn']['generator_checkpoint']
        dsc_checkpoint = config['transfer_learn']['discriminator_checkpoint']
        generator.load_transfer_data(torch.load(gen_checkpoint, map_location=device))
        discriminator.load_transfer_data(torch.load(dsc_checkpoint, map_location=device))

    train_params = config['train_params']

    trainer.loss_type = train_params['loss_type']
    trainer.seg_alpha = train_params['seg_alpha']

    trainer.train(train_data, val_data, args.n_epochs,
                  dsc_learning_rate=train_params['disc_learning_rate'],
                  gen_learning_rate=train_params['gen_learning_rate'],
                  lr_decay=train_params.get('decay_rate', None),
                  save_freq=train_params.get('save_freq', 10))
