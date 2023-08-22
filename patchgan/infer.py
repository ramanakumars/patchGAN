import torch
from torchinfo import summary
from patchgan.unet import UNet
from patchgan.disc import Discriminator
from patchgan.io import COCOStuffDataset
import yaml
import tqdm
import os
import numpy as np
import importlib.machinery
import argparse


def n_crop(image, size, overlap):
    c, height, width = image.shape

    effective_size = int(overlap * size)

    ncropsy = int(np.ceil(height / effective_size))
    ncropsx = int(np.ceil(width / effective_size))

    crops = torch.zeros((ncropsx * ncropsy, c, size, size), device=image.device)

    for j in range(ncropsy):
        for i in range(ncropsx):
            starty = j * effective_size
            startx = i * effective_size

            starty -= max([starty + size - height, 0])
            startx -= max([startx + size - width, 0])

            crops[j * ncropsy + i, :] = image[:, starty:starty + size, startx:startx + size]

    return crops


def build_mask(masks, crop_size, image_size, threshold, overlap):
    n, c, height, width = masks.shape
    image_height, image_width = image_size
    mask = np.zeros((c, *image_size))
    count = np.zeros((c, *image_size))

    effective_size = int(overlap * crop_size)

    ncropsy = int(np.ceil(image_height / effective_size))
    ncropsx = int(np.ceil(image_width / effective_size))

    for j in range(ncropsy):
        for i in range(ncropsx):
            starty = j * effective_size
            startx = i * effective_size
            starty -= max([starty + crop_size - image_height, 0])
            startx -= max([startx + crop_size - image_width, 0])
            endy = starty + crop_size
            endx = startx + crop_size

            mask[:, starty:endy, startx:endx] += masks[j * ncropsy + i, :]
            count[:, starty:endy, startx:endx] += 1
    mask = mask / count

    if threshold > 0:
        mask[mask >= threshold] = 1
        mask[mask < threshold] = 0

    if c > 1:
        return np.argmax(mask, axis=0)
    else:
        return mask[0]


def patchgan_infer():
    parser = argparse.ArgumentParser(
        prog='PatchGAN',
        description='Train the PatchGAN architecture'
    )

    parser.add_argument('-c', '--config_file', required=True, type=str, help='Location of the config YAML file')
    parser.add_argument('--dataloader_workers', default=4, type=int, help='Number of workers to use with dataloader (set to 0 to disable multithreading)')
    parser.add_argument('-d', '--device', default='auto', help='Device to use to train the model (CUDA=GPU)')
    parser.add_argument('--summary', default=True, action='store_true', help="Print summary of the models")

    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif args.device in ['cuda', 'cpu']:
        device = args.device

    print(f"Running with {device}")

    with open(args.config_file, 'r') as infile:
        config = yaml.safe_load(infile)

    dataset_params = config['dataset']
    dataset_path = dataset_params['dataset_path']

    size = dataset_params.get('size', 256)

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

    assert hasattr(Dataset, 'get_filename') and callable(Dataset.get_filename), \
        f"Dataset class {Dataset.__name__} must have the get_filename method which returns the image filename for a given index"

    datagen = Dataset(dataset_path, **dataset_kwargs)

    model_params = config['model_params']
    gen_filts = model_params['gen_filts']
    disc_filts = model_params['disc_filts']
    n_disc_layers = model_params['n_disc_layers']
    activation = model_params['activation']
    final_activation = model_params.get('final_activation', 'sigmoid')

    # create the generator
    generator = UNet(in_channels, out_channels, gen_filts, activation=activation, final_act=final_activation).to(device)

    # create the discriminator
    discriminator = Discriminator(in_channels + out_channels, disc_filts, n_layers=n_disc_layers).to(device)

    if args.summary:
        summary(generator, [1, in_channels, size, size], device=device)
        summary(discriminator, [1, in_channels + out_channels, size, size], device=device)

    checkpoint_paths = config['checkpoint_paths']
    gen_checkpoint = checkpoint_paths['generator']
    dsc_checkpoint = checkpoint_paths['discriminator']

    infer_params = config.get('infer_params', {})
    output_path = infer_params.get('output_path', 'predictions/')

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created folder {output_path}")

    generator.eval()
    discriminator.eval()

    generator.load_state_dict(torch.load(gen_checkpoint, map_location=device))
    discriminator.load_state_dict(torch.load(dsc_checkpoint, map_location=device))

    threshold = infer_params.get('threshold', 0)
    overlap = infer_params.get('overlap', 0.9)

    for i, data in enumerate(tqdm.tqdm(datagen, desc='Predicting', dynamic_ncols=True, ascii=True)):
        imgs = n_crop(data, size, overlap)
        out_fname, _ = os.path.splitext(datagen.get_filename(i))

        with torch.no_grad():
            img_tensor = torch.Tensor(imgs).to(device)
            masks = generator(img_tensor).cpu().numpy()

        mask = build_mask(masks, size, data.shape[1:], threshold, overlap)

        Dataset.save_mask(mask, output_path, out_fname)
