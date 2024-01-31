import torch
from torchinfo import summary
from .io import COCOStuffDataset
from .patchgan import PatchGAN
import yaml
import tqdm
import os
import importlib.machinery
import argparse
from torch import nn
from einops import rearrange
import lightning as L


class InferenceModel(L.LightningModule):
    '''
        Wrapper for running PatchGAN with a crop inference mode,
        where the input images are cropped with overlap into (patch_size x patch_size)
    '''

    def __init__(self, model: PatchGAN, patch_size: int):
        super().__init__()
        self.save_hyperparameters()
        self.model = model

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        '''
            run the image forward through the patch generation stage
            and then through the patchGAN
        '''
        C, H, W = img.shape

        assert H == W, "PatchGAN currently only supports square images"
        image_size = H

        patch_size = self.hparams.patch_size

        # find the optimal stride
        # this stride should cover the whole image with minimal overlap
        # essentially solving (n - 1) * (kernel_size + 1) >= image_size
        # for the n = number of overlapping patches in each dimension
        for i in range(2, 10):
            n = (image_size - patch_size) // patch_size + i
            stride = (image_size - patch_size) // (n - 1)
            if stride * (n - 1) + patch_size == image_size:
                break

        # check to make sure we got convergence
        if stride * (n - 1) + patch_size != image_size:
            raise ValueError(f"Could fit {image_size} into window of size {patch_size}")

        # create the parameter dict for torch's fold and unfold functions
        fold_params = {'kernel_size': patch_size, 'stride': stride, 'dilation': 1, 'padding': 0}

        # we will also apply this to a ones array to get the number of patches that cover each pixel
        # we will divide the final mask by this count to normalize the number of predictions per pixel
        count = self.fold(self.unfold(torch.ones_like(img), fold_params), fold_params, image_size)
        masks = self.fold(self.model(self.unfold(img, fold_params)), fold_params, image_size)

        return masks / count

    def fold(self, x, fold_params, image_size):
        '''
            Folding function. Given an input of (l, channels, patch_size, patch_size), returns the
            reconstructed image of (channels, image_size, image_size)
        '''
        x = rearrange(x, 'l c h w -> (c h w) l')
        return nn.functional.fold(x, output_size=(image_size, image_size), **fold_params)

    def unfold(self, x, fold_params):
        '''
            Unfolding function. Given an input of (channels, image_size, image_size) returns the set
            of overlapping patches of size (l, channels, patch_size, patch_size)
        '''
        x = nn.functional.unfold(x, **fold_params)
        return rearrange(x, '(c h w) l -> l c h w', c=self.model.hparams.input_channels,
                         h=self.hparams.patch_size, w=self.hparams.patch_size)


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

    patch_size = dataset_params.get('patch_size', 256)

    dataset_kwargs = {}
    if dataset_params['type'] == 'COCOStuff':
        Dataset = COCOStuffDataset
        labels = dataset_params.get('labels', [1])
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

    assert hasattr(Dataset, 'get_filename') and callable(Dataset.get_filename), \
        f"Dataset class {Dataset.__name__} must have the get_filename method which returns the image filename for a given index"

    assert hasattr(Dataset, 'save_mask') and callable(Dataset.save_mask), \
        f"Dataset class {Dataset.__name__} must have the save_mask method to save a mask cube for a given filename"

    datagen = Dataset(dataset_path, **dataset_kwargs)

    model_checkpoint = config['model_checkpoint']

    # create the patchGAN
    model = PatchGAN.load_from_checkpoint(model_checkpoint)

    infer_params = config.get('infer_params', {})
    output_path = infer_params.get('output_path', 'predictions/')

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created folder {output_path}")

    model.eval()

    inferencemodel = InferenceModel(model, patch_size)

    if args.summary:
        summary(inferencemodel, datagen[0].shape, device=device)

    for i, data in enumerate(tqdm.tqdm(datagen, desc='Predicting', dynamic_ncols=True, ascii=True)):
        out_fname, _ = os.path.splitext(datagen.get_filename(i))

        with torch.no_grad():
            img_tensor = torch.Tensor(data).to(device)
            mask = inferencemodel(img_tensor).cpu().numpy()

        Dataset.save_mask(mask, output_path, out_fname)
