import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Compose, Resize, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
from einops import rearrange
import rasterio


class FloatingForestsDataset(Dataset):
    augmentation = None

    def __init__(self, imgfolder, maskfolder, size=256, augmentation='randomcrop'):
        self.images = sorted(glob.glob(os.path.join(imgfolder, "*.tif")))
        self.masks = sorted(glob.glob(os.path.join(maskfolder, "*.tif")))
        self.size = size

        self.image_ids = [int(os.path.basename(image).replace('.tif', '')) for image in self.images]
        self.mask_ids = [int(os.path.basename(image).replace('.tif', '')) for image in self.masks]

        assert np.all(self.image_ids == self.mask_ids), "Image IDs and Mask IDs do not match!"

        if augmentation == 'randomcrop':
            self.augmentation = RandomCrop(size=(size, size))
        elif augmentation == 'randomcrop+flip':
            self.augmentation = Compose([
                RandomCrop(size=(size, size)),
                RandomHorizontalFlip(0.25),
                RandomVerticalFlip(0.25)
            ])

        print(f"Loaded {len(self)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_file = self.images[index]
        mask_file = self.masks[index]

        img = rasterio.open(image_file)
        img_stacked = np.dstack([img.read(1), img.read(2), img.read(3), img.read(4)])

        # clean up artifacts in the data
        img_stacked[np.abs(img_stacked) > 1.e20] = 0
        img_stacked[np.isnan(img_stacked)] = 0

        # remove negative signal
        img_stacked = img_stacked - np.percentile(img_stacked.flatten(), 2)

        norm = np.nansum(img_stacked, axis=-1, keepdims=True)
        img_stacked = img_stacked / (norm + 1.e-3)

        # normalize the image per-pixel
        img = np.clip(img_stacked, 0, 1)
        img[~np.isfinite(img)] = 0.

        # add the mask so we can crop it
        mask = rasterio.open(mask_file).read(1)
        mask[mask < 0.] = 0.
        data_stacked = np.concatenate((img, np.expand_dims(mask, -1)), axis=-1)
        data_stacked = rearrange(torch.Tensor(data_stacked), "h w c -> c h w")

        if self.augmentation is not None:
            data_stacked = self.augmentation(data_stacked)

        return data_stacked[:4, :, :], data_stacked[4, :, :].unsqueeze(0)


class COCOStuffDataset(Dataset):
    augmentation = None

    def __init__(self, imgfolder, maskfolder, labels=[1], size=256, augmentation='resize'):
        self.images = np.asarray(sorted(glob.glob(os.path.join(imgfolder, "*.jpg"))))
        self.masks = np.asarray(sorted(glob.glob(os.path.join(maskfolder, "*.png"))))
        self.size = size
        self.labels = np.sort(labels)

        self.image_ids = [int(os.path.basename(image).replace('.jpg', '')) for image in self.images]
        self.mask_ids = [int(os.path.basename(image).replace('.png', '')) for image in self.masks]

        assert np.all(self.image_ids == self.mask_ids), "Image IDs and Mask IDs do not match!"

        if augmentation == 'randomcrop':
            self.augmentation = Resize(size=(size, size), antialias=None)
        elif augmentation == 'randomcrop+flip':
            self.augmentation = Compose([
                Resize(size=(size, size), antialias=None),
                RandomHorizontalFlip(0.25),
                RandomVerticalFlip(0.25),
            ])

        print(f"Loaded {len(self)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_file = self.images[index]
        mask_file = self.masks[index]

        img = read_image(image_file, ImageReadMode.RGB) / 255.
        labels = read_image(mask_file, ImageReadMode.GRAY) + 1

        # add the mask so we can crop it
        data_stacked = torch.cat((img, labels), dim=0)

        if self.augmentation is not None:
            data_stacked = self.augmentation(data_stacked)

        img = data_stacked[:3, :]
        labels = data_stacked[3, :]

        mask = torch.zeros((len(self.labels), labels.shape[0], labels.shape[1]))
        for i, label in enumerate(self.labels):
            mask[i, labels == label] = 1

        return img, mask
