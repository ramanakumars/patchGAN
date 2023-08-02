import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms


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
            self.augmentation = transforms.Resize(size=(size, size), antialias=None)
        elif augmentation == 'randomcrop+flip':
            self.augmentation = transforms.Compose([
                transforms.Resize(size=(size, size), antialias=None),
                transforms.RandomHorizontalFlip(0.25),
                transforms.RandomVerticalFlip(0.25),
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
