import numpy as np
import netCDF4 as nc


class DataGenerator():
    def __init__(self, nc_file, batch_size, indices=None):
        self.nc_file = nc_file

        self.batch_size = batch_size

        if indices is not None:
            self.indices = indices
            self.ndata = len(indices)
        else:
            with nc.Dataset(nc_file, 'r') as dset:
                self.ndata = int(dset.dimensions['file'].size)
            self.indices = np.arange(self.ndata)

        print(f"Found data with {self.ndata} images")

    def shuffle(self):
        np.random.shuffle(self.indices)

    def __len__(self):
        return self.ndata // self.batch_size

    def __getitem__(self, index):
        batch_indices = self.indices[index *
                                     self.batch_size:(index + 1) * self.batch_size]

        if len(batch_indices) < 1:
            raise StopIteration

        return self.get_from_indices(batch_indices)

    def get_from_indices(self, indices):
        with nc.Dataset(self.nc_file, 'r') as dset:
            imgs = dset.variables['imgs'][indices,
                                          :, :, :].astype(float) / 255.
            mask = dset.variables['mask'][indices, :, :].astype(float)

        return imgs, np.expand_dims(mask, axis=1)

    def get_meta(self, key, index=None):
        if index is not None:
            batch_indices = self.indices[index *
                                         self.batch_size:(index + 1) * self.batch_size]
        else:
            batch_indices = self.indices

        with nc.Dataset(self.nc_file, 'r') as dset:
            var = dset.variables[key][batch_indices]

        return var


class MmapDataGenerator(DataGenerator):
    def __init__(self, img_file, mask_file, batch_size, indices=None):
        self.img_file = img_file
        self.mask_file = mask_file

        self.batch_size = batch_size

        self.imgs = np.load(self.img_file, mmap_mode='r')
        self.mask = np.load(self.mask_file, mmap_mode='r')

        self.ndata = len(self.imgs)

        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(self.ndata)

        print(f"Found data with {self.ndata} images")

    def get_from_indices(self, batch_indices):
        img = self.imgs[batch_indices, :].astype(float) / 255
        mask = self.mask[batch_indices, :].astype(float)

        return img, mask


def create_generators(generator, val_split=0.1, **kwargs):
    gen = generator(**kwargs)

    ndata = gen.ndata

    print(f"Creating generators from {ndata} images")

    inds = np.arange(ndata)
    np.random.shuffle(inds)

    val_split_ind = int(ndata * val_split)
    val_ind = inds[:val_split_ind]
    training_ind = inds[val_split_ind:]

    train_data = generator(**kwargs, indices=training_ind)
    val_data = generator(**kwargs, indices=val_ind)

    return train_data, val_data
