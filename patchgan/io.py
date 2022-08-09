import numpy as np
import netCDF4 as nc

class DataGenerator():
    def __init__(self, nc_file, batch_size, indices=None):
        self.nc_file = nc_file

        self.batch_size = batch_size

        if indices is not None:
            self.indices = indices
            self.ndata   = len(indices)
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
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        with nc.Dataset(self.nc_file, 'r') as dset:
            imgs = dset.variables['imgs'][batch_indices,:,:,:].astype(float)
            mask = dset.variables['mask'][batch_indices,:,:].astype(float)

        return imgs.data, np.expand_dims(mask, axis=1)

    def get_from_indices(self, indices):
        with nc.Dataset(self.nc_file, 'r') as dset:
            imgs = dset.variables['imgs'][indices,:,:,:].astype(float)/255.
            mask = dset.variables['mask'][indices,:,:].astype(float)

        return imgs, mask

    def get_meta(self, key, index=None):
        if index is not None:
            batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        else:
            batch_indices = self.indices

        with nc.Dataset(self.nc_file, 'r') as dset:
            var = dset.variables[key][batch_indices]

        return var

class MmapDataGenerator(DataGenerator):
    def __init__(self, img_file, mask_file, batch_size, indices=None):
        self.img_file  = img_file
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

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        if len(batch_indices) < 1:
            raise StopIteration

        img  = self.imgs[batch_indices,:].astype(float)/255
        mask = self.mask[batch_indices,:].astype(float)
        return img, mask
