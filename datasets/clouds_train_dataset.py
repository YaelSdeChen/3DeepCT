import glob
import torch
import os.path
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset


class CloudsDataset(Dataset):
    def __init__(self, data_paths, key: str = "beta"):
        self.images_paths = []
        self.clouds_paths = []
        self.cloud_indices = []
        self.key = key

        for data_path in data_paths:
            for cloud_path in glob.iglob(f'{data_path}/lwcs/cloud*.mat'):
                cloud_index = cloud_path.replace(f'{data_path}/lwcs/cloud', '').replace('.mat', '')
                satellites_images_path = f'{data_path}/satellites_images/satellites_images_{cloud_index}.mat'

                if os.path.isfile(cloud_path) and os.path.isfile(satellites_images_path):
                    self.clouds_paths.append(cloud_path)
                    self.images_paths.append(satellites_images_path)
                    self.cloud_indices.append(cloud_index)

        print(f'Finished creating dataset, size {len(self.images_paths)}')

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, index):
        images_path = self.images_paths[index]
        cloud_path = self.clouds_paths[index]
        cloud_index = self.cloud_indices[index]

        cloud = np.transpose(sio.loadmat(cloud_path)[self.key], (2, 0, 1))
        images = sio.loadmat(images_path)['satellites_images']

        if not torch.is_tensor(images):
            images = torch.tensor(images).float()
        if not torch.is_tensor(cloud):
            cloud = torch.tensor(cloud).float()

        return images, cloud, cloud_index
