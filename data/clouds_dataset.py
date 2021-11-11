import glob
import torch
import os.path
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset

# from Augmentations.noise import Noise


class CloudsDataset(Dataset):
    def __init__(self, is_train, data_paths, key: str = "beta"):
        self.is_train = is_train
        self.images_paths = []
        self.clouds_paths = []
        self.cloud_indices = []
        self.key = key
        # self.noise = Noise()

        for data_path in data_paths:
            for cloud_path in glob.iglob(f'{data_path}/lwcs/cloud*.mat'):
                cloud_index = cloud_path.replace(f'{data_path}/lwcs/cloud', '').replace('.mat', '')
                if (self.is_train and int(cloud_index) > 1000 and int(cloud_index) < 1010) or (not self.is_train and int(cloud_index) <= 1000 and int(cloud_index) >= 990):
                    satellites_images_path = f'{data_path}/satellites_images/satellites_images_{cloud_index}.mat'

                    if os.path.isfile(cloud_path) and os.path.isfile(satellites_images_path):
                        self.clouds_paths.append(cloud_path)
                        self.images_paths.append(satellites_images_path)
                        self.cloud_indices.append(cloud_index)

        print(f'Finished creating dataset, size {len(self.images_paths)}')

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, index):
        images_path, cloud_path, cloud_index = self.images_paths[index], self.clouds_paths[index], self.cloud_indices[
            index]
        cloud = np.transpose(sio.loadmat(cloud_path)[self.key], (2, 0, 1))
        images = sio.loadmat(images_path)['satellites_images']

        # images = self.noise.add_noise(images)

        if not torch.is_tensor(images):
            images = torch.tensor(images).float()
        if not torch.is_tensor(cloud):
            cloud = torch.tensor(cloud).float()

        return images, cloud, cloud_index
