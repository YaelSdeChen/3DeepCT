import torch
import numpy as np

from network.DeepCT import DeepCT
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets.clouds_train_dataset import CloudsDataset
from torch.utils.data.sampler import SubsetRandomSampler
from datasets.clouds_test_dataset import CloudsTestDataset


def get_dataloaders(dataloader_params: dict, batch_size: int):
    train_dataset = CloudsDataset(data_paths=dataloader_params["path"], key=dataloader_params["key"])
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(dataloader_params['eval_percent'] * num_train))

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    eval_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=dataloader_params['num_workers']
    )
    eval_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=eval_sampler, num_workers=dataloader_params['num_workers']
    )
    return train_loader, eval_loader


def get_net(network_params: dict):
    if network_params['name'] == "3DeepCT":
        return DeepCT(in_channels=network_params['in_channels'],
                      out_channels=network_params['out_channels'],
                      model_depth=network_params['model_depth'])
    else:
        print("No net type was chosen. Exiting")
        exit(-1)


def get_loss(loss_params: dict):
    if loss_params['name'] == "MSE":
        return torch.nn.MSELoss()
    if loss_params['name'] == "CrossEntropy":
        return torch.nn.CrossEntropyLoss()
    if loss_params['name'] == "L1":
        return torch.nn.L1Loss()
    if loss_params['name'] == "HardTanh":
        return torch.nn.Hardtanh(min_val=0, max_val=1)
    else:
        print("No loss. Exiting")
        exit(-1)


def get_scheduler(scheduler_params: dict, optimizer):
    if scheduler_params['name'] == "None":
        return None
    elif scheduler_params['name'] == "ReduceLROnPlateau":
        return ReduceLROnPlateau(optimizer, mode='max', factor=scheduler_params['factor'],
                                 patience=scheduler_params['patience'], verbose=False)
    elif scheduler_params['name'] == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_params['step_size'],
                                               gamma=scheduler_params['gamma'])
    elif scheduler_params['name'] == "ExponentialLR":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_params['gamma'], last_epoch=-1)
    else:
        print("No scheduler. Exiting")
        exit(-1)


def get_test_dataloader(dataloader_params: dict, batch_size: int):
    test_dataset = CloudsTestDataset(data_paths=dataloader_params["path"], key=dataloader_params["key"])

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=dataloader_params['num_workers']
    )
    return test_loader


def get_total_mass_error(clouds_gt, clouds_net):
    total_mass_error = torch.div(
        torch.sub(clouds_gt.flatten(start_dim=1).abs(), clouds_net.flatten(start_dim=1).abs()).sum(dim=1),
        clouds_gt.flatten(start_dim=1).abs().sum(dim=1))
    total_mass_error = 100 * torch.sum(total_mass_error) / clouds_gt.shape[0]

    return total_mass_error


def get_relative_average_error(clouds_gt, clouds_net):
    relative_average_error = torch.div(
        torch.sub(clouds_gt.flatten(start_dim=1), clouds_net.flatten(start_dim=1)).abs().sum(dim=1),
        clouds_gt.flatten(start_dim=1).abs().sum(dim=1))
    relative_average_error = 100 * torch.sum(relative_average_error) / clouds_gt.shape[0]

    return relative_average_error
