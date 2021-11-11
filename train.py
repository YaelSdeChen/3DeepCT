import os
import sys
import time
import json
import torch
import warnings
import numpy as np
import torch.optim as optim
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

from util import get_dataloaders, get_loss, get_scheduler, get_net, get_total_mass_error, get_relative_average_error

sys.dont_write_bytecode = True
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")


def train(exp_dir: str, net, optimizer_params: dict, training_params: dict, loss_params: dict,
          checkpoint_path: str, load_checkpoint: bool = False):
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", training_params['batch_size'])
    print("epochs=", training_params['max_epochs'])
    print("learning_rate=", optimizer_params['lr'])
    print("=" * 30)
    writer = SummaryWriter(log_dir=f"{exp_dir}/log_dir")

    train_loader, eval_loader = get_dataloaders(training_params['train_loader'], training_params['batch_size'])

    mse_loss = get_loss(loss_params)
    optimizer = optim.Adam(net.parameters(), lr=optimizer_params['lr'])
    scheduler = get_scheduler(training_params['scheduler'], optimizer)

    training_start_time = time.time()  # Time for printing
    start_epoch = 0

    if load_checkpoint:
        if training_params["use_pretrain"]:
            net.load_state_dict(torch.load(training_params["pretrain_path"]))

        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = get_scheduler(training_params['scheduler'], optimizer)
        start_epoch = checkpoint['epoch']

    # Loop for n_epochs
    for epoch in np.arange(start=start_epoch + 1, stop=training_params['max_epochs'] + 1):
        epoch_start_time = time.time()

        print("Epoch number: ", str(epoch))

        epoch_loss = 0
        eval_epoch_loss = 0

        for i, data in enumerate(train_loader, 0):
            images_gt, clouds_gt, _ = data
            images_gt, clouds_gt = images_gt.to(device), clouds_gt.to(device)

            clouds_net = net(images_gt)

            loss_val = mse_loss(clouds_net, clouds_gt)

            epoch_loss += loss_val.data
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

        epoch_loss /= len(train_loader)
        if scheduler is not None:
            scheduler.step(epoch_loss)

        writer.add_scalar('Train Loss', epoch_loss, epoch)

        # extra measurements
        relative_average_error = get_relative_average_error(clouds_gt, clouds_net)
        total_mass_error = get_total_mass_error(clouds_gt, clouds_net)

        writer.add_scalar('Epsilon', relative_average_error.data, epoch)
        writer.add_scalar('Delta', total_mass_error, epoch)

        if epoch % training_params['eval_epoch_gap'] == 0:
            net.eval()
            with torch.no_grad():
                for i, data in enumerate(eval_loader, 0):
                    images_gt, clouds_gt, _ = data
                    images_gt, clouds_gt = images_gt.to(device), clouds_gt.to(device)
                    clouds_net = net(images_gt)
                    loss_val = mse_loss(clouds_net, clouds_gt)
                    eval_epoch_loss += loss_val.data

                writer.add_scalar('Eval Loss', eval_epoch_loss / len(eval_loader), epoch)
            net.train()

        if epoch % training_params['save_model_gap'] == 0:
            torch.save(net.state_dict(), f'{exp_dir}/model_{epoch}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f'{exp_dir}/checkpoint_{epoch}')

        print(f'Epoch i={epoch} finished, took {time.time() - epoch_start_time} seconds, Training Loss={epoch_loss}')

    writer.close()
    print(f'Training finished successfully, took {time.time() - training_start_time} seconds')


if __name__ == '__main__':
    '''
    Example command:
    
    python train.py --exp_dir experiments/example
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    parser = ArgumentParser()
    parser.add_argument('--exp_dir', type=str, help='Path to experiment directory')
    args = parser.parse_args()

    with open(f'{args.exp_dir}/config.json') as json_file:
        config = json.load(json_file)
    optimizer_params = config['optimizer']
    training_params = config['training']
    network_params = config['network']
    loss_params = config['loss']

    net = get_net(network_params)
    net.to(device)

    # main
    train(args.exp_dir, net, optimizer_params, training_params, loss_params, load_checkpoint=training_params["use_pretrain"],
          checkpoint_path=training_params["pretrain_path"])
