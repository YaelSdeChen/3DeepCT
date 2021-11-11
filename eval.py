import os
import sys
import time
import json
import torch
import scipy.io as sio
from argparse import ArgumentParser
from util import get_loss, get_net, get_test_dataloader, get_relative_average_error, get_total_mass_error

sys.dont_write_bytecode = True
torch.backends.cudnn.benchmark = True


def evaluate(exp_dir, trained_model, testing_params: dict, loss_params: dict):
    with torch.no_grad():
        trained_model.eval()

        test_loader = get_test_dataloader(testing_params['test_loader'], testing_params['batch_size'])

        mse_loss = get_loss(loss_params)

        batch_clouds_net = None
        batch_images_gt = None
        batch_clouds_gt = None
        net_losses = []
        relative_average_errors = []
        total_mass_errors = []
        cloud_indices = []
        batch_time_net = []

        evaluation_loss = 0
        evaluation_start_time = time.time()  # Time for printing

        for i, data in enumerate(test_loader, 0):
            images_gt, clouds_gt, cloud_index = data
            images_gt, clouds_gt = images_gt.to(device), clouds_gt.to(device)

            net_start_time = time.time()
            clouds_net = trained_model(images_gt)
            time_net = time.time() - net_start_time

            loss_val = mse_loss(clouds_net, clouds_gt)
            evaluation_loss += loss_val.data
            net_losses.append(loss_val.data.cpu().item())

            # extra measurements
            relative_average_error = get_relative_average_error(clouds_gt, clouds_net)
            total_mass_error = get_total_mass_error(clouds_gt, clouds_net)

            relative_average_errors.append(relative_average_error.data.cpu().item())
            total_mass_errors.append(total_mass_error.data.cpu().item())

            if batch_clouds_net is not None:
                batch_clouds_net = torch.cat((batch_clouds_net, clouds_net), 0)
                batch_images_gt = torch.cat((batch_images_gt, images_gt), 0)
                batch_clouds_gt = torch.cat((batch_clouds_gt, clouds_gt), 0)
            else:
                batch_clouds_net = clouds_net
                batch_images_gt = images_gt
                batch_clouds_gt = clouds_gt

            cloud_indices.append(cloud_index)
            batch_time_net.append(time_net)

        result = {'key': testing_params['test_loader']['key'],
                  'images_gt': batch_images_gt.cpu().numpy(),
                  'clouds_gt': batch_clouds_gt.cpu().numpy(),
                  'loss': net_losses,
                  'relative_average_error': relative_average_errors,
                  'total_mass_error': total_mass_errors,
                  'clouds_net': batch_clouds_net.cpu().detach().numpy(),
                  'net_time': batch_time_net,
                  'cloud_indices': cloud_indices}
        sio.savemat(f'{exp_dir}/evaluation_result.mat', result)

        evaluation_loss /= len(test_loader)
        print(f'Evaluation finished successfully, took {time.time() - evaluation_start_time} seconds, '
              f'Evaluation Loss = {evaluation_loss}')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    parser = ArgumentParser()
    parser.add_argument('--exp_dir', type=str, help='Path to experiment directory')
    args = parser.parse_args()

    # exp_dir = "/home/yaelsc/PycharmProjects/3DeepCT/experiments/example"
    with open(f'{args.exp_dir}/config.json') as json_file:
        config = json.load(json_file)
    testing_params = config['testing']
    network_params = config['network']
    loss_params = config['loss']

    net = get_net(network_params)
    net.to(device)
    net.load_state_dict(torch.load(testing_params["model_path"]))

    evaluate(args.exp_dir, net, testing_params, loss_params)
