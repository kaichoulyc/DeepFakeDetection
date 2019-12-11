import argparse
import os

import torch
from torch.optim import Adam
import pandas as pd
from tqdm import tqdm
from models.model import get_model
import torch.nn as nn

from util.data_loader import get_dataloader
from util.losses import FocalLoss
from util.utils import accuracy

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default=None,
                        help='name of the experement')
    parser.add_argument('--test_path', type=str, default=None,
                        help='path to test data')
    parser.add_argument('--epoch', type=int, default=0,
                        help='epoch for test')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='type of the model to train')
    parser.add_argument('--model_type', type=str, default='xception',
                        help='type of the model to learn')
    parser.add_argument('--side_size', type=int, default=256,
                        help='image side size')
    parser.add_argument('--loss_type', type=str, default='cross',
                        choices=['cross', 'focal'],
                        help='type of the loss to use')

    args = parser.parse_args()
    return args

def test(model, criterion, device, test_loader,
         args, information, checkpoints_path):

    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    steps = 0
    with torch.no_grad():
        for images_batch, targets_batch in test_loader:
            images_batch = images_batch.to(device)
            targets_batch = targets_batch.to(device)
            predicts = model(images_batch)
            loss = criterion(predicts, targets_batch)
            epoch_loss += loss.item()
            epoch_acc += accuracy(predicts, targets_batch)[0].item()
            steps += 1
        epoch_loss /= steps
        epoch_acc /= steps

    print(f'Test finished! Test loss: {epoch_loss}, Test acc: {epoch_acc}')
    information[0].loc[0] = [epoch_loss, epoch_acc]
    information[0].to_csv(f'test_info/{information[1]}.csv', index=False)


def main(args):

    print(f'Name: {args.name}')
    print('Data preparing...')
    test_loader = get_dataloader(path=args.test_path,
                                 mode='test',
                                 side_size=args.side_size,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers)
    print('Data prepared!')
    checkpoints_path = f'checkpoints/{args.name}'

    losses = {
        'cross': nn.CrossEntropyLoss,
        'focal': FocalLoss
    }

    device = torch.device('cuda')

    model = get_model(args.model_type, args.num_classes)

    criterion = losses[args.loss_type]()

    weights = torch.load(
        os.path.join(checkpoints_path, f'epoch_{args.epoch}.pth'),
        map_location='cpu'
    )
    model.load_state_dict(weights['model'])

    columns = ['Test loss', 'Test accuracy']
    information = (pd.DataFrame(columns=columns),
                   f'{args.name}_test_{args.epoch}')
    test(model, criterion, device, test_loader,
         args, information, checkpoints_path)


if __name__ == '__main__':

    args = get_args()
    main(args)