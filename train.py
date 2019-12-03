import argparse
import os

import torch
from torch.optim import Adam
import pandas as pd
from tqdm import tqdm

from head.metrics import ArcFace
from models.irse import IR_50, HeadedModel
from util.data_loader import get_dataloader
from util.losses import FocalLoss
from util.utils import accuracy


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default=None,
                        help='name of the experement')
    parser.add_argument('--train_path', type=str, default=10,
                        help='path to training data')
    parser.add_argument('--valid_path', type=str, default=None,
                        help='path to validation data')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='starting epoch for resume')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=120,
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='type of the model to train')

    args = parser.parse_args()
    return args


def validate(model, criterion, device, valid_loader):

    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    steps = 0
    with torch.no_grad():
        for images_batch, targets_batch in valid_loader:
            images_batch = images_batch.to(device)
            targets_batch = targets_batch.to(device)
            predicts = model(images_batch, targets_batch)
            loss = criterion(predicts, targets_batch)
            epoch_loss += loss.item()
            epoch_acc += accuracy(predicts, targets_batch)[0].item()
            steps += 1
        epoch_loss /= steps
        epoch_acc /= steps

    return epoch_loss, epoch_acc


def train(model, optimizer, criterion, device, train_loader, valid_loader,
          args, information, checkpoints_path):

    model.to(device)
    print('Train started...')
    locer = 0
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        print(f'Epoch {epoch+1} started')
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        steps = 0
        for images_batch, targets_batch in train_loader:
            images_batch = images_batch.to(device)
            targets_batch = targets_batch.to(device)
            predicts = model(images_batch, targets_batch)
            loss = criterion(predicts, targets_batch)
            epoch_loss += loss.item()
            epoch_acc += accuracy(predicts, targets_batch)[0].item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1
        epoch_loss /= steps
        epoch_acc /= steps

        val_loss, val_accuracy = validate(model,
                                          criterion,
                                          device,
                                          valid_loader)
        epoch_info = f'Epoch finished! Train loss: {epoch_loss},' \
                     f'Train acc: {epoch_acc}, Val loss: {val_loss}, Val acc: {val_accuracy}'
        print(epoch_info)

        information[0].loc[locer] = [epoch_loss, epoch_acc, val_loss, val_accuracy]
        information[0].to_csv(f'info/{information[1]}.csv', index=False)
        torch.save({
            'model': model.state_dict(),
            'otimizer': optimizer.state_dict()
        }, os.path.join(checkpoints_path, f'epoch_{epoch}.pth'))
        locer += 1


def main(args):

    print('Data preparing...')
    train_loader = get_dataloader(path=args.train_path,
                                  mode='train',
                                  binary=(args.num_classes == 2),
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    valid_loader = get_dataloader(path=args.train_path,
                                  mode='valid',
                                  binary=(args.num_classes == 2),
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    print('Data prepared!')

    checkpoints_path = f'checkpoints/{args.name}'
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    device = torch.device('cuda')
    backbone = IR_50([224, 224])
    head = ArcFace(512, args.num_classes)
    model = HeadedModel(backbone, head)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = FocalLoss()

    if args.start_epoch:
        weights = torch.load(
            os.path.join(checkpoints_path, f'epoch_{args.start_epoch}.pth'),
            map_location='cpu'
        )
        model.load_state_dict(weights['model'])
        optimizer.load_state_dict(weights['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    columns = ['Train loss', 'Valid loss', 'Train accuracy', 'Valid accuracy']
    information = (pd.DataFrame(columns=columns),
                   f'{args.name}_from_{args.start_epoch}')

    train(model, optimizer, criterion, device, train_loader, valid_loader,
          args, information, checkpoints_path)


if __name__ == '__main__':

    args = get_args()
    main(args)
