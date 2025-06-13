import argparse
import os

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TrainDataset, ValDataset
from model import LPRModel


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(args):
    train_loader = DataLoader(
        dataset=TrainDataset(), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        dataset=ValDataset(), batch_size=args.batch_size)

    model = LPRModel()
    model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_writer = SummaryWriter(logdir=os.path.join(args.log_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(args.log_dir, 'val'))
    for epoch in range(1, args.epochs + 1):
        avg_loss = 0
        avg_acc = 0
        model.train()
        with tqdm(train_loader, ncols=0, leave=False) as pbar:
            pbar.set_description(
                'Epoch %3d/%3d' % (epoch, args.epochs))
            for image, label in pbar:
                image = image.to(device)
                label = label.to(device)
                predict = model(image)
                loss = loss_fn(predict.transpose(1, 2), label)
                optim.zero_grad()
                loss.backward()
                optim.step()

                predict = torch.argmax(predict, dim=-1)
                acc = (label == predict).float().mean(dim=1).sum().cpu()
                avg_loss += loss.item()
                avg_acc += acc.item()
                pbar.set_postfix(loss='%.4f' % loss)
            avg_loss = avg_loss / len(train_loader)
            avg_acc = avg_acc / len(train_loader.dataset)
            train_writer.add_scalar('loss', avg_loss, epoch)
            train_writer.add_scalar('acc', avg_acc, epoch)

        avg_val_loss = 0
        avg_val_acc = 0
        model.eval()
        with torch.no_grad():
            for image, label in val_loader:
                image = image.to(device)
                label = label.to(device)
                predict = model(image)
                loss = loss_fn(predict.transpose(1, 2), label)
                predict = torch.argmax(predict, dim=-1)
                acc = (label == predict).float().mean(dim=1).sum().cpu()
                avg_val_loss += loss.item()
                avg_val_acc += acc.item()
        avg_val_loss = avg_val_loss / len(val_loader)
        avg_val_acc = avg_val_acc / len(val_loader.dataset)
        val_writer.add_scalar('loss', avg_val_loss, epoch)
        val_writer.add_scalar('acc', avg_val_acc, epoch)
        print('Epoch %3d/%3d train_loss: %.4f, val_loss: %.4f, '
              'train_acc: %.4f, val_acc: %.4f' % (
                    epoch, args.epochs,
                    avg_loss, avg_val_loss, avg_acc, avg_val_acc))
        if epoch % args.checkpoint_period == 0:
            path = os.path.join(args.checkpoint_dir, 'ckpt%d.pt' % epoch)
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('License Plate Recognition')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--checkpoint-period', type=int, default=5)
    parser.add_argument('--checkpoint-dir', type=str, default="./checkpoints")
    parser.add_argument('--log-dir', type=str, default="./logs")
    args = parser.parse_args()

    main(args)
