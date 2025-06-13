import argparse
import random

import torch
import torchvision.transforms as transforms
from PIL import Image

from dataset import ValDataset, INT_TO_CODE, HEIGHT, WIDTH
from model import LPRModel


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def decode(predict):
    return "".join([INT_TO_CODE[num] for num in predict])


def main(args):
    if args.random:
        print("Randomly select image from ValDataset")
        val_dataset = ValDataset()
        rand_idx = random.randint(0, len(val_dataset) - 1)
        img, label = val_dataset[rand_idx]
    else:
        print("Loading image from {}".format(args.image))
        img = Image.open(args.image)
        img = transforms.functional.resize(img, (HEIGHT, WIDTH))
        img = transforms.functional.to_tensor(img)
    img = img.unsqueeze(0).to(device)

    ckpt = torch.load(
        args.checkpoint, map_location=lambda storage, loc: storage)
    model = LPRModel().to(device)
    model.load_state_dict(ckpt)
    model.eval()
    print('weight has been loaded')

    predict = model(img)
    predict = torch.argmax(predict, axis=-1).cpu()

    if args.random:
        print('label:', decode(label.numpy()))
    print('predictions:', decode(predict[0].numpy()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('License Plate Recognition')
    parser.add_argument(
        '--checkpoint', type=str, default='./pretrain/ckpt100.pt')
    parser.add_argument(
        '--image', type=str, default='./example/9B52145.png')
    parser.add_argument(
        '--random', action='store_true',
        help="Randomly select image from val set")
    args = parser.parse_args()

    main(args)
