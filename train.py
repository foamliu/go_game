import argparse
import warnings

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from config import train_ratio
from data_gen import GameGoDataset
from models import AlphaZeroModel

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--compressed', type=bool, default=False, help='compressed')
    parser.add_argument('--gpus', type=str, default=None, help='gpus')
    parser.add_argument('--input-size', type=str, default='(1, 19, 19)', help='input-size')
    parser.add_argument('--precision', type=int, default=32, help='input-size')
    parser.add_argument('--end-epoch', type=int, default=20, help='training epoch size.')
    parser.add_argument('--num-workers', type=int, default=8, help='number of workers.')

    args = parser.parse_args()
    print(args)

    # data
    dataset = GameGoDataset(args.compressed)
    num_samples = len(dataset)
    print('num_samples: ' + str(num_samples))

    num_train = int(num_samples * train_ratio)
    num_val = num_samples - num_train
    print('num_train: ' + str(num_train))
    print('num_val: ' + str(num_val))

    # 241169408
    train, val = random_split(dataset, [num_train, num_val])

    train_loader = DataLoader(train, batch_size=128, pin_memory=True, num_workers=args.num_workers,
                              persistent_workers=True)
    val_loader = DataLoader(val, batch_size=128, pin_memory=True, num_workers=args.num_workers, persistent_workers=True)

    # model
    model = AlphaZeroModel(eval(args.input_size))

    # Save the model periodically by monitoring val_acc
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max', dirpath='./checkpoints',
                                          save_weights_only=True, filename='checkpoint-{epoch:02d}-{val_acc:.4f}')

    # training
    trainer = pl.Trainer(gpus=eval(args.gpus), precision=args.precision, max_epochs=args.end_epoch,
                         callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)

    print(checkpoint_callback.best_model_path)
