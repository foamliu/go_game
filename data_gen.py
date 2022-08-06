import os

import numpy as np
from torch.utils.data import Dataset

from config import chunksize, feature_file_base, label_file_base


class GameGoDataset(Dataset):
    def __init__(self, compressed):
        self.compressed = compressed

        if compressed:
            self.feature_file_base = 'data/features_%d.npz'
            self.label_file_base = 'data/labels_%d.npz'
        else:
            self.feature_file_base = 'data/features_%d.npy'
            self.label_file_base = 'data/labels_%d.npy'

        num_samples = 0
        chunk = 0

        while True:
            feature_file = self.feature_file_base % chunk
            label_file = self.label_file_base % chunk
            # print(feature_file)
            # print(label_file)

            if os.path.isfile(feature_file) and os.path.isfile(label_file):
                # current_features = np.load(feature_file)
                # current_labels = np.load(label_file)
                # assert len(current_features) == len(current_labels)
                # num_samples += len(current_features)
                num_samples += chunksize
                chunk += 1

            else:
                break

        self.num_samples = num_samples
        self.num_chunk = chunk

        print('{} chunks, {} data loaded'.format(chunk, num_samples))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, i):
        chunk = i // chunksize
        idx = i % chunksize

        feature_file = self.feature_file_base % chunk
        label_file = self.label_file_base % chunk

        features = np.load(feature_file, allow_pickle=True)
        labels = np.load(label_file)

        if self.compressed:
            features = features['features']
            labels = labels['labels']

        data = features[idx]
        label = labels[idx]

        return data.astype(np.float32), label


if __name__ == '__main__':
    dataset = GameGoDataset()
    data = dataset[0][0]
    print(data.shape)
