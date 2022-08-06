import pickle
import random

import numpy as np

from dlgo.encoders.oneplane import OnePlaneEncoder

if __name__ == '__main__':
    encoder = OnePlaneEncoder((19, 19))

    with open('samples.pkl', 'rb') as fp:
        data = pickle.load(fp)

    sample = random.choice(data)

    idx = sample[0]
    label = sample[1]

    filename = 'data/{}.pkl'.format(idx)
    print(filename)

    with open(filename, 'rb') as fp:
        data = pickle.load(fp).astype(np.float32)

    print(data.astype(np.int))
    print(label)

    print(encoder.decode_point_index(label))
