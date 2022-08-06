import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

board_size = (19, 19)
input_size = (1, 19, 19)
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none
num_workers = 8

train_ratio = 0.9

num_classes = 19 * 19

DATA_DIR = '/mnt/sdb/go-dataset/ge5d'
# DATA_DIR = r'D:\code\go-dataset\7d\7d'
data_folder = 'data'

chunksize = 1024
feature_file_base = 'data/features_%d.npz'
label_file_base = 'data/labels_%d.npz'
