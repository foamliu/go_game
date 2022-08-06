import os
import shutil

from tqdm import tqdm

from utils import ensure_folder

if __name__ == '__main__':
    root_dir = '/mnt/sdb/go-dataset/'
    target_dir = '/mnt/sdb/go-dataset/ge5d'
    ensure_folder(target_dir)

    for sub_dir in ['5d', '6d', '7d', '8d', '9d']:
        print(sub_dir)
        dir_path = root_dir + '{0}/{0}'.format(sub_dir)
        files = os.listdir(dir_path)
        for file in tqdm(files):
            source_file = os.path.join(dir_path, file)
            target_file = os.path.join(target_dir, file)
            shutil.move(source_file, target_file)

    files = os.listdir(target_dir)
    files = [f for f in files if f.endswith('.sgf')]

    print('number of sgf files: ' + str(len(files)))
