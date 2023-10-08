import os
import tifffile
import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, image_set, wo_mean, dataset_name):
        super().__init__()
        self.input_dir = input_dir
        self.image_set = image_set
        self.wo_mean = wo_mean
        self.dataset_name = dataset_name

        self.input_list = os.listdir(self.input_dir)
        self.input_list.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1].split('.')[0])))

    def __getitem__(self, index):
        input = tifffile.imread(os.path.join(self.input_dir, self.input_list[index])).astype(np.float32)

        if self.wo_mean:
            input_mean = np.mean(input, axis=0, keepdims=True)
            input = input - input_mean
            np.save('mean_%s_%s.npy' % (self.image_set, self.dataset_name), input_mean)

        input = torch.from_numpy(input).unsqueeze(0)
        return input

    def __len__(self):
        return len(self.input_list)


def build_dataset(args, image_set):
    input_dir = os.path.join(args.dataset_dir, image_set)
    dataset = Dataset(input_dir, image_set, args.wo_mean, args.dataset_name)
    return dataset
