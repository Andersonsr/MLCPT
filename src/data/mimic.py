import json
import os
import logging
import glob
import torch
import numpy as np
import pickle
import gc
from PIL import Image
from torchvision.transforms import ToTensor

to_tensor = ToTensor()


class MimicDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, json_file, zeroed=False):
        assert os.path.exists(json_file), '{} does not exist'.format(json_file)
        assert os.path.isdir(root_dir), '{} is not a dir'.format(root_dir)

        self.root = root_dir
        self.data = json.load(open(json_file, 'r'))
        self.zeroed = zeroed
        self.mimic_folders = 'mimic-cxr-jpg' in self.root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        patient = self.data[i]['patient']
        folder = self.data[i]['patient'][:3]
        study = self.data[i]['study']
        image_name = self.data[i]['image_name']

        if self.mimic_folders:
            image = Image.open(os.path.join(self.root, folder, patient, study, image_name)).convert('RGB')

        else:
            image = Image.open(os.path.join(self.root, self.data[i]['imag_name'])).convert('RGB')

        return {'id': self.data[i]['id'],
                'findings': self.data[i]['findings'],
                'labels': self.data[i]['labels'],
                'image': image,
                'patient': patient,
                'study': study,
                'image_filename': image_name}

    def collate_fn(self, batch):
        data = {}

        for sample in batch:
            for key, item in sample.items():
                if key not in data.keys():
                    data[key] = []

                data[key].append(item)

        # reorganize labels
        reorganized_labels = {}

        for key in data['labels'][0].keys():
            reorganized_labels[key] = []

        # print(reorganized_labels)
        for labels in data['labels']:
            for key in reorganized_labels.keys():
                # print('old', labels[key])
                # print('new', 0 if labels[key] == 3 and self.zeroed else labels[key])
                if key in labels.keys():
                    reorganized_labels[key].append(0 if self.zeroed and labels[key] == 3 else labels[key])
                else:
                    reorganized_labels[key].append(0 if self.zeroed else 3)

        data['labels'] = reorganized_labels
        return data

    def get_loader(self, batch_size):
        indices = np.arange(len(self))
        sampler = torch.utils.data.SequentialSampler(indices)
        return torch.utils.data.DataLoader(self,
                                           batch_size=batch_size,
                                           sampler=sampler,
                                           shuffle=False,
                                           collate_fn=self.collate_fn)


if __name__ == '__main__':
    dataset = MimicDataset('E:\\datasets\\mimic\\mimic-cxr-jpg\\2.1.0\\files', 'E:\\datasets\\mimic\\preprocess\\train_split.json', zeroed=True)
    loader = dataset.get_loader(4)
    from tqdm import tqdm
    for batch in tqdm(loader):
        print(batch['labels'])
        break
