import torch
import random
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from PIL import Image
import os
import logging
import pickle
import numpy as np
import sys
path = os.path.normpath(os.path.join(os.path.join(os.path.abspath(__file__)), '..', '..'))
sys.path.append(path)


class COCOCaptions(Dataset):
    def __init__(self, annotation: str, root: str, split: str):
        self.coco = COCO(annotation)
        self.ids = self.coco.getImgIds()
        self.imgs = self.coco.loadImgs(self.ids)
        self.root = root
        self.split = split

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        image_path = '{}/{}2017/{}'.format(self.root, self.split, self.imgs[i]['file_name'])
        image = Image.open(image_path).convert('RGB')
        annotations = self.coco.loadAnns(self.coco.getAnnIds(self.ids[i]))
        captions = [annotation['caption'] for annotation in annotations]

        data = {'image_id': self.ids[i],
                'file_name': self.imgs[i]['file_name'],
                'caption': captions[random.randint(0, len(captions) - 1)],
                'image': image}

        return data

    def collate_fn(self, batch):
        new_batch = {}
        for sample in batch:
            for key in sample.keys():
                if key not in new_batch.keys():
                    new_batch[key] = [sample[key]]
                else:
                    new_batch[key].append(sample[key])
        return new_batch

    def get_loader(self, batch_size, shuffle):
        sampler = range(len(self))
        if shuffle:
            sampler = random.sample(sampler, len(sampler))

        return torch.utils.data.DataLoader(self,
                                           batch_size=batch_size,
                                           sampler=sampler,
                                           collate_fn=self.collate_fn)


class COCOEmbeddings(Dataset):
    def __init__(self, path):
        assert os.path.exists(path), '{} does not exist'.format(path)
        self.text_embeddings = []
        self.captions = []
        self.patch_embeddings = []
        with open(path, 'rb') as f:
            data = pickle.load(f)
            logging.debug('data keys: {}'.format(data.keys()))
            logging.debug('len captions : {}'.format(len(data['captions'])))
            logging.debug('len images : {}'.format(len(data['image_embeddings'])))

            self.captions = data['captions']
            self.image_embeddings = data['image_embeddings']
            self.image_id = data['image_id']
            self.image_name = data['image_name']

    def __len__(self):
        return len(self.image_embeddings)

    def __getitem__(self, index):
        # print('embedding shape 0', self.image_embeddings[index].shape)
        payload = {'image_id': self.image_id[index],
                   'file_name': self.image_name[index],
                   'image_embeddings': self.image_embeddings[index],
                   'caption': self.captions[index][random.randint(0, len(self.captions[index]) - 1)]}
        return payload

    def collate_fn(self, batch):
        new_batch = {}
        for sample in batch:
            for key in sample.keys():
                if key not in new_batch.keys():
                    new_batch[key] = [sample[key]]
                else:
                    new_batch[key].append(sample[key])
        new_batch['image_embeddings'] = torch.stack(new_batch['image_embeddings'])
        return new_batch

    def get_loader(self, batch_size, shuffle):
        sampler = range(len(self))
        if shuffle:
            sampler = random.sample(sampler, len(sampler))

        return torch.utils.data.DataLoader(self,
                                           batch_size=batch_size,
                                           sampler=sampler,
                                           collate_fn=self.collate_fn)


if __name__ == '__main__':
    from model.encoder import Encoder
    from model.decoder import Decoder
    from tqdm import tqdm

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataset = COCOCaptions('E:/datasets/coco_2017/annotations/captions_train2017.json', 'E:/datasets/coco_2017/', 'train')
    dataset = COCOEmbeddings('E:embeddings/coco/dinov2_train.pkl')
    dataloader = dataset.get_loader(4, False)

    # encoder = Encoder('facebook/dinov2-base', 'coco', False)
    # decoder = Decoder("facebook/opt-350m", device, input_dimension=768, precision=torch.float32)
    for batch in tqdm(dataloader):
        pass

    #     print(batch)
    #     # x = encoder(batch['image'])
    #     x = batch['image_embeddings']
    #     print(x.shape)
    #     decoder(x, batch['caption'])

