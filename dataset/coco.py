import torch
import random
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class CocoCaptions(Dataset):
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
        annotations = self.coco.loadAnns(self.coco.getAnnIds(self.ids[i]))
        captions = [annotation['caption'] for annotation in annotations]

        data = {'image_id': self.ids[i],
                'file_name': self.imgs[i]['file_name'],
                'caption': captions[random.randint(0, len(captions) - 1)],
                'image_path': image_path}

        return data

    def get_loader(self, batch_size, shuffle):
        sampler = range(len(self))
        return torch.utils.data.DataLoader(self,
                                           batch_size=batch_size,
                                           sampler=sampler,
                                           shuffle=shuffle)


if __name__ == '__main__':
    dataset = CocoCaptions('E:/datasets/coco_2017/annotations/captions_val2017.json', 'E:/datasets/coco_2017/', 'val')
    dataloader = dataset.get_loader(8, False)
    for batch in dataloader:
        print(len(batch['caption']))
        break