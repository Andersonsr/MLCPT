import argparse
import pickle
from tqdm import tqdm
import os, sys
path = os.path.normpath(os.path.join(os.path.join(os.path.abspath(__file__)), '..', '..'))
sys.path.append(path)
from dataset.coco import COCOCaptions
from model.encoder import Encoder
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--dataset_root', type=str, default='E:/datasets/coco_2017')
    parser.add_argument('--split', type=str, choices=['train', 'val'], required=True)
    parser.add_argument('--encoder_name', type=str, default='facebook/dinov2-base')
    parser.add_argument('--save_path', type=str, required=True)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'coco':
        data = COCOCaptions(f'{args.dataset_root}/annotations/captions_{args.split}2017.json',
                            args.dataset_root,
                            args.split)

    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    loader = data.get_loader(16, False)
    encoder = Encoder(args.encoder_name, False, False, device)
    embeddings = []
    image_id = []
    captions = []
    file_name = []

    for batch in tqdm(loader):
        embeddings += encoder(batch['image']).detach().cpu()
        image_id += batch['image_id']
        file_name += batch['file_name']
        captions += batch['caption']

    embeddings = torch.stack(embeddings, dim=0).unsqueeze(dim=1)
    print(embeddings.shape)
    with open(args.save_path, 'wb') as f:
        pickle.dump({'image_id': image_id,
                     'captions': captions,
                     'image_name': file_name,
                     'image_embeddings': embeddings}, f)
