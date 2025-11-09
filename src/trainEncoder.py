import argparse

import torch
import json
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from model.classification_head import MultiClassifier, Attention
from model.dinov3 import DINOv3
from data.mimic import MimicDataset
from tqdm import tqdm
import os

mimic_classifier_list = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
                         'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
                         'Pneumonia', 'Pneumothorax', 'Support Devices', 'No Finding']


def balance_weights(json_file, class_list, number_of_classes):
    counts = {}
    for class_name in class_list:
        counts[class_name] = []

    for dado in json.load(open(json_file, 'r')):
        for label in class_list:
            if label in dado['labels'].keys():
                if dado['labels'][label] < number_of_classes:
                    counts[label].append(dado['labels'][label])
                else:
                    counts[label].append(0)

    weights = {}
    for class_name in class_list:
        occurrences = np.unique(counts[class_name])
        # check if all possible values are present in data
        for value in range(number_of_classes):
            if value not in occurrences:
                counts[class_name].append(value)

        weight = compute_class_weight('balanced', classes=np.array(range(number_of_classes)), y=counts[class_name])
        weights[class_name] = torch.tensor(weight)
    # print(weights)
    return weights

def forward_backward(batch: dict, training: bool) -> torch.Tensor:
    polling.requires_grad = training
    classification_head.requires_grad = training

    features = encoder.get_features(batch['image'], crop_center=True)
    pooled_features = polling(features['patches'])['attn_output']
    outputs = classification_head(pooled_features)

    losses = []
    for name in mimic_classifier_list:
        target = torch.tensor(batch['labels'][name], dtype=torch.long, device=device)
        CE = torch.nn.CrossEntropyLoss(weight=weights[name].to(device, dtype=torch.float32))
        loss = CE(outputs[name], target)
        losses.append(loss)

    loss = sum(losses) / len(losses)
    if training:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, default='E:\\datasets\\mimic\\preprocess\\train_split.json')
    parser.add_argument('--root', type=str, default='E:\\datasets\\mimic\\mimic-cxr-jpg\\2.1.0\\files')
    parser.add_argument('--step', type=int, default=20000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='checkpoints/run/')
    args = parser.parse_args()

    json_file = args.json
    root = args.root
    step = args.step
    output_dir = args.output_dir
    epochs = args.epochs
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MimicDataset(root, json_file, zeroed=True)
    loader = dataset.get_loader(32)

    classification_head = MultiClassifier(mimic_classifier_list, 1024, 4).to(device)
    polling = Attention(len(mimic_classifier_list), 1024).to(device)
    encoder = DINOv3().to(device)

    weights = balance_weights(json_file, mimic_classifier_list, 4)
    optimizer = torch.optim.AdamW(classification_head.parameters())
    step_log = []
    step_counter = 0
    log = {'train_loss': [], 'train_step': [], 'val_loss': [],'val_step': []}
    pbar = tqdm(total=len(loader)*epochs, desc='training progress')

    for epoch in range(epochs):
        for i, batch in enumerate(loader):
            loss = forward_backward(batch, training=True)
            step_log.append(loss.detach().cpu().item())
            pbar.update(1)

            if (i+1) % step == 0 or i == len(loader) - 1:
                # update log
                log['train_loss'].append(sum(step_log) / len(step_log))
                log['train_step'].append(step_counter)
                step_counter += 1
                step_log = []

                if i == len(loader) - 1:
                    # validate
                    val_log = []
                    dataset = MimicDataset(root, json_file.replace('train', 'dev'), zeroed=True)
                    loader = dataset.get_loader(32)
                    for batch in loader:
                        loss = forward_backward(batch, training=False)
                        val_log.append(loss.detach().cpu().item())

                    log['val_loss'].append(sum(val_log) / len(val_log))
                    log['val_step'].append(step_counter)

                # save log and checkpoints
                with open(os.path.join(output_dir, 'log.json'), 'w') as f:
                    json.dump(log, f, indent=2)

                path = os.path.join(output_dir, 'classifiers.pth')
                torch.save(classification_head.state_dict(), path)
                path = os.path.join(output_dir, 'poller.pth')
                torch.save(polling.state_dict(), path)

