import torch
import json
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from model.classification_head import MultiClassifier
from model.dinov3 import DINOv3
from data.mimic import MimicDataset
from tqdm import tqdm


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


if __name__ == '__main__':
    json_file = 'E:\\datasets\\mimic\\preprocess\\train_split.json'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MimicDataset('E:\\datasets\\mimic\\mimic-cxr-jpg\\2.1.0\\files', json_file, zeroed=True)
    loader = dataset.get_loader(32)

    classification_head = MultiClassifier(mimic_classifier_list, 1024, 4).to(device)
    encoder = DINOv3().to(device)
    weights = balance_weights(json_file, mimic_classifier_list, 4)

    optimizer = torch.optim.AdamW(classification_head.parameters())

    for epoch in range(10):
        for batch in tqdm(loader):
            features = encoder.get_features(batch['image'], crop_center=True)
            # print(features['classification'].shape)
            outputs = classification_head(features['classification'])
            losses = []
            for name in mimic_classifier_list:
                target = torch.tensor(batch['labels'][name], dtype=torch.long, device=device)
                CE = torch.nn.CrossEntropyLoss(weight=weights[name].to(device, dtype=torch.float32))
                loss = CE(outputs[name], target)
                losses.append(loss)

            loss = sum(losses) / len(losses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

