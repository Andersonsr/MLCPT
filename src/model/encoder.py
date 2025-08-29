import os
import sys
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from model.classifiers import mimic_classifier_list, MultiClassifier
from torch import nn

path = os.path.normpath(os.path.join(os.path.join(os.path.abspath(__file__)), '..', '..'))
sys.path.append(path)


class Encoder(nn.Module):
    def __init__(self, model_name, dataset, build_head):
        super(Encoder, self).__init__()
        self.vision_preprocess = AutoImageProcessor.from_pretrained(model_name)
        self.mapper = None
        self.backbone = AutoModel.from_pretrained(
            model_name
        )
        if 'dinov2' in model_name:
            self.dim = self.backbone.embeddings.patch_embeddings.projection.weight.size()[0]

        elif 'dinov3' in model_name:
            self.dim = self.backbone.embeddings.patch_embeddings.weight.size()[0]

        else:
            raise ValueError('supported encoders are dinov2 and dinov3')

        if build_head:
            if dataset == 'mimic':
                self.classifiers = MultiClassifier(mimic_classifier_list, self.dim, 4)

            elif dataset == 'coco':
                self.classifiers = MultiClassifier(mimic_classifier_list, self.dim, 2)

    def forward(self, x, trainable=True):
        x = self.preprocess(x)
        if trainable:
            x = self.backbone(**x)

        else:
            with torch.no_grad():
                x = self.backbone(**x)

        if hasattr(self, 'classifiers'):
            return self.classifiers(x.pooler_output)

        return x.pooler_output

    def encode(self, image):
        output = self.backbone(**self.preprocess(image))
        return output.pooler_output

    def preprocess(self, image):
        if type(image) is str:
            inputs = self.vision_preprocess(images=Image.open(image), return_tensors="pt")

        elif type(image) is list:
            if len(image) > 1:
                inputs = self.vision_preprocess(images=image, return_tensors="pt")
                # inputs = torch.stack(inputs, dim=0)

            elif len(image) == 1:
                inputs = self.vision_preprocess(images=image[0], return_tensors="pt")

            else:
                raise IndexError('Image list is empty')

        else:
            inputs = self.vision_preprocess(images=image, return_tensors="pt")

        return inputs


if __name__ == '__main__':
    model_name = 'facebook/dinov2-base'
    # model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    vision_preprocess = AutoImageProcessor.from_pretrained(model_name)
    mapper = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = AutoModel.from_pretrained(
        model_name, )
    if 'dinov2' in model_name:
        print(backbone.embeddings.patch_embeddings.projection.weight.size()[0])

    if 'dinov3' in model_name:
        print(backbone.embeddings.patch_embeddings.weight.size()[0])
