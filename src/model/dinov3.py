from PIL import Image
from transformers import AutoModel
import torch
import torchvision.transforms.functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH_SIZE = 16
IMAGE_SIZE = 512


class DINOv3(torch.nn.Module):
    def __init__(self, model_name="facebook/dinov3-vitl16-pretrain-lvd1689m"):
        super(DINOv3, self).__init__()
        self.model = AutoModel.from_pretrained(
            model_name,
            device_map="auto",
        )

    def preprocess(self, mask_image: Image, image_size: int, crop_center: bool = True) -> torch.Tensor:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        patch_size = 16

        w, h = mask_image.size
        # if crop center is true the smaller dimension will have the target size,
        # if not the height dimension will have the target size as in the original dinov3 preprocess
        if not crop_center or w > h:
            h_patches = int(image_size / patch_size)
            w_patches = int((w * image_size) / (h * patch_size))

        else:
            w_patches = int(image_size / patch_size)
            h_patches = int((h * image_size) / (w * patch_size))

        input = TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))
        input = TF.normalize(input, mean=mean, std=std)

        if crop_center:
            _, w, h = input.shape
            center = (int(w / 2), int(h / 2))
            return input[:,
                   center[0] - int(image_size / 2):center[0] + int(image_size / 2),
                   center[1] - int(image_size / 2):center[1] + int(image_size / 2)]

        return input

    def get_features(self, images: list, crop_center=False) -> dict:
        image_tensors = []
        for image in images:
            image_tensors.append(self.preprocess(image, 512, crop_center=crop_center))
        inputs = torch.stack(image_tensors, dim=0).to(self.model.device)
        with torch.no_grad():
            feats = self.model(inputs)
            x = feats.last_hidden_state[0, 5:, :]
            cls = feats.pooler_output
            return {'classification': cls, 'patches': x}

