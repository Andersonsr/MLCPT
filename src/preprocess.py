import json
from PIL import ImageFile
from tqdm import tqdm
from PIL import Image
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def resize(mask_image: Image, image_size: int, ) -> Image:
    patch_size = 16

    w, h = mask_image.size
    if h < w:
        h_patches = int(image_size / patch_size)
        w_patches = int((w * image_size) / (h * patch_size))

    else:
        w_patches = int(image_size / patch_size)
        h_patches = int((h * image_size) / (w * patch_size))

    return mask_image.resize((w_patches * patch_size, h_patches * patch_size))


if __name__ == '__main__':
    json_files = ['E:\\datasets\\mimic\\preprocess\\train_split.json', 'E:\\datasets\\mimic\\preprocess\\test_split.json', 'E:\\datasets\\mimic\\preprocess\\dev_split.json']
    root = 'E:\\datasets\\mimic\\mimic-cxr-jpg\\2.1.0\\files'
    output_folder = 'D:\\mimic\\preprocess\\512'
    json_data = []
    for json_file in json_files:
        json_data += json.load(open(json_file))
        print(len(json_data))

    os.makedirs(output_folder, exist_ok=True)

    for sample in tqdm(json_data, total=len(json_data)):
        patient = sample['patient']
        folder = sample['patient'][:3]
        study = sample['study']
        image_name = sample['image_name']

        if not os.path.exists(os.path.join(output_folder, sample['image_name'])):
            image = Image.open(os.path.join(root, folder, patient, study, image_name)).convert('RGB')
            image = resize(image, 512)
            image.save(os.path.join(output_folder, sample['image_name']))


