import os
import pickle
import argparse

import img_utils

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tqdm import tqdm
from PIL import Image

class CropDataset(torch.utils.data.Dataset):
    '''
    Crop images based on bounding boxes and optional padding.
    '''
    def __init__(self, img_dir, img_names, bboxes, transform=None, padding=0):
        super(CropDataset, self).__init__()
        self.img_dir = img_dir
        self.img_names = img_names
        self.bboxes = bboxes
        self.transform = transform
        self.padding = padding
        self.count = 0

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # Load image with RGB, convert to numpy array, crop and convert back to PIL
        # This is done because CLIP transform assumes input is PIL image.
        img = np.array(Image.open(os.path.join(self.img_dir, self.img_names[idx])).convert('RGB'))

        # img_no_padding = Image.fromarray(img_utils.get_bbox(img, self.bboxes[idx], padding=0))
        # img_no_padding.save('img_no_padding.jpg')

        img = img_utils.get_bbox(img, self.bboxes[idx], padding=self.padding)
        img = Image.fromarray(img)
        # img.save('img_{}.jpg'.format(self.count))
        # self.count += 1
        # if self.count == 10:
            # raise Exception('Stop here')
        # raise Exception('Stop here')

        try:
            if self.transform:
                img = self.transform(img)
        except Exception as e:

            img = np.array(Image.open(os.path.join(self.img_dir, self.img_names[idx])).convert('RGB'))
            img = img_utils.get_bbox(img, self.bboxes[idx], padding=0)
            img = Image.fromarray(img)
            img.save('img_no_padding_{}.jpg'.format(self.img_names[idx]))

            img = np.array(Image.open(os.path.join(self.img_dir, self.img_names[idx])).convert('RGB'))
            img = img_utils.get_bbox(img, self.bboxes[idx], padding=self.padding)
            img = Image.fromarray(img)
            img.save('img_padding_{}.jpg'.format(self.img_names[idx]))

            print(e)
            print('Error with image:', self.img_names[idx])
            print('Error with bbox:', self.bboxes[idx], img.size)
            raise e
        return img

@torch.no_grad()
def extract_features(
        model_name:str, 
        img_dir,
        img_list,
        bounding_boxes,
        padding,
        batch_size:int=1,
        device:torch.device=torch.device("cuda")
        )-> np.array: 

    model = torch.hub.load('facebookresearch/dinov2', model_name)
    model = model.to(device)
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    print('Loaded', model_name)


    dataset = CropDataset(img_dir, img_list, bounding_boxes, padding=padding, 
                          transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=1, pin_memory=True)

    all_feats = []
    for batch in loader:
        images = batch.to(device)
        feats = model(images)
        # all_feats.append(feats.cpu())
        all_feats.append(feats)

    # for batch_idx, (images, _) in enumerate(tqdm(loader)):
    #     images = images.to(device)
    #     feats = model(images)
    #     all_feats.append(feats.cpu())
    
    all_feats = torch.cat(all_feats, dim=0)
    # all_feats = all_feats.numpy()

    return all_feats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, required=True, choices=[
        'dinov2_vitb14', 'dinov2_vitl14'
    ])
    parser.add_argument('--pickle', '-p', type=str, required=True)

    args = parser.parse_args()
    with open(args.pickle, 'rb') as f:
        data = pickle.load(f)

    embeddings = extract_features(
        args.model, data['img_dir'], data['img_list'], data['bboxes'],
        padding=data['padding'], device=torch.device(data['device'])
    )

    with open(args.pickle+'.out', 'wb') as f:
        data['embeddings'] = embeddings
        pickle.dump(data, f)

