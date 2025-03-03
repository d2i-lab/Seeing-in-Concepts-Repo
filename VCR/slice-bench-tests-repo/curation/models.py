import os
import subprocess
import string
import random
import pickle
import shutil

import img_utils

from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import clip

# Implement differnet methods for CLIP and DINO

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
        # if self.count == 50:
        #     raise Exception('Stop here')

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

def load_clip_model(name, device):
    model, preprocess = clip.load(name, device)
    return model, preprocess

def load_dino_model(name, device):
    pass
    # model = torch.hub.load('facebookresearch/dino:main', name)
    # model = model.to(device)
    # model.eval()
    # preprocess = transforms.Compose([
    #     transforms.Resize(256, interpolation=3),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    # return model, preprocess

def get_embeddings(model):
    # send parameters. set random file name. --> subprocess.run(['python', 'get_embeddings.py', model])
    # check if subprocess succeeded
    # load embeddings from output file
    # remove output file
    pass


def get_embeddings_dino(model, img_dir, img_list, bounding_boxes, batch_size=1,
                        padding=0, device='cuda'):
    # raise NotImplemented('DINO embeddings not implemented yet')
    N = 8
    res = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=N))
    
    input_dict = {
        'img_dir': img_dir,
        'img_list': img_list,
        'bboxes': bounding_boxes,
        'padding': padding,
        # 'batch_size': batch_size,
        'device': device
    }

    pickle_name = f'input_{res}.pkl'
    with open(pickle_name, 'wb') as f:
        pickle.dump(input_dict, f)
    
    mask_clip_env = '/data/home/jxu680/miniconda3/envs/maskclip/bin/python3'
    mask_clip_env += ' dino_process.py'
    mask_clip_env += ' --model {}'.format(model)
    mask_clip_env += ' --pickle {}'.format(pickle_name)

    process = subprocess.run(mask_clip_env, shell=True, stderr=subprocess.PIPE)
    # error = process.stderr
    
    # if error is not None:
    #     raise Exception('Error with subprocess:', error)

    if not os.path.exists(pickle_name+'.out'):
        raise Exception('Output file not found:', pickle_name+'.out')

    with open(pickle_name+'.out', 'rb') as f:
        embeddings = pickle.load(f)['embeddings']

    os.remove(pickle_name)
    os.remove(pickle_name+'.out')

    return embeddings



def get_embeddings_clip(model, img_dir, img_list, bounding_boxes, batch_size=1,
                        padding=0, device='cuda'):
    model, preprocess = load_clip_model(model, device)
    transform = transforms.Compose([
        preprocess
    ])
    dataset = CropDataset(img_dir, img_list, bounding_boxes, padding=padding, 
                          transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=1, pin_memory=True)
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            embeddings.append(model.encode_image(batch))

    embeddings = torch.cat(embeddings)
    return embeddings