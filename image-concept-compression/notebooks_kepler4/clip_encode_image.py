import argparse
import pickle

import torch
import numpy as np
import clip
from PIL import Image

def encode_img(img_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)

    return image_features.cpu().numpy()

def save_to_pickle(feature_dict, path):
    with open(path, "wb") as f:
        pickle.dump(feature_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    feature_dict = encode_img(args.img_path)
    save_to_pickle(feature_dict, args.output_path)