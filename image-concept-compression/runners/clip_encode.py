import argparse
import pickle

import torch
import numpy as np
import clip
from PIL import Image

def encode_text(word_list):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)
    text = clip.tokenize(word_list).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)

    # Normalize the feature to have magnitude of 1
    # text_features /= text_features.norm(dim=-1, keepdim=True)
    # for i in range(len(text_features)):
        # text_features[i] /= text_features[i].norm()

    feature_dict = {}
    for i, word in enumerate(word_list):
        feature_dict[word] = text_features[i].cpu().numpy() 

    # Normalize the feature to have magnitude of 1
    for word in feature_dict:
        feature_dict[word] /= np.linalg.norm(feature_dict[word], keepdims=True)

    return feature_dict

def save_to_pickle(feature_dict, path):
    with open(path, "wb") as f:
        pickle.dump(feature_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--word_list", type=str, nargs="+", required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    feature_dict = encode_text(args.word_list)
    save_to_pickle(feature_dict, args.output_path)