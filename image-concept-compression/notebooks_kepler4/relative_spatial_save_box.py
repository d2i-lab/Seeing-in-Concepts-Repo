import os
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
from functools import partial
import imagesize

import random
from tqdm import tqdm
import json

import pycocotools

from io import BytesIO
import os
import pickle
import numpy as np


def load_embeddings(fast_path_dir, seg_embeddings_dir):
    print('Running get embeddings call')
    if fast_path_dir and os.path.exists(fast_path_dir):
        with open(fast_path_dir, 'rb') as f:
            embed_dict = pickle.load(f)

        print('Fast path')
        embeddings = embed_dict['average_embeddings']
        segment_ids_across_images = embed_dict['segment_ids']
        return embeddings, segment_ids_across_images

    print('No fast path sir')
    average_embeddings_across_images = []
    segment_ids_across_images = [] 
    # img_idx = []
    imgs = sorted(os.listdir(seg_embeddings_dir))
    img_to_vec_list = {}
    vector_idx = 0
    vec_to_img = [] # Maps vector index to image index

    print(len(imgs))

    for idx, seg_emb in enumerate(imgs):
        seg_emb_file = os.path.join(seg_embeddings_dir, seg_emb)

        try:
            with open(seg_emb_file, "rb") as f:
                dictionary = pickle.load(f)
        except:
            print('Error loading embeddings', seg_emb)
            continue
    
        dictionary["average_embeddings"] = np.load(BytesIO(dictionary["average_embeddings"]))['a']
        average_embeddings = dictionary["average_embeddings"]
        segment_ids = dictionary["segment_ids"]
        if len(segment_ids) == 0:
            print('what')
            continue

        if segment_ids[0] == 0:
            average_embeddings = average_embeddings[1:]
            segment_ids = segment_ids[1:]

        if len(average_embeddings) == 0:
            continue

        # Have a dictionary of image names pointing to the start and end index of the embeddings
        img_name = seg_emb.split('.pkl')[0]
        start_idx = vector_idx
        # end_idx = start_idx + len(average_embeddings) - 1
        end_idx = start_idx + len(average_embeddings)

        segment_id_idx = len(segment_ids_across_images)
        img_to_vec_list[img_name] = (start_idx, end_idx, segment_id_idx)
        for i in range(start_idx, end_idx):
            # vec_to_img.append(idx)
            vec_to_img.append(img_name)

        average_embeddings_across_images.append(average_embeddings)
        segment_ids_across_images.append(segment_ids)

        vector_idx += len(average_embeddings)
        # img_idx.append(idx)


    average_embeddings_across_images = np.vstack(average_embeddings_across_images)
    
    return average_embeddings_across_images, segment_ids_across_images, img_to_vec_list, vec_to_img

def create_new_pickle(seg_embed_dir, pickle_out_path='asdf.pkl'):
    if os.path.exists(pickle_out_path):
        print('fast pickle exists loading...')
        with open(pickle_out_path, 'rb') as f:
            embed_dict = pickle.load(f)

        return embed_dict

    embeds, seg_ids, img_to_vec_list, vec_to_img = load_embeddings(None, seg_embed_dir)
    out_dict = {
        'average_embeddings': embeds, 
        'segment_ids': seg_ids, 
        'img_to_vec_list': img_to_vec_list,
        'vec_to_img': vec_to_img,
    }
    with open(pickle_out_path, "wb") as f:
        pickle.dump(out_dict, f)

    return out_dict

# img_dir = '/data/users/jie/data-slicing/COCO/train2017/'
# seg_dir = '/data/users/jie/data-slicing/COCO/sam/train2017_vit_l/'
# embed_dir = '/data/users/jie/data-slicing/COCO/embeds/train2017_vitl_fixed_maskclip'
# new_pickle_path = 'train_2017_vitl_fixed_maskclip.pkl'

img_dir = '/data/users/jie/data-slicing/bdd100k/images/100k/train/'
seg_dir = '/data/users/jie/data-slicing/bdd100k/sam/vit_l/'
embed_dir = '/data/users/jie/data-slicing/bdd100k/embeds/sam_maskclip/'
new_pickle_path = 'bdd100k_sam_maskclip.pkl'




# embed_dir = '/data/users/jie/data-slicing/COCO/embeds/train2017_fixed_clip_only'
# new_pickle_path = 'train_2017_fixed_clip_only.pkl'

# BDD100K
# img_dir = '/data/users/jie/data-slicing/bdd100k/images/100k/train/'
# seg_dir = '/data/users/jie/data-slicing/bdd100k/sam/vit_l/'
# embed_dir = '/data/users/jie/data-slicing/bdd100k/embeds/sam_maskclip/'
# new_pickle_path = 'bdd100k_sam_maskclip.pkl'

# img_dir = '/data/users/jie/data-slicing/LVIS/train2017'
# seg_dir = '/data/users/jie/data-slicing/LVIS/segmentation'
# embed_dir = '/data/users/jie/data-slicing/LVIS/embeds/train2017_maskclip/'
# new_pickle_path = 'lvis_train2017_maskclip.pkl'
embed_dict = create_new_pickle(embed_dir, new_pickle_path)

random.seed(43)

average_embeddings = embed_dict['average_embeddings']
img_to_vec_list = embed_dict['img_to_vec_list']
vec_to_img = embed_dict['vec_to_img']
img_to_vec = embed_dict['img_to_vec_list']
seg_ids = embed_dict['segment_ids']
embeds = average_embeddings

# n_imgs = 40_000
n_imgs = len(img_to_vec)
print("Number of images: ", n_imgs)
imgs = sorted(img_to_vec.keys())
idxs = random.sample(range(len(imgs)), n_imgs)

chosen_imgs = [(i, imgs[i]) for i in idxs]
data = [(i, img, img_to_vec[img]) for (i, img) in chosen_imgs]

def extract_segments(json_file):
    with open(json_file, 'r') as f:
        json_dict = json.load(f)
    segments = [seg['segmentation'] for seg in json_dict]
    decoded_segments = []
    for seg in segments:
        decoded_segments.append(pycocotools.mask.decode(seg))

    return decoded_segments

def load_masks(seg_path: str):
    with open(seg_path, 'r') as f:
        sam_output = json.load(f)
    
    masks = []
    for segment in sam_output:
        rle = segment['segmentation']
        mask = pycocotools.mask.decode(rle)
        masks.append(mask)
    
    return masks

def get_box(mask):
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return x_min, y_min, x_max, y_max


size_threshold = 0.01

def process_chunk(chunk, seg_ids, embeds, img_dir, seg_dir, size_threshold):
    results = []
    for (idx, img, vec_info) in chunk:
        start, end, seg_id_idx = vec_info
        seg_id = seg_ids[seg_id_idx]
        seg_emb = embeds[start:end]

        # Fix off-by-one seg_id
        seg_id = [i - 1 for i in seg_id]
        seg_id_to_emb = {i: emb for i, emb in zip(seg_id, seg_emb)}
        seg_file = os.path.join(seg_dir, img + '.json')
        if not os.path.exists(seg_file):
            continue

        decoded_segments = load_masks(seg_file)
        id_segments = list(enumerate(decoded_segments))

        # img_rgb = Image.open(os.path.join(img_dir, img)).convert('RGB')
        img_width, img_height = imagesize.get(os.path.join(img_dir, img))
        
        # img_width, img_height = img_rgb.size
        img_area = img_width * img_height

        for i, (curr_seg_id, segment) in enumerate(id_segments):
            try:
                x_min, y_min, x_max, y_max = get_box(segment)
                box_area = (x_max - x_min) * (y_max - y_min)
                if box_area < img_area * size_threshold:
                    continue

                b_box = (x_min / img_width, y_min / img_height, x_max / img_width, y_max / img_height)
                # b_embed = seg_emb[i]
                b_embed = seg_id_to_emb[curr_seg_id]
                results.append((b_box, b_embed, curr_seg_id, img))
            except Exception as e:
                print('Error processing', img, i)
                print(e)

    return results

def get_box_embed_data_parallel(data, seg_ids, embeds, img_dir, seg_dir, size_threshold=0.005, num_processes=None):
    if num_processes is None:
        num_processes = mp.cpu_count()

    # Split the data into chunks
    chunk_size = len(data) // num_processes
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    # Create a pool of worker processes
    with mp.Pool(num_processes) as pool:
        # Create a partial function with fixed arguments
        process_chunk_partial = partial(
            process_chunk,
            seg_ids=seg_ids,
            embeds=embeds,
            img_dir=img_dir,
            seg_dir=seg_dir,
            size_threshold=size_threshold
        )

        # Use tqdm to show progress
        results = list(tqdm(pool.imap(process_chunk_partial, chunks), total=len(chunks), desc="Processing chunks"))

    # Flatten the results
    flattened_results = [item for sublist in results for item in sublist]

    # Separate the results into individual lists
    boxes, box_embeds, box_seg_ids, box_img = zip(*flattened_results)

    return list(boxes), list(box_embeds), list(box_seg_ids), list(box_img)

# Usage example (to be run in a Jupyter notebook cell):

box_pickle_name = new_pickle_path.split('.pkl')[0] + '_boxes.pkl'
boxes, box_embeds, box_seg_ids, box_img = None, None, None, None

if os.path.exists(box_pickle_name):
    print("Load from file")
    with open(box_pickle_name, 'rb') as f:
        box_data = pickle.load(f)

    boxes = box_data['boxes']
    box_embeds = box_data['embeds']
    box_seg_ids = box_data['seg_ids']
    box_img = box_data['img'] 
    # boxes, box_embeds, box_seg_ids, box_img = box_data
else:
    print("Load manual (long ):)")
    boxes, box_embeds, box_seg_ids, box_img = get_box_embed_data_parallel(
        data, seg_ids, embeds, img_dir, seg_dir, num_processes=32
    )

    box_dict = {
        'boxes': boxes,
        'embeds': box_embeds,
        'seg_ids': box_seg_ids,
        'img': box_img
    }

    if not os.path.exists(box_pickle_name):
        with open(box_pickle_name, 'wb') as f:
            pickle.dump(box_dict, f)