from io import BytesIO
import os
import pickle
from collections import defaultdict

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
    n_empty_segment_ids = 0

    for idx, seg_emb in enumerate(imgs):
        seg_emb_file = os.path.join(seg_embeddings_dir, seg_emb)

        try:
            with open(seg_emb_file, "rb") as f:
                dictionary = pickle.load(f)
        except Exception as e:
            print('Error loading embeddings', seg_emb, e)
            continue
    
        dictionary["average_embeddings"] = np.load(BytesIO(dictionary["average_embeddings"]))['a']
        average_embeddings = dictionary["average_embeddings"]
        segment_ids = dictionary["segment_ids"]
        if len(segment_ids) == 0:
            # print('Empty segment ids')
            n_empty_segment_ids += 1
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
        end_idx = start_idx + len(average_embeddings) # This is exclusive. 

        segment_id_idx = len(segment_ids_across_images)
        img_to_vec_list[img_name] = (start_idx, end_idx, segment_id_idx)
        for i in range(start_idx, end_idx):
            # vec_to_img.append(idx)
            vec_to_img.append(img_name)

        average_embeddings_across_images.append(average_embeddings)
        segment_ids_across_images.append(segment_ids)

        vector_idx += len(average_embeddings)
        # img_idx.append(idx)


    print(f'{n_empty_segment_ids} empty segment ids')
    print(len(average_embeddings_across_images))

    average_embeddings_across_images = np.vstack(average_embeddings_across_images)
    
    return average_embeddings_across_images, segment_ids_across_images, img_to_vec_list, vec_to_img

def create_new_pickle(seg_embed_dir, pickle_out_path=None):
    if pickle_out_path is None:
        seg_embed_dir_name = os.path.basename(seg_embed_dir)
        pickle_out_path = f'{seg_embed_dir_name}-fast.pkl'

    if pickle_out_path == '-fast.pkl':
        raise Exception("Cannot use -fast.pkl as pickle out path")

    if os.path.exists(pickle_out_path):
        print(f'Pickle file {pickle_out_path} already exists')
        with open(pickle_out_path, 'rb') as f:
            embed_dict = pickle.load(f)
        return embed_dict

    print(f'Creating new pickle file at {pickle_out_path}')
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

def create_grid_visualization(query_img, top_k_imgs, output_dir, query_index):
    from PIL import Image
    import math
    
    # Open images
    images = [Image.open(query_img)] + [Image.open(img) for img in top_k_imgs]
    
    # Calculate grid size
    n_images = len(images)
    cols = math.ceil(math.sqrt(n_images))
    rows = math.ceil(n_images / cols)

    # Resize images
    max_size = (300, 300)
    for i, img in enumerate(images):
        images[i] = img.copy()  # Create a copy to avoid modifying the original
        images[i].thumbnail(max_size, Image.LANCZOS)

    # Create blank image
    grid_size = (cols * max_size[0], rows * max_size[1])
    grid_img = Image.new('RGB', grid_size, color='white')

    # Paste images into grid
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        grid_img.paste(img, (col * max_size[0], row * max_size[1]))

    # Save grid image
    os.makedirs(output_dir, exist_ok=True)
    grid_img.save(os.path.join(output_dir, f"visualization_{query_index}.jpg"), quality=95)