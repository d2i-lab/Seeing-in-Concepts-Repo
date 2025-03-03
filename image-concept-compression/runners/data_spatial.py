import os
import json
import pickle
from io import BytesIO
from functools import partial
import multiprocessing as mp
import hashlib

import bitpacking
from custom_pq import CustomProductQuantizer

import faiss
from PIL import Image
from tqdm import tqdm
import numpy as np
import imagesize
import pycocotools
from python_prtree import PRTree2D

def pack_data(img_id, n):
    if img_id < 0 or img_id >= (1 << 22):
        raise ValueError("img_id must be between 0 and 2^22 - 1")
    if n < 0 or n >= (1 << 10):
        raise ValueError("coarse_idx must be between 0 and 2^10 - 1")
    return np.uint32((img_id << 10) | n)

def unpack_data(packed_data):
    img_id = (packed_data >> 10) & 0x3FFFFF  # Changed to 0x3FFFFF (22 bits)
    coarse_idx = packed_data & 0x3FF
    return img_id, coarse_idx

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
    imgs = sorted(os.listdir(seg_embeddings_dir))
    img_to_vec_list = {}
    vector_idx = 0
    vec_to_img = [] # Maps vector index to image index

    print('Len of imgs:', len(imgs))

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
            vec_to_img.append(img_name)

        average_embeddings_across_images.append(average_embeddings)
        segment_ids_across_images.append(segment_ids)

        vector_idx += len(average_embeddings)


    average_embeddings_across_images = np.vstack(average_embeddings_across_images)
    return average_embeddings_across_images, segment_ids_across_images, img_to_vec_list, vec_to_img

def create_new_pickle(seg_embed_dir, pickle_out_path='xxxxx'):
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
    '''
    Get the bounding box of the mask
    '''
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return x_min, y_min, x_max, y_max


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
        img_area = img_width * img_height

        n_errs = 0
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
                n_errs += 1

        print('Num errs:', n_errs)

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

def get_embed_dict(embed_dir, pickle_path):
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            embed_dict = pickle.load(f)
        return embed_dict

    embed_dict = create_new_pickle(embed_dir, pickle_path)
    return embed_dict

def get_box_embed_data_wrapper(
    img_dir, seg_dir, embed_dir, pickle_path
):
    embed_dict = get_embed_dict(embed_dir, pickle_path)
    average_embeddings = embed_dict['average_embeddings']
    seg_ids = embed_dict['segment_ids']
    img_to_vecs = embed_dict['img_to_vec_list']
    vec_to_img = embed_dict['vec_to_img']
    embed_dict['dataset_hash'] = hashlib.md5(embed_dir.encode()).hexdigest()[:7]
    # For each image, we get their list of vectors (start, end, seg_id_idx)
    data = [
        (i, img, img_to_vecs[img]) for i, img in enumerate(vec_to_img)
    ]

    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            box_data = pickle.load(f)
        boxes = box_data['boxes']
        box_embeds = box_data['embeds']
        box_seg_ids = box_data['seg_ids']
        box_imgs = box_data['img']
    else:
        boxes, box_embeds, box_seg_ids, box_imgs = get_box_embed_data_parallel(
        data, seg_ids, average_embeddings, img_dir, seg_dir, 0.005)
        with open(pickle_path, 'wb') as f:
            pickle.dump({
                'boxes': boxes,
                'embeds': box_embeds,
                'seg_ids': box_seg_ids,
                'img': box_imgs
            }, f)


    return boxes, box_embeds, box_seg_ids, box_imgs, data, embed_dict

def get_indices(dim, k_coarse, m, cluster_bits, n_probes, 
                boxes, box_embeds, box_imgs,
                use_custom_pq=False, train_sample_size=None,
                embed_dict=None,
                build_ivf_flat=False):

    if embed_dict is None:
        raise ValueError('Embed dict is required')
    if 'dataset_hash' not in embed_dict:
        raise ValueError('Dataset hash is required')

    all_images = sorted(embed_dict['img_to_vec_list'])
    np.random.seed(int(embed_dict['dataset_hash'], 16))
    embeddings = box_embeds

    # all_images = sorted(embed_dict['img_to_vec_list'])
    train_indices = np.arange(embeddings.shape[0])
    if train_sample_size is not None:
        print('Training on subset', train_sample_size / embeddings.shape[0])
        train_indices = np.random.choice(embeddings.shape[0], train_sample_size, replace=False)
    training_embeds = embeddings[train_indices]

    # Build IVFPQ index
    print('Building IVFPQ index')
    n_cells = k_coarse
    nbits_per_idx = cluster_bits
    quantizer = faiss.IndexFlatL2(dim)
    index_pq = faiss.IndexIVFPQ(quantizer, dim, n_cells, m, nbits_per_idx)
    index_pq.nprobe = n_probes
    index_pq.train(training_embeds)
    index_pq.add(embeddings)

    # Get coarse centroids
    coarse_centroids = index_pq.quantizer.reconstruct_n(0, index_pq.nlist)
    kmeans = faiss.Kmeans(index_pq.d, index_pq.nlist, niter=index_pq.cp.niter)
    kmeans.train(coarse_centroids)
    print('Kmeans trained (hax)', len(coarse_centroids))

    # Build flat index (not cached)
    print('Building Flat index')
    index_flat_cpu = faiss.IndexFlatL2(dim)

    if not build_ivf_flat:
        index_flat_cpu.add(embeddings)

    index_ivf_flat_cpu = None
    if build_ivf_flat:
        print('Building IVF-Flat index')
        index_ivf_flat_cpu = faiss.IndexIVFFlat(index_flat_cpu, dim, index_pq.nlist)
        index_ivf_flat_cpu.nprobe = n_probes
        index_ivf_flat_cpu.train(coarse_centroids)
        index_ivf_flat_cpu.add(embeddings)
    
    # Build our custom PQ
    d, cluster_ids = kmeans.index.search(training_embeds, 1)
    centroids = kmeans.centroids[cluster_ids].squeeze()
    train_residuals = training_embeds - centroids

    pqq = None
    if use_custom_pq:
        pqq = CustomProductQuantizer(dim, m, cluster_bits)
    else:
        pqq = faiss.ProductQuantizer(dim, m, cluster_bits)
        
    pqq.train(train_residuals)
    d, cluster_ids = kmeans.index.search(embeddings, 1)
    centroids = kmeans.centroids[cluster_ids].squeeze()
    box_codes = pqq.compute_codes(embeddings - centroids)

    packd = bitpacking.NumpyBitpackedDict()
    coarse_lists = [] # Store info per centroid
    for i in range(k_coarse):
        coarse_lists.append({
            'codes': [],
            'boxes': [],
            'imgs': [],
            'offsets': [], # Offset in the packd list
        })

    img_to_idx = {img: i for i, img in enumerate(all_images)}
    for i in tqdm(range(len(embeddings))):
        coarse_idx = cluster_ids[i][0]
        coarse_lists[coarse_idx]['codes'].append(box_codes[i])
        coarse_lists[coarse_idx]['boxes'].append(boxes[i])
        coarse_lists[coarse_idx]['imgs'].append(box_imgs[i])

        img_idx = img_to_idx[box_imgs[i]]
        if (img_idx, coarse_idx) not in packd:
            coarse_lists[coarse_idx]['offsets'].append(0)
            packd[(img_idx, coarse_idx)] = [box_codes[i]]
        else:
            next_offset = len(packd[(img_idx, coarse_idx)])
            coarse_lists[coarse_idx]['offsets'].append(next_offset)
            packd[(img_idx, coarse_idx)].append(box_codes[i])

    return index_pq, index_flat_cpu, index_ivf_flat_cpu, packd, all_images, pqq, kmeans, coarse_lists

def build_trees(k_coarse, coarse_lists, embed_dict,
                boxes):
    trees_ours = []
    packed_to_idx = {}
    all_images = sorted(embed_dict['img_to_vec_list'])
    img_to_idx = {img: i for i, img in enumerate(all_images)}
    for concept_idx in range(k_coarse):
        tree_ids = []
        concept_data = coarse_lists[concept_idx]
        offsets = concept_data['offsets']
        for j in range(len(concept_data['boxes'])):
            img_idx = img_to_idx[concept_data['img'][j]]
            offset = offsets[j]
            packed_to_idx[pack_data(img_idx, offset), concept_idx] = offset

        tree_ids = np.array(tree_ids)
        boxes_arr = np.array(concept_data['boxes']).astype(np.float16)
        tree = PRTree2D(tree_ids, boxes_arr)
        trees_ours.append(tree)

    # Construct boxes without coarse separation
    flat_box_ids = []
    flat_boxes = []
    # These flat box ids will point to box_embeds
    # since they are directly related
    for i in range(len(boxes)):
        flat_box_ids.append(i)
        flat_boxes.append(boxes[i])

    flat_boxes = np.array(flat_boxes).astype(np.float16)
    flat_tree = PRTree2D(flat_box_ids, flat_boxes)

    return trees_ours, flat_tree




