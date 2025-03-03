from io import BytesIO
import os
import pickle
from collections import defaultdict

import numpy as np

import faiss

def calculate_avg_precision(gt, pred):
    """
    Compute the Average Precision (AP) based on the given formula.
    
    Args:
    gt (list): Ground truth list of relevant items (documents).
    pred (list): Predicted list of items (documents), sorted by relevance score.
    
    Returns:
    float: The Average Precision score.
    """
    raise Exception("This function is implemented in metrics.py")
    if not gt:
        return 0.0
    
    score = 0.0
    num_hits = 0.0
    relevant_items = set(gt)  # To check relevance
    # total_relevant = len(relevant_items)  # Number of relevant items
    
    for i, p in enumerate(pred):
        if p in relevant_items:
            num_hits += 1
            precision_at_i = num_hits / (i + 1)  # Precision@i
            score += precision_at_i  # Add Precision@i to the score
    
    # Normalize by the total number of relevant items
    # return score / total_relevant if total_relevant > 0 else 0.0
    return score / len(gt) 


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

    for idx, seg_emb in enumerate(imgs):
        seg_emb_file = os.path.join(seg_embeddings_dir, seg_emb)

        try:
            with open(seg_emb_file, "rb") as f:
                dictionary = pickle.load(f)
        except:
            print('Error loading', seg_emb)
            continue
    
        dictionary["average_embeddings"] = np.load(BytesIO(dictionary["average_embeddings"]))['a']
        average_embeddings = dictionary["average_embeddings"]
        segment_ids = dictionary["segment_ids"]

        if len(segment_ids) == 0:
            print('what', idx)
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

    average_embeddings_across_images = np.vstack(average_embeddings_across_images)
    
    return average_embeddings_across_images, segment_ids_across_images, img_to_vec_list, vec_to_img

def create_new_pickle(seg_embed_dir, pickle_out_path='coco-2014-val-clip-embeds-fast-2.pkl'):
    if os.path.exists(pickle_out_path):
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

def get_query_invlists_info(index, query):
    """
    Get information about which inverted lists a query falls into and their sizes.
    
    Args:
    - index: A trained FAISS IVF-PQ index
    - query: A numpy array of shape (d,) where d is the dimension of the index
    
    Returns:
    - A list of tuples, each containing (invlist_id, invlist_size)
    """
    if not isinstance(index, faiss.IndexIVFPQ) and not isinstance(index, faiss.IndexIVFFlat):
        raise ValueError("The provided index must be an instance of faiss.IndexIVFPQ")
    
    # Ensure the query is a 2D array
    if query.ndim == 1:
        query = query.reshape(1, -1)
    
    # Compute the distances to the centroids
    # print("Searching with nprobes=", index.nprobe)
    _, assigned_ids = index.quantizer.search(query, index.nprobe)

    # print("assigned_ids", assigned_ids)
    # print("assigned ids[0]", assigned_ids[0])
    
    # result = []
    # for invlist_id in assigned_ids[0]:
    #     invlist = index.invlists.list_size(int(invlist_id))
    #     result.append((int(invlist_id), invlist))
    #     print("result", result)

    # print("this is results", result)
    # raise Exception("huh")


    actual_results = []
    for batch in range(0, len(assigned_ids)):
        for invlist_id in assigned_ids[batch]:
            invlist = index.invlists.list_size(int(invlist_id))
            actual_results.append((int(invlist_id), invlist))

    print("this is actual results", actual_results)

    
    # return result
    return actual_results

def get_invlist(invlists, l):
    """ returns the inverted lists content. """
    ls = invlists.list_size(l)
    list_ids = np.zeros(ls, dtype='int64')
    x = invlists.get_ids(l)
    faiss.memcpy(faiss.swig_ptr(list_ids), x, list_ids.nbytes)
    invlists.release_ids(l, x)
    x = invlists.get_codes(l)
    list_codes = np.zeros((ls, invlists.code_size), dtype='uint8')
    faiss.memcpy(faiss.swig_ptr(list_codes), x, list_codes.nbytes)
    invlists.release_codes(l, x)    
    return list_ids, list_codes

# def get_query_invlists_info(index, query):
#     """
#     Get information about which inverted lists a query falls into and their sizes.
    
#     Args:
#     - index: A trained FAISS IVF-PQ index
#     - query: A numpy array of shape (d,) where d is the dimension of the index
    
#     Returns:
#     - A list of tuples, each containing (invlist_id, invlist_size)
#     """
#     if not isinstance(index, faiss.IndexIVFPQ):
#         raise ValueError("The provided index must be an instance of faiss.IndexIVFPQ")
    
#     # Ensure the query is a 2D array
#     if query.ndim == 1:
#         query = query.reshape(1, -1)
    
#     # Compute the distances to the centroids
#     _, assigned_ids = index.quantizer.search(query, index.nprobe)
    
#     result = []
#     for invlist_id in assigned_ids[0]:
#         invlist = index.invlists.list_size(int(invlist_id))
#         result.append((int(invlist_id), invlist))
    
#     return result